/**
 * FootprintManager - Simple explicit footprint loading
 *
 * DESIGN:
 * - Footprints load ONLY when loadFootprints() is called
 * - No camera listeners, no automatic updates
 * - Loaded footprints are static snapshots
 * - Reload replaces all previous footprints
 */

import * as Cesium from "cesium";
import { getInstrumentCesiumColor, type InstrumentId } from "../config/instrumentRegistry";
import { normalizeLonForMap } from "./coordinates";

export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CTX_MOSAIC" | "HIRISE_DTM" | "CRISM_TRR3";

export interface LoadResult {
  instrument: InstrumentType;
  count: number;
  truncated: boolean;
  total: number;
}

export interface FootprintFeature {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  properties: { product_id: string; [key: string]: any };
  geometry: {
    type: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    coordinates: any;
  };
}

interface FootprintResponse {
  features: FootprintFeature[];
  metadata: {
    truncated: boolean;
    returned: number;
    total_estimate: number;
  };
}

export interface FootprintManagerConfig {
  viewer: Cesium.Viewer;
  ellipsoid: Cesium.Ellipsoid;
  onLoadStart?: (instrument: InstrumentType) => void;
  onLoadEnd?: (instrument: InstrumentType, result: LoadResult) => void;
  onError?: (instrument: InstrumentType, error: Error) => void;
  highResOnly?: boolean;
}

export interface ProductBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface FeatureMetadata {
  instrument: InstrumentType;
  productId: string;
  properties: Record<string, unknown>;
  bounds: ProductBounds;
  outlineIndex: number;
}

// Using shared normalizeLonForMap from coordinates.ts
const normalizeLon = normalizeLonForMap;

export class FootprintManager {
  private viewer: Cesium.Viewer;
  private ellipsoid: Cesium.Ellipsoid;
  private instanceIds: Map<InstrumentType, Set<string>> = new Map();
  private features: Map<InstrumentType, FootprintFeature[]> = new Map();
  private abortControllers: Map<InstrumentType, AbortController> = new Map();
  private requestIds: Map<InstrumentType, number> = new Map(); // Track request IDs for idempotency
  private nextRequestId = 0;
  // In-flight promise deduplication: if loadFootprints is called while a previous load is pending, return the same promise
  private inFlightLoads: Map<InstrumentType, Promise<LoadResult | null>> = new Map();
  // Store the bbox that was used when loading each instrument's footprints
  private loadedBboxes: Map<InstrumentType, [number, number, number, number]> = new Map();

  private fillPrimitives: Map<InstrumentType, Cesium.Primitive> = new Map();
  private outlineCollections: Map<InstrumentType, Cesium.PolylineCollection> = new Map();

  private featureMetadata: Map<string, FeatureMetadata> = new Map();
  private featureVisibility: Map<string, boolean> = new Map();

  private hoverLabelEntity: Cesium.Entity | null = null;

  private onLoadStart?: (instrument: InstrumentType) => void;
  private onLoadEnd?: (instrument: InstrumentType, result: LoadResult) => void;
  private onError?: (instrument: InstrumentType, error: Error) => void;
  private _highResOnly: boolean = false;

  constructor(config: FootprintManagerConfig) {
    this.viewer = config.viewer;
    this.ellipsoid = config.ellipsoid;
    this.onLoadStart = config.onLoadStart;
    this.onLoadEnd = config.onLoadEnd;
    this.onError = config.onError;
    this._highResOnly = config.highResOnly ?? false;

    // Initialize empty collections for each instrument
    this.instanceIds.set("CRISM", new Set());
    this.instanceIds.set("HIRISE", new Set());
    this.instanceIds.set("SHARAD", new Set());
    this.instanceIds.set("SHARAD_HIGHRES", new Set());
    this.instanceIds.set("CTX", new Set());
    this.instanceIds.set("HIRISE_DTM", new Set());
    this.instanceIds.set("CRISM_TRR3", new Set());
    this.features.set("CRISM", []);
    this.features.set("HIRISE", []);
    this.features.set("SHARAD", []);
    this.features.set("SHARAD_HIGHRES", []);
    this.features.set("CTX", []);
    this.features.set("HIRISE_DTM", []);
    this.features.set("CRISM_TRR3", []);
  }

  /**
   * Get current viewport bounding box using Cesium's computeViewRectangle
   * Falls back to screen corner picking if that fails
   */
  private getViewportBbox(): [number, number, number, number] | null {
    const scene = this.viewer.scene;

    // Try to get the view rectangle directly from Cesium
    const viewRect = this.viewer.camera.computeViewRectangle(scene.globe.ellipsoid);

    if (viewRect) {
      const west = Cesium.Math.toDegrees(viewRect.west);
      const south = Cesium.Math.toDegrees(viewRect.south);
      const east = Cesium.Math.toDegrees(viewRect.east);
      const north = Cesium.Math.toDegrees(viewRect.north);

      // Cesium returns west > east when crossing antimeridian
      const crossesAntimeridian = west > east;

      // Calculate actual longitude span
      const lonSpan = crossesAntimeridian
        ? (180 - west) + (180 + east)  // e.g., west=150, east=-160 -> 30 + 20 = 50°
        : (east - west);

      const latSpan = north - south;

      // Sanity check - if spans are too large, fallback to corner picking
      if (lonSpan < 350 && latSpan < 170) {
        return [west, south, east, north];
      }
    }

    // Fallback: pick screen corners
    return this.getViewportBboxFromCorners();
  }

  /**
   * Fallback method: compute viewport bbox by picking screen corners
   */
  private getViewportBboxFromCorners(): [number, number, number, number] | null {
    const camera = this.viewer.camera;
    const canvas = this.viewer.scene.canvas;

    // Sample many points across the screen for better coverage
    const samplePoints: Cesium.Cartesian2[] = [];
    const gridSize = 5; // 5x5 grid

    for (let i = 0; i <= gridSize; i++) {
      for (let j = 0; j <= gridSize; j++) {
        samplePoints.push(new Cesium.Cartesian2(
          (canvas.width * i) / gridSize,
          (canvas.height * j) / gridSize
        ));
      }
    }

    const lons: number[] = [];
    const lats: number[] = [];

    for (const point of samplePoints) {
      const cartesian = camera.pickEllipsoid(point, this.ellipsoid);
      if (cartesian) {
        const carto = Cesium.Cartographic.fromCartesian(cartesian, this.ellipsoid);
        lons.push(Cesium.Math.toDegrees(carto.longitude));
        lats.push(Cesium.Math.toDegrees(carto.latitude));
      }
    }

    if (lons.length < 4) {
      // Not enough points picked - probably viewing full globe or off-planet
      return [-180, -90, 180, 90];
    }

    const south = Math.min(...lats);
    const north = Math.max(...lats);

    // Detect antimeridian crossing: if we have both high positive and low negative longitudes
    const hasHighPositive = lons.some(lon => lon > 90);
    const hasLowNegative = lons.some(lon => lon < -90);
    const crossesAntimeridian = hasHighPositive && hasLowNegative;

    let west: number;
    let east: number;

    if (crossesAntimeridian) {
      // Viewport crosses antimeridian - compute west/east correctly
      // West = minimum of positive values (left edge near +180)
      // East = maximum of negative values (right edge near -180)
      const positiveLons = lons.filter(lon => lon > 0);
      const negativeLons = lons.filter(lon => lon < 0);

      if (positiveLons.length > 0 && negativeLons.length > 0) {
        west = Math.min(...positiveLons);
        east = Math.max(...negativeLons);
      } else {
        // Fallback if something went wrong
        west = Math.min(...lons);
        east = Math.max(...lons);
      }
    } else {
      // Normal case: simple min/max
      west = Math.min(...lons);
      east = Math.max(...lons);
    }

    // If longitude span > 300 (and not antimeridian crossing), we're viewing most of the globe
    const span = crossesAntimeridian ? (180 - west) + (180 + east) : (east - west);
    if (span > 300) {
      west = -180;
      east = 180;
    }

    return [west, south, east, north];
  }

  /**
   * Load footprints for current viewport - EXPLICIT CALL ONLY
   * Each call clears previous footprints and loads fresh data.
   * Concurrent calls for the same instrument are deduplicated (returns same promise).
   */
  async loadFootprints(instrument: InstrumentType): Promise<LoadResult | null> {
    // Deduplicate: if a load is already in-flight for this instrument, return it
    const existing = this.inFlightLoads.get(instrument);
    if (existing) {
      return existing;
    }

    const promise = this._doLoadFootprints(instrument);
    this.inFlightLoads.set(instrument, promise);
    try {
      return await promise;
    } finally {
      this.inFlightLoads.delete(instrument);
    }
  }

  private async _doLoadFootprints(instrument: InstrumentType): Promise<LoadResult | null> {
    const bbox = this.getViewportBbox();
    if (!bbox) {

      return null;
    }


    // Generate unique request ID
    const requestId = ++this.nextRequestId;
    this.requestIds.set(instrument, requestId);

    // Cancel any pending request
    this.abortControllers.get(instrument)?.abort();
    const controller = new AbortController();
    this.abortControllers.set(instrument, controller);

    // Clear existing footprints FIRST (before async operations)
    this.clearFootprints(instrument);

    this.onLoadStart?.(instrument);

    try {
      const hrParam = this._highResOnly ? "&highres_only=true" : "";
      const url = `/api/footprints?instrument=${instrument}&bbox=${bbox.join(",")}&limit=2000${hrParam}`;
      const res = await fetch(url, { signal: controller.signal });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data: FootprintResponse = await res.json();

      // Check if this request is still the current one (idempotency check)
      if (this.requestIds.get(instrument) !== requestId) {
        return null;
      }


      // Store loaded bbox and render footprints
      this.loadedBboxes.set(instrument, bbox);
      this.features.set(instrument, data.features);
      await this.renderFeatures(instrument, data.features, requestId);

      const result: LoadResult = {
        instrument,
        count: data.features.length,
        truncated: data.metadata.truncated,
        total: data.metadata.total_estimate,
      };

      this.onLoadEnd?.(instrument, result);
      return result;

    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        return null;
      }
      console.error(`[FootprintManager] Error loading ${instrument}:`, err);
      this.onError?.(instrument, err instanceof Error ? err : new Error(String(err)));
      return null;
    } finally {
      // Only delete controller if it's still ours (prevents race condition)
      if (this.abortControllers.get(instrument) === controller) {
        this.abortControllers.delete(instrument);
      }
    }
  }

  /**
   * Clear all footprints for an instrument
   */
  clearFootprints(instrument: InstrumentType): void {
    // Guard: viewer may already be destroyed during error cascade
    const sceneAlive = this.viewer && !this.viewer.isDestroyed();

    const fill = this.fillPrimitives.get(instrument);
    if (fill && sceneAlive) {
      this.viewer.scene.primitives.remove(fill);
    }
    this.fillPrimitives.delete(instrument);

    const outlines = this.outlineCollections.get(instrument);
    if (outlines && sceneAlive) {
      this.viewer.scene.primitives.remove(outlines);
    }
    this.outlineCollections.delete(instrument);

    for (const [key, metadata] of this.featureMetadata) {
      if (metadata.instrument === instrument) {
        this.featureMetadata.delete(key);
        this.featureVisibility.delete(key);
      }
    }

    this.instanceIds.get(instrument)?.clear();
    this.features.set(instrument, []);
    if (sceneAlive) this.viewer.scene.requestRender();
  }

  /**
   * Get loaded features for an instrument
   */
  getFeatures(instrument: InstrumentType): FootprintFeature[] {
    return this.features.get(instrument) ?? [];
  }

  /**
   * Get the bbox that was used when loading an instrument's footprints.
   * Returns [west, south, east, north] or null if not loaded.
   */
  getLoadedBbox(instrument: InstrumentType): [number, number, number, number] | null {
    return this.loadedBboxes.get(instrument) ?? null;
  }

  /**
   * Get all loaded bboxes (for rendering on map).
   */
  getAllLoadedBboxes(): Map<InstrumentType, [number, number, number, number]> {
    return this.loadedBboxes;
  }

  private getFeatureBoundsFromGeometry(
    instrument: InstrumentType,
    geom: FootprintFeature["geometry"],
  ): ProductBounds | null {
    if (geom.type === "Polygon") {
      const ring = geom.coordinates?.[0] as [number, number][] | undefined;
      if (!ring || ring.length === 0) return null;

      let west = Infinity;
      let east = -Infinity;
      let south = Infinity;
      let north = -Infinity;

      for (const [lon, lat] of ring) {
        // Use raw lon — antimeridian polygons may use >180 (e.g. 180.1)
        if (lon < west) west = lon;
        if (lon > east) east = lon;
        if (lat < south) south = lat;
        if (lat > north) north = lat;
      }

      if (east - west > 180) return null;
      return { west, south, east, north };
    }

    if (geom.type === "LineString") {
      const coords = geom.coordinates as [number, number][] | undefined;
      if (!coords || coords.length < 2) return null;

      let west = Infinity;
      let east = -Infinity;
      let south = Infinity;
      let north = -Infinity;

      for (const [lon, lat] of coords) {
        const nlon = normalizeLon(lon);
        if (nlon < west) west = nlon;
        if (nlon > east) east = nlon;
        if (lat < south) south = lat;
        if (lat > north) north = lat;
      }

      return { west, south, east, north };
    }

    if (geom.type === "Point") {
      const coords = geom.coordinates as [number, number] | undefined;
      if (!coords || coords.length < 2) return null;
      const [lon, lat] = coords;
      if (lon === 0 && lat === 0) return null;

      const nlon = normalizeLon(lon);
      // HiRISE: narrow & tall strip; others: roughly square
      const halfW = instrument === "HIRISE" ? 0.03 : 0.07;
      const halfH = instrument === "HIRISE" ? 0.1 : 0.06;
      return {
        west: nlon - halfW,
        east: nlon + halfW,
        south: lat - halfH,
        north: lat + halfH,
      };
    }

    return null;
  }

  private async renderFeatures(
    instrument: InstrumentType,
    features: FootprintFeature[],
    requestId: number,
  ): Promise<void> {
    let ids = this.instanceIds.get(instrument);
    if (!ids) {
      ids = new Set<string>();
      this.instanceIds.set(instrument, ids);
    }

    const color = this.getColor(instrument);
    const geometryInstances: Cesium.GeometryInstance[] = [];
    const outlineCollection = new Cesium.PolylineCollection();
    const chunkSize = 200;

    for (let index = 0; index < features.length; index += chunkSize) {
      if (this.requestIds.get(instrument) !== requestId) {
        return;
      }

      const chunk = features.slice(index, index + chunkSize);
      for (const feature of chunk) {
        const productId = feature.properties?.product_id;
        const geom = feature.geometry;
        if (!productId || !geom) continue;

        const entityId = `${instrument}_FP_${productId}`;
        if (ids.has(entityId) || this.featureMetadata.has(entityId)) continue;

        const bounds = this.getFeatureBoundsFromGeometry(instrument, geom);
        if (!bounds) continue;

        let outlineIndex = -1;

        if (geom.type === "Polygon") {
          // Use actual polygon coordinates (content-fitted, not bbox)
          const ring = geom.coordinates?.[0] as [number, number][] | undefined;
          const positions = ring && ring.length >= 3
            ? Cesium.Cartesian3.fromDegreesArray(
                ring.flatMap(([lon, lat]) => [normalizeLon(lon), lat]),
                this.ellipsoid,
              )
            : Cesium.Cartesian3.fromDegreesArray(
                [bounds.west, bounds.south, bounds.west, bounds.north,
                 bounds.east, bounds.north, bounds.east, bounds.south],
                this.ellipsoid,
              );

          geometryInstances.push(
            new Cesium.GeometryInstance({
              geometry: new Cesium.PolygonGeometry({
                polygonHierarchy: new Cesium.PolygonHierarchy(positions),
              }),
              id: entityId,
              attributes: {
                color: Cesium.ColorGeometryInstanceAttribute.fromColor(color.withAlpha(0.12)),
                show: new Cesium.ShowGeometryInstanceAttribute(true),
              },
            }),
          );

          outlineIndex = outlineCollection.length;
          // Outline follows actual polygon shape
          const outlinePositions = ring && ring.length >= 3
            ? Cesium.Cartesian3.fromDegreesArray(
                [...ring.flatMap(([lon, lat]) => [normalizeLon(lon), lat])],
                this.ellipsoid,
              )
            : Cesium.Cartesian3.fromDegreesArray(
                [bounds.west, bounds.south, bounds.west, bounds.north,
                 bounds.east, bounds.north, bounds.east, bounds.south,
                 bounds.west, bounds.south],
                this.ellipsoid,
              );
          outlineCollection.add({
            positions: outlinePositions,
            width: 1.0,
            material: Cesium.Material.fromType("Color", { color }),
            id: entityId,
          });
        } else if (geom.type !== "LineString") {
          // Point or other types: use bbox rectangle
          geometryInstances.push(
            new Cesium.GeometryInstance({
              geometry: new Cesium.RectangleGeometry({
                rectangle: Cesium.Rectangle.fromDegrees(
                  bounds.west, bounds.south, bounds.east, bounds.north,
                ),
              }),
              id: entityId,
              attributes: {
                color: Cesium.ColorGeometryInstanceAttribute.fromColor(color.withAlpha(0.12)),
                show: new Cesium.ShowGeometryInstanceAttribute(true),
              },
            }),
          );

          outlineIndex = outlineCollection.length;
          outlineCollection.add({
            positions: Cesium.Cartesian3.fromDegreesArray(
              [bounds.west, bounds.south, bounds.west, bounds.north,
               bounds.east, bounds.north, bounds.east, bounds.south,
               bounds.west, bounds.south],
              this.ellipsoid,
            ),
            width: 1.0,
            material: Cesium.Material.fromType("Color", { color }),
            id: entityId,
          });
        } else {
          const coords = geom.coordinates as [number, number][];
          const positions: number[] = [];
          for (const [lon, lat] of coords) {
            positions.push(normalizeLon(lon), lat);
          }
          outlineIndex = outlineCollection.length;
          outlineCollection.add({
            positions: Cesium.Cartesian3.fromDegreesArray(positions, this.ellipsoid),
            width: 3.0,
            material: Cesium.Material.fromType("Color", { color }),
            id: entityId,
          });
        }

        const metadata: FeatureMetadata = {
          instrument,
          productId,
          properties: { ...feature.properties, instrument } as Record<string, unknown>,
          bounds,
          outlineIndex,
        };
        this.featureMetadata.set(entityId, metadata);
        this.featureVisibility.set(entityId, true);
        ids.add(entityId);
      }

      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }

    if (this.requestIds.get(instrument) !== requestId) {
      return;
    }

    if (geometryInstances.length > 0) {
      const primitive = new Cesium.Primitive({
        geometryInstances,
        appearance: new Cesium.PerInstanceColorAppearance({
          flat: true,
          translucent: true,
        }),
        asynchronous: true,
      });
      this.viewer.scene.primitives.add(primitive);
      this.fillPrimitives.set(instrument, primitive);
      void (async () => {
        while (this.requestIds.get(instrument) === requestId && !primitive.ready) {
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        }
        if (this.requestIds.get(instrument) !== requestId || !primitive.ready) return;
        for (const id of ids) {
          const attrs = primitive.getGeometryInstanceAttributes(id);
          if (!attrs) continue;
          attrs.show = Cesium.ShowGeometryInstanceAttribute.toValue(
            this.featureVisibility.get(id) ?? true,
          );
        }
        this.viewer.scene.requestRender();
      })();
    }

    this.viewer.scene.primitives.add(outlineCollection);
    this.outlineCollections.set(instrument, outlineCollection);
    this.viewer.scene.requestRender();
  }

  /**
   * Set high-res only filter flag
   */
  set highResOnly(value: boolean) {
    this._highResOnly = value;
  }

  get highResOnly(): boolean {
    return this._highResOnly;
  }

  /**
   * Set visibility of footprints
   */
  setVisible(instrument: InstrumentType, visible: boolean): void {
    const fill = this.fillPrimitives.get(instrument);
    if (fill) fill.show = visible;
    const outlines = this.outlineCollections.get(instrument);
    if (outlines) outlines.show = visible;

    if (!visible) {
      this.hideHoverLabel();
    }

    this.viewer.scene.requestRender();
  }

  /**
   * Set visibility of footprint fills only (keep outlines visible).
   * Used when CTX mosaic replaces individual footprint fills.
   */
  setFillVisible(instrument: InstrumentType, visible: boolean): void {
    const fill = this.fillPrimitives.get(instrument);
    if (fill) fill.show = visible;
    this.viewer.scene.requestRender();
  }

  getFeatureMetadata(entityId: string): FeatureMetadata | null {
    return this.featureMetadata.get(entityId) ?? null;
  }

  getFeatureBounds(entityId: string): ProductBounds | null {
    const metadata = this.featureMetadata.get(entityId);
    return metadata ? metadata.bounds : null;
  }

  /**
   * Find all footprint features whose bounds contain the given lat/lon.
   * Returns matches sorted by area (smallest first).
   */
  getFeaturesAtPosition(lat: number, lon: number): FeatureMetadata[] {
    const results: { meta: FeatureMetadata; area: number }[] = [];
    for (const meta of this.featureMetadata.values()) {
      const b = meta.bounds;
      if (lat >= b.south && lat <= b.north && lon >= b.west && lon <= b.east) {
        results.push({ meta, area: (b.east - b.west) * (b.north - b.south) });
      }
    }
    results.sort((a, b) => a.area - b.area);
    return results.map((r) => r.meta);
  }

  setFeatureVisible(instrument: InstrumentType, productId: string, visible: boolean): void {
    const id = `${instrument}_FP_${productId}`;
    this.featureVisibility.set(id, visible);

    const primitive = this.fillPrimitives.get(instrument);
    if (primitive?.ready) {
      const attrs = primitive.getGeometryInstanceAttributes(id);
      if (attrs) {
        attrs.show = Cesium.ShowGeometryInstanceAttribute.toValue(visible);
      }
    }

    const metadata = this.featureMetadata.get(id);
    if (metadata) {
      const outlines = this.outlineCollections.get(instrument);
      if (outlines) {
        const polyline = outlines.get(metadata.outlineIndex);
        if (polyline) polyline.show = visible;
      }
    }

    this.viewer.scene.requestRender();
  }

  showHoverLabel(position: Cesium.Cartesian3, text: string): void {
    if (!this.hoverLabelEntity) {
      this.hoverLabelEntity = this.viewer.entities.add({
        position,
        label: {
          text,
          font: "11px sans-serif",
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.CENTER,
          horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    } else {
      this.hoverLabelEntity.position = new Cesium.ConstantPositionProperty(position);
      const labelText = this.hoverLabelEntity.label?.text;
      if (labelText instanceof Cesium.ConstantProperty) {
        labelText.setValue(text);
      } else if (this.hoverLabelEntity.label) {
        this.hoverLabelEntity.label.text = new Cesium.ConstantProperty(text);
      }
      this.hoverLabelEntity.show = true;
    }

    this.viewer.scene.requestRender();
  }

  hideHoverLabel(): void {
    if (this.hoverLabelEntity) {
      this.hoverLabelEntity.show = false;
      this.viewer.scene.requestRender();
    }
  }

  hasFeature(entityId: string): boolean {
    return this.featureMetadata.has(entityId);
  }

  /**
   * Check if footprints are loaded
   */
  hasFootprints(instrument: InstrumentType): boolean {
    return (this.instanceIds.get(instrument)?.size ?? 0) > 0;
  }

  /**
   * Get instrument color
   */
  private getColor(instrument: InstrumentType): Cesium.Color {
    const rgb = getInstrumentCesiumColor(instrument.toLowerCase() as InstrumentId);
    return new Cesium.Color(rgb.r, rgb.g, rgb.b, 1.0);
  }

  /**
   * Cleanup
   */
  dispose(): void {
    for (const controller of this.abortControllers.values()) {
      controller.abort();
    }

    // Guard: viewer may already be destroyed (e.g. render error cascade)
    const viewerAlive = this.viewer && !this.viewer.isDestroyed();

    if (viewerAlive) {
      for (const primitive of this.fillPrimitives.values()) {
        this.viewer.scene.primitives.remove(primitive);
      }
      for (const outlines of this.outlineCollections.values()) {
        this.viewer.scene.primitives.remove(outlines);
      }
      if (this.hoverLabelEntity) {
        this.viewer.entities.remove(this.hoverLabelEntity);
      }
    }

    this.fillPrimitives.clear();
    this.outlineCollections.clear();
    this.featureMetadata.clear();
    this.featureVisibility.clear();
    this.hoverLabelEntity = null;

    // Skip clearFootprints if viewer is dead — it calls viewer.scene.requestRender()
    if (viewerAlive) {
      this.clearFootprints("CRISM");
      this.clearFootprints("HIRISE");
      this.clearFootprints("SHARAD");
      this.clearFootprints("SHARAD_HIGHRES");
      this.clearFootprints("CTX");
      this.clearFootprints("HIRISE_DTM");
      this.clearFootprints("CRISM_TRR3");
    } else {
      // Just clear data structures without touching viewer
      for (const inst of this.features.keys()) {
        this.features.set(inst, []);
        this.instanceIds.get(inst)?.clear();
      }
    }
  }
}

export default FootprintManager;
