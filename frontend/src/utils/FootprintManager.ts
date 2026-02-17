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

export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3";

export interface LoadResult {
  instrument: InstrumentType;
  count: number;
  truncated: boolean;
  total: number;
}

interface FootprintFeature {
  properties: { product_id: string; [key: string]: any };
  geometry: {
    type: string;
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
}

// Using shared normalizeLonForMap from coordinates.ts
const normalizeLon = normalizeLonForMap;

export class FootprintManager {
  private viewer: Cesium.Viewer;
  private ellipsoid: Cesium.Ellipsoid;
  private entityIds: Map<InstrumentType, Set<string>> = new Map();
  private features: Map<InstrumentType, FootprintFeature[]> = new Map();
  private abortControllers: Map<InstrumentType, AbortController> = new Map();
  private requestIds: Map<InstrumentType, number> = new Map(); // Track request IDs for idempotency
  private nextRequestId = 0;
  // In-flight promise deduplication: if loadFootprints is called while a previous load is pending, return the same promise
  private inFlightLoads: Map<InstrumentType, Promise<LoadResult | null>> = new Map();
  // Store the bbox that was used when loading each instrument's footprints
  private loadedBboxes: Map<InstrumentType, [number, number, number, number]> = new Map();

  private onLoadStart?: (instrument: InstrumentType) => void;
  private onLoadEnd?: (instrument: InstrumentType, result: LoadResult) => void;
  private onError?: (instrument: InstrumentType, error: Error) => void;

  constructor(config: FootprintManagerConfig) {
    this.viewer = config.viewer;
    this.ellipsoid = config.ellipsoid;
    this.onLoadStart = config.onLoadStart;
    this.onLoadEnd = config.onLoadEnd;
    this.onError = config.onError;

    // Initialize empty collections for each instrument
    this.entityIds.set("CRISM", new Set());
    this.entityIds.set("HIRISE", new Set());
    this.entityIds.set("SHARAD", new Set());
    this.entityIds.set("SHARAD_HIGHRES", new Set());
    this.entityIds.set("CTX", new Set());
    this.entityIds.set("HIRISE_DTM", new Set());
    this.entityIds.set("CRISM_TRR3", new Set());
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
      console.warn("[FootprintManager] Cannot compute viewport");
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
      const url = `/api/footprints?instrument=${instrument}&bbox=${bbox.join(",")}&limit=2000`;
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
      this.renderFeatures(instrument, data.features);

      const entityCount = this.entityIds.get(instrument)?.size ?? 0;

      const result: LoadResult = {
        instrument,
        count: data.features.length,
        truncated: data.metadata.truncated,
        total: data.metadata.total_estimate,
      };

      this.onLoadEnd?.(instrument, result);
      return result;

    } catch (err: any) {
      if (err.name === "AbortError") {
        return null;
      }
      console.error(`[FootprintManager] Error loading ${instrument}:`, err);
      this.onError?.(instrument, err);
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
    const ids = this.entityIds.get(instrument);
    if (ids && ids.size > 0) {
      this.viewer.entities.suspendEvents();
      for (const id of ids) {
        const entity = this.viewer.entities.getById(id);
        if (entity) this.viewer.entities.remove(entity);
      }
      ids.clear();
      this.viewer.entities.resumeEvents();
      this.viewer.scene.requestRender();
    }
    this.features.set(instrument, []);
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

  /**
   * Render features as Cesium entities
   * Passes ALL feature properties to entity for popup access
   */
  private renderFeatures(instrument: InstrumentType, features: FootprintFeature[]): void {
    const ids = this.entityIds.get(instrument)!;
    const color = this.getColor(instrument);

    this.viewer.entities.suspendEvents();

    for (const feature of features) {
      const productId = feature.properties?.product_id;
      if (!productId) continue;

      const geom = feature.geometry;
      if (!geom) continue;

      const entityId = `${instrument}_FP_${productId}`;

      // Skip duplicates (same product_id appearing twice in data)
      if (ids.has(entityId) || this.viewer.entities.getById(entityId)) continue;

      // Pass ALL feature properties to the entity (for popup access)
      const entityProps = { ...feature.properties, instrument };

      if (geom.type === "Polygon") {
        const ring = geom.coordinates?.[0];
        if (!ring || ring.length === 0) continue;

        // Compute bounding box
        let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
        for (const [lon, lat] of ring) {
          const nlon = normalizeLon(lon);
          if (nlon < west) west = nlon;
          if (nlon > east) east = nlon;
          if (lat < south) south = lat;
          if (lat > north) north = lat;
        }

        // Skip if crosses dateline (would render incorrectly)
        if (east - west > 180) continue;

        const centerLon = (west + east) / 2;
        const centerLat = (south + north) / 2;

        // Add rectangle with light fill (visible but subtle)
        this.viewer.entities.add({
          id: entityId,
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
            material: color.withAlpha(0.10),
            outline: true,
            outlineColor: color,
            outlineWidth: 1,
            height: 0,
          },
          properties: entityProps,
        });
        ids.add(entityId);

        // Add label (hidden by default, shown on hover)
        const labelId = `${instrument}_LBL_${productId}`;
        this.viewer.entities.add({
          id: labelId,
          show: false,
          position: Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 0, this.ellipsoid),
          label: {
            text: productId,
            font: "11px sans-serif",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.CENTER,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e6),
          },
          properties: entityProps,
        });
        ids.add(labelId);

      } else if (geom.type === "LineString") {
        const coords = geom.coordinates;
        if (!coords || coords.length < 2) continue;

        // Convert coordinates to Cesium positions
        const positions = coords.map(([lon, lat]: [number, number]) =>
          Cesium.Cartesian3.fromDegrees(normalizeLon(lon), lat, 0, this.ellipsoid)
        );

        this.viewer.entities.add({
          id: entityId,
          polyline: {
            positions,
            width: 3,
            material: color,
            clampToGround: true,
            arcType: Cesium.ArcType.GEODESIC, // Proper great circle path for polar orbits
          },
          properties: entityProps,
        });
        ids.add(entityId);

        // Compute label position at geodesic midpoint
        const startPos = positions[0];
        const endPos = positions[positions.length - 1];
        const midPos = Cesium.Cartesian3.midpoint(startPos, endPos, new Cesium.Cartesian3());

        const labelId = `${instrument}_LBL_${productId}`;
        this.viewer.entities.add({
          id: labelId,
          show: false,
          position: midPos,
          label: {
            text: productId,
            font: "11px sans-serif",
            fillColor: Cesium.Color.ORANGE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            pixelOffset: new Cesium.Cartesian2(0, -5),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e6),
          },
          properties: entityProps,
        });
        ids.add(labelId);

      } else if (geom.type === "Point") {
        const [lon, lat] = geom.coordinates;

        // Skip invalid placeholder coordinates (0, 0)
        if (lon === 0 && lat === 0) continue;

        const nlon = normalizeLon(lon);

        this.viewer.entities.add({
          id: entityId,
          position: Cesium.Cartesian3.fromDegrees(nlon, lat, 0, this.ellipsoid),
          point: {
            pixelSize: 8,
            color: color.withAlpha(0.9),
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 1,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          properties: entityProps,
        });
        ids.add(entityId);

        // Add label (hidden by default, shown on hover)
        const labelId = `${instrument}_LBL_${productId}`;
        this.viewer.entities.add({
          id: labelId,
          show: false,
          position: Cesium.Cartesian3.fromDegrees(nlon, lat, 0, this.ellipsoid),
          label: {
            text: productId,
            font: "10px sans-serif",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            pixelOffset: new Cesium.Cartesian2(0, -10),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 3e6),
          },
          properties: entityProps,
        });
        ids.add(labelId);
      }
    }

    this.viewer.entities.resumeEvents();
    this.viewer.scene.requestRender();
  }

  /**
   * Set visibility of footprints
   */
  setVisible(instrument: InstrumentType, visible: boolean): void {
    const ids = this.entityIds.get(instrument);
    if (!ids || ids.size === 0) return;

    this.viewer.entities.suspendEvents();
    for (const id of ids) {
      const entity = this.viewer.entities.getById(id);
      if (!entity) continue;
      // Labels stay hidden (hover-only) — only toggle footprint entities
      if (id.includes('_LBL_')) {
        entity.show = false;
      } else {
        entity.show = visible;
      }
    }
    this.viewer.entities.resumeEvents();
    this.viewer.scene.requestRender();
  }

  /**
   * Check if footprints are loaded
   */
  hasFootprints(instrument: InstrumentType): boolean {
    return (this.entityIds.get(instrument)?.size ?? 0) > 0;
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
    this.clearFootprints("CRISM");
    this.clearFootprints("HIRISE");
    this.clearFootprints("SHARAD");
    this.clearFootprints("SHARAD_HIGHRES");
    this.clearFootprints("CTX");
    this.clearFootprints("HIRISE_DTM");
    this.clearFootprints("CRISM_TRR3");
  }
}

export default FootprintManager;
