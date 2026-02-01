/**
 * FootprintManager - Viewport-based footprint loading with LOD and caching
 *
 * This module provides:
 * - Camera moveEnd listener with debounce
 * - Bbox computation from camera view
 * - LOD determination based on camera height
 * - LRU cache for responses
 * - AbortController for in-flight requests
 * - Diff-based rendering (add/remove only changed features)
 */

import * as Cesium from "cesium";

// ============================================================
// Types
// ============================================================

export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD";
export type LODType = "none" | "point" | "poly";
export type SimplifyLevel = "low" | "mid" | "high";

export interface FootprintFeature {
  type: "Feature";
  properties: {
    product_id: string;
    instrument?: string;
    [key: string]: any;
  };
  geometry: {
    type: "Point" | "Polygon" | "LineString";
    coordinates: number[] | number[][] | number[][][];
  };
}

export interface FootprintResponse {
  type: "FeatureCollection";
  features: FootprintFeature[];
  metadata: {
    truncated: boolean;
    returned: number;
    total_estimate: number;
    lod: LODType;
    simplify: SimplifyLevel | null;
    bbox: [number, number, number, number];
    instrument: string;
  };
}

export interface ViewportState {
  bbox: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  cameraHeight: number;
  lod: LODType;
  simplify: SimplifyLevel | null;
}

export interface FootprintManagerConfig {
  viewer: Cesium.Viewer;
  ellipsoid: Cesium.Ellipsoid;
  debounceMs?: number;
  maxCacheSize?: number;
  onTruncated?: (instrument: InstrumentType, returned: number, total: number) => void;
  onLoadStart?: (instrument: InstrumentType) => void;
  onLoadEnd?: (instrument: InstrumentType, count: number) => void;
  onError?: (instrument: InstrumentType, error: Error) => void;
  onLODChange?: (lod: LODType, cameraHeight: number) => void;
}

// ============================================================
// LOD Thresholds (camera height in meters)
// STRICT ZOOM GATING - These thresholds are NON-NEGOTIABLE
// ============================================================

const LOD_THRESHOLDS = {
  // FAR VIEW: camera height > 2,000 km → NO footprints at all
  FAR: 2_000_000, // 2,000 km in meters
  // MID VIEW: 2,000 km >= height > 500 km → points only
  MID: 500_000, // 500 km in meters
  // CLOSE VIEW: height <= 500 km → polygons allowed
};

// Simplification levels based on camera height (for CLOSE VIEW only)
const SIMPLIFY_THRESHOLDS = {
  // < 200 km → high detail (minimal simplification)
  HIGH: 200_000, // 200 km
  // 200-500 km → mid simplification
  MID: 500_000, // 500 km
};

// ============================================================
// LRU Cache
// ============================================================

class LRUCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;

  constructor(maxSize: number) {
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, value);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

// ============================================================
// Utility Functions
// ============================================================

function normalizeLon(lon: number): number {
  while (lon > 180) lon -= 360;
  while (lon < -180) lon += 360;
  return lon;
}

function computeBboxKey(
  instrument: InstrumentType,
  bbox: [number, number, number, number],
  lod: LODType,
  simplify: SimplifyLevel | null
): string {
  // Round bbox to reduce cache misses for small camera movements
  const precision = 2; // 2 decimal places (~1km precision)
  const roundedBbox = bbox.map((v) => v.toFixed(precision)).join(",");
  return `${instrument}:${roundedBbox}:${lod}:${simplify || "none"}`;
}

function determineLOD(cameraHeight: number): LODType {
  if (cameraHeight > LOD_THRESHOLDS.FAR) return "none";
  if (cameraHeight > LOD_THRESHOLDS.MID) return "point";
  return "poly";
}

function determineSimplify(cameraHeight: number, lod: LODType): SimplifyLevel | null {
  if (lod !== "poly") return null;
  if (cameraHeight < SIMPLIFY_THRESHOLDS.HIGH) return "high";
  if (cameraHeight < SIMPLIFY_THRESHOLDS.MID) return "mid";
  return "low";
}

// ============================================================
// FootprintManager Class
// ============================================================

export class FootprintManager {
  private viewer: Cesium.Viewer;
  private ellipsoid: Cesium.Ellipsoid;
  private debounceMs: number;
  private cache: LRUCache<string, FootprintResponse>;

  // Abort controllers for in-flight requests
  private abortControllers: Map<InstrumentType, AbortController> = new Map();

  // Current state per instrument
  private currentFeatures: Map<InstrumentType, Map<string, FootprintFeature>> = new Map();
  private enabled: Map<InstrumentType, boolean> = new Map();

  // Cesium primitives
  private pointCollections: Map<InstrumentType, Cesium.PointPrimitiveCollection> = new Map();
  private polygonPrimitives: Map<InstrumentType, Cesium.GroundPrimitive | null> = new Map();
  private entityIds: Map<InstrumentType, Set<string>> = new Map();

  // Debounce timer
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  // Camera event listener
  private moveEndListener: Cesium.Event.RemoveCallback | null = null;

  // Callbacks
  private onTruncated?: (instrument: InstrumentType, returned: number, total: number) => void;
  private onLoadStart?: (instrument: InstrumentType) => void;
  private onLoadEnd?: (instrument: InstrumentType, count: number) => void;
  private onError?: (instrument: InstrumentType, error: Error) => void;
  private onLODChange?: (lod: LODType, cameraHeight: number) => void;

  // Current LOD state
  private currentLOD: LODType = "none";
  private currentCameraHeight: number = Infinity;

  // Colors
  private static COLORS: Record<InstrumentType, Cesium.Color> = {
    CRISM: Cesium.Color.CYAN,
    HIRISE: Cesium.Color.YELLOW,
    SHARAD: Cesium.Color.ORANGE,
  };

  constructor(config: FootprintManagerConfig) {
    this.viewer = config.viewer;
    this.ellipsoid = config.ellipsoid;
    this.debounceMs = config.debounceMs ?? 300;
    this.cache = new LRUCache(config.maxCacheSize ?? 100);
    this.onTruncated = config.onTruncated;
    this.onLoadStart = config.onLoadStart;
    this.onLoadEnd = config.onLoadEnd;
    this.onError = config.onError;
    this.onLODChange = config.onLODChange;

    // Initialize state for each instrument
    (["CRISM", "HIRISE", "SHARAD"] as InstrumentType[]).forEach((inst) => {
      this.currentFeatures.set(inst, new Map());
      this.enabled.set(inst, false);
      this.entityIds.set(inst, new Set());
    });

    // Set up camera moveEnd listener
    this.setupCameraListener();
  }

  private setupCameraListener(): void {
    this.moveEndListener = this.viewer.camera.moveEnd.addEventListener(() => {
      this.onCameraMoveEnd();
    });
  }

  private onCameraMoveEnd(): void {
    // Debounce
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.updateFootprints();
    }, this.debounceMs);
  }

  private getViewportState(): ViewportState | null {
    const camera = this.viewer.camera;

    // Compute camera height
    const cartographic = camera.positionCartographic;
    const cameraHeight = cartographic ? cartographic.height : camera.positionWC.z;

    // Compute view rectangle
    let viewRect = camera.computeViewRectangle(this.ellipsoid);

    // Fallback: compute from canvas corners
    if (!viewRect) {
      const canvas = this.viewer.scene.canvas;
      const corners = [
        new Cesium.Cartesian2(0, 0),
        new Cesium.Cartesian2(canvas.width, 0),
        new Cesium.Cartesian2(canvas.width, canvas.height),
        new Cesium.Cartesian2(0, canvas.height),
      ];

      let west = Infinity,
        east = -Infinity,
        south = Infinity,
        north = -Infinity;

      for (const corner of corners) {
        const cartesian = camera.pickEllipsoid(corner, this.ellipsoid);
        if (cartesian) {
          const carto = Cesium.Cartographic.fromCartesian(cartesian, this.ellipsoid);
          west = Math.min(west, carto.longitude);
          east = Math.max(east, carto.longitude);
          south = Math.min(south, carto.latitude);
          north = Math.max(north, carto.latitude);
        }
      }

      if (west !== Infinity) {
        viewRect = new Cesium.Rectangle(west, south, east, north);
      }
    }

    if (!viewRect) {
      return null;
    }

    // Convert to degrees
    const minLon = normalizeLon(Cesium.Math.toDegrees(viewRect.west));
    const maxLon = normalizeLon(Cesium.Math.toDegrees(viewRect.east));
    const minLat = Cesium.Math.toDegrees(viewRect.south);
    const maxLat = Cesium.Math.toDegrees(viewRect.north);

    const lod = determineLOD(cameraHeight);
    const simplify = determineSimplify(cameraHeight, lod);

    return {
      bbox: [minLon, minLat, maxLon, maxLat],
      cameraHeight,
      lod,
      simplify,
    };
  }

  private async updateFootprints(): Promise<void> {
    const viewport = this.getViewportState();
    if (!viewport) return;

    const { lod, cameraHeight } = viewport;

    // Notify LOD change
    if (lod !== this.currentLOD || Math.abs(cameraHeight - this.currentCameraHeight) > 10000) {
      this.currentLOD = lod;
      this.currentCameraHeight = cameraHeight;
      this.onLODChange?.(lod, cameraHeight);
    }

    // FAR VIEW: Clear all footprints and do NOT fetch
    if (lod === "none") {
      const instruments: InstrumentType[] = ["CRISM", "HIRISE"];
      for (const instrument of instruments) {
        if (this.enabled.get(instrument)) {
          this.clearInstrumentFootprints(instrument);
        }
      }
      return;
    }

    // Update each enabled instrument
    const instruments: InstrumentType[] = ["CRISM", "HIRISE"];

    for (const instrument of instruments) {
      if (!this.enabled.get(instrument)) continue;
      await this.updateInstrument(instrument, viewport);
    }
  }

  private clearInstrumentFootprints(instrument: InstrumentType): void {
    const viewer = this.viewer;
    const entityIdSet = this.entityIds.get(instrument)!;

    // Remove all entities for this instrument
    for (const id of entityIdSet) {
      const entity = viewer.entities.getById(id);
      if (entity) {
        viewer.entities.remove(entity);
      }
    }
    entityIdSet.clear();

    // Clear current features
    this.currentFeatures.get(instrument)?.clear();

    viewer.scene.requestRender();
  }

  private async updateInstrument(
    instrument: InstrumentType,
    viewport: ViewportState
  ): Promise<void> {
    const { bbox, lod, simplify } = viewport;

    // Check cache
    const cacheKey = computeBboxKey(instrument, bbox, lod, simplify);
    let response = this.cache.get(cacheKey);

    if (!response) {
      // Abort any in-flight request
      const existingController = this.abortControllers.get(instrument);
      if (existingController) {
        existingController.abort();
      }

      // Create new abort controller
      const controller = new AbortController();
      this.abortControllers.set(instrument, controller);

      try {
        this.onLoadStart?.(instrument);

        const params = new URLSearchParams({
          instrument,
          bbox: bbox.join(","),
          lod,
          limit: "2000",
        });

        if (simplify) {
          params.set("simplify", simplify);
        }

        const res = await fetch(`/api/footprints?${params}`, {
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        response = await res.json();
        this.cache.set(cacheKey, response!);

        // Notify if truncated
        if (response!.metadata.truncated) {
          this.onTruncated?.(
            instrument,
            response!.metadata.returned,
            response!.metadata.total_estimate
          );
        }
      } catch (err: any) {
        if (err.name === "AbortError") {
          // Request was aborted, ignore
          return;
        }
        this.onError?.(instrument, err);
        return;
      } finally {
        this.abortControllers.delete(instrument);
      }
    }

    if (!response) return;

    // Diff and update rendering
    this.updateRendering(instrument, response.features, viewport.lod);
    this.onLoadEnd?.(instrument, response.features.length);
  }

  private updateRendering(
    instrument: InstrumentType,
    features: FootprintFeature[],
    lod: LODType
  ): void {
    const viewer = this.viewer;
    const currentMap = this.currentFeatures.get(instrument)!;
    const entityIdSet = this.entityIds.get(instrument)!;

    // Build new feature map
    const newMap = new Map<string, FootprintFeature>();
    for (const f of features) {
      const id = f.properties.product_id;
      if (id) {
        newMap.set(id, f);
      }
    }

    // Find features to remove
    const toRemove: string[] = [];
    for (const id of currentMap.keys()) {
      if (!newMap.has(id)) {
        toRemove.push(id);
      }
    }

    // Find features to add
    const toAdd: FootprintFeature[] = [];
    for (const [id, f] of newMap) {
      if (!currentMap.has(id)) {
        toAdd.push(f);
      }
    }

    // Remove old entities
    for (const id of toRemove) {
      const entityId = `${instrument}_VP_${id}`;
      const entity = viewer.entities.getById(entityId);
      if (entity) {
        viewer.entities.remove(entity);
      }
      entityIdSet.delete(entityId);

      // Also remove label and point entities
      const labelId = `${instrument}_VP_LABEL_${id}`;
      const pointId = `${instrument}_VP_POINT_${id}`;
      const labelEnt = viewer.entities.getById(labelId);
      const pointEnt = viewer.entities.getById(pointId);
      if (labelEnt) viewer.entities.remove(labelEnt);
      if (pointEnt) viewer.entities.remove(pointEnt);
      entityIdSet.delete(labelId);
      entityIdSet.delete(pointId);
    }

    // Add new features
    const color = FootprintManager.COLORS[instrument];

    for (const f of toAdd) {
      const id = f.properties.product_id;
      const geom = f.geometry;

      if (!geom) continue;

      const entityId = `${instrument}_VP_${id}`;

      // MID VIEW: ONLY render points, never polygons
      if (lod === "point") {
        // Render as point (centroid)
        let lon: number, lat: number;
        if (geom.type === "Point") {
          const coords = geom.coordinates as number[];
          lon = normalizeLon(coords[0]);
          lat = coords[1];
        } else if (geom.type === "Polygon") {
          // Compute centroid from polygon
          const coords = geom.coordinates as number[][][];
          const ring = coords[0];
          const lons = ring.map((c) => c[0]);
          const lats = ring.map((c) => c[1]);
          lon = normalizeLon(lons.reduce((a, b) => a + b, 0) / lons.length);
          lat = lats.reduce((a, b) => a + b, 0) / lats.length;
        } else {
          continue; // Skip unsupported geometry types in point mode
        }

        viewer.entities.add({
          id: entityId,
          position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, this.ellipsoid),
          point: {
            pixelSize: 6,
            color: color.withAlpha(0.8),
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 1,
            disableDepthTestDistance: Infinity,
          },
          properties: {
            product_id: id,
            instrument,
            kind: "FOOTPRINT_POINT",
          },
        });
        entityIdSet.add(entityId);
      } else if (lod === "poly" && geom.type === "Polygon") {
        // CLOSE VIEW: Render polygons with minimal material, NO outline
        const coords = geom.coordinates as number[][][];
        const ring = coords[0];

        const lons = ring.map((c) => normalizeLon(c[0]));
        const lats = ring.map((c) => c[1]);
        const west = Math.min(...lons);
        const east = Math.max(...lons);
        const south = Math.min(...lats);
        const north = Math.max(...lats);

        // Check for antimeridian crossing
        const width = east - west;
        const rects: Cesium.Rectangle[] = [];

        if (width > 180) {
          // Split into two rectangles
          rects.push(Cesium.Rectangle.fromDegrees(east, south, 180, north));
          rects.push(Cesium.Rectangle.fromDegrees(-180, south, west, north));
        } else {
          rects.push(Cesium.Rectangle.fromDegrees(west, south, east, north));
        }

        rects.forEach((rect, i) => {
          const rectEntityId = i === 0 ? entityId : `${entityId}_${i}`;
          viewer.entities.add({
            id: rectEntityId,
            rectangle: {
              coordinates: rect,
              // Minimal material - solid color, no translucency for performance
              material: color.withAlpha(0.4),
              // NO outline for performance
              outline: false,
              height: 0,
            },
            properties: {
              product_id: id,
              instrument,
              kind: "FOOTPRINT_RECT",
            },
          });
          entityIdSet.add(rectEntityId);
        });

        // Add label at center
        const centerLon = (west + east) / 2;
        const centerLat = (south + north) / 2;
        const labelId = `${instrument}_VP_LABEL_${id}`;

        viewer.entities.add({
          id: labelId,
          position: Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 0, this.ellipsoid),
          label: {
            text: id,
            font: "12px sans-serif",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 3,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            pixelOffset: new Cesium.Cartesian2(0, -12),
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            disableDepthTestDistance: Infinity,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, 9.0e7),
          },
          properties: {
            product_id: id,
            instrument,
            kind: "FOOTPRINT_LABEL",
          },
        });
        entityIdSet.add(labelId);

        // Add center point
        const pointId = `${instrument}_VP_POINT_${id}`;
        viewer.entities.add({
          id: pointId,
          position: Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 0, this.ellipsoid),
          point: {
            pixelSize: 6,
            color,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 1,
            disableDepthTestDistance: Infinity,
          },
          properties: {
            product_id: id,
            instrument,
            kind: "FOOTPRINT_POINT",
          },
        });
        entityIdSet.add(pointId);
      } else if (geom.type === "LineString") {
        // Render as polyline (SHARAD)
        const coords = geom.coordinates as number[][];
        const positions = coords.map((c) =>
          Cesium.Cartesian3.fromDegrees(normalizeLon(c[0]), c[1], 0, this.ellipsoid)
        );

        viewer.entities.add({
          id: entityId,
          polyline: {
            positions,
            width: 3,
            material: color.withAlpha(0.8),
            clampToGround: true,
          },
          properties: {
            product_id: id,
            instrument,
            ...f.properties,
          },
        });
        entityIdSet.add(entityId);
      }
    }

    // Update current features map
    this.currentFeatures.set(instrument, newMap);

    viewer.scene.requestRender();
  }

  /**
   * Enable or disable an instrument's footprints.
   */
  setEnabled(instrument: InstrumentType, enabled: boolean): void {
    this.enabled.set(instrument, enabled);

    if (!enabled) {
      // Hide all entities for this instrument
      const entityIdSet = this.entityIds.get(instrument)!;
      for (const id of entityIdSet) {
        const entity = this.viewer.entities.getById(id);
        if (entity) {
          entity.show = false;
        }
      }
    } else {
      // Show all entities and trigger refresh
      const entityIdSet = this.entityIds.get(instrument)!;
      for (const id of entityIdSet) {
        const entity = this.viewer.entities.getById(id);
        if (entity) {
          entity.show = true;
        }
      }
      // Trigger update
      this.onCameraMoveEnd();
    }

    this.viewer.scene.requestRender();
  }

  /**
   * Force a refresh of all enabled instruments.
   */
  refresh(): void {
    this.onCameraMoveEnd();
  }

  /**
   * Clear cache and reload.
   */
  clearCache(): void {
    this.cache.clear();
    this.refresh();
  }

  /**
   * Get current features for an instrument.
   */
  getFeatures(instrument: InstrumentType): FootprintFeature[] {
    const map = this.currentFeatures.get(instrument);
    return map ? Array.from(map.values()) : [];
  }

  /**
   * Get all visible product IDs.
   */
  getVisibleProductIds(instrument: InstrumentType): string[] {
    const map = this.currentFeatures.get(instrument);
    return map ? Array.from(map.keys()) : [];
  }

  /**
   * Check if manager is enabled for an instrument.
   */
  isEnabled(instrument: InstrumentType): boolean {
    return this.enabled.get(instrument) ?? false;
  }

  /**
   * Get current LOD level.
   */
  getCurrentLOD(): LODType {
    return this.currentLOD;
  }

  /**
   * Get current camera height in meters.
   */
  getCameraHeight(): number {
    return this.currentCameraHeight;
  }

  /**
   * Get LOD thresholds for UI display.
   */
  static getLODThresholds() {
    return { ...LOD_THRESHOLDS };
  }

  /**
   * Dispose and clean up resources.
   */
  dispose(): void {
    // Clear debounce timer
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    // Remove camera listener
    if (this.moveEndListener) {
      this.moveEndListener();
    }

    // Abort any in-flight requests
    for (const controller of this.abortControllers.values()) {
      controller.abort();
    }

    // Remove all entities
    for (const [instrument, entityIdSet] of this.entityIds) {
      for (const id of entityIdSet) {
        const entity = this.viewer.entities.getById(id);
        if (entity) {
          this.viewer.entities.remove(entity);
        }
      }
      entityIdSet.clear();
    }

    // Clear cache
    this.cache.clear();
  }
}

export default FootprintManager;
