/**
 * FootprintManager - Viewport-based footprint loading with LOD and caching
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Batches entity operations (suspendEvents/resumeEvents)
 * - Pre-computed geometry caching
 * - Aggressive bbox rounding for better cache hits
 * - Diff-based rendering (add/remove only changed features)
 * - Parallel instrument updates
 */

import * as Cesium from "cesium";
import { perf } from "./perfMonitor";
import { getInstrumentCesiumColor, type InstrumentId } from "../config/instrumentRegistry";

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
  bbox: [number, number, number, number];
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
// LOD Thresholds
// ============================================================

const LOD_THRESHOLDS = {
  FAR: 15_000_000,
  MID: 5_000_000,
};

const SIMPLIFY_THRESHOLDS = {
  HIGH: 2_000_000,
  MID: 5_000_000,
};

// ============================================================
// Pre-computed geometry cache
// ============================================================

interface CachedGeometry {
  centroid: { lon: number; lat: number };
  bounds?: { west: number; south: number; east: number; north: number };
}

const geometryCache = new Map<string, CachedGeometry>();

function getCachedGeometry(feature: FootprintFeature): CachedGeometry | null {
  const id = feature.properties.product_id;
  if (!id) return null;

  const cached = geometryCache.get(id);
  if (cached) return cached;

  const geom = feature.geometry;
  if (!geom) return null;

  let result: CachedGeometry;

  if (geom.type === "Point") {
    const coords = geom.coordinates as number[];
    result = {
      centroid: { lon: normalizeLon(coords[0]), lat: coords[1] }
    };
  } else if (geom.type === "Polygon") {
    const coords = geom.coordinates as number[][][];
    const ring = coords[0];
    if (!ring || ring.length === 0) return null;

    const lons = ring.map((c) => normalizeLon(c[0]));
    const lats = ring.map((c) => c[1]);

    const west = Math.min(...lons);
    const east = Math.max(...lons);
    const south = Math.min(...lats);
    const north = Math.max(...lats);

    result = {
      centroid: { lon: (west + east) / 2, lat: (south + north) / 2 },
      bounds: { west, south, east, north }
    };
  } else {
    return null;
  }

  geometryCache.set(id, result);
  return result;
}

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
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
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
  // Round bbox for better cache hits
  const precision = 1;
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

  private abortControllers: Map<InstrumentType, AbortController> = new Map();
  private currentFeatures: Map<InstrumentType, Map<string, FootprintFeature>> = new Map();
  private enabled: Map<InstrumentType, boolean> = new Map();
  private entityIds: Map<InstrumentType, Set<string>> = new Map();

  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private moveEndListener: Cesium.Event.RemoveCallback | null = null;

  private onTruncated?: (instrument: InstrumentType, returned: number, total: number) => void;
  private onLoadStart?: (instrument: InstrumentType) => void;
  private onLoadEnd?: (instrument: InstrumentType, count: number) => void;
  private onError?: (instrument: InstrumentType, error: Error) => void;
  private onLODChange?: (lod: LODType, cameraHeight: number) => void;

  private currentLOD: LODType = "none";
  private currentCameraHeight: number = Infinity;
  private renderedLOD: Map<InstrumentType, LODType> = new Map();

  /**
   * Get Cesium color for an instrument from the registry
   */
  private static getInstrumentColor(instrument: InstrumentType): Cesium.Color {
    const rgb = getInstrumentCesiumColor(instrument.toLowerCase() as InstrumentId);
    return new Cesium.Color(rgb.r, rgb.g, rgb.b, 1.0);
  }

  constructor(config: FootprintManagerConfig) {
    this.viewer = config.viewer;
    this.ellipsoid = config.ellipsoid;
    this.debounceMs = config.debounceMs ?? 300;
    this.cache = new LRUCache(config.maxCacheSize ?? 200);
    this.onTruncated = config.onTruncated;
    this.onLoadStart = config.onLoadStart;
    this.onLoadEnd = config.onLoadEnd;
    this.onError = config.onError;
    this.onLODChange = config.onLODChange;

    const instruments: InstrumentType[] = ["CRISM", "HIRISE", "SHARAD"];
    instruments.forEach((inst) => {
      this.currentFeatures.set(inst, new Map());
      this.enabled.set(inst, false);
      this.entityIds.set(inst, new Set());
    });

    this.setupCameraListener();
  }

  private setupCameraListener(): void {
    this.moveEndListener = this.viewer.camera.moveEnd.addEventListener(() => {
      this.onCameraMoveEnd();
    });
  }

  private onCameraMoveEnd(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.updateFootprints();
    }, this.debounceMs);
  }

  private getViewportState(): ViewportState | null {
    const camera = this.viewer.camera;
    const cartographic = camera.positionCartographic;
    const cameraHeight = cartographic ? cartographic.height : camera.positionWC.z;

    let viewRect = camera.computeViewRectangle(this.ellipsoid);

    if (!viewRect) {
      const canvas = this.viewer.scene.canvas;
      const corners = [
        new Cesium.Cartesian2(0, 0),
        new Cesium.Cartesian2(canvas.width, 0),
        new Cesium.Cartesian2(canvas.width, canvas.height),
        new Cesium.Cartesian2(0, canvas.height),
      ];

      let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;

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

    if (!viewRect) return null;

    const minLon = normalizeLon(Cesium.Math.toDegrees(viewRect.west));
    const maxLon = normalizeLon(Cesium.Math.toDegrees(viewRect.east));
    const minLat = Cesium.Math.toDegrees(viewRect.south);
    const maxLat = Cesium.Math.toDegrees(viewRect.north);

    const lod = determineLOD(cameraHeight);
    const simplify = determineSimplify(cameraHeight, lod);

    return { bbox: [minLon, minLat, maxLon, maxLat], cameraHeight, lod, simplify };
  }

  private async updateFootprints(): Promise<void> {
    perf.start('footprint-update-total');
    const viewport = this.getViewportState();
    if (!viewport) {
      perf.end('footprint-update-total');
      return;
    }

    const { lod, cameraHeight } = viewport;

    if (lod !== this.currentLOD || Math.abs(cameraHeight - this.currentCameraHeight) > 50000) {
      this.currentLOD = lod;
      this.currentCameraHeight = cameraHeight;
      this.onLODChange?.(lod, cameraHeight);
    }

    if (lod === "none") {
      const instruments: InstrumentType[] = ["CRISM", "HIRISE"];
      for (const instrument of instruments) {
        if (this.enabled.get(instrument)) {
          this.clearInstrumentFootprints(instrument);
        }
      }
      perf.end('footprint-update-total');
      return;
    }

    const instruments: InstrumentType[] = ["CRISM", "HIRISE"];
    const promises = instruments
      .filter((inst) => this.enabled.get(inst))
      .map((inst) => this.updateInstrument(inst, viewport));

    await Promise.all(promises);
    perf.end('footprint-update-total');
  }

  private clearInstrumentFootprints(instrument: InstrumentType): void {
    const viewer = this.viewer;
    const entityIdSet = this.entityIds.get(instrument)!;

    if (entityIdSet.size > 0) {
      viewer.entities.suspendEvents();
      for (const id of entityIdSet) {
        const entity = viewer.entities.getById(id);
        if (entity) viewer.entities.remove(entity);
      }
      entityIdSet.clear();
      viewer.entities.resumeEvents();
    }

    this.currentFeatures.get(instrument)?.clear();
    viewer.scene.requestRender();
  }

  private async updateInstrument(
    instrument: InstrumentType,
    viewport: ViewportState
  ): Promise<void> {
    const { bbox, lod, simplify } = viewport;
    const cacheKey = computeBboxKey(instrument, bbox, lod, simplify);
    let response = this.cache.get(cacheKey);

    if (!response) {
      const existingController = this.abortControllers.get(instrument);
      if (existingController) {
        existingController.abort();
      }

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

        if (simplify) params.set("simplify", simplify);

        const res = await fetch(`/api/footprints?${params}`, {
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        response = await res.json();
        this.cache.set(cacheKey, response!);

        if (response!.metadata.truncated) {
          this.onTruncated?.(instrument, response!.metadata.returned, response!.metadata.total_estimate);
        }
      } catch (err: any) {
        if (err.name === "AbortError") return;
        this.onError?.(instrument, err);
        return;
      } finally {
        this.abortControllers.delete(instrument);
      }
    }

    if (!response) return;

    this.updateRendering(instrument, response.features, viewport.lod);
    this.onLoadEnd?.(instrument, response.features.length);
  }

  private updateRendering(
    instrument: InstrumentType,
    features: FootprintFeature[],
    lod: LODType
  ): void {
    perf.start(`render-${instrument}-${lod}`);
    const viewer = this.viewer;
    const currentMap = this.currentFeatures.get(instrument)!;
    const entityIdSet = this.entityIds.get(instrument)!;

    // Check LOD change
    const previousLOD = this.renderedLOD.get(instrument);
    const lodChanged = previousLOD !== undefined && previousLOD !== lod;

    if (lodChanged) {
      console.log(`[FootprintManager] LOD changed for ${instrument}: ${previousLOD} -> ${lod}`);
      viewer.entities.suspendEvents();
      for (const id of entityIdSet) {
        const entity = viewer.entities.getById(id);
        if (entity) viewer.entities.remove(entity);
      }
      entityIdSet.clear();
      viewer.entities.resumeEvents();
      currentMap.clear();
    }

    this.renderedLOD.set(instrument, lod);

    // Build new feature map
    const newMap = new Map<string, FootprintFeature>();
    for (const f of features) {
      const id = f.properties.product_id;
      if (id) newMap.set(id, f);
    }

    // Diff
    const toRemove: string[] = [];
    for (const id of currentMap.keys()) {
      if (!newMap.has(id)) toRemove.push(id);
    }

    const toAdd: FootprintFeature[] = [];
    for (const [id, f] of newMap) {
      if (!currentMap.has(id)) toAdd.push(f);
    }

    // Batch operations
    viewer.entities.suspendEvents();

    // Remove old entities
    for (const id of toRemove) {
      const entityId = `${instrument}_VP_${id}`;
      const entity = viewer.entities.getById(entityId);
      if (entity) viewer.entities.remove(entity);
      entityIdSet.delete(entityId);

      for (let i = 1; i < 4; i++) {
        const splitId = `${entityId}_${i}`;
        const splitEnt = viewer.entities.getById(splitId);
        if (splitEnt) viewer.entities.remove(splitEnt);
        entityIdSet.delete(splitId);
      }

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
    const color = FootprintManager.getInstrumentColor(instrument);

    for (const f of toAdd) {
      const id = f.properties.product_id;
      const geom = getCachedGeometry(f);
      if (!geom) continue;

      const entityId = `${instrument}_VP_${id}`;

      if (lod === "point") {
        // POINT MODE: Simple point entity
        viewer.entities.add({
          id: entityId,
          position: Cesium.Cartesian3.fromDegrees(geom.centroid.lon, geom.centroid.lat, 0, this.ellipsoid),
          point: {
            pixelSize: 6,
            color: color.withAlpha(0.8),
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 1,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          properties: {
            product_id: id,
            instrument,
            kind: "FOOTPRINT_POINT",
          },
        });
        entityIdSet.add(entityId);
      } else if (lod === "poly" && geom.bounds) {
        // POLY MODE: Rectangle entities
        const { west, south, east, north } = geom.bounds;

        const width = east - west;
        const rects: Array<{ rect: Cesium.Rectangle; suffix: string }> = [];

        if (width > 180) {
          rects.push({ rect: Cesium.Rectangle.fromDegrees(east, south, 180, north), suffix: "" });
          rects.push({ rect: Cesium.Rectangle.fromDegrees(-180, south, west, north), suffix: "_1" });
        } else {
          rects.push({ rect: Cesium.Rectangle.fromDegrees(west, south, east, north), suffix: "" });
        }

        for (const { rect, suffix } of rects) {
          const rectEntityId = entityId + suffix;
          viewer.entities.add({
            id: rectEntityId,
            rectangle: {
              coordinates: rect,
              material: color.withAlpha(0.4),
              outline: false,
              height: 0,
            },
            properties: { product_id: id, instrument, kind: "FOOTPRINT_RECT" },
          });
          entityIdSet.add(rectEntityId);
        }

        // Label
        const labelId = `${instrument}_VP_LABEL_${id}`;
        viewer.entities.add({
          id: labelId,
          position: Cesium.Cartesian3.fromDegrees(geom.centroid.lon, geom.centroid.lat, 0, this.ellipsoid),
          label: {
            text: id,
            font: "11px sans-serif",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            pixelOffset: new Cesium.Cartesian2(0, -10),
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e7),
          },
          properties: { product_id: id, instrument, kind: "FOOTPRINT_LABEL" },
        });
        entityIdSet.add(labelId);

        // Center point
        const pointId = `${instrument}_VP_POINT_${id}`;
        viewer.entities.add({
          id: pointId,
          position: Cesium.Cartesian3.fromDegrees(geom.centroid.lon, geom.centroid.lat, 0, this.ellipsoid),
          point: {
            pixelSize: 6,
            color,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 1,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          properties: { product_id: id, instrument, kind: "FOOTPRINT_POINT" },
        });
        entityIdSet.add(pointId);
      }
    }

    viewer.entities.resumeEvents();

    // Update current features map
    this.currentFeatures.set(instrument, newMap);
    viewer.scene.requestRender();
    perf.end(`render-${instrument}-${lod}`);
  }

  setEnabled(instrument: InstrumentType, enabled: boolean): void {
    const wasEnabled = this.enabled.get(instrument);
    this.enabled.set(instrument, enabled);

    const entityIdSet = this.entityIds.get(instrument)!;

    if (!enabled) {
      for (const id of entityIdSet) {
        const entity = this.viewer.entities.getById(id);
        if (entity) entity.show = false;
      }
    } else {
      for (const id of entityIdSet) {
        const entity = this.viewer.entities.getById(id);
        if (entity) entity.show = true;
      }

      if (!wasEnabled) {
        this.onCameraMoveEnd();
      }
    }

    this.viewer.scene.requestRender();
  }

  refresh(): void {
    this.onCameraMoveEnd();
  }

  clearCache(): void {
    this.cache.clear();
    geometryCache.clear();
    this.refresh();
  }

  getFeatures(instrument: InstrumentType): FootprintFeature[] {
    const map = this.currentFeatures.get(instrument);
    return map ? Array.from(map.values()) : [];
  }

  getVisibleProductIds(instrument: InstrumentType): string[] {
    const map = this.currentFeatures.get(instrument);
    return map ? Array.from(map.keys()) : [];
  }

  isEnabled(instrument: InstrumentType): boolean {
    return this.enabled.get(instrument) ?? false;
  }

  getCurrentLOD(): LODType {
    return this.currentLOD;
  }

  getCameraHeight(): number {
    return this.currentCameraHeight;
  }

  static getLODThresholds() {
    return { ...LOD_THRESHOLDS };
  }

  dispose(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    if (this.moveEndListener) this.moveEndListener();

    for (const controller of this.abortControllers.values()) {
      controller.abort();
    }

    this.viewer.entities.suspendEvents();
    for (const [, entityIdSet] of this.entityIds) {
      for (const id of entityIdSet) {
        const entity = this.viewer.entities.getById(id);
        if (entity) this.viewer.entities.remove(entity);
      }
      entityIdSet.clear();
    }
    this.viewer.entities.resumeEvents();

    this.cache.clear();
    geometryCache.clear();
  }
}

export default FootprintManager;
