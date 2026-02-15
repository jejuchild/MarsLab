/**
 * overlapFilter.ts — Pure spatial overlap computation for Multi-Instrument Overlap Filter
 *
 * No Cesium or React dependencies. Operates on GeoJSON features from FootprintManager.
 */

import type { InstrumentType } from "./FootprintManager";

/** Axis-aligned bounding box in degrees */
export interface BBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

/** GeoJSON feature as stored by FootprintManager */
export interface OverlapFeature {
  properties: { product_id: string; [key: string]: any };
  geometry: { type: string; coordinates: any };
}

/** Instrument → Set of passing product IDs */
export type OverlapResult = Map<InstrumentType, Set<string>>;

/** Stats for UI display */
export interface OverlapStats {
  totalChecked: number;
  totalPassing: number;
  perInstrument: Map<InstrumentType, { checked: number; passing: number }>;
}

// Small buffer (degrees) for Point geometries so they have spatial extent
const POINT_BUFFER_DEG = 0.05;

/**
 * Extract axis-aligned bounding box from a GeoJSON geometry.
 * - Polygon: min/max of outer ring coordinates
 * - LineString: min/max of all points
 * - Point: small buffer around the point
 */
export function getBoundingBox(feature: OverlapFeature): BBox | null {
  const geom = feature.geometry;
  if (!geom || !geom.coordinates) return null;

  if (geom.type === "Polygon") {
    const ring: number[][] = geom.coordinates[0];
    if (!ring || ring.length === 0) return null;
    let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
    for (const [lon, lat] of ring) {
      if (lon < west) west = lon;
      if (lon > east) east = lon;
      if (lat < south) south = lat;
      if (lat > north) north = lat;
    }
    return { west, south, east, north };
  }

  if (geom.type === "LineString") {
    const coords: number[][] = geom.coordinates;
    if (!coords || coords.length === 0) return null;
    let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
    for (const [lon, lat] of coords) {
      if (lon < west) west = lon;
      if (lon > east) east = lon;
      if (lat < south) south = lat;
      if (lat > north) north = lat;
    }
    return { west, south, east, north };
  }

  if (geom.type === "Point") {
    const [lon, lat] = geom.coordinates;
    return {
      west: lon - POINT_BUFFER_DEG,
      south: lat - POINT_BUFFER_DEG,
      east: lon + POINT_BUFFER_DEG,
      north: lat + POINT_BUFFER_DEG,
    };
  }

  return null;
}

/**
 * Test whether two bounding boxes overlap.
 * Handles antimeridian crossing (west > east).
 */
export function bboxIntersects(a: BBox, b: BBox): boolean {
  // Latitude overlap is straightforward
  if (a.north < b.south || b.north < a.south) return false;

  const aCrosses = a.west > a.east;
  const bCrosses = b.west > b.east;

  if (!aCrosses && !bCrosses) {
    // Neither crosses antimeridian: standard overlap
    return !(a.east < b.west || b.east < a.west);
  }

  if (aCrosses && bCrosses) {
    // Both cross: they always overlap in longitude
    return true;
  }

  // One crosses, one doesn't. The crossing box wraps around,
  // so it overlaps the normal box if the normal box's range
  // intersects either side of the wrapping box.
  const [crossing, normal] = aCrosses ? [a, b] : [b, a];
  return normal.east >= crossing.west || normal.west <= crossing.east;
}

/**
 * Liang-Barsky line clipping: returns true if segment (x1,y1)-(x2,y2)
 * intersects the axis-aligned rectangle [xmin,xmax] x [ymin,ymax].
 *
 * Same algorithm as backend proximity_router.py `_liang_barsky_clip`.
 */
function liangBarskyClip(
  x1: number, y1: number, x2: number, y2: number,
  xmin: number, ymin: number, xmax: number, ymax: number,
): boolean {
  const dx = x2 - x1;
  const dy = y2 - y1;
  let t0 = 0.0, t1 = 1.0;

  const edges: [number, number][] = [
    [-dx, x1 - xmin], [dx, xmax - x1],
    [-dy, y1 - ymin], [dy, ymax - y1],
  ];

  for (const [p, q] of edges) {
    if (p === 0) {
      if (q < 0) return false;
    } else {
      const r = q / p;
      if (p < 0) {
        t0 = Math.max(t0, r);
      } else {
        t1 = Math.min(t1, r);
      }
      if (t0 > t1) return false;
    }
  }
  return true;
}

/**
 * Check if a LineString (array of [lon, lat]) intersects a bounding box.
 * Tests each segment using Liang-Barsky clipping.
 *
 * Same algorithm as backend proximity_router.py `_line_intersects_bbox`.
 */
function lineIntersectsBBox(coords: number[][], bbox: BBox): boolean {
  const { west: xmin, south: ymin, east: xmax, north: ymax } = bbox;
  for (let i = 0; i < coords.length - 1; i++) {
    const [x1, y1] = coords[i];
    const [x2, y2] = coords[i + 1];
    // Skip antimeridian-crossing segments (large longitude jump).
    // These are orbital track dateline transitions, not real ground coverage.
    if (Math.abs(x2 - x1) > 180) continue;
    if (liangBarskyClip(x1, y1, x2, y2, xmin, ymin, xmax, ymax)) {
      return true;
    }
  }
  return false;
}

/** Pre-computed feature with its geometry info */
interface IndexedFeature {
  productId: string;
  bbox: BBox;
  isLine: boolean;
  coords: number[][] | null;  // LineString coordinates for Liang-Barsky testing
}

/**
 * Geometry-aware overlap test between two features.
 * Uses Liang-Barsky line clipping for LineString geometries (SHARAD)
 * to precisely test segment-vs-bbox intersection.
 *
 * Same logic as backend proximity_router.py `_geometries_overlap`.
 */
function featuresOverlap(a: IndexedFeature, b: IndexedFeature): boolean {
  // Quick rejection: overall AABBs don't overlap
  if (!bboxIntersects(a.bbox, b.bbox)) return false;

  if (a.isLine && b.isLine) {
    // Both lines: check if either line's segments intersect the other's bbox
    return (
      lineIntersectsBBox(a.coords!, b.bbox) ||
      lineIntersectsBBox(b.coords!, a.bbox)
    );
  }

  if (a.isLine) {
    // a is LineString, b is Polygon — check line segments vs b's bbox
    return lineIntersectsBBox(a.coords!, b.bbox);
  }

  if (b.isLine) {
    // a is Polygon, b is LineString — check line segments vs a's bbox
    return lineIntersectsBBox(b.coords!, a.bbox);
  }

  // Both polygons: bbox overlap is sufficient
  return true;
}

/**
 * Simple spatial grid index for fast overlap candidate lookup.
 * Divides lat/lon space into cells; features are inserted into
 * all cells their bbox intersects. Queries return only features
 * in overlapping cells (O(1) lookup vs O(n) linear scan).
 */
const GRID_CELL_SIZE = 5; // degrees — 72×36 cells for full Mars

class SpatialGrid {
  private cells = new Map<string, IndexedFeature[]>();

  private getCellKey(latIdx: number, lonIdx: number): string {
    return `${latIdx},${lonIdx}`;
  }

  /** Insert into all grid cells that a bbox covers (handles antimeridian crossing) */
  insert(feature: IndexedFeature): void {
    const { bbox } = feature;
    const minLatIdx = Math.floor(bbox.south / GRID_CELL_SIZE);
    const maxLatIdx = Math.floor(bbox.north / GRID_CELL_SIZE);

    // Handle antimeridian crossing (west > east)
    const crossesAM = bbox.west > bbox.east;
    const minLonIdx = Math.floor(bbox.west / GRID_CELL_SIZE);
    const maxLonIdx = Math.floor(bbox.east / GRID_CELL_SIZE);

    const addToCell = (li: number, lo: number) => {
      const key = this.getCellKey(li, lo);
      let cell = this.cells.get(key);
      if (!cell) {
        cell = [];
        this.cells.set(key, cell);
      }
      cell.push(feature);
    };

    for (let li = minLatIdx; li <= maxLatIdx; li++) {
      if (crossesAM) {
        // West side: from west to +180
        const maxLonCellWest = Math.floor(180 / GRID_CELL_SIZE);
        for (let lo = minLonIdx; lo <= maxLonCellWest; lo++) addToCell(li, lo);
        // East side: from -180 to east
        const minLonCellEast = Math.floor(-180 / GRID_CELL_SIZE);
        for (let lo = minLonCellEast; lo <= maxLonIdx; lo++) addToCell(li, lo);
      } else {
        for (let lo = minLonIdx; lo <= maxLonIdx; lo++) addToCell(li, lo);
      }
    }
  }

  /** Get candidate features that MAY overlap the given bbox (handles antimeridian) */
  getCandidates(bbox: BBox): IndexedFeature[] {
    const minLatIdx = Math.floor(bbox.south / GRID_CELL_SIZE);
    const maxLatIdx = Math.floor(bbox.north / GRID_CELL_SIZE);

    const crossesAM = bbox.west > bbox.east;
    const minLonIdx = Math.floor(bbox.west / GRID_CELL_SIZE);
    const maxLonIdx = Math.floor(bbox.east / GRID_CELL_SIZE);

    const seen = new Set<string>();
    const candidates: IndexedFeature[] = [];

    const collectFromCell = (li: number, lo: number) => {
      const cell = this.cells.get(this.getCellKey(li, lo));
      if (!cell) return;
      for (const f of cell) {
        if (!seen.has(f.productId)) {
          seen.add(f.productId);
          candidates.push(f);
        }
      }
    };

    for (let li = minLatIdx; li <= maxLatIdx; li++) {
      if (crossesAM) {
        const maxLonCellWest = Math.floor(180 / GRID_CELL_SIZE);
        for (let lo = minLonIdx; lo <= maxLonCellWest; lo++) collectFromCell(li, lo);
        const minLonCellEast = Math.floor(-180 / GRID_CELL_SIZE);
        for (let lo = minLonCellEast; lo <= maxLonIdx; lo++) collectFromCell(li, lo);
      } else {
        for (let lo = minLonIdx; lo <= maxLonIdx; lo++) collectFromCell(li, lo);
      }
    }
    return candidates;
  }
}

/**
 * Core overlap computation.
 *
 * For each product p from instrument X in selectedInstruments:
 *   p passes iff for at least one other instrument Y (Y ≠ X),
 *   there exists at least one product q in Y that overlaps p.
 *
 * Uses a spatial grid index for O(k) candidate lookup instead of O(n) linear scan.
 */
export function computeOverlapFilter(
  featuresByInstrument: Map<InstrumentType, OverlapFeature[]>,
  selectedInstruments: InstrumentType[],
): { result: OverlapResult; stats: OverlapStats } {
  // 1. Pre-compute bboxes and build spatial grid per instrument
  const indexed = new Map<InstrumentType, IndexedFeature[]>();
  const grids = new Map<InstrumentType, SpatialGrid>();

  for (const inst of selectedInstruments) {
    const features = featuresByInstrument.get(inst);
    if (!features) continue;
    const items: IndexedFeature[] = [];
    const grid = new SpatialGrid();
    for (const f of features) {
      const bbox = getBoundingBox(f);
      if (bbox && f.properties.product_id) {
        const geom = f.geometry;
        const isLine = geom.type === "LineString" && geom.coordinates?.length > 1;
        const item: IndexedFeature = {
          productId: f.properties.product_id,
          bbox,
          isLine,
          coords: isLine ? geom.coordinates : null,
        };
        items.push(item);
        grid.insert(item);
      }
    }
    if (items.length > 0) {
      indexed.set(inst, items);
      grids.set(inst, grid);
    }
  }

  const result: OverlapResult = new Map();
  const perInstrument = new Map<InstrumentType, { checked: number; passing: number }>();
  let totalChecked = 0;
  let totalPassing = 0;

  // 2. For each instrument, check each product against spatial grid of other instruments
  for (const instX of selectedInstruments) {
    const itemsX = indexed.get(instX);
    if (!itemsX) {
      perInstrument.set(instX, { checked: 0, passing: 0 });
      continue;
    }

    const passingIds = new Set<string>();
    const otherInstruments = selectedInstruments.filter(i => i !== instX && grids.has(i));

    for (const px of itemsX) {
      let passes = false;

      for (const instY of otherInstruments) {
        // Use spatial grid for candidate lookup instead of scanning all features
        const grid = grids.get(instY)!;
        const candidates = grid.getCandidates(px.bbox);

        for (const py of candidates) {
          if (featuresOverlap(px, py)) {
            passes = true;
            break;
          }
        }

        if (passes) break;
      }

      if (passes) {
        passingIds.add(px.productId);
      }
    }

    result.set(instX, passingIds);
    perInstrument.set(instX, { checked: itemsX.length, passing: passingIds.size });
    totalChecked += itemsX.length;
    totalPassing += passingIds.size;
  }

  return { result, stats: { totalChecked, totalPassing, perInstrument } };
}
