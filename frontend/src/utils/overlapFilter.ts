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

// Small buffer (degrees) for LineString bboxes so they have spatial extent
const LINE_BUFFER_DEG = 0.05;

/**
 * Extract axis-aligned bounding box from a GeoJSON geometry.
 * - Polygon: min/max of outer ring coordinates
 * - LineString: min/max of all points + small buffer
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
    // Buffer to give lines spatial extent
    return {
      west: west - LINE_BUFFER_DEG,
      south: south - LINE_BUFFER_DEG,
      east: east + LINE_BUFFER_DEG,
      north: north + LINE_BUFFER_DEG,
    };
  }

  if (geom.type === "Point") {
    const [lon, lat] = geom.coordinates;
    return {
      west: lon - LINE_BUFFER_DEG,
      south: lat - LINE_BUFFER_DEG,
      east: lon + LINE_BUFFER_DEG,
      north: lat + LINE_BUFFER_DEG,
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
  // crossing covers [crossing.west, 180] ∪ [-180, crossing.east]
  // normal covers [normal.west, normal.east]
  return normal.east >= crossing.west || normal.west <= crossing.east;
}

/** Pre-computed feature with its bbox */
interface IndexedFeature {
  productId: string;
  bbox: BBox;
}

/**
 * Core overlap computation.
 *
 * For each product p from instrument X in selectedInstruments:
 *   p passes iff for EVERY other instrument Y in selectedInstruments (Y ≠ X),
 *   there exists at least one product q in Y whose bbox overlaps p's bbox.
 */
export function computeOverlapFilter(
  featuresByInstrument: Map<InstrumentType, OverlapFeature[]>,
  selectedInstruments: InstrumentType[],
): { result: OverlapResult; stats: OverlapStats } {
  // 1. Pre-compute bboxes for all features
  const indexed = new Map<InstrumentType, IndexedFeature[]>();
  for (const inst of selectedInstruments) {
    const features = featuresByInstrument.get(inst);
    if (!features) continue;
    const items: IndexedFeature[] = [];
    for (const f of features) {
      const bbox = getBoundingBox(f);
      if (bbox && f.properties.product_id) {
        items.push({ productId: f.properties.product_id, bbox });
      }
    }
    if (items.length > 0) {
      indexed.set(inst, items);
    }
  }

  const result: OverlapResult = new Map();
  const perInstrument = new Map<InstrumentType, { checked: number; passing: number }>();
  let totalChecked = 0;
  let totalPassing = 0;

  // 2. For each instrument, check each product against all other selected instruments
  for (const instX of selectedInstruments) {
    const itemsX = indexed.get(instX);
    if (!itemsX) {
      perInstrument.set(instX, { checked: 0, passing: 0 });
      continue;
    }

    const passingIds = new Set<string>();
    const otherInstruments = selectedInstruments.filter(i => i !== instX && indexed.has(i));

    for (const px of itemsX) {
      let passes = true;

      for (const instY of otherInstruments) {
        const itemsY = indexed.get(instY)!;
        let foundOverlap = false;

        for (const py of itemsY) {
          if (bboxIntersects(px.bbox, py.bbox)) {
            foundOverlap = true;
            break;
          }
        }

        if (!foundOverlap) {
          passes = false;
          break;
        }
      }

      if (passes && otherInstruments.length > 0) {
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
