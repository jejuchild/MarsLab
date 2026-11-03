/**
 * Geographic bounds of a product (cached), from its PDS label or index.
 * Shared by the map overlays and the image viewers (moved verbatim from
 * MapView so viewers can place the inspected point on the image).
 */
import { parseLBLValue, loadCRISMLBL } from "./pdsLabels";
import { LRUMap } from "./LRUMap";
import { normalizeLonForMap } from "./coordinates";

const normalizeLonTo180 = normalizeLonForMap;
/**
 * Extract CRISM observation ID from full product ID
 * e.g., "frt0001fd76_07_if166j_mtr3" -> "frt0001fd76"
 *
 * Pattern: 3-letter prefix + 8 hex characters
 */
export function extractCrismObsId(productId: string): string {
  // Match 3 letters followed by 8 hex digits at the start
  const match = productId.match(/^([a-z]{3}[0-9a-f]{8})/i);
  const obsId = match?.[1];
  return obsId ? obsId.toLowerCase() : productId.toLowerCase();
}

/* ==================================================
 * LBL loaders with bounds caching
 * ==================================================*/
const hiriseLBLCache = new Map<string, string>();
const HIRISE_LBL_BASE = "/hirise_lbl";

// PERFORMANCE: Cache parsed bounds to avoid re-parsing LBL files
export interface ProductBounds {
  west: number;
  south: number;
  east: number;
  north: number;
  lines?: number;
  samples?: number;
  polygon?: [number, number][];  // actual polygon corners [lon, lat][] for precise overlay
}
export const boundsCache = new LRUMap<string, ProductBounds>(500);

export async function loadHiRISELBL(id: string): Promise<string | null> {
  if (hiriseLBLCache.has(id)) return hiriseLBLCache.get(id)!;

  // Try multiple path patterns: subdirectory (new downloads) and flat (legacy)
  const patterns = [
    `${HIRISE_LBL_BASE}/${id}_RED/${id}_RED.LBL`,
    `${HIRISE_LBL_BASE}/${id}_RED.lbl`,
    `${HIRISE_LBL_BASE}/${id}_RED/${id}_RED.lbl`,
  ];

  for (const url of patterns) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const text = await res.text();
      if (!text.includes("IMAGE_MAP_PROJECTION")) continue;
      hiriseLBLCache.set(id, text);
      return text;
    } catch {
      continue;
    }
  }

  return null;
}


type GeoJsonIndexFeature = {
  properties?: {
    product_id?: string;
    west?: number;
    east?: number;
    south?: number;
    north?: number;
  };
  geometry?: {
    coordinates?: number[][][];
  };
};

type GeoJsonIndex = {
  features?: GeoJsonIndexFeature[];
};

// Cache for HiRISE index
let hiriseIndexCache: GeoJsonIndex | null = null;
// Cache for HiRISE DTM index
let hiriseDTMIndexCache: GeoJsonIndex | null = null;
// Cache for CRISM TRR3 index
let crismTRR3IndexCache: GeoJsonIndex | null = null;

// Compute bounding box from GeoJSON polygon coordinates.
// Longitudes are normalized to [-180, 180] (indexes mix 0-360 and ±180), so
// a footprint straddling 0°/360° no longer yields a ~360°-wide box. A
// footprint straddling ±180° gets west > east, which Cesium's Rectangle
// treats as crossing the antimeridian. Previously both cases produced a
// near-global rectangle with the image stretched across the planet.
export function boundsFromPolygon(coords: number[][]): ProductBounds | null {
  if (!coords || coords.length < 3) return null;
  const lons: number[] = [];
  let south = Infinity, north = -Infinity;
  for (const pair of coords) {
    const lon = pair[0];
    const lat = pair[1];
    if (lon == null || lat == null) continue;
    lons.push(lon > 180 ? lon - 360 : lon < -180 ? lon + 360 : lon);
    if (lat < south) south = lat;
    if (lat > north) north = lat;
  }
  if (lons.length < 3) return null;
  let west = Math.min(...lons);
  let east = Math.max(...lons);
  if (east - west > 180) {
    // Crosses the antimeridian: measure in 0-360 instead.
    const shifted = lons.map((l) => (l < 0 ? l + 360 : l));
    west = Math.min(...shifted);
    east = Math.max(...shifted) - 360;
  }
  return { west, east, south, north };
}

// Get cached bounds or parse from LBL
export async function getProductBounds(productId: string): Promise<ProductBounds | null> {
  if (boundsCache.has(productId)) {
    return boundsCache.get(productId)!;
  }

  // Check for HiRISE DTM products (start with DTE, e.g. DTEEC_, DTEED_)
  const isHiRISEDTM = productId.startsWith("DTE");

  if (isHiRISEDTM) {
    if (!hiriseDTMIndexCache) {
      try {
        const res = await fetch("/hirise_dtm_index.geojson");
        if (res.ok) {
          hiriseDTMIndexCache = await res.json();
        }
      } catch {
        return null;
      }
    }

    if (hiriseDTMIndexCache?.features) {
      for (const feature of hiriseDTMIndexCache.features) {
        if (feature.properties?.product_id === productId) {
          const props = feature.properties;
          if (
            props.west == null ||
            props.east == null ||
            props.south == null ||
            props.north == null
          ) {
            continue;
          }
          const bounds: ProductBounds = {
            west: props.west,
            east: props.east,
            south: props.south,
            north: props.north,
          };
          boundsCache.set(productId, bounds);
          return bounds;
        }
      }
    }
    return null;
  }

  // Check for CRISM TRR3 products — bounds from index.geojson polygon
  const isTRR3 = /^(frs|msv|frt|hrl|hrs|arl|atl)[0-9a-f]+_\d{2}$/i.test(productId);
  if (isTRR3) {
    if (!crismTRR3IndexCache) {
      try {
        const res = await fetch("/crism_trr3_index.geojson");
        if (res.ok) {
          crismTRR3IndexCache = await res.json();
        }
      } catch {
        return null;
      }
    }

    if (crismTRR3IndexCache?.features) {
      for (const feature of crismTRR3IndexCache.features) {
        if (feature.properties?.product_id === productId) {
          const coords = feature.geometry?.coordinates?.[0] as [number, number][] | undefined;
          if (!coords || coords.length < 4) continue;
          const bbox = boundsFromPolygon(coords);
          if (bbox) {
            const bounds: ProductBounds = {
              ...bbox,
              polygon: coords.slice(0, -1),  // strip closing vertex
            };
            boundsCache.set(productId, bounds);
            return bounds;
          }
        }
      }
    }
    return null;
  }

  // Check for HiRISE products — bounds from index.geojson polygon
  const isHiRISE = productId.startsWith("ESP_") || productId.startsWith("PSP_");
  if (isHiRISE) {
    if (!hiriseIndexCache) {
      try {
        const res = await fetch("/hirise_index.geojson");
        if (res.ok) {
          hiriseIndexCache = await res.json();
        }
      } catch {
        // fall through to LBL parsing
      }
    }

    if (hiriseIndexCache?.features) {
      for (const feature of hiriseIndexCache.features) {
        if (feature.properties?.product_id === productId) {
          const props = feature.properties;
          const coords = feature.geometry?.coordinates?.[0] as [number, number][] | undefined;
          // Use polygon coordinates for precise overlay placement
          if (coords && coords.length >= 4) {
            const bbox = boundsFromPolygon(coords);
            if (bbox) {
              const bounds: ProductBounds = {
                ...bbox,
                polygon: coords.slice(0, -1),  // strip closing vertex
              };
              boundsCache.set(productId, bounds);
              return bounds;
            }
          }
          // Fallback to LBL bbox properties
          if (props.west != null && props.east != null && props.south != null && props.north != null) {
            const bounds: ProductBounds = {
              west: props.west, east: props.east,
              south: props.south, north: props.north,
            };
            boundsCache.set(productId, bounds);
            return bounds;
          }
          break;
        }
      }
    }
  }

  // Fallback: parse LBL file for precise bounds
  const lbl = isHiRISE ? await loadHiRISELBL(productId) : await loadCRISMLBL(productId);
  if (!lbl) return null;

  const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
  const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
  const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
  const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");
  const lines = parseLBLValue(lbl, "LINES");
  const samples = parseLBLValue(lbl, "LINE_SAMPLES");

  if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
    return null;
  }

  const bounds: ProductBounds = {
    west: normalizeLonTo180(westLon360),
    east: normalizeLonTo180(eastLon360),
    south: Math.min(minLat, maxLat),
    north: Math.max(minLat, maxLat),
    lines: lines ?? undefined,
    samples: samples ?? undefined,
  };

  boundsCache.set(productId, bounds);
  return bounds;
}

