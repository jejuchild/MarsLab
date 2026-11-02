import { normalizeLonForMap } from "./coordinates";

/** Numeric value of `KEY = value` in a PDS3 label (null if absent). */
export function parseLBLValue(
  block: string | null | undefined,
  key: string
): number | null {
  if (!block) return null;
  const m = block.match(new RegExp(`${key}\\s*=\\s*([-+0-9.eE]+)`, "i"));
  return m ? Number(m[1]) : null;
}

const crismLBLCache = new Map<string, string>();

/** Fetch (and cache) the map-projected CRISM MTRDR label for a product. */
export async function loadCRISMLBL(id: string): Promise<string | null> {
  if (crismLBLCache.has(id)) return crismLBLCache.get(id)!;
  const CRISM_LBL_BASE = "/crism_lbl";

  // Extract base_key from product_id
  const baseKey = id.replace(/_(?:if|br)[0-9a-z]+_mtr3$/i, "");

  const patterns = [
    `${CRISM_LBL_BASE}/${baseKey}_brcarj_mtr3.lbl`,
    `${CRISM_LBL_BASE}/${baseKey}.lbl`,
    `${CRISM_LBL_BASE}/${id}.lbl`,
    // Newer downloads live in per-observation folders (218 products);
    // without this they never got a click spectrum.
    `${CRISM_LBL_BASE}/${baseKey}/${id}.lbl`,
    `${CRISM_LBL_BASE}/${baseKey}/${baseKey}_if166j_mtr3.lbl`,
  ];

  for (const url of patterns) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;

      const text = await res.text();
      if (!text.includes("IMAGE_MAP_PROJECTION")) continue;

      crismLBLCache.set(id, text);
      return text;
    } catch {
      continue;
    }
  }

  return null;
}

/**
 * Pixel (line, sample) of a CRISM MTRDR product at lat/lon, or null.
 *
 * MTRDR products are map-projected. The linear bounding-box mapping is
 * valid for EQUIRECTANGULAR labels only; ~7% of products are POLAR
 * STEREOGRAPHIC, where it picked the wrong pixel — return null there
 * rather than show a spectrum from the wrong place. Also null when the
 * point lies outside the product.
 */
export async function crismPixelAt(
  productId: string,
  lat: number,
  lon: number,
): Promise<{ line: number; sample: number } | null> {
  const lbl = await loadCRISMLBL(productId);
  if (!lbl) return null;
  const projection = lbl.match(/MAP_PROJECTION_TYPE\s*=\s*"?([A-Z _]+)"?/i)?.[1]?.trim().toUpperCase();
  if (projection !== "EQUIRECTANGULAR") return null;

  const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
  const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
  const westRaw = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
  const eastRaw = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");
  const lines = parseLBLValue(lbl, "LINES");
  const samples = parseLBLValue(lbl, "LINE_SAMPLES");
  if (minLat == null || maxLat == null || westRaw == null || eastRaw == null || !lines || !samples) return null;

  const west = normalizeLonForMap(westRaw);
  const east = normalizeLonForMap(eastRaw);
  const south = Math.min(minLat, maxLat);
  const north = Math.max(minLat, maxLat);
  const latFrac = (north - lat) / (north - south);
  const lonFrac = (normalizeLonForMap(lon) - west) / (east - west);
  if (!(latFrac >= 0 && latFrac <= 1 && lonFrac >= 0 && lonFrac <= 1)) return null;
  return {
    line: Math.min(lines - 1, Math.floor(latFrac * lines)),
    sample: Math.min(samples - 1, Math.floor(lonFrac * samples)),
  };
}
