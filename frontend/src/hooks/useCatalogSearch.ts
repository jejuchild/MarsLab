import { useCallback, useMemo } from "react";
import catalogData from "../data/mars_catalog.json";

/* =========================================================
 * Types
 * =======================================================*/

export type Lane = "SHARAD" | "CRISM" | "HIRISE" | "CTX";

export type CatalogEntry = {
  id: string;
  name: string;
  lat: number;
  lon: number;
  zoom_km: number;
  keywords: string[];
  description: string;
};

export type SearchResult =
  | { type: "catalog"; entry: CatalogEntry }
  | { type: "coordinate"; lat: number; lon: number }
  | { type: "product"; productId: string; instrument: Lane }
  | { type: "easter_egg"; eggId: string }
  | { type: "none" };

const CATALOG: CatalogEntry[] = catalogData as CatalogEntry[];

/* =========================================================
 * Easter eggs (kept per Phase 1 Q4=A decision)
 * =======================================================*/

const EASTER_EGGS: Record<string, string> = {
  game: "game",
  "space game": "game",
  shooter: "game",
  terraform: "terraform",
  watney: "watney",
  "mark watney": "watney",
  curiosity: "curiosity",
  "curiosity selfie": "curiosity",
  selfie: "curiosity",
};

/* =========================================================
 * Coordinate parser
 * =======================================================*/

/** Parse strings like "18.4, 77.7" / "18.4N 77.7E" / "-23.98 -33.30" */
function parseCoordinate(query: string): { lat: number; lon: number } | null {
  const trimmed = query.trim();

  // Pattern A: "lat, lon" or "lat lon" with optional N/S/E/W suffix
  const m = trimmed.match(
    /^([+-]?\d+(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,\s]\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*([EWew])?$/
  );
  if (!m) return null;

  let lat = parseFloat(m[1]!);
  const latSuffix = m[2]?.toUpperCase();
  let lon = parseFloat(m[3]!);
  const lonSuffix = m[4]?.toUpperCase();

  if (latSuffix === "S") lat = -lat;
  if (lonSuffix === "W") lon = -lon;

  if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;

  return { lat, lon };
}

/* =========================================================
 * Product ID detector
 * =======================================================*/

/**
 * Detect a product_id and infer its instrument from the prefix.
 * Patterns:
 *   ESP_NNNNNN_NNNN, PSP_NNNNNN_NNNN  → HIRISE
 *   frtNNNNNNNN, hrlNNNNNNNN, frsNNNNNNNN, atoNNNNNNNN, atuNNNNNNNN, mrlNNNNNNNN  → CRISM
 *   s_NNNNNNNN_NNN  → SHARAD
 *   B01_NNNNNN_NNNN, J01_NNNNNN_NNNN, etc.  → CTX
 */
function parseProductId(query: string): { productId: string; instrument: Lane } | null {
  const q = query.trim();
  if (!q) return null;

  if (/^(ESP|PSP)_\d{6}_\d{4}/i.test(q)) return { productId: q, instrument: "HIRISE" };
  if (/^(frt|hrl|frs|ato|atu|mrl)\w{8}/i.test(q)) return { productId: q, instrument: "CRISM" };
  if (/^s_\d{8}_\d{3}/i.test(q)) return { productId: q, instrument: "SHARAD" };
  if (/^[A-J]\d{2}_\d{6}_\d{4}/i.test(q)) return { productId: q, instrument: "CTX" };
  return null;
}

/* =========================================================
 * Catalog matcher (fuzzy)
 * =======================================================*/

/** Score 0-100. Higher is better match. 0 means no match. */
function scoreCatalogMatch(entry: CatalogEntry, query: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 0;

  const name = entry.name.toLowerCase();
  const id = entry.id.toLowerCase();

  // Exact name/id match → 100
  if (name === q || id === q) return 100;
  // Name starts with query → 90
  if (name.startsWith(q)) return 90;
  // ID starts with query → 85
  if (id.startsWith(q)) return 85;
  // Name contains query as a word → 75
  if (name.includes(q)) return 75;
  // Keyword exact → 70
  if (entry.keywords.some((k) => k.toLowerCase() === q)) return 70;
  // Keyword startswith → 60
  if (entry.keywords.some((k) => k.toLowerCase().startsWith(q))) return 60;
  // Keyword includes → 50
  if (entry.keywords.some((k) => k.toLowerCase().includes(q))) return 50;
  // Description includes → 30
  if (entry.description.toLowerCase().includes(q)) return 30;

  return 0;
}

/* =========================================================
 * Hook
 * =======================================================*/

export interface UseCatalogSearchReturn {
  /** All catalog entries (read-only). */
  catalog: CatalogEntry[];
  /**
   * Parse a query string into a SearchResult.
   * Priority: easter egg → catalog → coordinate → product_id → none.
   */
  parse: (query: string) => SearchResult;
  /**
   * Returns the top suggestions for a partial query (live dropdown).
   * Includes catalog matches sorted by score, plus a coordinate hint
   * and a product_id hint if those parse cleanly.
   */
  suggestions: (query: string, limit?: number) => SearchResult[];
}

export default function useCatalogSearch(): UseCatalogSearchReturn {
  const catalog = useMemo(() => CATALOG, []);

  const parse = useCallback((query: string): SearchResult => {
    const q = query.trim();
    if (!q) return { type: "none" };

    // 1. Easter egg
    const eggKey = q.toLowerCase();
    if (EASTER_EGGS[eggKey]) {
      return { type: "easter_egg", eggId: EASTER_EGGS[eggKey]! };
    }

    // 2. Catalog (best fuzzy match — only if score >= 50)
    let best: { entry: CatalogEntry; score: number } | null = null;
    for (const entry of CATALOG) {
      const score = scoreCatalogMatch(entry, q);
      if (score >= 50 && (!best || score > best.score)) {
        best = { entry, score };
      }
    }
    if (best) return { type: "catalog", entry: best.entry };

    // 3. Coordinate
    const coord = parseCoordinate(q);
    if (coord) return { type: "coordinate", ...coord };

    // 4. Product ID
    const product = parseProductId(q);
    if (product) return { type: "product", ...product };

    return { type: "none" };
  }, []);

  const suggestions = useCallback(
    (query: string, limit = 5): SearchResult[] => {
      const q = query.trim();
      if (!q) return [];

      const out: SearchResult[] = [];

      // Catalog matches (top N by score)
      const matches = CATALOG
        .map((entry) => ({ entry, score: scoreCatalogMatch(entry, q) }))
        .filter((m) => m.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
      for (const m of matches) {
        out.push({ type: "catalog", entry: m.entry });
      }

      // Coordinate hint
      const coord = parseCoordinate(q);
      if (coord) out.push({ type: "coordinate", ...coord });

      // Product hint
      const product = parseProductId(q);
      if (product) out.push({ type: "product", ...product });

      return out.slice(0, limit);
    },
    []
  );

  return { catalog, parse, suggestions };
}
