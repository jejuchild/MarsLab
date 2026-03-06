/**
 * API client for the Mars Ice Accessibility algorithm.
 */

/* =========================================================
 * Types
 * =======================================================*/

export interface AccessibilityScore {
  lat: number;
  lon: number;
  score: number;
  ice_presence: number;
  ice_depth: number;
  excavation: number;
  landing: number;
  weights: Record<string, number>;
  inputs: Record<string, number | string | null>;
  layers_available: number;
  layers_total: number;
  confidence: "high" | "medium" | "low" | "insufficient";
}

export interface AccessibilityExplanation extends AccessibilityScore {
  explanation: string;
}

export interface LandformCacheEntry {
  product_id: string;
  lat: number;
  lon: number;
  dominant_class: string;
  confidence: number;
  model_version: string;
  classified_at: string;
}

export interface LandformCacheResponse {
  count: number;
  entries: LandformCacheEntry[];
}

export interface AccessibilityWeights {
  ice_presence: number;
  ice_depth: number;
  excavation: number;
  landing: number;
}

export const DEFAULT_WEIGHTS: AccessibilityWeights = {
  ice_presence: 0.35,
  ice_depth: 0.25,
  excavation: 0.20,
  landing: 0.20,
};

/* =========================================================
 * Fetchers
 * =======================================================*/

export async function fetchAccessibilityScore(
  lat: number,
  lon: number,
  weights?: Partial<AccessibilityWeights>,
  landform?: string,
  signal?: AbortSignal,
): Promise<AccessibilityScore> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  if (landform) params.set("landform", landform);
  if (weights) {
    if (weights.ice_presence != null) params.set("w_ice", weights.ice_presence.toString());
    if (weights.ice_depth != null) params.set("w_depth", weights.ice_depth.toString());
    if (weights.excavation != null) params.set("w_excavation", weights.excavation.toString());
    if (weights.landing != null) params.set("w_landing", weights.landing.toString());
  }
  const res = await fetch(`/api/accessibility/score?${params}`, { signal });
  if (!res.ok) throw new Error(`Accessibility query failed: ${res.status}`);
  return res.json();
}

/* =========================================================
 * Tile URL helper
 * =======================================================*/

export function getAccessibilityTileUrl(
  weights?: Partial<AccessibilityWeights>,
): string {
  let base = "/api/accessibility/tile/{z}/{x}/{y}.png";
  if (weights) {
    const params = new URLSearchParams();
    if (weights.ice_presence != null) params.set("w_ice", weights.ice_presence.toString());
    if (weights.ice_depth != null) params.set("w_depth", weights.ice_depth.toString());
    if (weights.excavation != null) params.set("w_excavation", weights.excavation.toString());
    if (weights.landing != null) params.set("w_landing", weights.landing.toString());
    const qs = params.toString();
    if (qs) base += `?${qs}`;
  }
  return base;
}


/* =========================================================
 * Explain fetcher (score + LLM explanation)
 * =======================================================*/

export async function fetchAccessibilityExplanation(
  lat: number,
  lon: number,
  weights?: Partial<AccessibilityWeights>,
  landform?: string,
): Promise<AccessibilityExplanation> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  if (landform) params.set("landform", landform);
  if (weights) {
    if (weights.ice_presence != null) params.set("w_ice", weights.ice_presence.toString());
    if (weights.ice_depth != null) params.set("w_depth", weights.ice_depth.toString());
    if (weights.excavation != null) params.set("w_excavation", weights.excavation.toString());
    if (weights.landing != null) params.set("w_landing", weights.landing.toString());
  }
  const res = await fetch(`/api/accessibility/explain?${params}`);
  if (!res.ok) throw new Error(`Accessibility explain failed: ${res.status}`);
  return res.json();
}

/* =========================================================
 * Fusion tile URL helper
 * =======================================================*/

export function getFusionTileUrl(
  weights?: Partial<AccessibilityWeights>,
): string {
  let base = "/api/accessibility/fusion-tile/{z}/{x}/{y}.png";
  if (weights) {
    const params = new URLSearchParams();
    if (weights.ice_presence != null) params.set("w_ice", weights.ice_presence.toString());
    if (weights.ice_depth != null) params.set("w_depth", weights.ice_depth.toString());
    if (weights.excavation != null) params.set("w_excavation", weights.excavation.toString());
    if (weights.landing != null) params.set("w_landing", weights.landing.toString());
    const qs = params.toString();
    if (qs) base += `?${qs}`;
  }
  return base;
}

/* =========================================================
 * Landform cache API (fusion integration)
 * =======================================================*/

export async function fetchLandformCache(): Promise<LandformCacheResponse> {
  if (!_landformCachePromise) {
    _landformCachePromise = fetch("/api/accessibility/landform-cache")
      .then((res) => {
        if (!res.ok) throw new Error(`Landform cache fetch failed: ${res.status}`);
        return res.json();
      })
      .catch((error) => {
        _landformCachePromise = null;
        throw error;
      });
  }
  return _landformCachePromise;
}

let _landformCachePromise: Promise<LandformCacheResponse> | null = null;

export function clearLandformCache(): void {
  _landformCachePromise = null;
}

export async function registerLandform(
  productId: string,
  lat: number,
  lon: number,
  dominantClass: string,
  confidence: number,
  modelVersion: string = "",
): Promise<void> {
  const params = new URLSearchParams({
    product_id: productId,
    lat: lat.toString(),
    lon: lon.toString(),
    dominant_class: dominantClass,
    confidence: confidence.toString(),
    model_version: modelVersion,
  });
  const res = await fetch(`/api/accessibility/register-landform?${params}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Landform registration failed: ${res.status}`);
}
