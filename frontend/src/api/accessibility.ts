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
): Promise<AccessibilityScore> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  if (weights) {
    if (weights.ice_presence != null) params.set("w_ice", weights.ice_presence.toString());
    if (weights.ice_depth != null) params.set("w_depth", weights.ice_depth.toString());
    if (weights.excavation != null) params.set("w_excavation", weights.excavation.toString());
    if (weights.landing != null) params.set("w_landing", weights.landing.toString());
  }
  const res = await fetch(`/api/accessibility/score?${params}`);
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
): Promise<AccessibilityExplanation> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
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