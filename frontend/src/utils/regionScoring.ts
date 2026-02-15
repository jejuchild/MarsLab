/**
 * Region scoring types and helpers for the Mars Region Dashboard.
 *
 * Scores are fetched from the backend `/api/regions/scores` endpoint
 * which computes real values from DEM, CRISM, SHARAD, and local indices.
 * Sub-scores can be `null` when insufficient data exists for a category.
 */

export interface RegionScores {
  ice: number | null;
  slope: number | null;
  subsurface: number | null;
  coverage: number | null;
}

export interface RegionScoresWithComposite extends RegionScores {
  composite: number | null;
}

export interface RegionScoreDetails {
  slope_safety: string | null;
  slope_mean: number | null;
  slope_favorable_pct: number | null;
  crism_count: number;
  scored_count: number;
  high_ice_count: number;
  sharad_track_count: number;
  instruments_with_data: string[];
  product_counts: Record<string, number>;
}

export interface RegionScoreEntry extends RegionScoresWithComposite {
  details: RegionScoreDetails;
}

export interface RegionScoresResponse {
  scores: Record<string, RegionScoreEntry>;
  computed_at: string;
  is_cached: boolean;
  computation_time_s: number;
  regions_scored: number;
}

/* ---------------------------------------------------------------
 * Composite calculation (handles null sub-scores)
 * --------------------------------------------------------------- */

const WEIGHTS = {
  subsurface: 0.30,
  ice: 0.25,
  slope: 0.25,
  coverage: 0.20,
};

/**
 * Weighted composite of available (non-null) scores.
 * Weights are renormalized to sum to 1 over available scores only.
 * Returns null if all sub-scores are null.
 */
export function computeComposite(scores: RegionScores): number | null {
  const parts: { value: number; weight: number }[] = [];

  if (scores.subsurface !== null) parts.push({ value: scores.subsurface, weight: WEIGHTS.subsurface });
  if (scores.ice !== null) parts.push({ value: scores.ice, weight: WEIGHTS.ice });
  if (scores.slope !== null) parts.push({ value: scores.slope, weight: WEIGHTS.slope });
  if (scores.coverage !== null) parts.push({ value: scores.coverage, weight: WEIGHTS.coverage });

  if (parts.length === 0) return null;

  const totalWeight = parts.reduce((sum, p) => sum + p.weight, 0);
  return Math.round(
    parts.reduce((sum, p) => sum + p.value * (p.weight / totalWeight), 0),
  );
}

/* ---------------------------------------------------------------
 * Color helpers (null-safe)
 * --------------------------------------------------------------- */

/**
 * Tailwind text-color class. Returns neutral gray for null.
 *   >70 green, 40-70 amber, <40 red, null gray
 */
export function scoreColor(score: number | null): string {
  if (score === null) return "text-[#4a5a78]";
  if (score > 70) return "text-emerald-400";
  if (score >= 40) return "text-amber-400";
  return "text-red-400";
}

/**
 * Raw hex color for canvas / inline-style. Returns gray for null.
 */
export function scoreColorHex(score: number | null): string {
  if (score === null) return "#4a5a78";
  if (score > 70) return "#34d399";
  if (score >= 40) return "#fbbf24";
  return "#f87171";
}

/**
 * Tailwind bg-color class for bar fills.
 */
export function scoreBarBg(score: number | null): string {
  if (score === null) return "bg-[#2a3444]";
  if (score > 70) return "bg-emerald-400";
  if (score >= 40) return "bg-amber-400";
  return "bg-red-400";
}

/** Default empty scores (all null). */
export const EMPTY_SCORES: RegionScoresWithComposite = {
  ice: null,
  slope: null,
  subsurface: null,
  coverage: null,
  composite: null,
};
