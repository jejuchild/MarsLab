/**
 * API client for Crater Stratigraphy Analyzer.
 */

export interface CraterInfo {
  lat: number;
  lon: number;
  diameter_km: number;
  depth_m?: number | null;
  morphology?: string | null;
  n_terraces: number;
  terrace_depths: Record<string, unknown>[];
  dtm_source: string;
}

export interface RadialProfilePoint {
  radius_m: number;
  elevation_m: number;
  elevation_min?: number | null;
  elevation_max?: number | null;
}

export interface SharadPickInfo {
  product_id: string;
  n_picks: number;
  median_twt_us: number;
  twt_unc_us: number;
  min_distance_km: number;
}

export interface EpsilonEstimateInfo {
  depth_metric: string;
  depth_m: number;
  depth_unc_m: number;
  twt_us: number;
  epsilon_r: number;
  epsilon_low: number;
  epsilon_high: number;
  interpretation: string;
  quality: string;
  quality_note: string;
  sharad_product: string;
  n_picks: number;
}

export interface StratigraphyOverlayPoint {
  lat: number;
  lon: number;
  epsilon_r?: number | null;
  color: [number, number, number, number];
}

export interface StratigraphySummary {
  n_sharad_tracks: number;
  n_epsilon_estimates: number;
  best_epsilon_r?: number | null;
  best_interpretation?: string | null;
  best_quality?: string | null;
  dem_source: string;
  total_sharad_picks: number;
}

export interface StratigraphyParameters {
  crater_lat: number;
  crater_lon: number;
  diameter_km: number;
  buffer_km: number;
  n_azimuths: number;
  max_radius_m: number;
  min_snr: number;
}

export interface StratigraphyResult {
  success: boolean;
  error?: string | null;
  crater_info?: CraterInfo | null;
  summary?: StratigraphySummary | null;
  radial_profile: RadialProfilePoint[];
  epsilon_estimates: EpsilonEstimateInfo[];
  sharad_picks: SharadPickInfo[];
  overlay_points: StratigraphyOverlayPoint[];
  parameters?: StratigraphyParameters | null;
}

export async function fetchStratigraphyAnalysis(
  craterLat: number,
  craterLon: number,
  diameterKm: number = 0,
  bufferKm: number = 30,
  nAzimuths: number = 36,
  maxRadiusM: number = 3000,
  minSnr: number = 3.5,
): Promise<StratigraphyResult> {
  const params = new URLSearchParams({
    crater_lat: craterLat.toString(),
    crater_lon: craterLon.toString(),
    diameter_km: diameterKm.toString(),
    buffer_km: bufferKm.toString(),
    n_azimuths: nAzimuths.toString(),
    max_radius_m: maxRadiusM.toString(),
    min_snr: minSnr.toString(),
  });

  const res = await fetch(`/api/stratigraphy/analyze?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Stratigraphy analysis failed (${res.status})`);
  }
  return res.json();
}

export function getStratigraphyCsvUrl(
  craterLat: number,
  craterLon: number,
  diameterKm: number = 0,
  bufferKm: number = 30,
  nAzimuths: number = 36,
  maxRadiusM: number = 3000,
  minSnr: number = 3.5,
): string {
  const params = new URLSearchParams({
    crater_lat: craterLat.toString(),
    crater_lon: craterLon.toString(),
    diameter_km: diameterKm.toString(),
    buffer_km: bufferKm.toString(),
    n_azimuths: nAzimuths.toString(),
    max_radius_m: maxRadiusM.toString(),
    min_snr: minSnr.toString(),
  });
  return `/api/stratigraphy/export_csv?${params}`;
}
