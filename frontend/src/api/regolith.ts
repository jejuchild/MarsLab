/**
 * API client for Regolith Thickness Estimator (RTE).
 */

export interface RegolithSample {
  trace_idx: number;
  lat: number;
  lon: number;
  along_track_km: number;
  surface_elev_m: number;
  interface_detected: boolean;
  delta_bins?: number | null;
  twt_us?: number | null;
  thickness_m?: number | null;
  thickness_low_m?: number | null;
  thickness_high_m?: number | null;
  snr?: number | null;
  confidence?: number | null;
  ringing_rejected?: boolean;
  clutter_available?: boolean;
  clutter_flagged?: boolean;
  clutter_snr?: number | null;
  mode?: string;
}

export interface OverlaySegment {
  start_lat: number;
  start_lon: number;
  end_lat: number;
  end_lon: number;
  thickness_m?: number | null;
  color: [number, number, number, number]; // RGBA 0-255
}

export interface RegolithSummary {
  product_id: string;
  epsilon_r: number;
  total_traces: number;
  valid_traces: number;
  detection_rate: number;
  thickness_mean_m?: number | null;
  thickness_median_m?: number | null;
  thickness_std_m?: number | null;
  thickness_min_m?: number | null;
  thickness_max_m?: number | null;
  mean_snr?: number | null;
  mean_confidence?: number | null;
  dem_source: string;
  total_distance_km: number;
  shallow_mode_enabled?: boolean;
  ring_reject_rate?: number;
  clutter_available?: boolean;
  clutter_flag_rate?: number;
  epsilon_uncertainty?: number;
}

export interface RegolithParameters {
  epsilon_r: number;
  snr_threshold: number;
  search_lo: number;
  search_hi: number;
  dem_source: string;
  speed_of_light_mps: number;
  sample_interval_us: number;
  mode?: string;
  epsilon_uncertainty?: number;
  clutter_mode?: string;
  clutter_snr_threshold?: number;
  clutter_bin_tolerance?: number;
}

export interface RegolithResult {
  success: boolean;
  error?: string | null;
  summary?: RegolithSummary | null;
  profile: RegolithSample[];
  overlay_segments: OverlaySegment[];
  parameters?: RegolithParameters | null;
}

export async function fetchRegolithProfile(
  productId: string,
  epsilonR: number = 2.5,
  snrThreshold: number = 3.5,
  searchLo: number = 10,
  searchHi: number = 150,
  options?: {
    dtm_product_id?: string;
    mode?: string;
    epsilon_uncertainty?: number;
    clutter_mode?: string;
    clutter_snr_threshold?: number;
    clutter_bin_tolerance?: number;
  },
): Promise<RegolithResult> {
  const params = new URLSearchParams({
    product_id: productId,
    epsilon_r: epsilonR.toString(),
    snr_threshold: snrThreshold.toString(),
    search_lo: searchLo.toString(),
    search_hi: searchHi.toString(),
  });

  if (options?.dtm_product_id) params.set("dtm_product_id", options.dtm_product_id);
  if (options?.mode) params.set("mode", options.mode);
  if (options?.epsilon_uncertainty !== undefined) params.set("epsilon_uncertainty", options.epsilon_uncertainty.toString());
  if (options?.clutter_mode) params.set("clutter_mode", options.clutter_mode);
  if (options?.clutter_snr_threshold !== undefined) params.set("clutter_snr_threshold", options.clutter_snr_threshold.toString());
  if (options?.clutter_bin_tolerance !== undefined) params.set("clutter_bin_tolerance", options.clutter_bin_tolerance.toString());

  const res = await fetch(`/api/regolith/thickness_profile?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `RTE failed (${res.status})`);
  }
  return res.json();
}

export function getExportCsvUrl(
  productId: string,
  epsilonR: number,
  snrThreshold: number,
  searchLo: number,
  searchHi: number,
  options?: {
    dtm_product_id?: string;
    mode?: string;
    epsilon_uncertainty?: number;
    clutter_mode?: string;
    clutter_snr_threshold?: number;
    clutter_bin_tolerance?: number;
  },
): string {
  const params = new URLSearchParams({
    product_id: productId,
    epsilon_r: epsilonR.toString(),
    snr_threshold: snrThreshold.toString(),
    search_lo: searchLo.toString(),
    search_hi: searchHi.toString(),
  });

  if (options?.dtm_product_id) params.set("dtm_product_id", options.dtm_product_id);
  if (options?.mode) params.set("mode", options.mode);
  if (options?.epsilon_uncertainty !== undefined) params.set("epsilon_uncertainty", options.epsilon_uncertainty.toString());
  if (options?.clutter_mode) params.set("clutter_mode", options.clutter_mode);
  if (options?.clutter_snr_threshold !== undefined) params.set("clutter_snr_threshold", options.clutter_snr_threshold.toString());
  if (options?.clutter_bin_tolerance !== undefined) params.set("clutter_bin_tolerance", options.clutter_bin_tolerance.toString());

  return `/api/regolith/export_csv?${params}`;
}
