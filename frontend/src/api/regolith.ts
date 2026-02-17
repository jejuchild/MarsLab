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
  snr?: number | null;
  confidence?: number | null;
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
}

export interface RegolithParameters {
  epsilon_r: number;
  snr_threshold: number;
  search_lo: number;
  search_hi: number;
  dem_source: string;
  speed_of_light_mps: number;
  sample_interval_us: number;
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
  dtmProductId: string = "",
): Promise<RegolithResult> {
  const params = new URLSearchParams({
    product_id: productId,
    epsilon_r: epsilonR.toString(),
    snr_threshold: snrThreshold.toString(),
    search_lo: searchLo.toString(),
    search_hi: searchHi.toString(),
  });
  if (dtmProductId) params.set("dtm_product_id", dtmProductId);

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
  dtmProductId: string = "",
): string {
  const params = new URLSearchParams({
    product_id: productId,
    epsilon_r: epsilonR.toString(),
    snr_threshold: snrThreshold.toString(),
    search_lo: searchLo.toString(),
    search_hi: searchHi.toString(),
  });
  if (dtmProductId) params.set("dtm_product_id", dtmProductId);
  return `/api/regolith/export_csv?${params}`;
}
