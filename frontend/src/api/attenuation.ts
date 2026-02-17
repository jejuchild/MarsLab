/**
 * API client for Radar Attenuation Mapper.
 */

export interface AttenuationSample {
  trace_idx: number;
  lat: number;
  lon: number;
  along_track_km: number;
  surface_elev_m: number;
  interface_detected: boolean;
  surface_power_dB?: number | null;
  subsurface_power_dB?: number | null;
  depth_m?: number | null;
  alpha_dBm?: number | null;
  transparency?: string | null;
  snr?: number | null;
  confidence?: number | null;
}

export interface AttenuationOverlaySegment {
  start_lat: number;
  start_lon: number;
  end_lat: number;
  end_lon: number;
  alpha_dBm?: number | null;
  color: [number, number, number, number];
}

export interface AttenuationSummary {
  product_id: string;
  epsilon_r: number;
  total_traces: number;
  valid_traces: number;
  detection_rate: number;
  alpha_mean_dBm?: number | null;
  alpha_median_dBm?: number | null;
  alpha_std_dBm?: number | null;
  dominant_transparency?: string | null;
  transparency_counts: Record<string, number>;
  dem_source: string;
  total_distance_km: number;
}

export interface AttenuationParameters {
  epsilon_r: number;
  snr_threshold: number;
  search_lo: number;
  search_hi: number;
  dem_source: string;
}

export interface AttenuationResult {
  success: boolean;
  error?: string | null;
  summary?: AttenuationSummary | null;
  profile: AttenuationSample[];
  overlay_segments: AttenuationOverlaySegment[];
  parameters?: AttenuationParameters | null;
}

export async function fetchAttenuationProfile(
  productId: string,
  epsilonR: number = 2.5,
  snrThreshold: number = 3.5,
  searchLo: number = 10,
  searchHi: number = 150,
  dtmProductId: string = "",
): Promise<AttenuationResult> {
  const params = new URLSearchParams({
    product_id: productId,
    epsilon_r: epsilonR.toString(),
    snr_threshold: snrThreshold.toString(),
    search_lo: searchLo.toString(),
    search_hi: searchHi.toString(),
  });
  if (dtmProductId) params.set("dtm_product_id", dtmProductId);

  const res = await fetch(`/api/attenuation/profile?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Attenuation analysis failed (${res.status})`);
  }
  return res.json();
}

export function getAttenuationCsvUrl(
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
  return `/api/attenuation/export_csv?${params}`;
}
