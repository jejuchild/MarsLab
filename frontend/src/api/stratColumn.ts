/**
 * API client for Stratigraphic Column Builder.
 */

export interface ColumnLayer {
  layer_idx: number;
  depth_top_m: number;
  depth_bottom_m: number;
  thickness_m: number;
  source: string;
  instrument: string;
  mineral_name?: string | null;
  geochem_group?: string | null;
  epsilon_r?: number | null;
  material_class?: string | null;
  color: [number, number, number, number];
  confidence?: number | null;
}

export interface ColumnSummary {
  crater_lat: number;
  crater_lon: number;
  diameter_km: number;
  n_layers: number;
  total_depth_m: number;
  instruments_used: string[];
  dtm_source: string;
  has_crism: boolean;
  has_sharad_subsurface: boolean;
  dominant_material?: string | null;
}

export interface ColumnParameters {
  crater_lat: number;
  crater_lon: number;
  diameter_km: number;
  buffer_km: number;
  include_crism: boolean;
  include_sharad: boolean;
}

export interface StratColumnResult {
  success: boolean;
  error?: string | null;
  summary?: ColumnSummary | null;
  layers: ColumnLayer[];
  rim_elevation_m?: number | null;
  parameters?: ColumnParameters | null;
}

export async function fetchStratColumn(
  craterLat: number,
  craterLon: number,
  diameterKm: number = 0,
  bufferKm: number = 30,
  includeCrism: boolean = true,
  includeSharad: boolean = true,
): Promise<StratColumnResult> {
  const params = new URLSearchParams({
    crater_lat: craterLat.toString(),
    crater_lon: craterLon.toString(),
    diameter_km: diameterKm.toString(),
    buffer_km: bufferKm.toString(),
    include_crism: includeCrism.toString(),
    include_sharad: includeSharad.toString(),
  });

  const res = await fetch(`/api/strat-column/build?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Stratigraphic column build failed (${res.status})`);
  }
  return res.json();
}

export function getStratColumnCsvUrl(
  craterLat: number,
  craterLon: number,
  diameterKm: number = 0,
  bufferKm: number = 30,
  includeCrism: boolean = true,
  includeSharad: boolean = true,
): string {
  const params = new URLSearchParams({
    crater_lat: craterLat.toString(),
    crater_lon: craterLon.toString(),
    diameter_km: diameterKm.toString(),
    buffer_km: bufferKm.toString(),
    include_crism: includeCrism.toString(),
    include_sharad: includeSharad.toString(),
  });
  return `/api/strat-column/export_csv?${params}`;
}
