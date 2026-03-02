/**
 * API client for SWIM (Subsurface Water Ice Mapping) ice detection endpoints.
 */

/* =========================================================
 * Types — per-method point responses
 * =======================================================*/

export interface SwimNeutronPoint {
  lat: number;
  lon: number;
  consistency_score: number | null;
  water_equivalent_h: number | null;
  depth_range: string;
  data_quality: string;
}

export interface SwimThermalPoint {
  lat: number;
  lon: number;
  consistency_score: number | null;
  thermal_inertia: number | null;
  depth_range: string;
  data_quality: string;
}

export interface SwimRadarSurfacePoint {
  lat: number;
  lon: number;
  consistency_score: number | null;
  surface_return_power: number | null;
  depth_range: string;
  data_quality: string;
}

export interface SwimRadarDielectricPoint {
  lat: number;
  lon: number;
  consistency_score: number | null;
  dielectric_constant: number | null;
  depth_range: string;
  data_quality: string;
}

export interface SwimGeomorphicPoint {
  lat: number;
  lon: number;
  consistency_score: number | null;
  landform_type: string | null;
  depth_range: string;
  data_quality: string;
}

/* =========================================================
 * Consistency / Fusion
 * =======================================================*/

export interface MethodScore {
  method: string;
  score: number | null;
  weight: number;
}

export interface SwimConsistencyPoint {
  lat: number;
  lon: number;
  consistency_score: number | null;
  method_scores: MethodScore[];
  mode: "precomputed" | "live";
  depth_to_ice_estimate_m: number | null;
}

export interface ConsistencyRegionStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  coverage_pct: number;
}

export interface SwimConsistencyRegion {
  bounds: { north: number; south: number; east: number; west: number };
  stats_0_1m: ConsistencyRegionStats;
  stats_1_5m: ConsistencyRegionStats;
  stats_5m_plus: ConsistencyRegionStats;
  tile_urls: Record<string, string>;
}

/* =========================================================
 * Depth range type
 * =======================================================*/

export type DepthRange = "0-1m" | "1-5m" | "5m-plus";

export const SWIM_METHODS = [
  "neutron",
  "thermal",
  "radar_surface",
  "radar_dielectric",
  "geomorphic",
] as const;

export type SwimMethod = (typeof SWIM_METHODS)[number];

/* =========================================================
 * Fetchers — individual methods
 * =======================================================*/

export async function fetchSwimNeutronPoint(
  lat: number,
  lon: number,
): Promise<SwimNeutronPoint> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  const res = await fetch(`/api/swim-ice/neutron/point?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM neutron query failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSwimThermalPoint(
  lat: number,
  lon: number,
): Promise<SwimThermalPoint> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  const res = await fetch(`/api/swim-ice/thermal/point?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM thermal query failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSwimRadarSurfacePoint(
  lat: number,
  lon: number,
): Promise<SwimRadarSurfacePoint> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  const res = await fetch(`/api/swim-ice/radar_surface/point?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM radar surface query failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSwimRadarDielectricPoint(
  lat: number,
  lon: number,
): Promise<SwimRadarDielectricPoint> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  const res = await fetch(`/api/swim-ice/radar_dielectric/point?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM radar dielectric query failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSwimGeomorphicPoint(
  lat: number,
  lon: number,
): Promise<SwimGeomorphicPoint> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  const res = await fetch(`/api/swim-ice/geomorphic/point?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM geomorphic query failed (${res.status})`);
  }
  return res.json();
}

/* =========================================================
 * Fetchers — consistency / fusion
 * =======================================================*/

export async function fetchSwimConsistency(
  lat: number,
  lon: number,
  mode?: "precomputed" | "live",
): Promise<SwimConsistencyPoint> {
  const params = new URLSearchParams({ lat: lat.toString(), lon: lon.toString() });
  if (mode) params.set("mode", mode);
  const res = await fetch(`/api/swim-ice/consistency/point?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM consistency query failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSwimCustomFusion(
  lat: number,
  lon: number,
  methods: string[],
  weights?: Record<string, number>,
): Promise<SwimConsistencyPoint> {
  const res = await fetch("/api/swim-ice/consistency/custom", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lat,
      lon,
      enabled_methods: methods,
      custom_weights: weights ?? {},
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM custom fusion failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSwimConsistencyRegion(
  north: number,
  south: number,
  east: number,
  west: number,
): Promise<SwimConsistencyRegion> {
  const params = new URLSearchParams({
    north: north.toString(),
    south: south.toString(),
    east: east.toString(),
    west: west.toString(),
  });
  const res = await fetch(`/api/swim-ice/consistency/region?${params}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `SWIM region query failed (${res.status})`);
  }
  return res.json();
}

/* =========================================================
 * Tile URL helpers
 * =======================================================*/

export function getSwimMethodTileUrl(method: SwimMethod, depth: DepthRange): string {
  return `/api/swim-ice/${method}/tile/{z}/{x}/{y}.png?depth=${depth}`;
}

export function getSwimConsistencyTileUrl(depth: DepthRange): string {
  return `/api/swim-ice/consistency/tile/{z}/{x}/{y}.png?depth=${depth}`;
}
