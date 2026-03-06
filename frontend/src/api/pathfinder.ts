/**
 * API client for the Mars Pathfinder route-planning service.
 * Supports SSE streaming for real-time route computation progress.
 */

/* =========================================================
 * Types
 * =======================================================*/

export interface CostWeights {
  slope: number;
  roughness: number;
  hazard: number;
  elevation: number;
}

export const DEFAULT_COST_WEIGHTS: CostWeights = {
  slope: 0.4,
  roughness: 0.3,
  hazard: 0.2,
  elevation: 0.1,
};

export interface PlanRequest {
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  rover_type: string;
  waypoint_spacing_m: number;
  cost_weights?: CostWeights;
}

export interface Waypoint {
  lat: number;
  lon: number;
  elevation: number;
  distance_from_start: number;
  slope: number;
  heading: number;
}

export interface RouteSummary {
  total_distance_m: number;
  total_time_hours: number;
  total_elevation_gain_m: number;
  total_elevation_loss_m: number;
  max_slope_deg: number;
  mean_slope_deg: number;
  n_waypoints: number;
  waypoint_spacing_m: number;
  generation_time_ms: number;
  rover: string;
}

export interface RouteProfile {
  distance: number[];
  elevation: number[];
  slope: number[];
}

export interface SolPlan {
  sol_number: number;
  distance_m: number;
  start_wp_id: number;
  end_wp_id: number;
  time_hours: number;
  n_waypoints: number;
}

export interface RouteGeoPoint {
  lat: number;
  lon: number;
}

export interface TerrainZone {
  zone_id: number;
  terrain_type: "bedrock" | "sand" | "regolith" | "rocky" | "ice_rich" | "mixed";
  confidence: number;
  traversability: "easy" | "moderate" | "difficult" | "impassable";
  hazards: string[];
  description: string;
  bbox_pct: [number, number, number, number];
}

export interface VLMAnalysis {
  zones: TerrainZone[];
  overall_assessment: string;
  recommended_corridors: string[];
  risk_level: "low" | "moderate" | "high" | "extreme";
  analysis_model: string;
  terrain_image_b64?: string;
}
export interface RouteResult {
  waypoints: Waypoint[];
  summary: RouteSummary;
  profiles: RouteProfile;
  sol_plan: SolPlan[];
  route_geo: RouteGeoPoint[];
  vlm_analysis?: VLMAnalysis;
}

export interface RoverProfile {
  name: string;
  id: string;
  max_slope_deg: number;
  max_speed_m_per_hr: number;
  drive_hours_per_sol: number;
  wheel_diameter_m: number;
  mass_kg: number;
}

export interface RoverProfiles {
  rovers: RoverProfile[];
}

export interface PathfinderStatus {
  status: string;
  dem_loaded: boolean;
  cost_maps_available: string[];
  version: string;
}

export interface SegmentRequest {
  start: { lat: number; lon: number };
  end: { lat: number; lon: number };
}

export interface SegmentAnalysis {
  distance_m: number;
  elevation_change_m: number;
  max_slope: number;
  avg_slope: number;
  terrain_type: string;
  traversability: "safe" | "marginal" | "unsafe";
  hazards: string[];
}

export type SSEEvent =
  | { event: "progress"; data: { stage: string; message: string; pct: number } }
  | { event: "result"; data: RouteResult }
  | { event: "error"; data: { error: string } };

/* =========================================================
 * SSE Streaming — Plan Route
 * =======================================================*/

export async function* planRoute(
  req: PlanRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  // Transform frontend PlanRequest to backend schema
  const backendBody = {
    start_lat: req.start.lat,
    start_lon: req.start.lon,
    goal_lat: req.goal.lat,
    goal_lon: req.goal.lon,
    rover_type: req.rover_type,
    waypoint_spacing_m: req.waypoint_spacing_m,
    ...(req.cost_weights ? {
      w_slope: req.cost_weights.slope,
      w_roughness: req.cost_weights.roughness,
      w_hazard: req.cost_weights.hazard,
      w_elevation: req.cost_weights.elevation,
    } : {}),
  };
  const response = await fetch("/api/pathfinder/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(backendBody),
    signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    yield {
      event: "error",
      data: { error: body?.error || `HTTP ${response.status}` },
    };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    yield { event: "error", data: { error: "No response body" } };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("event: ")) {
          currentEvent = trimmed.slice(7).trim();
        } else if (trimmed.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            if (currentEvent) {
              yield { event: currentEvent, data: parsed } as SSEEvent;
            } else {
              yield parsed as SSEEvent;
            }
          } catch {
            // skip malformed JSON
          }
          currentEvent = "";
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/* =========================================================
 * REST Fetchers
 * =======================================================*/

export async function fetchRovers(signal?: AbortSignal): Promise<RoverProfiles> {
  const res = await fetch("/api/pathfinder/rovers", { signal });
  if (!res.ok) throw new Error(`Failed to fetch rovers: ${res.status}`);
  return res.json();
}

export async function fetchPathfinderStatus(signal?: AbortSignal): Promise<PathfinderStatus> {
  const res = await fetch("/api/pathfinder/status", { signal });
  if (!res.ok) throw new Error(`Failed to fetch pathfinder status: ${res.status}`);
  return res.json();
}

export async function analyzeSegment(
  req: SegmentRequest,
  signal?: AbortSignal,
): Promise<SegmentAnalysis> {
  const res = await fetch("/api/pathfinder/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error || `Segment analysis failed: ${res.status}`);
  }
  return res.json();
}

/* =========================================================
 * Suggested Routes
 * =======================================================*/

export interface SuggestedRoute {
  id: string;
  name: string;
  description: string;
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  tags: string[];
  difficulty: "easy" | "moderate" | "hard";
  estimated_distance_km: number;
  science_interest: string;
}

export async function fetchSuggestedRoutes(signal?: AbortSignal): Promise<SuggestedRoute[]> {
  const res = await fetch("/api/pathfinder/suggest-routes", { signal });
  if (!res.ok) throw new Error(`Failed to fetch suggested routes: ${res.status}`);
  const data = await res.json();
  return data.routes ?? [];
}
