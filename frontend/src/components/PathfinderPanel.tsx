import { useState, useCallback, useRef, useEffect, type MouseEvent as ReactMouseEvent } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  planRoute,
  fetchRovers,
  type PlanRequest,
  type RouteResult,
  type RoverProfile,
  type CostWeights,
  type VLMAnalysis,
  type TerrainZone,
  DEFAULT_COST_WEIGHTS,
} from "../api/pathfinder";
import type { RoverTelemetry, SpeedOption } from "../hooks/useRoverSimulation";
import { SPEED_OPTIONS } from "../hooks/useRoverSimulation";

/* =========================================================
 * Props
 * =======================================================*/
export interface SimulationControls {
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  setSpeed: (speed: SpeedOption) => void;
  seek: (progress: number) => void;
  reset: () => void;
  toggleCamera: () => void;
}

export interface PathfinderPanelProps {
  startPoint: { lat: number; lon: number } | null;
  goalPoint: { lat: number; lon: number } | null;
  onRouteReady?: (route: RouteResult) => void;
  onClear?: () => void;
  /* Simulation */
  simPlaying?: boolean;
  simSpeed?: SpeedOption;
  simProgress?: number;
  simTelemetry?: RoverTelemetry | null;
  simComplete?: boolean;
  simCameraFollow?: boolean;
  simControls?: SimulationControls;
}

/* =========================================================
 * Helpers
 * =======================================================*/

function fmtCoord(lat: number, lon: number): string {
  const latStr = `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"}`;
  const lonStr = `${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;
  return `${latStr}, ${lonStr}`;
}

function fmtDist(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`;
}

function slopeColor(deg: number): string {
  if (deg < 5) return "text-emerald-400";
  if (deg < 10) return "text-amber-400";
  return "text-red-400";
}

type ProgressData = { stage: string; message: string; pct: number };

/* =========================================================
 * Custom Tooltip
 * =======================================================*/
function ProfileTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const elev = payload.find((p: any) => p.dataKey === "elev");
  const slope = payload.find((p: any) => p.dataKey === "slope");

  return (
    <div className="bg-[#0a0f18] border border-[#232f48] rounded-md px-3 py-2 text-[11px] shadow-xl">
      <p className="text-slate-500 mb-1">{(label as number).toFixed(1)} km</p>
      {elev && (
        <p className="text-blue-400">
          Elevation: <span className="font-mono">{(elev.value as number).toFixed(0)} m</span>
        </p>
      )}
      {slope && (
        <p className={slopeColor(slope.value as number)}>
          Slope: <span className="font-mono">{(slope.value as number).toFixed(1)}°</span>
        </p>
      )}
    </div>
  );
}

/* =========================================================
 * Component
 * =======================================================*/
export default function PathfinderPanel({
  startPoint,
  goalPoint,
  onRouteReady,
  onClear,
  simPlaying = false,
  simSpeed = 1,
  simProgress = 0,
  simTelemetry = null,
  simComplete = false,
  simCameraFollow = true,
  simControls,
}: PathfinderPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [roverType, setRoverType] = useState("perseverance");
  const [waypointSpacing, setWaypointSpacing] = useState(10);
  const [costWeights, setCostWeights] = useState<CostWeights>({ ...DEFAULT_COST_WEIGHTS });
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [planning, setPlanning] = useState(false);
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [result, setResult] = useState<RouteResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [rovers, setRovers] = useState<Record<string, RoverProfile>>({});
  const abortRef = useRef<AbortController | null>(null);

  // Fetch rover profiles once
  useEffect(() => {
    fetchRovers()
      .then((data) => {
        if (data.rovers) {
          const roversMap: Record<string, RoverProfile> = {};
          if (Array.isArray(data.rovers)) {
            for (const r of data.rovers) roversMap[r.id] = r;
          } else {
            for (const [k, v] of Object.entries(data.rovers as Record<string, any>)) {
              roversMap[k] = v as RoverProfile;
            }
          }
          setRovers(roversMap);
        }
      })
      .catch(() => {});
  }, []);

  // ── Plan Route ──────────────────────────────────────────────
  const handlePlan = useCallback(async () => {
    if (!startPoint || !goalPoint) return;

    // Cancel previous
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPlanning(true);
    setProgress(null);
    setResult(null);
    setError(null);

    const req: PlanRequest = {
      start: startPoint,
      goal: goalPoint,
      rover_type: roverType,
      waypoint_spacing_m: waypointSpacing,
      cost_weights: costWeights,
    };

    try {
      for await (const event of planRoute(req, controller.signal)) {
        if (controller.signal.aborted) break;

        if (event.event === "progress") {
          setProgress(event.data as ProgressData);
        } else if (event.event === "result") {
          const routeResult = event.data as RouteResult;
          setResult(routeResult);
          setPlanning(false);
          setProgress(null);
          onRouteReady?.(routeResult);
        } else if (event.event === "error") {
          setError((event.data as { error: string }).error);
          setPlanning(false);
          setProgress(null);
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "Planning failed");
      }
      setPlanning(false);
      setProgress(null);
    }
  }, [startPoint, goalPoint, roverType, waypointSpacing, costWeights, onRouteReady]);

  // ── Cancel ──────────────────────────────────────────────────
  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setPlanning(false);
    setProgress(null);
  }, []);

  // ── Clear ───────────────────────────────────────────────────
  const handleClear = useCallback(() => {
    abortRef.current?.abort();
    setPlanning(false);
    setProgress(null);
    setResult(null);
    setError(null);
    onClear?.();
  }, [onClear]);

  // ── Elevation Profile Data ──────────────────────────────────
  const profileData =
    result?.profiles
      ? result.profiles.distance.map((d, i) => ({
          dist: d / 1000, // km
          elev: result.profiles.elevation[i] ?? 0,
          slope: result.profiles.slope[i] ?? 0,
        }))
      : [];

  const canPlan = !!startPoint && !!goalPoint && !planning;

  // ── Render ──────────────────────────────────────────────────
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 shadow-xl overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-700/50 hover:bg-slate-700 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="material-icons text-orange-400 text-lg">route</span>
          <span className="text-white font-semibold text-sm">Pathfinder</span>
          {result && (
            <span className="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 px-1.5 py-0.5 rounded">
              Route Ready
            </span>
          )}
        </div>
        <span
          className={`material-icons text-slate-400 text-sm transform transition-transform ${
            collapsed ? "" : "rotate-180"
          }`}
        >
          expand_more
        </span>
      </button>

      {!collapsed && (
        <div className="p-4 space-y-4 max-h-[80vh] overflow-y-auto">
          {/* ── Points ──────────────────────────────────── */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="material-icons text-emerald-400 text-sm">trip_origin</span>
              <span className="text-slate-300 text-xs flex-1">
                {startPoint ? fmtCoord(startPoint.lat, startPoint.lon) : "Click map to set start"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="material-icons text-red-400 text-sm">flag</span>
              <span className="text-slate-300 text-xs flex-1">
                {goalPoint ? fmtCoord(goalPoint.lat, goalPoint.lon) : "Right-click map to set goal"}
              </span>
            </div>
          </div>

          {/* ── Rover Selector ──────────────────────────── */}
          <div>
            <label className="text-slate-400 text-xs block mb-1">Rover Profile</label>
            <select
              value={roverType}
              onChange={(e) => setRoverType(e.target.value)}
              className="w-full bg-slate-900 text-slate-200 text-xs rounded border border-slate-600 px-2 py-1.5 focus:outline-none focus:border-orange-500"
            >
              {Object.keys(rovers).length > 0 ? (
                Object.entries(rovers).map(([id, r]) => (
                  <option key={id} value={id}>{r.name}</option>
                ))
              ) : (
                <>
                  <option value="perseverance">Perseverance (Mars 2020)</option>
                  <option value="curiosity">Curiosity (MSL)</option>
                  <option value="generic_small">Generic Small Rover</option>
                </>
              )}
            </select>
          </div>

          {/* ── Waypoint Spacing ────────────────────────── */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-slate-400 text-xs">Waypoint Spacing</label>
              <span className="text-orange-400 text-xs font-mono">{waypointSpacing} m</span>
            </div>
            <input
              type="range"
              min={1}
              max={100}
              value={waypointSpacing}
              onChange={(e) => setWaypointSpacing(Number(e.target.value))}
              className="w-full h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-orange-500"
            />
          </div>

          {/* ── Advanced Settings ───────────────────────── */}
          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-slate-400 text-xs hover:text-slate-200 transition-colors"
            >
              <span
                className={`material-icons text-xs transform transition-transform ${
                  showAdvanced ? "rotate-180" : ""
                }`}
              >
                expand_more
              </span>
              Cost Weights
            </button>
            {showAdvanced && (
              <div className="mt-2 space-y-2 pl-4 border-l border-slate-600">
                {(["slope", "roughness", "hazard", "elevation"] as const).map((key) => (
                  <div key={key}>
                    <div className="flex justify-between items-center">
                      <label className="text-slate-500 text-[10px] capitalize">{key}</label>
                      <span className="text-slate-400 text-[10px] font-mono">
                        {costWeights[key].toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={Math.round(costWeights[key] * 100)}
                      onChange={(e) =>
                        setCostWeights((prev) => ({
                          ...prev,
                          [key]: Number(e.target.value) / 100,
                        }))
                      }
                      className="w-full h-0.5 bg-slate-700 rounded appearance-none cursor-pointer accent-orange-500"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Action Buttons ──────────────────────────── */}
          <div className="flex gap-2">
            <button
              onClick={handlePlan}
              disabled={!canPlan}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded text-sm font-medium transition-colors ${
                canPlan
                  ? "bg-orange-600 hover:bg-orange-500 text-white"
                  : "bg-slate-700 text-slate-500 cursor-not-allowed"
              }`}
            >
              <span className="material-icons text-sm">
                {planning ? "hourglass_top" : "route"}
              </span>
              {planning ? "Planning..." : "Plan Route"}
            </button>
            {planning && (
              <button
                onClick={handleCancel}
                className="px-3 py-2 rounded text-sm bg-red-900/50 hover:bg-red-800/50 text-red-400 transition-colors"
              >
                Cancel
              </button>
            )}
            {result && (
              <button
                onClick={handleClear}
                className="px-3 py-2 rounded text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
              >
                Clear
              </button>
            )}
          </div>

          {/* ── Progress Bar ───────────────────────────── */}
          {planning && progress && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-slate-400">
                <span>{progress.message}</span>
                <span className="font-mono">{progress.pct}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-1.5">
                <div
                  className="bg-orange-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${progress.pct}%` }}
                />
              </div>
            </div>
          )}

          {/* ── Error ──────────────────────────────────── */}
          {error && (
            <div className="flex items-start gap-2 bg-red-900/20 border border-red-500/30 rounded p-3">
              <span className="material-icons text-red-400 text-sm mt-0.5">error</span>
              <span className="text-red-300 text-xs">{error}</span>
            </div>
          )}

          {/* ── Route Summary ──────────────────────────── */}
          {result?.summary && (
            <div className="space-y-3">
              <div className="text-xs font-semibold text-white flex items-center gap-1.5">
                <span className="material-icons text-emerald-400 text-sm">check_circle</span>
                Route Summary
              </div>

              <div className="grid grid-cols-2 gap-2">
                <StatCard
                  icon="straighten"
                  label="Distance"
                  value={fmtDist(result.summary.total_distance_m)}
                />
                <StatCard
                  icon="schedule"
                  label="Est. Time"
                  value={`${result.summary.total_time_hours.toFixed(1)} hrs`}
                />
                <StatCard
                  icon="wb_sunny"
                  label="Sols Needed"
                  value={String(result.sol_plan?.length ?? 1)}
                  color="text-amber-400"
                />
                <StatCard
                  icon="pin_drop"
                  label="Waypoints"
                  value={String(result.summary.n_waypoints)}
                />
                <StatCard
                  icon="trending_up"
                  label="Max Slope"
                  value={`${result.summary.max_slope_deg.toFixed(1)}°`}
                  color={
                    result.summary.max_slope_deg > 10
                      ? "text-red-400"
                      : result.summary.max_slope_deg > 5
                        ? "text-amber-400"
                        : "text-emerald-400"
                  }
                />
                <StatCard
                  icon="height"
                  label="Elev. Gain"
                  value={`+${result.summary.total_elevation_gain_m.toFixed(0)} m`}
                />
              </div>

              {/* ── Elevation Profile ────────────────────── */}
              {profileData.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-slate-300 mb-2 flex items-center gap-1">
                    <span className="material-icons text-blue-400 text-sm">show_chart</span>
                    Elevation Profile
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-2">
                    <ResponsiveContainer width="100%" height={140}>
                      <ComposedChart data={profileData}>
                        <defs>
                          <linearGradient id="elevGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis
                          dataKey="dist"
                          tick={{ fill: "#64748b", fontSize: 9 }}
                          tickFormatter={(v) => `${v.toFixed(1)}`}
                          label={{ value: "km", position: "insideBottomRight", fill: "#64748b", fontSize: 9, offset: -5 }}
                        />
                        <YAxis
                          yAxisId="elev"
                          orientation="left"
                          tick={{ fill: "#3b82f6", fontSize: 9 }}
                          tickFormatter={(v) => `${v.toFixed(0)}`}
                          label={{ value: "m", angle: -90, position: "insideLeft", fill: "#3b82f6", fontSize: 9 }}
                        />
                        <YAxis
                          yAxisId="slope"
                          orientation="right"
                          tick={{ fill: "#f97316", fontSize: 9 }}
                          tickFormatter={(v) => `${v.toFixed(0)}°`}
                        />
                        <Tooltip content={<ProfileTooltip />} />
                        <Area
                          yAxisId="elev"
                          type="monotone"
                          dataKey="elev"
                          stroke="#3b82f6"
                          fill="url(#elevGrad)"
                          strokeWidth={1.5}
                        />
                        <Line
                          yAxisId="slope"
                          type="monotone"
                          dataKey="slope"
                          stroke="#f97316"
                          strokeWidth={1}
                          dot={false}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* ── Sol Plan Table ────────────────────────── */}
              {result.sol_plan && result.sol_plan.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-slate-300 mb-2 flex items-center gap-1">
                    <span className="material-icons text-amber-400 text-sm">calendar_today</span>
                    Sol Drive Plan ({result.sol_plan.length} sols)
                  </div>
                  <div className="bg-slate-900/50 rounded-lg overflow-hidden">
                    <table className="w-full text-[10px] text-slate-300">
                      <thead>
                        <tr className="bg-slate-800/80 text-slate-500">
                          <th className="px-2 py-1.5 text-left">Sol</th>
                          <th className="px-2 py-1.5 text-right">Distance</th>
                          <th className="px-2 py-1.5 text-right">Time</th>
                          <th className="px-2 py-1.5 text-right">WPs</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.sol_plan.map((sol) => (
                          <tr key={sol.sol_number} className="border-t border-slate-800/50 hover:bg-slate-800/30">
                            <td className="px-2 py-1 text-amber-400 font-mono">Sol {sol.sol_number}</td>
                            <td className="px-2 py-1 text-right font-mono">{fmtDist(sol.distance_m)}</td>
                            <td className="px-2 py-1 text-right font-mono">{sol.time_hours.toFixed(1)}h</td>
                            <td className="px-2 py-1 text-right font-mono">
                              {sol.start_wp_id}–{sol.end_wp_id}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* ── VLM Terrain Analysis ─────────────────── */}
              {result.vlm_analysis && (
                <VLMAnalysisSection vlm={result.vlm_analysis} />
              )}

              {/* ── Digital Twin Simulation ──────────────── */}
              {simControls && (
                <SimulationSection
                  playing={simPlaying}
                  speed={simSpeed}
                  progress={simProgress}
                  telemetry={simTelemetry ?? undefined}
                  complete={simComplete}
                  cameraFollow={simCameraFollow}
                  controls={simControls}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* =========================================================
 * Stat Card sub-component
 * =======================================================*/
function StatCard({
  icon,
  label,
  value,
  color = "text-white",
}: {
  icon: string;
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-slate-900/50 rounded p-2 flex items-center gap-2">
      <span className="material-icons text-slate-500 text-sm">{icon}</span>
      <div className="min-w-0">
        <div className="text-slate-500 text-[9px] leading-tight">{label}</div>
        <div className={`text-xs font-mono font-semibold ${color}`}>{value}</div>
      </div>
    </div>
  );
}

/* =========================================================
 * VLM Terrain Analysis Section
 * =======================================================*/

const TERRAIN_TYPE_STYLES: Record<string, { icon: string; color: string; label: string }> = {
  bedrock:  { icon: "landscape",    color: "text-blue-400",   label: "Bedrock" },
  sand:     { icon: "grain",        color: "text-yellow-400", label: "Sand" },
  regolith: { icon: "terrain",      color: "text-slate-300",  label: "Regolith" },
  rocky:    { icon: "filter_hdr",   color: "text-orange-400", label: "Rocky" },
  ice_rich: { icon: "ac_unit",      color: "text-cyan-400",   label: "Ice-rich" },
  mixed:    { icon: "blur_on",      color: "text-purple-400", label: "Mixed" },
};

const RISK_BADGE: Record<string, { bg: string; text: string }> = {
  low:      { bg: "bg-green-900/50",  text: "text-green-400" },
  moderate: { bg: "bg-yellow-900/50", text: "text-yellow-400" },
  high:     { bg: "bg-orange-900/50", text: "text-orange-400" },
  extreme:  { bg: "bg-red-900/50",    text: "text-red-400" },
};

function VLMAnalysisSection({ vlm }: { vlm: VLMAnalysis }) {
  const risk = RISK_BADGE[vlm.risk_level] ?? RISK_BADGE.moderate;

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-slate-300 flex items-center gap-1">
          <span className="material-icons text-purple-400 text-sm">psychology</span>
          AI Terrain Analysis
        </div>
        <div className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase ${risk.bg} ${risk.text}`}>
          {vlm.risk_level} risk
        </div>
      </div>

      {/* Overall assessment */}
      <div className="bg-slate-900/50 rounded-lg p-3">
        <p className="text-[11px] text-slate-300 leading-relaxed">{vlm.overall_assessment}</p>
        <div className="mt-2 flex items-center gap-1 text-[9px] text-slate-500">
          <span className="material-icons text-[10px]">smart_toy</span>
          {vlm.analysis_model}
        </div>
      </div>

      {/* Recommended corridors */}
      {vlm.recommended_corridors.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[9px] text-slate-500">Safe corridors:</span>
          {vlm.recommended_corridors.map((c) => (
            <span
              key={c}
              className="px-1.5 py-0.5 bg-green-900/30 text-green-400 rounded text-[9px] font-medium"
            >
              {c}
            </span>
          ))}
        </div>
      )}

      {/* Terrain zones */}
      <div className="space-y-1.5">
        {vlm.zones.map((zone) => (
          <TerrainZoneCard key={zone.zone_id} zone={zone} />
        ))}
      </div>

      {/* Composite terrain image */}
      {vlm.terrain_image_b64 && (
        <div>
          <div className="text-[9px] text-slate-500 mb-1">Composite Terrain Map (R=slope, G=traversability, B=hazard)</div>
          <img
            src={`data:image/png;base64,${vlm.terrain_image_b64}`}
            alt="VLM Terrain Composite"
            className="w-full rounded-lg border border-slate-700/50"
          />
        </div>
      )}
    </div>
  );
}

function TerrainZoneCard({ zone }: { zone: TerrainZone }) {
  const style = TERRAIN_TYPE_STYLES[zone.terrain_type] ?? TERRAIN_TYPE_STYLES.mixed;
  const confPct = Math.round(zone.confidence * 100);

  return (
    <div className="bg-slate-900/40 rounded-lg p-2 flex items-start gap-2">
      <span className={`material-icons text-sm mt-0.5 ${style.color}`}>{style.icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className={`text-[11px] font-medium ${style.color}`}>{style.label}</span>
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-slate-500">{confPct}%</span>
            <div className="w-12 h-1 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-400"
                style={{ width: `${confPct}%` }}
              />
            </div>
          </div>
        </div>
        <p className="text-[10px] text-slate-400 mt-0.5 leading-snug">{zone.description}</p>
        {zone.hazards.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {zone.hazards.map((h) => (
              <span
                key={h}
                className="px-1 py-0.5 bg-red-900/30 text-red-400 rounded text-[8px]"
              >
                {h.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* =========================================================
 * Digital Twin Simulation Section
 * =======================================================*/

const TERRAIN_ICON_MAP: Record<string, string> = {
  bedrock: "landscape", sand: "grain", regolith: "terrain",
  rocky: "filter_hdr", ice_rich: "ac_unit", mixed: "blur_on",
  "N/A": "help_outline",
};

function SimulationSection({
  playing,
  speed,
  progress,
  telemetry,
  complete,
  cameraFollow,
  controls,
}: {
  playing: boolean;
  speed: number;
  progress: number;
  telemetry?: RoverTelemetry;
  complete: boolean;
  cameraFollow: boolean;
  controls: SimulationControls;
}) {
  const progressBarRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);

  const handleProgressBarClick = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      const bar = progressBarRef.current;
      if (!bar) return;
      const rect = bar.getBoundingClientRect();
      const p = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      controls.seek(p);
    },
    [controls],
  );

  const pct = Math.round(progress * 100);
  const distKm = telemetry ? (telemetry.distanceTraveled / 1000).toFixed(2) : "0.00";
  const totalKm = telemetry ? (telemetry.totalDistance / 1000).toFixed(2) : "0.00";

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-white flex items-center gap-1.5">
          <span className="material-icons text-yellow-400 text-sm">smart_toy</span>
          Digital Twin Simulation
        </div>
        {complete && (
          <span className="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 px-1.5 py-0.5 rounded">
            Mission Complete
          </span>
        )}
      </div>

      {/* Play / Speed controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={controls.togglePlayPause}
          className="flex items-center justify-center w-8 h-8 rounded-lg bg-yellow-600 hover:bg-yellow-500 text-black transition-colors"
          title={playing ? "Pause" : "Play"}
        >
          <span className="material-icons text-base">
            {complete ? "replay" : playing ? "pause" : "play_arrow"}
          </span>
        </button>

        <div className="flex rounded-md overflow-hidden border border-slate-600">
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => controls.setSpeed(s)}
              className={`px-2 py-1 text-[10px] font-mono transition-colors ${
                speed === s
                  ? "bg-yellow-600/30 text-yellow-400 font-bold"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        <button
          onClick={controls.reset}
          className="ml-auto p-1.5 rounded hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          title="Reset simulation"
        >
          <span className="material-icons text-sm">restart_alt</span>
        </button>
      </div>

      {/* Progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-[10px] text-slate-400">
          <span>Sol {telemetry?.currentSol ?? 1} / {telemetry?.totalSols ?? 1}</span>
          <span className="font-mono">{distKm} / {totalKm} km</span>
          <span className="font-mono">{pct}%</span>
        </div>
        <div
          ref={progressBarRef}
          onClick={handleProgressBarClick}
          className="w-full h-2 bg-slate-700 rounded-full cursor-pointer relative group"
        >
          <div
            className="h-full bg-gradient-to-r from-yellow-500 to-lime-400 rounded-full transition-[width] duration-100"
            style={{ width: `${pct}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow border-2 border-yellow-500 opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ left: `calc(${pct}% - 6px)` }}
          />
        </div>
      </div>

      {/* Telemetry cards */}
      {telemetry && (
        <div className="grid grid-cols-3 gap-1.5">
          <SimStatCard icon="straighten" label="Distance" value={`${distKm} km`} />
          <SimStatCard
            icon="trending_up"
            label="Slope"
            value={`${telemetry.currentSlope.toFixed(1)}°`}
            color={
              telemetry.currentSlope > 10 ? "text-red-400"
              : telemetry.currentSlope > 5 ? "text-amber-400"
              : "text-emerald-400"
            }
          />
          <SimStatCard icon="height" label="Elevation" value={`${telemetry.currentElevation.toFixed(0)} m`} />
          <SimStatCard icon="explore" label="Heading" value={`${telemetry.currentHeading.toFixed(0)}°`} />
          <SimStatCard icon="speed" label="Speed" value={`${telemetry.speedMPerS.toFixed(3)} m/s`} />
          <SimStatCard
            icon={TERRAIN_ICON_MAP[telemetry.currentTerrainType] ?? "blur_on"}
            label="Terrain"
            value={telemetry.currentTerrainType}
            color={
              telemetry.currentTerrainType === "rocky" ? "text-orange-400"
              : telemetry.currentTerrainType === "sand" ? "text-yellow-400"
              : telemetry.currentTerrainType === "ice_rich" ? "text-cyan-400"
              : "text-slate-300"
            }
          />
        </div>
      )}

      {/* Camera mode toggle */}
      <button
        onClick={controls.toggleCamera}
        className={`w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
          cameraFollow
            ? "bg-yellow-600/20 text-yellow-400 border border-yellow-500/40"
            : "bg-slate-800 text-slate-400 border border-slate-600 hover:bg-slate-700"
        }`}
      >
        <span className="material-icons text-sm">{cameraFollow ? "videocam" : "videocam_off"}</span>
        Camera: {cameraFollow ? "Follow Rover" : "Free"}
      </button>
    </div>
  );
}

function SimStatCard({
  icon,
  label,
  value,
  color = "text-white",
}: {
  icon: string;
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-slate-900/60 rounded p-1.5 min-w-0">
      <div className="flex items-center gap-1 mb-0.5">
        <span className="material-icons text-slate-500 text-[10px]">{icon}</span>
        <span className="text-slate-500 text-[8px] uppercase tracking-wide">{label}</span>
      </div>
      <div className={`text-[11px] font-mono font-semibold truncate ${color}`}>{value}</div>
    </div>
  );
}
