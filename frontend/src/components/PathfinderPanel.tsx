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
  fetchSuggestedRoutes,
  type PlanRequest,
  type RouteResult,
  type RoverProfile,
  type CostWeights,
  type VLMAnalysis,
  type TerrainZone,
  type SuggestedRoute,
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
  onSuggestRoute?: (start: { lat: number; lon: number }, goal: { lat: number; lon: number }) => void;
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
  onSuggestRoute,
  simPlaying = false,
  simSpeed = 1,
  simProgress = 0,
  simTelemetry = null,
  simComplete = false,
  simCameraFollow = true,
  simControls,
}: PathfinderPanelProps) {
  const [panelWidth, setPanelWidth] = useState(420);
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
  const [suggestedRoutes, setSuggestedRoutes] = useState<SuggestedRoute[]>([]);
  const [showSuggest, setShowSuggest] = useState(false);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.6);
      setPanelWidth(Math.max(320, Math.min(maxW, startW + delta)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [panelWidth]);

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

  // Fetch suggested routes once (on first expand)
  const handleToggleSuggest = useCallback(() => {
    setShowSuggest((prev) => {
      if (!prev && suggestedRoutes.length === 0) {
        fetchSuggestedRoutes()
          .then((routes) => setSuggestedRoutes(routes))
          .catch(() => {});
      }
      return !prev;
    });
  }, [suggestedRoutes.length]);

  const handlePickSuggestedRoute = useCallback((route: SuggestedRoute) => {
    onSuggestRoute?.(route.start, route.goal);
    setShowSuggest(false);
  }, [onSuggestRoute]);

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
    <aside className="relative flex h-full flex-col border-l border-[#232f48] bg-[#101622]" style={{ width: panelWidth }}>
      <div
        onMouseDown={handleResizeStart}
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-orange-500/30 active:bg-orange-500/50 transition-colors"
      />

      <div className="flex items-center gap-2 px-4 py-3 border-b border-[#232f48] bg-[#0d1219]">
        <span className="material-symbols-outlined text-orange-400 text-lg">route</span>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-bold text-white truncate">Pathfinder</h2>
          <p className="text-[10px] text-[#6b7c9c] truncate">AI Rover Route Planning</p>
        </div>
        {result && (
          <span className="text-[9px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full font-medium">
            Route Ready
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-dark p-4 space-y-4">
        <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3 space-y-2">
          <h3 className="text-[10px] font-bold uppercase text-[#6b7c9c] mb-2 tracking-wider flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[12px]">pin_drop</span>
            ROUTE POINTS
          </h3>
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-emerald-400 text-sm">trip_origin</span>
            <span className="text-slate-300 text-xs flex-1">
              {startPoint ? fmtCoord(startPoint.lat, startPoint.lon) : "Click map to set start"}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-red-400 text-sm">flag</span>
            <span className="text-slate-300 text-xs flex-1">
              {goalPoint ? fmtCoord(goalPoint.lat, goalPoint.lon) : "Right-click map to set goal"}
            </span>
          </div>
        </div>

        {/* Suggest Route */}
        <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
          <button
            onClick={handleToggleSuggest}
            className="w-full flex items-center justify-between gap-2 text-left"
          >
            <div className="flex items-center gap-1.5">
              <span className="material-symbols-outlined text-orange-400 text-sm">auto_awesome</span>
              <span className="text-[11px] font-medium text-[#92a4c9]">Suggest Route</span>
            </div>
            <svg className={`w-3 h-3 text-[#6b7c9c] transform transition-transform ${showSuggest ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
          </button>
          {showSuggest && (
            <div className="mt-2 space-y-1.5">
              {suggestedRoutes.length === 0 && (
                <p className="text-[10px] text-[#6b7c9c] italic">Loading routes...</p>
              )}
              {suggestedRoutes.map((route) => {
                const diffColor = route.difficulty === "easy" ? "text-emerald-400" : route.difficulty === "hard" ? "text-red-400" : "text-amber-400";
                return (
                  <button
                    key={route.id}
                    onClick={() => handlePickSuggestedRoute(route)}
                    className="w-full text-left bg-[#0a0f18] border border-[#232f48] rounded-lg p-2 hover:border-orange-500/30 hover:bg-[#0d1420] transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[11px] font-medium text-white group-hover:text-orange-300 transition-colors">{route.name}</span>
                      <span className={`text-[8px] uppercase font-bold ${diffColor}`}>{route.difficulty}</span>
                    </div>
                    <p className="text-[9px] text-[#6b7c9c] leading-snug mb-1">{route.description}</p>
                    <div className="flex items-center gap-2 text-[8px] text-[#4a5a7c]">
                      <span>~{route.estimated_distance_km} km</span>
                      <span>•</span>
                      <span className="capitalize">{route.science_interest.replace("_", " ")} interest</span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {route.tags.slice(0, 3).map((tag) => (
                        <span key={tag} className="px-1 py-0.5 bg-orange-500/5 border border-orange-500/10 text-orange-400/70 rounded text-[7px]">{tag}</span>
                      ))}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3 space-y-3">
          <h3 className="text-[10px] font-bold uppercase text-[#6b7c9c] mb-2 tracking-wider flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[12px]">settings</span>
            ROUTE SETTINGS
          </h3>

          <div>
            <label className="text-[#6b7c9c] text-[10px] font-medium block mb-1">Rover Profile</label>
            <select
              value={roverType}
              onChange={(e) => setRoverType(e.target.value)}
              className="w-full bg-[#0a0f18] border border-[#232f48] rounded px-2 py-1.5 text-[11px] text-slate-300 focus:outline-none focus:border-orange-500/50"
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

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-[#6b7c9c] text-[10px] font-medium">Waypoint Spacing</label>
              <span className="text-orange-400 text-xs font-mono">{waypointSpacing} m</span>
            </div>
            <input
              type="range"
              min={1}
              max={100}
              value={waypointSpacing}
              onChange={(e) => setWaypointSpacing(Number(e.target.value))}
              className="w-full h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer accent-orange-500"
            />
          </div>

          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-[#92a4c9] text-xs hover:text-white transition-colors"
            >
              <svg className={`w-3 h-3 transform transition-transform ${showAdvanced ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
              Cost Weights
            </button>
            {showAdvanced && (
              <div className="mt-2 space-y-2 pl-4 border-l border-[#232f48]">
                {(["slope", "roughness", "hazard", "elevation"] as const).map((key) => (
                  <div key={key}>
                    <div className="flex justify-between items-center">
                      <label className="text-[#6b7c9c] text-[10px] font-medium capitalize">{key}</label>
                      <span className="text-[#92a4c9] text-[10px] font-mono">{costWeights[key].toFixed(2)}</span>
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
                      className="w-full h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer accent-orange-500"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <button
              onClick={handlePlan}
              disabled={!canPlan}
              className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-orange-500/20 border border-orange-500/30 rounded-lg text-orange-400 text-xs font-bold uppercase tracking-wider hover:bg-orange-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span className="material-symbols-outlined text-sm">{planning ? "hourglass_top" : "route"}</span>
              {planning ? "Planning..." : "Plan Route"}
            </button>
            {planning && (
              <button
                onClick={handleCancel}
                className="px-3 py-2 rounded-lg text-xs bg-red-500/10 border border-red-500/20 text-red-300 hover:bg-red-500/20 transition-colors"
              >
                Cancel
              </button>
            )}
            {result && (
              <button
                onClick={handleClear}
                className="px-3 py-2 rounded-lg text-xs bg-[#0a0f18] border border-[#232f48] text-[#92a4c9] hover:bg-[#1a2333] transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {planning && progress && (
          <div className="space-y-1 bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
            <div className="flex justify-between text-[11px] text-[#92a4c9]">
              <span>{progress.message}</span>
              <span className="font-mono">{progress.pct}%</span>
            </div>
            <div className="w-full bg-[#232f48] rounded-full h-1.5">
              <div
                className="bg-orange-500 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${progress.pct}%` }}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
            <span className="material-symbols-outlined text-red-400 text-sm mt-0.5">error</span>
            <span className="text-red-300 text-[11px]">{error}</span>
          </div>
        )}

        {result?.summary && (
          <div className="space-y-4">
            <div>
              <h3 className="text-[10px] font-bold uppercase text-[#6b7c9c] mb-2 tracking-wider flex items-center gap-1.5">
                <span className="material-symbols-outlined text-[12px]">check_circle</span>
                ROUTE SUMMARY
              </h3>
              <div className="grid grid-cols-2 gap-2">
                <StatCard icon="straighten" label="Distance" value={fmtDist(result.summary.total_distance_m)} />
                <StatCard icon="schedule" label="Est. Time" value={`${result.summary.total_time_hours.toFixed(1)} hrs`} />
                <StatCard icon="wb_sunny" label="Sols Needed" value={String(result.sol_plan?.length ?? 1)} color="text-amber-400" />
                <StatCard icon="pin_drop" label="Waypoints" value={String(result.summary.n_waypoints)} />
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
                <StatCard icon="height" label="Elev. Gain" value={`+${result.summary.total_elevation_gain_m.toFixed(0)} m`} />
              </div>
            </div>

            {profileData.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-[#6b7c9c] mb-2 tracking-wider flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[12px]">show_chart</span>
                  ELEVATION PROFILE
                </h3>
                <div className="bg-[#0a0f18] rounded-lg border border-[#232f48] p-2">
                  <ResponsiveContainer width="100%" height={140}>
                    <ComposedChart data={profileData}>
                      <defs>
                        <linearGradient id="elevGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e2a40" />
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
                      <Area yAxisId="elev" type="monotone" dataKey="elev" stroke="#3b82f6" fill="url(#elevGrad)" strokeWidth={1.5} />
                      <Line yAxisId="slope" type="monotone" dataKey="slope" stroke="#f97316" strokeWidth={1} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {result.sol_plan && result.sol_plan.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-[#6b7c9c] mb-2 tracking-wider flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[12px]">calendar_today</span>
                  SOL DRIVE PLAN ({result.sol_plan.length} SOLS)
                </h3>
                <div className="bg-[#0a0f18] rounded-lg border border-[#232f48] overflow-hidden">
                  <table className="w-full text-[10px] text-slate-300">
                    <thead>
                      <tr className="bg-[#0d1219] text-[#6b7c9c]">
                        <th className="px-2 py-1.5 text-left">Sol</th>
                        <th className="px-2 py-1.5 text-right">Distance</th>
                        <th className="px-2 py-1.5 text-right">Time</th>
                        <th className="px-2 py-1.5 text-right">WPs</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.sol_plan.map((sol) => (
                        <tr key={sol.sol_number} className="border-t border-[#232f48] hover:bg-[#1a2333]">
                          <td className="px-2 py-1 text-amber-400 font-mono">Sol {sol.sol_number}</td>
                          <td className="px-2 py-1 text-right font-mono">{fmtDist(sol.distance_m)}</td>
                          <td className="px-2 py-1 text-right font-mono">{sol.time_hours.toFixed(1)}h</td>
                          <td className="px-2 py-1 text-right font-mono">{sol.start_wp_id}-{sol.end_wp_id}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {result.vlm_analysis && <VLMAnalysisSection vlm={result.vlm_analysis} />}

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
    </aside>
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
    <div className="bg-[#0a0f18] p-2.5 rounded-lg border border-[#232f48]">
      <p className="text-[#6b7c9c] text-[9px] uppercase font-bold flex items-center gap-1">
        <span className="material-symbols-outlined text-[12px]">{icon}</span>
        {label}
      </p>
      <div className="min-w-0">
        <div className={`text-sm font-mono font-semibold ${color}`}>{value}</div>
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
  low:      { bg: "bg-green-500/10 border border-green-500/20", text: "text-green-400" },
  moderate: { bg: "bg-yellow-500/10 border border-yellow-500/20", text: "text-yellow-400" },
  high:     { bg: "bg-orange-500/10 border border-orange-500/20", text: "text-orange-400" },
  extreme:  { bg: "bg-red-500/10 border border-red-500/20", text: "text-red-400" },
};

function VLMAnalysisSection({ vlm }: { vlm: VLMAnalysis }) {
  const risk = RISK_BADGE[vlm.risk_level] ?? { bg: "bg-yellow-500/10 border border-yellow-500/20", text: "text-yellow-400" };

  return (
    <div className="space-y-3 bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[10px] font-bold uppercase text-[#6b7c9c] tracking-wider flex items-center gap-1.5">
          <span className="material-symbols-outlined text-[12px] text-orange-400">psychology</span>
          AI TERRAIN ANALYSIS
        </h3>
        <div className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase ${risk.bg} ${risk.text}`}>
          {vlm.risk_level} risk
        </div>
      </div>

      <div className="bg-[#0a0f18] rounded-lg border border-[#232f48] p-3">
        <p className="text-[11px] text-slate-300 leading-relaxed">{vlm.overall_assessment}</p>
        <div className="mt-2 flex items-center gap-1 text-[9px] text-[#6b7c9c]">
          <span className="material-symbols-outlined text-[10px]">smart_toy</span>
          {vlm.analysis_model}
        </div>
      </div>

      {vlm.recommended_corridors.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[9px] text-[#6b7c9c]">Safe corridors:</span>
          {vlm.recommended_corridors.map((c) => (
            <span
              key={c}
              className="px-1.5 py-0.5 bg-green-500/10 border border-green-500/20 text-green-400 rounded text-[9px] font-medium"
            >
              {c}
            </span>
          ))}
        </div>
      )}

      <div className="space-y-1.5">
        {vlm.zones.map((zone) => (
          <TerrainZoneCard key={zone.zone_id} zone={zone} />
        ))}
      </div>

      {vlm.terrain_image_b64 && (
        <div className="group">
          <div className="flex items-center gap-1 text-[9px] text-[#6b7c9c] mb-1">
            <span className="material-symbols-outlined text-[10px]">satellite_alt</span>
            AI Terrain Analysis Map
            <span className="hidden group-hover:inline text-[8px] text-[#4a5a7c] ml-1">(slope / traversability / hazard)</span>
          </div>
          <img
            src={`data:image/png;base64,${vlm.terrain_image_b64}`}
            alt="AI Terrain Analysis"
            className="w-full rounded-lg border border-[#232f48]"
          />
        </div>
      )}
    </div>
  );
}

function TerrainZoneCard({ zone }: { zone: TerrainZone }) {
  const style = TERRAIN_TYPE_STYLES[zone.terrain_type] ?? { icon: "blur_on", color: "text-purple-400", label: "Mixed" };
  const confPct = Math.round(zone.confidence * 100);

  return (
    <div className="bg-[#0a0f18] border border-[#232f48] rounded-lg p-2 flex items-start gap-2">
      <span className={`material-symbols-outlined text-sm mt-0.5 ${style.color}`}>{style.icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className={`text-[11px] font-medium ${style.color}`}>{style.label}</span>
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-[#6b7c9c]">{confPct}%</span>
            <div className="w-12 h-1 bg-[#232f48] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-400"
                style={{ width: `${confPct}%` }}
              />
            </div>
          </div>
        </div>
        <p className="text-[10px] text-[#92a4c9] mt-0.5 leading-snug">{zone.description}</p>
        {zone.hazards.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {zone.hazards.map((h) => (
              <span
                key={h}
                className="px-1 py-0.5 bg-red-500/10 border border-red-500/20 text-red-400 rounded text-[8px]"
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
  // isDraggingRef reserved for future drag-to-seek

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
    <div className="space-y-3 bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-white flex items-center gap-1.5">
          <span className="material-symbols-outlined text-yellow-400 text-sm">smart_toy</span>
          Digital Twin Simulation
        </div>
        {complete && (
          <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1.5 py-0.5 rounded">
            Mission Complete
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={controls.togglePlayPause}
          className="flex items-center justify-center w-8 h-8 rounded-lg bg-yellow-500/20 border border-yellow-500/40 hover:bg-yellow-500/30 text-yellow-300 transition-colors"
          title={playing ? "Pause" : "Play"}
        >
          <span className="material-symbols-outlined text-base">
            {complete ? "replay" : playing ? "pause" : "play_arrow"}
          </span>
        </button>

        <div className="flex rounded-md overflow-hidden border border-[#232f48]">
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => controls.setSpeed(s)}
              className={`px-2 py-1 text-[10px] font-mono transition-colors ${
                speed === s
                  ? "bg-yellow-500/20 text-yellow-400 font-bold"
                  : "bg-[#0a0f18] text-[#92a4c9] hover:bg-[#1a2333]"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        <button
          onClick={controls.reset}
          className="ml-auto p-1.5 rounded border border-[#232f48] bg-[#0a0f18] hover:bg-[#1a2333] text-[#92a4c9] hover:text-white transition-colors"
          title="Reset simulation"
        >
          <span className="material-symbols-outlined text-sm">restart_alt</span>
        </button>
      </div>

      <div className="space-y-1">
        <div className="flex justify-between text-[10px] text-[#92a4c9]">
          <span>Sol {telemetry?.currentSol ?? 1} / {telemetry?.totalSols ?? 1}</span>
          <span className="font-mono">{distKm} / {totalKm} km</span>
          <span className="font-mono">{pct}%</span>
        </div>
        <div
          ref={progressBarRef}
          onClick={handleProgressBarClick}
          className="w-full h-2 bg-[#232f48] rounded-full cursor-pointer relative group"
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
            ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40"
            : "bg-[#0a0f18] text-[#92a4c9] border border-[#232f48] hover:bg-[#1a2333]"
        }`}
      >
        <span className="material-symbols-outlined text-sm">{cameraFollow ? "videocam" : "videocam_off"}</span>
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
    <div className="bg-[#0a0f18] border border-[#232f48] rounded-lg p-1.5 min-w-0">
      <div className="flex items-center gap-1 mb-0.5">
        <span className="material-symbols-outlined text-[#6b7c9c] text-[10px]">{icon}</span>
        <span className="text-[#6b7c9c] text-[8px] uppercase tracking-wide">{label}</span>
      </div>
      <div className={`text-[11px] font-mono font-semibold truncate ${color}`}>{value}</div>
    </div>
  );
}
