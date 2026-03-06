import { useState, useCallback, useEffect, useRef } from "react";
import {
  fetchAccessibilityScore,
  fetchLandformCache,
  DEFAULT_WEIGHTS,
  type AccessibilityScore,
  type AccessibilityWeights,
  type LandformCacheEntry,
} from "../api/accessibility";

/* =========================================================
 * Props
 * =======================================================*/
export interface AccessibilityPanelProps {
  lat: number | null;
  lon: number | null;
}

/* =========================================================
 * Helpers
 * =======================================================*/

function scoreColor(score: number): string {
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.6) return "text-lime-400";
  if (score >= 0.3) return "text-amber-400";
  return "text-red-400";
}

function scoreLabel(score: number): string {
  if (score >= 0.8) return "Excellent";
  if (score >= 0.6) return "Good";
  if (score >= 0.3) return "Moderate";
  return "Poor";
}

function barColor(v: number): string {
  if (v >= 0.8) return "bg-emerald-500";
  if (v >= 0.6) return "bg-lime-500";
  if (v >= 0.3) return "bg-amber-500";
  return "bg-red-500";
}

function confidenceBadge(c: string): { text: string; cls: string } {
  switch (c) {
    case "high":
      return { text: "HIGH", cls: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40" };
    case "medium":
      return { text: "MED", cls: "bg-amber-500/20 text-amber-400 border-amber-500/40" };
    case "low":
      return { text: "LOW", cls: "bg-red-500/20 text-red-400 border-red-500/40" };
    default:
      return { text: "N/A", cls: "bg-slate-500/20 text-slate-400 border-slate-500/40" };
  }
}

const SUB_SCORE_LABELS: { key: keyof AccessibilityWeights; label: string; icon: string }[] = [
  { key: "ice_presence", label: "Ice Presence", icon: "ac_unit" },
  { key: "ice_depth", label: "Ice Depth", icon: "layers" },
  { key: "excavation", label: "Excavation", icon: "construction" },
  { key: "landing", label: "Landing Safety", icon: "flight_land" },
];

/* =========================================================
 * Component
 * =======================================================*/
export default function AccessibilityPanel({ lat, lon }: AccessibilityPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [result, setResult] = useState<AccessibilityScore | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [weights, setWeights] = useState<AccessibilityWeights>({ ...DEFAULT_WEIGHTS });
  const [showWeights, setShowWeights] = useState(false);
  const [landformMatch, setLandformMatch] = useState<LandformCacheEntry | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const weightsRef = useRef(weights);
  const landformMatchCacheRef = useRef<Map<string, LandformCacheEntry | null>>(new Map());

  useEffect(() => {
    weightsRef.current = weights;
  }, [weights]);

  const findNearestLandform = useCallback(async (queryLat: number, queryLon: number) => {
    const cacheKey = `${queryLat.toFixed(5)},${queryLon.toFixed(5)}`;
    if (landformMatchCacheRef.current.has(cacheKey)) {
      return landformMatchCacheRef.current.get(cacheKey) ?? null;
    }

    try {
      const cache = await fetchLandformCache();
      let bestEntry: LandformCacheEntry | null = null;
      let bestDist = Infinity;
      for (const entry of cache.entries) {
        const dist = Math.sqrt((entry.lat - queryLat) ** 2 + (entry.lon - queryLon) ** 2);
        if (dist < 0.5 && dist < bestDist) {
          bestDist = dist;
          bestEntry = entry;
        }
      }

      const matched = bestEntry && bestEntry.dominant_class !== "OTHER" ? bestEntry : null;
      landformMatchCacheRef.current.set(cacheKey, matched);
      return matched;
    } catch {
      return null;
    }
  }, []);

  const queryPoint = useCallback(async (queryLat: number, queryLon: number) => {
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);
    setLandformMatch(null);

    try {
      const matched = await findNearestLandform(queryLat, queryLon);
      if (controller.signal.aborted) return;

      let landform: string | undefined;
      if (matched) {
        landform = matched.dominant_class;
        setLandformMatch(matched);
      }

      const data = await fetchAccessibilityScore(
        queryLat,
        queryLon,
        weightsRef.current,
        landform,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setResult(data);
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, [findNearestLandform]);

  const queryCurrentPoint = useCallback(() => {
    if (lat == null || lon == null) return;
    void queryPoint(lat, lon);
  }, [lat, lon, queryPoint]);

  useEffect(() => {
    if (lat == null || lon == null) return;
    const timeoutId = window.setTimeout(() => {
      void queryPoint(lat, lon);
    }, 500);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [lat, lon, queryPoint]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const updateWeight = (key: keyof AccessibilityWeights, rawValue: number) => {
    const next = { ...weights, [key]: rawValue / 100 };
    // Normalise so they sum to 1
    const total = Object.values(next).reduce((a, b) => a + b, 0);
    if (total > 0) {
      for (const k of Object.keys(next) as (keyof AccessibilityWeights)[]) {
        next[k] = next[k] / total;
      }
    }
    setWeights(next);
  };

  const resetWeights = () => setWeights({ ...DEFAULT_WEIGHTS });

  /* ── Render ──────────────────────────────────────────── */
  return (
    <section className="flex flex-col border-b border-[#232f48]">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center gap-1.5 bg-[#0d1520] px-3 py-2 text-left hover:bg-[#111b2a]"
      >
        <span className="material-symbols-outlined text-[16px] text-emerald-400">explore</span>
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-200">
          Ice Accessibility
        </span>
        <span className="ml-auto material-symbols-outlined text-[14px] text-slate-500">
          {collapsed ? "expand_more" : "expand_less"}
        </span>
      </button>

      {!collapsed && (
        <div className="space-y-2 px-3 py-2 text-[11px] text-slate-300">
          {/* Query button */}
          <button
            onClick={queryCurrentPoint}
            disabled={lat == null || lon == null || loading}
            className={`w-full rounded py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
              loading
                ? "bg-emerald-500/10 text-emerald-400/50 border border-emerald-500/20 cursor-wait"
                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30"
            }`}
          >
            {loading ? "Refreshing…" : "Refresh Accessibility"}
          </button>

          {loading && (
            <div className="space-y-1.5 rounded border border-emerald-500/20 bg-emerald-500/5 p-2 animate-pulse">
              <div className="h-2 w-28 rounded bg-emerald-500/20" />
              <div className="h-1.5 w-full rounded bg-slate-700/60" />
              <div className="h-1.5 w-5/6 rounded bg-slate-700/50" />
              <div className="h-1.5 w-2/3 rounded bg-slate-700/40" />
            </div>
          )}

          {error && (
            <div className="flex items-center justify-between gap-2 rounded border border-red-500/20 bg-red-500/5 px-2 py-1.5">
              <p className="text-[10px] text-red-400">{error}</p>
              <button
                onClick={queryCurrentPoint}
                disabled={lat == null || lon == null || loading}
                className="rounded border border-red-500/30 bg-red-500/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-red-300 hover:bg-red-500/20 disabled:opacity-50"
              >
                Retry
              </button>
            </div>
          )}

          {/* Results */}
          {result && !error && (
            <div className="space-y-2.5">
              {/* Score header */}
              <div className="flex items-center justify-between">
                <div className="flex items-baseline gap-2">
                  <span className={`text-2xl font-bold tabular-nums ${scoreColor(result.score)}`}>
                    {(result.score * 100).toFixed(0)}
                  </span>
                  <span className="text-[10px] text-slate-500">/100</span>
                  <span className={`text-[10px] ${scoreColor(result.score)}`}>
                    {scoreLabel(result.score)}
                  </span>
                </div>
                <span
                  className={`rounded border px-1.5 py-0.5 text-[8px] font-bold uppercase ${
                    confidenceBadge(result.confidence).cls
                  }`}
                >
                  {confidenceBadge(result.confidence).text}
                </span>
              </div>

              {/* Landform enhancement badge */}
              {landformMatch && (
                <div className="flex items-center gap-1.5 rounded border border-violet-500/30 bg-violet-500/10 px-2 py-1">
                  <span className="material-symbols-outlined text-[12px] text-violet-400">microscope</span>
                  <span className="text-[9px] text-violet-300">
                    Enhanced: <span className="font-bold">{landformMatch.dominant_class}</span>{" "}
                    ({Math.round(landformMatch.confidence * 100)}%)
                  </span>
                  <span className="text-[8px] text-slate-500 ml-auto">HiRISE</span>
                </div>
              )}

              {/* Sub-scores */}
              <div className="space-y-1.5">
                {SUB_SCORE_LABELS.map(({ key, label, icon }) => {
                  const val = result[key];
                  return (
                    <div key={key} className="flex items-center gap-1.5">
                      <span className="material-symbols-outlined text-[12px] text-slate-500 w-4">
                        {icon}
                      </span>
                      <span className="w-20 text-[10px] text-slate-400 truncate">{label}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-[#1a2333] overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${barColor(val)}`}
                          style={{ width: `${Math.max(2, val * 100)}%` }}
                        />
                      </div>
                      <span className={`w-8 text-right text-[10px] tabular-nums ${scoreColor(val)}`}>
                        {(val * 100).toFixed(0)}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Layers info */}
              <div className="text-[9px] text-slate-500">
                {result.layers_available}/{result.layers_total} data layers •{" "}
                ({result.lat.toFixed(2)}°, {result.lon.toFixed(2)}°)
              </div>

              {/* Inputs detail (collapsible) */}
              <details className="text-[9px]">
                <summary className="cursor-pointer text-slate-500 hover:text-slate-400">
                  Raw inputs ▸
                </summary>
                <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 text-slate-500">
                  {Object.entries(result.inputs).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="truncate">{k}:</span>
                      <span className="text-slate-400 ml-1">
                        {v == null ? "—" : typeof v === "number" ? v.toFixed(1) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}

          {/* Weight adjustment */}
          <div>
            <button
              onClick={() => setShowWeights(!showWeights)}
              className="flex items-center gap-1 text-[9px] text-slate-500 hover:text-slate-400"
            >
              <span className="material-symbols-outlined text-[12px]">tune</span>
              Adjust Weights {showWeights ? "▾" : "▸"}
            </button>

            {showWeights && (
              <div className="mt-1.5 space-y-1.5">
                {SUB_SCORE_LABELS.map(({ key, label }) => (
                  <div key={key} className="flex items-center gap-1.5">
                    <span className="w-20 text-[9px] text-slate-500 truncate">{label}</span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={Math.round(weights[key] * 100)}
                      onChange={(e) => updateWeight(key, parseInt(e.target.value))}
                      className="flex-1 h-1 accent-emerald-500"
                    />
                    <span className="w-8 text-right text-[9px] text-slate-400 tabular-nums">
                      {Math.round(weights[key] * 100)}%
                    </span>
                  </div>
                ))}
                <button
                  onClick={resetWeights}
                  className="text-[8px] text-slate-500 hover:text-slate-400 underline"
                >
                  Reset defaults
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
