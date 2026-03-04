import { useState, useCallback } from "react";
import {
  fetchAccessibilityScore,
  DEFAULT_WEIGHTS,
  type AccessibilityScore,
  type AccessibilityWeights,
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

  const queryPoint = useCallback(async () => {
    if (lat == null || lon == null) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAccessibilityScore(lat, lon, weights);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }, [lat, lon, weights]);

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
            onClick={queryPoint}
            disabled={lat == null || lon == null || loading}
            className={`w-full rounded py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
              loading
                ? "bg-emerald-500/10 text-emerald-400/50 border border-emerald-500/20 cursor-wait"
                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30"
            }`}
          >
            {loading ? "Computing…" : "Query Accessibility"}
          </button>

          {error && (
            <p className="text-[10px] text-red-400">{error}</p>
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
