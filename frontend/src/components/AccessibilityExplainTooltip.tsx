import { useEffect, useState } from "react";
import {
  fetchAccessibilityExplanation,
  type AccessibilityExplanation,
} from "../api/accessibility";

/* =========================================================
 * Props
 * =======================================================*/
export interface AccessibilityExplainTooltipProps {
  lat: number;
  lon: number;
  onClose: () => void;
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

function scoreBg(score: number): string {
  if (score >= 0.8) return "border-emerald-500/50";
  if (score >= 0.6) return "border-lime-500/50";
  if (score >= 0.3) return "border-amber-500/50";
  return "border-red-500/50";
}

function scoreEmoji(score: number): string {
  if (score >= 0.8) return "🟢";
  if (score >= 0.6) return "🟡";
  if (score >= 0.3) return "🟠";
  return "🔴";
}

/* =========================================================
 * Component
 * =======================================================*/
export default function AccessibilityExplainTooltip({
  lat,
  lon,
  onClose,
}: AccessibilityExplainTooltipProps) {
  const [result, setResult] = useState<AccessibilityExplanation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setResult(null);

    fetchAccessibilityExplanation(lat, lon)
      .then((data) => {
        if (!cancelled) setResult(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [lat, lon]);

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-40 pointer-events-auto">
      <div
        className={`relative bg-[#0d1520]/95 backdrop-blur-sm border rounded-lg shadow-2xl
          max-w-sm w-[340px] overflow-hidden transition-all ${
            result ? scoreBg(result.score) : "border-[#232f48]"
          }`}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-1.5 right-1.5 p-0.5 rounded hover:bg-[#232f48] transition-colors text-slate-500 hover:text-slate-300 z-10"
        >
          <span className="material-symbols-outlined text-[14px]">close</span>
        </button>

        {/* Loading state */}
        {loading && (
          <div className="px-4 py-5 flex items-center gap-2 text-[11px] text-slate-400">
            <span className="material-symbols-outlined text-[14px] animate-spin">progress_activity</span>
            Analyzing ({lat.toFixed(2)}°, {lon.toFixed(2)}°)…
          </div>
        )}

        {/* Error state */}
        {error && !loading && (
          <div className="px-4 py-3 text-[11px] text-red-400">
            <span className="material-symbols-outlined text-[12px] align-middle mr-1">error</span>
            {error}
          </div>
        )}

        {/* Result */}
        {result && !loading && !error && (
          <div className="px-4 py-3 space-y-2">
            {/* Header: score + coordinates */}
            <div className="flex items-center gap-2">
              <span className="text-sm">{scoreEmoji(result.score)}</span>
              <span className={`text-lg font-bold tabular-nums ${scoreColor(result.score)}`}>
                {(result.score * 100).toFixed(0)}
              </span>
              <span className="text-[9px] text-slate-500">/100</span>
              <span className="ml-auto text-[9px] text-slate-500 tabular-nums">
                {result.lat.toFixed(2)}°, {result.lon.toFixed(2)}°
              </span>
            </div>

            {/* Mini sub-score bars */}
            <div className="grid grid-cols-4 gap-1">
              {(
                [
                  { key: "ice_presence", label: "Ice", icon: "ac_unit" },
                  { key: "ice_depth", label: "Depth", icon: "layers" },
                  { key: "excavation", label: "Dig", icon: "construction" },
                  { key: "landing", label: "Land", icon: "flight_land" },
                ] as const
              ).map(({ key, label, icon }) => {
                const val = result[key];
                return (
                  <div key={key} className="text-center">
                    <span className="material-symbols-outlined text-[10px] text-slate-500">
                      {icon}
                    </span>
                    <div className="h-1 rounded-full bg-[#1a2333] overflow-hidden mt-0.5">
                      <div
                        className={`h-full rounded-full ${
                          val >= 0.6 ? "bg-emerald-500" : val >= 0.3 ? "bg-amber-500" : "bg-red-500"
                        }`}
                        style={{ width: `${Math.max(4, val * 100)}%` }}
                      />
                    </div>
                    <span className="text-[8px] text-slate-500 mt-0.5 block">{label}</span>
                  </div>
                );
              })}
            </div>

            {/* LLM explanation */}
            <p className="text-[11px] text-slate-300 leading-relaxed">
              {result.explanation}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
