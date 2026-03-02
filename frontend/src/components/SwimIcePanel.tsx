import { useState, useCallback } from "react";
import DepthRangeSelector from "./DepthRangeSelector";
import SwimMethodLayer from "./SwimMethodLayer";
import IceConsistencyLegend from "./IceConsistencyLegend";
import {
  fetchSwimConsistency,
  SWIM_METHODS,
  type DepthRange,
  type SwimMethod,
  type SwimConsistencyPoint,
  type MethodScore,
} from "../api/swim_ice";

/* =========================================================
 * Props
 * =======================================================*/
export interface SwimIcePanelProps {
  lat: number | null;
  lon: number | null;
  onLayerToggle?: (method: SwimMethod, visible: boolean) => void;
  onLayerOpacity?: (method: SwimMethod, opacity: number) => void;
  onDepthChange?: (depth: DepthRange) => void;
}

/* =========================================================
 * Helpers
 * =======================================================*/

/** Map consistency score to color class */
function scoreColorClass(score: number | null): string {
  if (score == null) return "text-slate-500";
  if (score >= 0.7) return "text-blue-400";
  if (score >= 0.3) return "text-blue-300";
  if (score >= -0.3) return "text-slate-400";
  if (score >= -0.7) return "text-red-300";
  return "text-red-400";
}

/** Map consistency score to label */
function scoreLabel(score: number | null): string {
  if (score == null) return "No Data";
  if (score >= 0.7) return "Strong Ice Evidence";
  if (score >= 0.3) return "Moderate Ice Evidence";
  if (score >= -0.3) return "Ambiguous";
  if (score >= -0.7) return "Moderate Against";
  return "Strong Against Ice";
}

/* =========================================================
 * Component
 * =======================================================*/
export default function SwimIcePanel({
  lat,
  lon,
  onLayerToggle,
  onLayerOpacity,
  onDepthChange,
}: SwimIcePanelProps) {
  const [depth, setDepth] = useState<DepthRange>("0-1m");
  const [visibleMethods, setVisibleMethods] = useState<Set<SwimMethod>>(new Set());
  const [collapsed, setCollapsed] = useState(false);
  const [showMethods, setShowMethods] = useState(true);

  // Query state
  const [consistency, setConsistency] = useState<SwimConsistencyPoint | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /* ── Depth change ────────────────────────────────────── */
  const handleDepthChange = useCallback(
    (d: DepthRange) => {
      setDepth(d);
      onDepthChange?.(d);
    },
    [onDepthChange],
  );

  /* ── Method toggle ───────────────────────────────────── */
  const handleMethodToggle = useCallback(
    (method: SwimMethod, visible: boolean) => {
      setVisibleMethods((prev) => {
        const next = new Set(prev);
        if (visible) next.add(method);
        else next.delete(method);
        return next;
      });
      onLayerToggle?.(method, visible);
    },
    [onLayerToggle],
  );

  /* ── Query consistency ───────────────────────────────── */
  const queryPoint = useCallback(async () => {
    if (lat == null || lon == null) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSwimConsistency(lat, lon);
      setConsistency(data);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Query failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [lat, lon]);

  /* ── Render ──────────────────────────────────────────── */
  return (
    <section className="flex flex-col border-b border-[#232f48]">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center gap-1.5 bg-[#0d1520] px-3 py-2 text-left hover:bg-[#111b2a]"
      >
        <span className="material-symbols-outlined text-[16px] text-blue-400">
          ac_unit
        </span>
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-200">
          SWIM Ice Detection
        </span>
        <span className="material-symbols-outlined ml-auto text-[14px] text-[#92a4c9]">
          {collapsed ? "expand_more" : "expand_less"}
        </span>
      </button>

      {!collapsed && (
        <div className="flex flex-col gap-2 bg-[#0d1520] px-3 pb-3 pt-1">
          {/* Depth selector */}
          <DepthRangeSelector value={depth} onChange={handleDepthChange} />

          {/* Method layers section */}
          <div className="flex flex-col gap-1">
            <button
              onClick={() => setShowMethods(!showMethods)}
              className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-[#92a4c9]"
            >
              <span className="material-symbols-outlined text-[12px]">
                {showMethods ? "expand_less" : "expand_more"}
              </span>
              Method Layers
            </button>
            {showMethods && (
              <div className="flex flex-col gap-1">
                {SWIM_METHODS.map((method) => (
                  <SwimMethodLayer
                    key={method}
                    method={method}
                    depth={depth}
                    visible={visibleMethods.has(method)}
                    onToggle={handleMethodToggle}
                    onOpacityChange={onLayerOpacity}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Point query */}
          {lat != null && lon != null && (
            <div className="flex flex-col gap-1.5 rounded border border-[#232f48] bg-[#111b2a] p-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-wider text-[#92a4c9]">
                  Point Query
                </span>
                <button
                  onClick={queryPoint}
                  disabled={loading}
                  className="rounded bg-blue-600/20 px-2 py-0.5 text-[10px] font-medium text-blue-300 transition-colors hover:bg-blue-600/30 disabled:opacity-50"
                >
                  {loading ? "Querying…" : "Query Ice"}
                </button>
              </div>
              <div className="flex gap-3 font-mono text-[10px] text-slate-400">
                <span>Lat: {lat.toFixed(3)}°</span>
                <span>Lon: {lon.toFixed(3)}°</span>
              </div>

              {/* Error */}
              {error && (
                <div className="rounded border border-red-500/20 bg-red-500/5 px-2 py-1 text-[10px] text-red-400">
                  {error}
                </div>
              )}

              {/* Results */}
              {consistency && !error && (
                <div className="flex flex-col gap-1.5">
                  {/* Overall score */}
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[#92a4c9]">Score:</span>
                    <span
                      className={`font-mono text-[14px] font-bold ${scoreColorClass(consistency.consistency_score)}`}
                    >
                      {consistency.consistency_score != null
                        ? consistency.consistency_score.toFixed(2)
                        : "—"}
                    </span>
                    <span className={`text-[10px] ${scoreColorClass(consistency.consistency_score)}`}>
                      {scoreLabel(consistency.consistency_score)}
                    </span>
                  </div>

                  {/* Depth estimate */}
                  {consistency.depth_to_ice_estimate_m != null && (
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-[#92a4c9]">Est. depth:</span>
                      <span className="font-mono text-[11px] text-slate-200">
                        {consistency.depth_to_ice_estimate_m.toFixed(1)} m
                      </span>
                    </div>
                  )}

                  {/* Method breakdown */}
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[9px] uppercase tracking-wider text-[#92a4c9]">
                      Method Scores
                    </span>
                    {consistency.method_scores.map((ms: MethodScore) => (
                      <div
                        key={ms.method}
                        className="flex items-center gap-1.5 text-[10px]"
                      >
                        <span className="w-[90px] truncate text-[#92a4c9]">
                          {ms.method}
                        </span>
                        <div className="h-1 flex-1 overflow-hidden rounded-full bg-[#1a2744]">
                          {ms.score != null && (
                            <div
                              className={`h-full rounded-full ${
                                ms.score >= 0 ? "bg-blue-500" : "bg-red-500"
                              }`}
                              style={{
                                width: `${Math.abs(ms.score) * 50}%`,
                                marginLeft: ms.score >= 0 ? "50%" : `${50 - Math.abs(ms.score) * 50}%`,
                              }}
                            />
                          )}
                        </div>
                        <span
                          className={`min-w-[32px] text-right font-mono ${scoreColorClass(ms.score)}`}
                        >
                          {ms.score != null ? ms.score.toFixed(2) : "—"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Legend */}
          <IceConsistencyLegend />
        </div>
      )}
    </section>
  );
}
