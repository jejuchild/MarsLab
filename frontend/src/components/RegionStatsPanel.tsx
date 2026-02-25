import { useEffect, useState, useCallback } from "react";

/* =========================================================
 * Types
 * =======================================================*/
interface RegionStatsProps {
  vertices: { lat: number; lon: number }[];
  onClose: () => void;
  onClearPolygon: () => void;
}

type RegionStatsData = {
  area_km2: number;
  elevation: { min: number; max: number; mean: number; std: number };
  slope: { min: number; max: number; mean: number; std: number };
  slope_distribution: {
    safe_0_3: number;
    marginal_3_5: number;
    hazardous_5_plus: number;
  };
  safety_rating: "FAVORABLE" | "MARGINAL" | "UNFAVORABLE" | "UNKNOWN";
  pixel_count: number;
  vertices_count: number;
};

const SAFETY_CONFIG = {
  FAVORABLE: {
    icon: "check_circle",
    bg: "bg-green-500/10",
    border: "border-green-500/20",
    text: "text-green-500",
    label: "FAVORABLE",
  },
  MARGINAL: {
    icon: "warning",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/20",
    text: "text-yellow-500",
    label: "MARGINAL",
  },
  UNFAVORABLE: {
    icon: "dangerous",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
    text: "text-red-500",
    label: "UNFAVORABLE",
  },
  UNKNOWN: {
    icon: "help",
    bg: "bg-slate-500/10",
    border: "border-slate-500/20",
    text: "text-slate-500",
    label: "UNKNOWN",
  },
};

/* =========================================================
 * RegionStatsPanel Component
 * =======================================================*/
export default function RegionStatsPanel({
  vertices,
  onClose,
  onClearPolygon,
}: RegionStatsProps) {
  const [data, setData] = useState<RegionStatsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Panel width (resizable)
  const [panelWidth, setPanelWidth] = useState(384);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.6);
      setPanelWidth(Math.max(280, Math.min(maxW, startW + delta)));
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

  // Fetch region stats when vertices change (need at least 3)
  useEffect(() => {
    if (vertices.length < 3) {
      setData(null);
      setError(null);
      return;
    }

    const controller = new AbortController();
    let cancelled = false;

    async function fetchStats() {
      setLoading(true);
      setError(null);
      setData(null);

      try {
        const res = await fetch("/api/terrain/region_stats", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(vertices),
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.error || `HTTP ${res.status}`);
        }
        const json: RegionStatsData = await res.json();
        if (!cancelled) setData(json);
      } catch (e: unknown) {
        if (!cancelled && e instanceof Error && e.name !== "AbortError") {
          setError(e.message ?? "Unknown error");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchStats();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [vertices]);

  const safetyConfig = data ? SAFETY_CONFIG[data.safety_rating] : SAFETY_CONFIG.UNKNOWN;

  return (
    <aside
      className="relative flex h-full flex-col border-l border-border-dark bg-surface-dark/40"
      style={{ width: panelWidth }}
    >
      {/* Resize handle (left edge) */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
        onMouseDown={handleResizeStart}
      />

      {/* Header */}
      <div className="p-4 border-b border-border-dark">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-8 rounded-lg bg-indigo-500/20 flex items-center justify-center text-indigo-400">
              <span className="material-symbols-outlined text-lg">pentagon</span>
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Region Analysis</h2>
              <span className="text-[10px] text-slate-500">
                {vertices.length} vertices
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex items-center justify-center p-1 text-slate-500 hover:text-red-400 transition-colors"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto scrollbar-dark">
        {/* Instructions when collecting vertices */}
        {vertices.length < 3 && (
          <div className="p-4 border-b border-border-dark">
            <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="material-symbols-outlined text-indigo-400 text-lg">info</span>
                <span className="text-sm font-medium text-indigo-300">Draw a polygon</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Click on the map to add vertices. You need at least 3 vertices to define a polygon region.
                Statistics will be computed automatically once the polygon is complete.
              </p>
              <div className="mt-3 flex items-center gap-2 text-[10px] text-slate-500">
                <span className="material-symbols-outlined text-xs">ads_click</span>
                {vertices.length === 0
                  ? "Click map to place first vertex"
                  : `${vertices.length}/3 vertices placed`}
              </div>
            </div>
          </div>
        )}

        {/* Vertex list */}
        {vertices.length > 0 && (
          <div className="p-4 border-b border-border-dark">
            <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
              Polygon Vertices
            </h3>
            <div className="space-y-1 max-h-32 overflow-y-auto scrollbar-dark">
              {vertices.map((v, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 px-2 py-1.5 rounded bg-[#0a0f18] border border-[#232f48]"
                >
                  <span className="text-[9px] font-bold text-indigo-400 w-4">{i + 1}</span>
                  <span className="text-[10px] font-mono text-[#92a4c9]">
                    {v.lat.toFixed(4)}, {v.lon.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center h-64">
            <span className="material-symbols-outlined animate-spin text-3xl text-indigo-400 mb-3">
              progress_activity
            </span>
            <p className="text-sm text-slate-400">Computing region statistics...</p>
            <p className="text-[10px] text-slate-600 mt-1">Analysing DEM data within polygon</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="p-4">
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-center">
              <span className="material-symbols-outlined text-2xl text-red-400 mb-2 block">
                error
              </span>
              <p className="text-sm text-red-400 mb-1">Analysis failed</p>
              <p className="text-[11px] text-slate-500">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {data && !loading && (
          <>
            {/* Area */}
            <div className="p-4 border-b border-border-dark">
              <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
                Area
              </h3>
              <div className="bg-indigo-500/10 rounded-lg p-4 border border-indigo-500/20 text-center">
                <p className="text-3xl font-mono font-bold text-white">
                  {data.area_km2.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                </p>
                <p className="text-xs text-indigo-300 mt-1">km²</p>
              </div>
              <p className="text-[9px] text-slate-600 mt-2 text-center">
                {data.pixel_count.toLocaleString()} pixels analysed
              </p>
            </div>

            {/* Elevation */}
            <div className="p-4 border-b border-border-dark">
              <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
                Elevation
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
                  <p className="text-slate-500 text-[10px] font-bold">Min</p>
                  <p className="text-white font-mono text-base">
                    {(data.elevation.min / 1000).toFixed(2)}{" "}
                    <span className="text-xs text-slate-400">km</span>
                  </p>
                </div>
                <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
                  <p className="text-slate-500 text-[10px] font-bold">Max</p>
                  <p className="text-white font-mono text-base">
                    {(data.elevation.max / 1000).toFixed(2)}{" "}
                    <span className="text-xs text-slate-400">km</span>
                  </p>
                </div>
                <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
                  <p className="text-slate-500 text-[10px] font-bold">Mean</p>
                  <p className="text-white font-mono text-base">
                    {(data.elevation.mean / 1000).toFixed(2)}{" "}
                    <span className="text-xs text-slate-400">km</span>
                  </p>
                </div>
                <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
                  <p className="text-slate-500 text-[10px] font-bold">Std Dev</p>
                  <p className="text-white font-mono text-base">
                    {data.elevation.std.toFixed(1)}{" "}
                    <span className="text-xs text-slate-400">m</span>
                  </p>
                </div>
              </div>

              {/* Elevation range bar */}
              <div className="mt-3">
                <div className="flex justify-between text-[9px] text-slate-500 mb-1">
                  <span>{(data.elevation.min / 1000).toFixed(2)} km</span>
                  <span>{(data.elevation.max / 1000).toFixed(2)} km</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-600 via-cyan-400 to-emerald-400"
                    style={{
                      width: "100%",
                    }}
                  />
                </div>
                <div className="flex justify-center mt-1">
                  <span className="text-[9px] text-cyan-400 font-mono">
                    Mean: {(data.elevation.mean / 1000).toFixed(2)} km
                  </span>
                </div>
              </div>
            </div>

            {/* Slope */}
            <div className="p-4 border-b border-border-dark">
              <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
                Slope
              </h3>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div>
                  <p className="text-slate-500 text-[10px] font-bold">Mean Slope</p>
                  <p className="text-white text-xl font-mono">
                    {data.slope.mean.toFixed(1)}{"\u00b0"}
                  </p>
                </div>
                <div>
                  <p className="text-slate-500 text-[10px] font-bold">Max Slope</p>
                  <p
                    className={`text-xl font-mono ${
                      data.slope.max >= 5 ? "text-red-400" : "text-white"
                    }`}
                  >
                    {data.slope.max.toFixed(1)}{"\u00b0"}
                  </p>
                </div>
              </div>

              {/* Slope distribution - stacked horizontal bar */}
              <p className="text-slate-500 text-[10px] uppercase font-bold mb-2">
                Slope Distribution
              </p>
              <div className="h-5 flex rounded-full overflow-hidden mb-2">
                {data.slope_distribution.safe_0_3 > 0 && (
                  <div
                    className="bg-green-500 h-full transition-all duration-500"
                    style={{ width: `${data.slope_distribution.safe_0_3}%` }}
                    title={`0-3\u00b0: ${data.slope_distribution.safe_0_3.toFixed(1)}%`}
                  />
                )}
                {data.slope_distribution.marginal_3_5 > 0 && (
                  <div
                    className="bg-yellow-500 h-full transition-all duration-500"
                    style={{ width: `${data.slope_distribution.marginal_3_5}%` }}
                    title={`3-5\u00b0: ${data.slope_distribution.marginal_3_5.toFixed(1)}%`}
                  />
                )}
                {data.slope_distribution.hazardous_5_plus > 0 && (
                  <div
                    className="bg-red-500 h-full transition-all duration-500"
                    style={{ width: `${data.slope_distribution.hazardous_5_plus}%` }}
                    title={`>5\u00b0: ${data.slope_distribution.hazardous_5_plus.toFixed(1)}%`}
                  />
                )}
              </div>
              <div className="flex gap-4 text-[9px]">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-green-500" />
                  <span className="text-slate-400">
                    0-3{"\u00b0"} ({data.slope_distribution.safe_0_3.toFixed(1)}%)
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-yellow-500" />
                  <span className="text-slate-400">
                    3-5{"\u00b0"} ({data.slope_distribution.marginal_3_5.toFixed(1)}%)
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-red-500" />
                  <span className="text-slate-400">
                    {"\u2265"}5{"\u00b0"} ({data.slope_distribution.hazardous_5_plus.toFixed(1)}%)
                  </span>
                </div>
              </div>
            </div>

            {/* Safety Assessment */}
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold">
                  Safety Assessment
                </h3>
                <div
                  className={`flex items-center gap-1.5 ${safetyConfig.bg} ${safetyConfig.text} border ${safetyConfig.border} px-3 py-1 rounded-full`}
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {safetyConfig.icon}
                  </span>
                  <span className="text-[11px] font-bold">{safetyConfig.label}</span>
                </div>
              </div>
              <div className="bg-slate-800/20 rounded-lg p-3 border border-slate-700/30">
                <p className="text-slate-300 text-xs leading-relaxed">
                  {data.safety_rating === "FAVORABLE"
                    ? `All terrain within the polygon is below the 5\u00b0 safety threshold. Mean slope ${data.slope.mean.toFixed(1)}\u00b0 indicates favorable landing conditions.`
                    : data.safety_rating === "MARGINAL"
                    ? `${data.slope_distribution.hazardous_5_plus.toFixed(1)}% of the region exceeds the 5\u00b0 safety threshold. Mean slope ${data.slope.mean.toFixed(1)}\u00b0 is below the limit but localised steep areas require further analysis.`
                    : data.safety_rating === "UNFAVORABLE"
                    ? `${data.slope_distribution.hazardous_5_plus.toFixed(1)}% of the region exceeds the 5\u00b0 safety threshold. Mean slope ${data.slope.mean.toFixed(1)}\u00b0 indicates potential landing hazards. Not recommended for landing.`
                    : "No valid terrain data available in this region."}
                </p>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border-dark bg-bg-dark p-4 space-y-2">
        {vertices.length >= 3 && (
          <button
            onClick={onClearPolygon}
            className="flex w-full items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-bold uppercase tracking-widest bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:text-red-400 hover:border-red-500/30 transition-all active:scale-[0.98]"
          >
            <span className="material-symbols-outlined text-sm">delete</span>
            Clear Polygon
          </button>
        )}
        <button
          disabled
          title="Export region report (coming soon)"
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary/40 py-3 text-xs font-bold uppercase tracking-widest text-white/50 cursor-not-allowed transition-all"
        >
          <span className="material-symbols-outlined text-sm">download</span>
          Export Region Report
        </button>
      </div>
    </aside>
  );
}
