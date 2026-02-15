import { useEffect, useState, useCallback } from "react";

/* =========================================================
 * Types
 * =======================================================*/
export type TerrainPoint = {
  lat: number;
  lon: number;
};

type SlopeStats = {
  mean_slope: number;
  std_slope: number;
  max_slope: number;
  count: number;
  elevation_m: number;
  distribution: {
    "0_3": number;
    "3_5": number;
    "5_plus": number;
  };
  safety: "FAVORABLE" | "MARGINAL" | "UNFAVORABLE" | "UNKNOWN";
  safety_description: string;
  lat: number;
  lon: number;
  radius_m: number;
};

type ThermalInertia = {
  thermal_inertia: number;
  unit: string;
  elevation_m: number;
  interpretation: string;
  material_estimate: string;
  note: string;
};

const SAFETY_CONFIG = {
  FAVORABLE: {
    color: "green",
    icon: "check_circle",
    bg: "bg-green-500/10",
    border: "border-green-500/20",
    text: "text-green-500",
  },
  MARGINAL: {
    color: "yellow",
    icon: "warning",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/20",
    text: "text-yellow-500",
  },
  UNFAVORABLE: {
    color: "red",
    icon: "dangerous",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
    text: "text-red-500",
  },
  UNKNOWN: {
    color: "slate",
    icon: "help",
    bg: "bg-slate-500/10",
    border: "border-slate-500/20",
    text: "text-slate-500",
  },
};

const DIST_BARS: {
  key: keyof SlopeStats["distribution"];
  label: string;
  color: string;
  tag?: string;
}[] = [
  { key: "0_3", label: "0-3\u00b0", color: "bg-green-500", tag: "Safe" },
  { key: "3_5", label: "3-5\u00b0", color: "bg-yellow-500", tag: "Marginal" },
  { key: "5_plus", label: "\u22655\u00b0", color: "bg-red-500", tag: "Unsafe" },
];

/* =========================================================
 * Thermal Inertia color helper
 * =======================================================*/
function tiColorClass(ti: number): { bg: string; border: string; text: string; dot: string } {
  if (ti < 100) return { bg: "bg-sky-500/10", border: "border-sky-500/20", text: "text-sky-400", dot: "bg-sky-400" };
  if (ti < 250) return { bg: "bg-sky-500/10", border: "border-sky-500/20", text: "text-sky-300", dot: "bg-sky-300" };
  if (ti < 450) return { bg: "bg-amber-500/10", border: "border-amber-500/20", text: "text-amber-400", dot: "bg-amber-400" };
  if (ti < 800) return { bg: "bg-orange-500/10", border: "border-orange-500/20", text: "text-orange-400", dot: "bg-orange-400" };
  return { bg: "bg-red-500/10", border: "border-red-500/20", text: "text-red-400", dot: "bg-red-400" };
}

/* =========================================================
 * SlopeAnalysis Component
 * =======================================================*/
export default function SlopeAnalysis({
  point,
  onClose,
  radiusM = 1000,
}: {
  point: TerrainPoint;
  onClose: () => void;
  radiusM?: number;
}) {
  const [data, setData] = useState<SlopeStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Thermal inertia state
  const [tiData, setTiData] = useState<ThermalInertia | null>(null);
  const [tiLoading, setTiLoading] = useState(false);

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

  useEffect(() => {
    const controller = new AbortController();

    async function fetchSlope() {
      setLoading(true);
      setError(null);
      setData(null);

      try {
        const params = new URLSearchParams({
          lat: point.lat.toString(),
          lon: point.lon.toString(),
          radius_m: radiusM.toString(),
        });
        const res = await fetch(`/terrain/slope_stats?${params}`, {
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.error || `HTTP ${res.status}`);
        }
        const json: SlopeStats = await res.json();
        setData(json);
      } catch (e: any) {
        if (e.name !== "AbortError") {
          setError(e.message ?? "Unknown error");
        }
      } finally {
        setLoading(false);
      }
    }

    fetchSlope();
    return () => controller.abort();
  }, [point.lat, point.lon, radiusM]);

  // Fetch thermal inertia
  useEffect(() => {
    const controller = new AbortController();

    async function fetchTI() {
      setTiLoading(true);
      setTiData(null);

      try {
        const params = new URLSearchParams({
          lat: point.lat.toString(),
          lon: point.lon.toString(),
        });
        const res = await fetch(`/terrain/thermal_inertia?${params}`, {
          signal: controller.signal,
        });
        if (res.ok) {
          const json: ThermalInertia = await res.json();
          setTiData(json);
        }
      } catch {
        // Silently ignore -- thermal inertia is supplemental
      } finally {
        setTiLoading(false);
      }
    }

    fetchTI();
    return () => controller.abort();
  }, [point.lat, point.lon]);

  return (
    <aside className="relative flex h-full flex-col border-l border-border-dark bg-surface-dark/40" style={{ width: panelWidth }}>
      {/* Resize handle (left edge) */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
        onMouseDown={handleResizeStart}
      />
      {/* Header */}
      <div className="p-4 border-b border-border-dark">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary">
              <span className="material-symbols-outlined text-lg">analytics</span>
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Slope Analysis</h2>
              <span className="text-[10px] text-slate-500">
                {radiusM >= 1000 ? `${radiusM / 1000} km` : `${radiusM} m`} radius
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
        {/* Section 1: Coordinates */}
        <div className="p-4 border-b border-border-dark">
          <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
            Region &amp; Coordinates
          </h3>
          <p className="text-primary font-mono text-sm bg-primary/10 inline-block px-2 py-0.5 rounded">
            {Math.abs(point.lat).toFixed(4)}{"\u00b0"}
            {point.lat >= 0 ? "N" : "S"},{" "}
            {Math.abs(point.lon).toFixed(4)}{"\u00b0"}
            {point.lon >= 0 ? "E" : "W"}
          </p>

          {/* Elevation + Ellipsoid */}
          {data && (
            <div className="flex gap-3 mt-3">
              <div className="flex-1 bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
                <p className="text-slate-500 text-[10px] uppercase font-bold">MOLA Elevation</p>
                <p className="text-white font-mono text-base">
                  {(data.elevation_m / 1000).toFixed(2)}{" "}
                  <span className="text-xs text-slate-400">km</span>
                </p>
              </div>
              <div className="flex-1 bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
                <p className="text-slate-500 text-[10px] uppercase font-bold">Ellipsoid</p>
                <p className="text-white font-mono text-base">IAU 2000</p>
              </div>
            </div>
          )}
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center h-64">
            <span className="material-symbols-outlined animate-spin text-3xl text-primary mb-3">
              progress_activity
            </span>
            <p className="text-sm text-slate-400">Computing slope statistics...</p>
            <p className="text-[10px] text-slate-600 mt-1">Reading DEM data</p>
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
            {/* Section 2: Terrain Slope Statistics */}
            <div className="p-4 border-b border-border-dark">
              <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-4">
                Terrain Slope Statistics
              </h3>

              {/* 2x2 stat grid */}
              <div className="grid grid-cols-2 gap-3 mb-5">
                <div>
                  <p className="text-slate-500 text-[10px] font-bold">Mean Slope</p>
                  <p className="text-white text-xl font-mono">{data.mean_slope.toFixed(1)}{"\u00b0"}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-[10px] font-bold">Std Dev</p>
                  <p className="text-white text-xl font-mono">{data.std_slope.toFixed(1)}{"\u00b0"}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-[10px] font-bold">Max Slope</p>
                  <p className={`text-xl font-mono ${data.max_slope >= 5 ? "text-red-400" : "text-white"}`}>
                    {data.max_slope.toFixed(1)}{"\u00b0"}
                  </p>
                </div>
                <div>
                  <p className="text-slate-500 text-[10px] font-bold">Pixel Count</p>
                  <p className="text-white text-xl font-mono">{data.count.toLocaleString()}</p>
                </div>
              </div>

              {/* Slope distribution bars */}
              <div className="space-y-2.5">
                <p className="text-slate-500 text-[10px] uppercase font-bold">
                  Slope Distribution (%)
                </p>
                {DIST_BARS.map((bar) => (
                  <div key={bar.key} className="flex items-center gap-3">
                    <span className="text-[10px] font-mono text-slate-400 w-8">{bar.label}</span>
                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`${bar.color} h-full rounded-full transition-all duration-500`}
                        style={{ width: `${data.distribution[bar.key]}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-slate-500 w-10 text-right">
                      {data.distribution[bar.key].toFixed(1)}%
                    </span>
                    {bar.tag && (
                      <span className="text-[8px] text-slate-600 w-12">{bar.tag}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Section 3: Safety Assessment */}
            <div className="p-4 border-b border-border-dark">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold">
                  Safety Assessment
                </h3>
                <div
                  className={`flex items-center gap-1.5 ${SAFETY_CONFIG[data.safety].bg} ${SAFETY_CONFIG[data.safety].text} border ${SAFETY_CONFIG[data.safety].border} px-3 py-1 rounded-full`}
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {SAFETY_CONFIG[data.safety].icon}
                  </span>
                  <span className="text-[11px] font-bold">{data.safety}</span>
                </div>
              </div>
              <div className="bg-slate-800/20 rounded-lg p-3 border border-slate-700/30">
                <p className="text-slate-300 text-xs leading-relaxed">
                  {data.safety_description}
                </p>
              </div>
            </div>

            {/* Section 4: Thermal Inertia */}
            <div className="p-4 border-b border-border-dark">
              <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
                Thermal Inertia
              </h3>

              {tiLoading && (
                <div className="flex items-center gap-2 text-slate-500 text-xs">
                  <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                  Estimating...
                </div>
              )}

              {tiData && !tiLoading && (() => {
                const colors = tiColorClass(tiData.thermal_inertia);
                return (
                  <div className="space-y-3">
                    {/* Value + color indicator */}
                    <div className="flex items-center gap-3">
                      <div className={`size-3 rounded-full ${colors.dot}`} />
                      <div>
                        <p className="text-white font-mono text-lg">
                          {tiData.thermal_inertia.toFixed(1)}{" "}
                          <span className="text-[10px] text-slate-400">{tiData.unit}</span>
                        </p>
                      </div>
                    </div>

                    {/* Material estimate badge */}
                    <div className={`inline-flex items-center gap-1.5 ${colors.bg} ${colors.text} border ${colors.border} px-2.5 py-1 rounded-full`}>
                      <span className="material-symbols-outlined text-xs">layers</span>
                      <span className="text-[11px] font-bold">{tiData.material_estimate}</span>
                    </div>

                    {/* Interpretation */}
                    <div className="bg-slate-800/20 rounded-lg p-3 border border-slate-700/30">
                      <p className="text-slate-300 text-xs leading-relaxed">
                        {tiData.interpretation}
                      </p>
                    </div>

                    {/* Model note */}
                    <p className="text-[9px] text-slate-600 italic">{tiData.note}</p>
                  </div>
                );
              })()}

              {!tiData && !tiLoading && (
                <p className="text-[10px] text-slate-600">Not available</p>
              )}
            </div>

            {/* Section 5: GeoTIFF Export */}
            <div className="p-4">
              <h3 className="text-slate-400 text-[11px] uppercase tracking-[0.1em] font-bold mb-3">
                GIS Export
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={() =>
                    window.open(
                      `/terrain/export_geotiff?lat=${point.lat}&lon=${point.lon}&radius_km=5&data_type=elevation`
                    )
                  }
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-[10px] font-medium bg-emerald-500/20 border border-emerald-500/30 rounded text-emerald-400 hover:bg-emerald-500/30 transition-colors"
                >
                  <span className="material-symbols-outlined text-xs">download</span>
                  Export Elevation
                </button>
                <button
                  onClick={() =>
                    window.open(
                      `/terrain/export_geotiff?lat=${point.lat}&lon=${point.lon}&radius_km=5&data_type=slope`
                    )
                  }
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-[10px] font-medium bg-amber-500/20 border border-amber-500/30 rounded text-amber-400 hover:bg-amber-500/30 transition-colors"
                >
                  <span className="material-symbols-outlined text-xs">download</span>
                  Export Slope
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border-dark bg-bg-dark p-4">
        <button className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary py-3 text-xs font-bold uppercase tracking-widest text-white shadow-lg shadow-primary/20 hover:brightness-110 active:scale-[0.98] transition-all">
          <span className="material-symbols-outlined text-sm">download</span>
          Export Feasibility Report
        </button>
      </div>
    </aside>
  );
}
