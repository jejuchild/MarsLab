import { useEffect, useState, useCallback, useRef } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  fetchAttenuationProfile,
  getAttenuationCsvUrl,
  type AttenuationResult,
  type AttenuationOverlaySegment,
} from "../api/attenuation";

/* =========================================================
 * Props
 * =======================================================*/
export interface AttenuationPanelProps {
  productId: string;
  onClose: () => void;
  onOverlayUpdate?: (segments: AttenuationOverlaySegment[]) => void;
}

/* =========================================================
 * Transparency band colors
 * =======================================================*/
const TRANSPARENCY_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  "Pure ice":                   { bg: "bg-blue-500/10",   border: "border-blue-500/20",   text: "text-blue-400" },
  "Clean ice / low-loss rock":  { bg: "bg-cyan-500/10",   border: "border-cyan-500/20",   text: "text-cyan-400" },
  "Dusty ice / porous basalt":  { bg: "bg-yellow-500/10", border: "border-yellow-500/20", text: "text-yellow-400" },
  "Moderate loss (mixed)":      { bg: "bg-orange-500/10", border: "border-orange-500/20", text: "text-orange-400" },
  "Clay-rich / briny":          { bg: "bg-red-500/10",    border: "border-red-500/20",    text: "text-red-400" },
};

/* =========================================================
 * Component
 * =======================================================*/
export default function AttenuationPanel({
  productId,
  onClose,
  onOverlayUpdate,
}: AttenuationPanelProps) {
  const [result, setResult] = useState<AttenuationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Parameters
  const [epsilonR, setEpsilonR] = useState(2.5);
  const [snrThreshold, setSnrThreshold] = useState(3.5);
  const [searchLo, setSearchLo] = useState(10);
  const [searchHi, setSearchHi] = useState(150);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Panel width
  const [panelWidth, setPanelWidth] = useState(420);

  const lastRunRef = useRef({ epsilonR: 2.5, snrThreshold: 3.5, searchLo: 10, searchHi: 150 });
  const paramsChanged =
    epsilonR !== lastRunRef.current.epsilonR ||
    snrThreshold !== lastRunRef.current.snrThreshold ||
    searchLo !== lastRunRef.current.searchLo ||
    searchHi !== lastRunRef.current.searchHi;

  /* ── Resize ─────────────────────────────────────────── */
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
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
    },
    [panelWidth],
  );

  /* ── Fetch analysis ─────────────────────────────────── */
  const runAnalysis = useCallback(
    async (er: number, snr: number, lo: number, hi: number) => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchAttenuationProfile(productId, er, snr, lo, hi);
        setResult(data);
        lastRunRef.current = { epsilonR: er, snrThreshold: snr, searchLo: lo, searchHi: hi };
        onOverlayUpdate?.(data.overlay_segments);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Analysis failed";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [productId, onOverlayUpdate],
  );

  useEffect(() => {
    runAnalysis(epsilonR, snrThreshold, searchLo, searchHi);
    return () => onOverlayUpdate?.([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productId]);

  /* ── Chart data ─────────────────────────────────────── */
  const chartData =
    result?.profile
      .filter((s) => s.alpha_dBm != null)
      .map((s) => ({
        km: s.along_track_km,
        alpha: s.alpha_dBm,
        depth: s.depth_m,
      })) ?? [];

  const summary = result?.summary;

  /* ── Detection rate badge ───────────────────────────── */
  const detRate = summary ? summary.detection_rate * 100 : 0;
  const detColor =
    detRate >= 30
      ? { bg: "bg-green-500/10", border: "border-green-500/20", text: "text-green-400" }
      : detRate >= 10
        ? { bg: "bg-yellow-500/10", border: "border-yellow-500/20", text: "text-yellow-400" }
        : { bg: "bg-red-500/10", border: "border-red-500/20", text: "text-red-400" };

  /* ── Dominant transparency badge ────────────────────── */
  const domTrans = summary?.dominant_transparency;
  const transStyle = domTrans ? TRANSPARENCY_COLORS[domTrans] : null;

  return (
    <aside
      className="relative flex h-full flex-col border-l border-border-dark bg-surface-dark/40"
      style={{ width: panelWidth }}
    >
      {/* Resize handle */}
      <div
        onMouseDown={handleResizeStart}
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 transition-colors"
      />

      {/* ── Header ────────────────────────────────────── */}
      <div className="flex items-center gap-2 p-4 border-b border-border-dark">
        <span className="material-symbols-outlined text-primary text-lg">radar</span>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-bold text-white truncate">Radar Attenuation</h2>
          <p className="text-[10px] text-slate-500 font-mono truncate">{productId}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined text-lg">close</span>
        </button>
      </div>

      {/* ── Content ───────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto scrollbar-dark flex flex-col gap-4 p-4">
        {loading ? (
          <div className="flex-1 flex flex-col items-center justify-center py-16">
            <span className="material-symbols-outlined animate-spin text-3xl text-primary mb-3">
              progress_activity
            </span>
            <p className="text-xs text-slate-500">Computing radar attenuation...</p>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400 text-xs font-medium">{error}</p>
          </div>
        ) : result && summary ? (
          <>
            {/* ── Detection Rate Badge ────────────────── */}
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-lg ${detColor.bg} border ${detColor.border}`}
            >
              <span className={`material-symbols-outlined text-sm ${detColor.text}`}>
                {detRate >= 30 ? "check_circle" : detRate >= 10 ? "warning" : "error"}
              </span>
              <span className={`text-[11px] font-bold ${detColor.text}`}>
                {detRate.toFixed(1)}% Valid Attenuation
              </span>
              <span className="text-[10px] text-slate-500 ml-auto">
                {summary.valid_traces} / {summary.total_traces} traces
              </span>
            </div>

            {/* ── Dominant Transparency Badge ─────────── */}
            {domTrans && transStyle && (
              <div
                className={`flex items-center gap-2 px-3 py-2.5 rounded-lg ${transStyle.bg} border ${transStyle.border}`}
              >
                <span className={`material-symbols-outlined text-base ${transStyle.text}`}>
                  visibility
                </span>
                <div>
                  <p className={`text-[11px] font-bold ${transStyle.text}`}>{domTrans}</p>
                  <p className="text-[9px] text-slate-500">Dominant material transparency</p>
                </div>
              </div>
            )}

            {/* ── Summary Stats ───────────────────────── */}
            <div className="grid grid-cols-2 gap-2">
              <StatCard
                label="Mean α"
                value={summary.alpha_mean_dBm}
                unit="dB/m"
                decimals={4}
                icon="straighten"
              />
              <StatCard
                label="Median α"
                value={summary.alpha_median_dBm}
                unit="dB/m"
                decimals={4}
                icon="vertical_align_center"
              />
              <StatCard
                label="Std Dev"
                value={summary.alpha_std_dBm}
                unit="dB/m"
                decimals={4}
                icon="functions"
              />
              <StatCard
                label="Distance"
                value={summary.total_distance_km}
                unit="km"
                decimals={1}
                icon="route"
              />
            </div>

            {/* ── Transparency Counts ─────────────────── */}
            {Object.keys(summary.transparency_counts).length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Transparency Distribution
                </h3>
                <div className="space-y-1.5">
                  {Object.entries(summary.transparency_counts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, count]) => {
                      const style = TRANSPARENCY_COLORS[cls];
                      const pct = summary.valid_traces
                        ? ((count / summary.valid_traces) * 100).toFixed(0)
                        : "0";
                      return (
                        <div
                          key={cls}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-md ${style?.bg ?? "bg-slate-800/40"} border ${style?.border ?? "border-slate-700/50"}`}
                        >
                          <span className={`text-[10px] flex-1 ${style?.text ?? "text-slate-400"}`}>{cls}</span>
                          <span className="text-[10px] font-mono text-slate-400">{count}</span>
                          <span className="text-[9px] text-slate-600">({pct}%)</span>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

            {/* DEM + epsilon badges */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="inline-flex items-center gap-1 bg-slate-800/60 border border-slate-700/50 px-2.5 py-1 rounded-full text-[10px] text-slate-300">
                <span className="material-symbols-outlined text-[12px]">terrain</span>
                {summary.dem_source}
              </span>
              <span className="inline-flex items-center gap-1 bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-1 rounded-full text-[10px] text-indigo-400">
                <span className="material-symbols-outlined text-[12px]">electric_bolt</span>
                εr = {summary.epsilon_r}
              </span>
            </div>

            {/* ── Attenuation Profile Chart ────────────── */}
            {chartData.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Attenuation Profile
                </h3>
                <div className="bg-[#101622] rounded-lg border border-slate-700/50 p-2">
                  <ResponsiveContainer width="100%" height={220}>
                    <ComposedChart
                      data={chartData}
                      margin={{ top: 8, right: 8, left: -10, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e2a40" />
                      <XAxis
                        dataKey="km"
                        stroke="#4a5568"
                        tick={{ fontSize: 9, fill: "#6b7c9c" }}
                        label={{
                          value: "km",
                          position: "insideBottomRight",
                          offset: -2,
                          style: { fontSize: 9, fill: "#6b7c9c" },
                        }}
                      />
                      <YAxis
                        stroke="#4a5568"
                        tick={{ fontSize: 9, fill: "#6b7c9c" }}
                        label={{
                          value: "α (dB/m)",
                          angle: -90,
                          position: "insideLeft",
                          offset: 20,
                          style: { fontSize: 9, fill: "#6b7c9c" },
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#101622",
                          border: "1px solid #232f48",
                          fontSize: 11,
                          borderRadius: 6,
                        }}
                        labelFormatter={(v) => `${v} km`}
                        formatter={(value: number | undefined, name?: string) => {
                          if (name === "alpha")
                            return [`${(value ?? 0).toFixed(4)} dB/m`, "Attenuation"];
                          if (name === "depth")
                            return [`${(value ?? 0).toFixed(1)} m`, "Depth"];
                          return [value ?? 0, name ?? ""];
                        }}
                      />
                      {/* Transparency threshold reference lines */}
                      <ReferenceLine y={0.005} stroke="#3b82f6" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <ReferenceLine y={0.02}  stroke="#06b6d4" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <ReferenceLine y={0.05}  stroke="#eab308" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <ReferenceLine y={0.08}  stroke="#f97316" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <Line
                        type="monotone"
                        dataKey="alpha"
                        stroke="#4f9cf7"
                        strokeWidth={1.5}
                        isAnimationActive={false}
                        dot={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                  <div className="flex items-center justify-center gap-3 text-[9px] text-slate-500 mt-1 flex-wrap">
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-0.5 bg-[#4f9cf7] rounded" />
                      α (dB/m)
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-px bg-blue-500/50 border-dashed" />
                      Ice threshold
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* ── Parameters / Controls ───────────────── */}
            <div>
              <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                Parameters
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[10px] text-slate-400">
                      Dielectric Constant (εr)
                    </label>
                    <span className="text-xs font-mono text-white">{epsilonR.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min={1.5}
                    max={8.0}
                    step={0.1}
                    value={epsilonR}
                    onChange={(e) => setEpsilonR(+e.target.value)}
                    className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
                  />
                  <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                    <span>1.5 (dry regolith)</span>
                    <span>8.0 (ice-rich)</span>
                  </div>
                </div>

                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
                >
                  <span className="material-symbols-outlined text-[14px]">
                    {showAdvanced ? "expand_less" : "expand_more"}
                  </span>
                  Advanced Parameters
                </button>

                {showAdvanced && (
                  <div className="space-y-3 pl-2 border-l border-slate-700/50">
                    <SliderParam label="SNR Threshold" value={snrThreshold} min={1} max={20} step={0.5} onChange={setSnrThreshold} />
                    <SliderParam label="Search Lo (bins)" value={searchLo} min={1} max={100} step={1} onChange={setSearchLo} />
                    <SliderParam label="Search Hi (bins)" value={searchHi} min={50} max={500} step={5} onChange={setSearchHi} />
                  </div>
                )}

                {paramsChanged && (
                  <button
                    onClick={() => runAnalysis(epsilonR, snrThreshold, searchLo, searchHi)}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary/20 border border-primary/30 rounded-lg text-primary text-xs font-bold uppercase tracking-wider hover:bg-primary/30 transition-colors disabled:opacity-50"
                  >
                    <span className="material-symbols-outlined text-sm">refresh</span>
                    Re-run Analysis
                  </button>
                )}
              </div>
            </div>
          </>
        ) : null}
      </div>

      {/* ── Footer: Export ─────────────────────────────── */}
      {result?.success && (
        <div className="border-t border-border-dark bg-bg-dark p-4">
          <button
            onClick={() => {
              const url = getAttenuationCsvUrl(
                productId, epsilonR, snrThreshold, searchLo, searchHi,
              );
              window.open(url, "_blank");
            }}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-primary py-3 text-xs font-bold uppercase tracking-widest text-white hover:brightness-110 transition-all"
          >
            <span className="material-symbols-outlined text-sm">download</span>
            Export CSV
          </button>
        </div>
      )}
    </aside>
  );
}

/* =========================================================
 * Sub-components
 * =======================================================*/
function StatCard({
  label,
  value,
  unit,
  icon,
  decimals = 1,
}: {
  label: string;
  value?: number | null;
  unit?: string;
  icon?: string;
  decimals?: number;
}) {
  return (
    <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
      <p className="text-slate-500 text-[10px] uppercase font-bold flex items-center gap-1">
        {icon && (
          <span className="material-symbols-outlined text-[12px]">{icon}</span>
        )}
        {label}
      </p>
      <p className="text-white font-mono text-sm">
        {value != null ? value.toFixed(decimals) : "\u2014"}
        {unit && value != null && (
          <span className="text-[10px] text-slate-400 ml-1">{unit}</span>
        )}
      </p>
    </div>
  );
}

function SliderParam({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-[10px] text-slate-400">{label}</label>
        <span className="text-xs font-mono text-white">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
      />
    </div>
  );
}
