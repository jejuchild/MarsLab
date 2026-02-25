import { useEffect, useState, useCallback, useRef } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  fetchStratigraphyAnalysis,
  getStratigraphyCsvUrl,
  type StratigraphyResult,
  type EpsilonEstimateInfo,
} from "../api/stratigraphy";

/* =========================================================
 * Props
 * =======================================================*/
interface DetectedFeature {
  id: string;
  type: string;
  lat: number;
  lon: number;
  diameter_km?: number;
  depth_m?: number;
  morphology?: string;
  n_terraces?: number;
  confidence: number;
  terrace_depth_m?: number;
}

export interface StratigraphyPanelProps {
  craterFeature: DetectedFeature;
  onClose: () => void;
}

/* =========================================================
 * Quality badge colors
 * =======================================================*/
const QUALITY_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  good:       { bg: "bg-green-500/10",  border: "border-green-500/20",  text: "text-green-400" },
  marginal:   { bg: "bg-yellow-500/10", border: "border-yellow-500/20", text: "text-yellow-400" },
  unreliable: { bg: "bg-red-500/10",    border: "border-red-500/20",    text: "text-red-400" },
};

/* =========================================================
 * Component
 * =======================================================*/
export default function StratigraphyPanel({
  craterFeature,
  onClose,
}: StratigraphyPanelProps) {
  const [result, setResult] = useState<StratigraphyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Parameters
  const [bufferKm, setBufferKm] = useState(30);
  const [minSnr, setMinSnr] = useState(3.5);
  const [maxRadiusM, setMaxRadiusM] = useState(3000);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Panel width
  const [panelWidth, setPanelWidth] = useState(440);

  // Track param changes
  const lastRunRef = useRef({ bufferKm: 30, minSnr: 3.5, maxRadiusM: 3000 });
  const paramsChanged =
    bufferKm !== lastRunRef.current.bufferKm ||
    minSnr !== lastRunRef.current.minSnr ||
    maxRadiusM !== lastRunRef.current.maxRadiusM;

  /* ── Resize ─────────────────────────────────────────── */
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = panelWidth;
      const onMove = (ev: MouseEvent) => {
        const delta = startX - ev.clientX;
        const maxW = Math.floor(window.innerWidth * 0.6);
        setPanelWidth(Math.max(340, Math.min(maxW, startW + delta)));
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
    async (buf: number, snr: number, maxR: number) => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchStratigraphyAnalysis(
          craterFeature.lat,
          craterFeature.lon,
          craterFeature.diameter_km ?? 0,
          buf, 36, maxR, snr,
        );
        setResult(data);
        lastRunRef.current = { bufferKm: buf, minSnr: snr, maxRadiusM: maxR };
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Analysis failed";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [craterFeature],
  );

  // Auto-run on mount
  useEffect(() => {
    runAnalysis(bufferKm, minSnr, maxRadiusM);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [craterFeature.id]);

  /* ── Chart data ─────────────────────────────────────── */
  const chartData = result?.radial_profile.map((p) => ({
    r: p.radius_m,
    elev: p.elevation_m,
    elevMin: p.elevation_min,
    elevMax: p.elevation_max,
  })) ?? [];

  const summary = result?.summary;
  const craterInfo = result?.crater_info;
  const bestEst = result?.epsilon_estimates?.find(e => e.quality === "good")
    ?? result?.epsilon_estimates?.[0];

  /* ── Quality badge ──────────────────────────────────── */
  const qc = QUALITY_COLORS[bestEst?.quality ?? ""] ?? QUALITY_COLORS.unreliable!;

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
          <h2 className="text-sm font-bold text-white truncate">Crater Stratigraphy</h2>
          <p className="text-[10px] text-slate-500 font-mono truncate">
            ({craterFeature.lat.toFixed(3)}, {craterFeature.lon.toFixed(3)})
          </p>
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
            <p className="text-xs text-slate-500">Analyzing crater stratigraphy...</p>
            <p className="text-[10px] text-slate-600 mt-1">Searching SHARAD tracks + DTMs</p>
          </div>
        ) : error && !result ? (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400 text-xs font-medium">{error}</p>
          </div>
        ) : result ? (
          <>
            {/* Warning banner for partial results */}
            {result.error && (
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <p className="text-yellow-400 text-[11px]">{result.error}</p>
              </div>
            )}

            {/* ── Crater Info ──────────────────────────── */}
            {craterInfo && (
              <div className="space-y-2">
                <h3 className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">
                  Crater Info
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  <StatCard label="Diameter" value={craterInfo.diameter_km} unit="km" icon="adjust" />
                  <StatCard label="Terraces" value={craterInfo.n_terraces} icon="stacked_line_chart" />
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge icon="terrain" text={craterInfo.dtm_source} />
                  {craterInfo.morphology && (
                    <Badge icon="category" text={craterInfo.morphology} />
                  )}
                  {summary && (
                    <Badge icon="satellite_alt" text={`${summary.n_sharad_tracks} SHARAD tracks`} />
                  )}
                </div>
              </div>
            )}

            {/* ── Epsilon Result ────────────────────────── */}
            {bestEst && (
              <div className="space-y-2">
                <h3 className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">
                  Dielectric Constant
                </h3>
                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${qc.bg} border ${qc.border}`}>
                  <span className={`material-symbols-outlined text-sm ${qc.text}`}>
                    {bestEst.quality === "good" ? "check_circle" : bestEst.quality === "marginal" ? "warning" : "error"}
                  </span>
                  <div className="flex-1 min-w-0">
                    <span className={`text-sm font-bold ${qc.text}`}>
                      εr = {bestEst.epsilon_r.toFixed(2)}
                    </span>
                    <span className="text-[10px] text-slate-500 ml-2">
                      [{bestEst.epsilon_low.toFixed(1)} – {bestEst.epsilon_high.toFixed(1)}]
                    </span>
                  </div>
                  <span className={`text-[10px] uppercase font-bold ${qc.text}`}>
                    {bestEst.quality}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {bestEst.interpretation}
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <StatCard label="True Depth" value={bestEst.depth_m} unit="m" small />
                  <StatCard label="TWT" value={bestEst.twt_us} unit="µs" small />
                </div>
              </div>
            )}

            {/* ── Radial Profile Chart ──────────────────── */}
            {chartData.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Radial Elevation Profile
                </h3>
                <div className="bg-[#101622] rounded-lg border border-slate-700/50 p-2">
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart
                      data={chartData}
                      margin={{ top: 8, right: 8, left: -10, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e2a40" />
                      <XAxis
                        dataKey="r"
                        stroke="#4a5568"
                        tick={{ fontSize: 9, fill: "#6b7c9c" }}
                        label={{
                          value: "Radius (m)",
                          position: "insideBottomRight",
                          offset: -2,
                          style: { fontSize: 9, fill: "#6b7c9c" },
                        }}
                      />
                      <YAxis
                        stroke="#4a5568"
                        tick={{ fontSize: 9, fill: "#6b7c9c" }}
                        label={{
                          value: "Elevation (m)",
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
                        labelFormatter={(v) => `${v} m from center`}
                        formatter={(value: number | undefined, name?: string) => {
                          if (name === "elev") return [`${(value ?? 0).toFixed(1)} m`, "Median"];
                          return [(value ?? 0).toFixed(1), name ?? ""];
                        }}
                      />
                      {/* Min/max envelope */}
                      <Area
                        type="monotone"
                        dataKey="elevMax"
                        stroke="none"
                        fill="#4f9cf7"
                        fillOpacity={0.08}
                        isAnimationActive={false}
                      />
                      <Area
                        type="monotone"
                        dataKey="elevMin"
                        stroke="none"
                        fill="#101622"
                        fillOpacity={1}
                        isAnimationActive={false}
                      />
                      {/* Median profile */}
                      <Area
                        type="monotone"
                        dataKey="elev"
                        stroke="#4f9cf7"
                        fill="url(#elevGrad)"
                        strokeWidth={1.5}
                        isAnimationActive={false}
                        dot={false}
                      />
                      {/* Crater radius marker */}
                      {craterInfo && craterInfo.diameter_km > 0 && (
                        <ReferenceLine
                          x={craterInfo.diameter_km * 500}
                          stroke="#f59e42"
                          strokeDasharray="4 4"
                          label={{
                            value: "Rim",
                            position: "top",
                            style: { fontSize: 9, fill: "#f59e42" },
                          }}
                        />
                      )}
                      <defs>
                        <linearGradient id="elevGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#4f9cf7" stopOpacity={0.3} />
                          <stop offset="100%" stopColor="#4f9cf7" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                    </AreaChart>
                  </ResponsiveContainer>
                  <div className="flex items-center justify-center gap-4 text-[9px] text-slate-500 mt-1">
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-0.5 bg-[#4f9cf7] rounded" />
                      Median elevation
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-0.5 bg-[#f59e42] rounded" style={{ borderTop: "1px dashed #f59e42" }} />
                      Crater rim
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* ── SHARAD Picks ──────────────────────────── */}
            {result.sharad_picks.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  SHARAD Subsurface Picks
                </h3>
                <div className="space-y-1.5">
                  {result.sharad_picks.map((sp) => (
                    <div
                      key={sp.product_id}
                      className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50"
                    >
                      <p className="text-[10px] font-mono text-white truncate">{sp.product_id}</p>
                      <div className="flex items-center gap-3 mt-1 text-[10px] text-slate-400">
                        <span>{sp.n_picks} picks</span>
                        <span>TWT: {sp.median_twt_us.toFixed(3)} µs</span>
                        <span>{sp.min_distance_km.toFixed(1)} km</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── All Epsilon Estimates ──────────────────── */}
            {result.epsilon_estimates.length > 1 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  All Estimates
                </h3>
                <div className="space-y-1.5">
                  {result.epsilon_estimates.map((est, i) => (
                    <EpsilonRow key={i} est={est} />
                  ))}
                </div>
              </div>
            )}

            {/* ── Parameters / Controls ─────────────────── */}
            <div>
              <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                Parameters
              </h3>
              <div className="space-y-3">
                <SliderParam
                  label="Search Buffer"
                  value={bufferKm}
                  min={5} max={100} step={5}
                  onChange={setBufferKm}
                  unit="km"
                />

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
                    <SliderParam
                      label="Min SNR"
                      value={minSnr}
                      min={1} max={20} step={0.5}
                      onChange={setMinSnr}
                    />
                    <SliderParam
                      label="Max Radius"
                      value={maxRadiusM}
                      min={500} max={20000} step={500}
                      onChange={setMaxRadiusM}
                      unit="m"
                    />
                  </div>
                )}

                {paramsChanged && (
                  <button
                    onClick={() => runAnalysis(bufferKm, minSnr, maxRadiusM)}
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
      {result?.success && result.epsilon_estimates.length > 0 && (
        <div className="border-t border-border-dark bg-bg-dark p-4">
          <button
            onClick={() => {
              const url = getStratigraphyCsvUrl(
                craterFeature.lat, craterFeature.lon,
                craterFeature.diameter_km ?? 0,
                bufferKm, 36, maxRadiusM, minSnr,
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
  small,
}: {
  label: string;
  value?: number | null;
  unit?: string;
  icon?: string;
  small?: boolean;
}) {
  return (
    <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
      <p className="text-slate-500 text-[10px] uppercase font-bold flex items-center gap-1">
        {icon && <span className="material-symbols-outlined text-[12px]">{icon}</span>}
        {label}
      </p>
      <p className={`text-white font-mono ${small ? "text-sm" : "text-base"}`}>
        {value != null ? (Number.isInteger(value) ? value : value.toFixed(1)) : "\u2014"}
        {unit && value != null && (
          <span className="text-[10px] text-slate-400 ml-1">{unit}</span>
        )}
      </p>
    </div>
  );
}

function Badge({ icon, text }: { icon: string; text: string }) {
  return (
    <span className="inline-flex items-center gap-1 bg-slate-800/60 border border-slate-700/50 px-2.5 py-1 rounded-full text-[10px] text-slate-300">
      <span className="material-symbols-outlined text-[12px]">{icon}</span>
      {text}
    </span>
  );
}

function EpsilonRow({ est }: { est: EpsilonEstimateInfo }) {
  const qc = QUALITY_COLORS[est.quality] ?? QUALITY_COLORS.unreliable!;
  return (
    <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
      <div className="flex items-center gap-2">
        <span className={`text-xs font-bold font-mono ${qc.text}`}>
          εr = {est.epsilon_r.toFixed(2)}
        </span>
        <span className="text-[10px] text-slate-500">
          [{est.epsilon_low.toFixed(1)} – {est.epsilon_high.toFixed(1)}]
        </span>
        <span className={`text-[9px] uppercase font-bold ml-auto px-1.5 py-0.5 rounded ${qc.bg} ${qc.text}`}>
          {est.quality}
        </span>
      </div>
      <p className="text-[10px] text-slate-500 mt-0.5">
        {est.depth_metric.replace(/_/g, " ")} | depth: {est.depth_m.toFixed(0)}m | TWT: {est.twt_us.toFixed(3)}µs
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
  unit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-[10px] text-slate-400">{label}</label>
        <span className="text-xs font-mono text-white">
          {value}{unit ? ` ${unit}` : ""}
        </span>
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
