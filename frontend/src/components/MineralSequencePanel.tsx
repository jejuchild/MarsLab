import { useEffect, useState, useCallback, useRef } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  fetchMineralSequence,
  getMineralSequenceCsvUrl,
  type MineralSequenceResult,
} from "../api/mineralSequence";

/* =========================================================
 * Geochemical group colors
 * =======================================================*/
const GROUP_COLORS: Record<string, string> = {
  "Fe/Mg phyllosilicates": "#228b22",
  "Al phyllosilicates":    "#90ee90",
  "Sulfates":              "#ffd700",
  "Silica/Zeolite":        "#c8c8c8",
  "Ices":                  "#87cefa",
  "Fe oxides/hydroxides":  "#b22222",
  "Other hydrated":        "#ba55d3",
};

const GROUP_STYLES: Record<string, { bg: string; border: string; text: string }> = {
  "Fe/Mg phyllosilicates": { bg: "bg-green-700/20",  border: "border-green-600/30",  text: "text-green-400" },
  "Al phyllosilicates":    { bg: "bg-green-400/15",  border: "border-green-400/25",  text: "text-green-300" },
  "Sulfates":              { bg: "bg-yellow-500/15", border: "border-yellow-500/25", text: "text-yellow-400" },
  "Silica/Zeolite":        { bg: "bg-gray-400/15",  border: "border-gray-400/25",  text: "text-gray-300" },
  "Ices":                  { bg: "bg-sky-400/15",   border: "border-sky-400/25",   text: "text-sky-400" },
  "Fe oxides/hydroxides":  { bg: "bg-red-700/20",   border: "border-red-600/30",   text: "text-red-400" },
  "Other hydrated":        { bg: "bg-purple-500/15", border: "border-purple-500/25", text: "text-purple-400" },
};

const ENV_ICONS: Record<string, string> = {
  "Evaporite lake":        "water",
  "Acid leaching":         "science",
  "Deep alteration":       "layers",
  "Groundwater upwelling": "arrow_upward",
  "Ice-mineral contact":   "ac_unit",
};

/* =========================================================
 * Props
 * =======================================================*/
export interface MineralSequencePanelProps {
  obsId: string;
  onClose: () => void;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function MineralSequencePanel({
  obsId,
  onClose,
}: MineralSequencePanelProps) {
  const [result, setResult] = useState<MineralSequenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [direction, setDirection] = useState<"NS" | "EW">("NS");
  const [offset, setOffset] = useState(0.5);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [panelWidth, setPanelWidth] = useState(420);

  const lastRunRef = useRef({ direction: "NS", offset: 0.5 });
  const paramsChanged =
    direction !== lastRunRef.current.direction ||
    offset !== lastRunRef.current.offset;

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
    async (dir: string, off: number) => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchMineralSequence(obsId, dir, off);
        setResult(data);
        lastRunRef.current = { direction: dir, offset: off };
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Analysis failed";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [obsId],
  );

  useEffect(() => {
    runAnalysis(direction, offset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [obsId]);

  const summary = result?.summary;

  /* ── Histogram chart data ───────────────────────────── */
  const histogramData = result
    ? Object.entries(result.group_histogram)
        .sort(([, a], [, b]) => b - a)
        .map(([group, count]) => ({ group, count, color: GROUP_COLORS[group] || "#6b7c9c" }))
    : [];

  /* ── Classification rate badge ──────────────────────── */
  const classRate = summary ? summary.classification_rate * 100 : 0;
  const classColor =
    classRate >= 40
      ? { bg: "bg-green-500/10", border: "border-green-500/20", text: "text-green-400" }
      : classRate >= 15
        ? { bg: "bg-yellow-500/10", border: "border-yellow-500/20", text: "text-yellow-400" }
        : { bg: "bg-red-500/10", border: "border-red-500/20", text: "text-red-400" };

  return (
    <aside
      className="relative flex h-full flex-col border-l border-border-dark bg-surface-dark/40"
      style={{ width: panelWidth }}
    >
      <div
        onMouseDown={handleResizeStart}
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 transition-colors"
      />

      {/* ── Header ────────────────────────────────────── */}
      <div className="flex items-center gap-2 p-4 border-b border-border-dark">
        <span className="material-symbols-outlined text-primary text-lg">science</span>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-bold text-white truncate">Mineral Sequence</h2>
          <p className="text-[10px] text-slate-500 font-mono truncate">{obsId}</p>
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
            <p className="text-xs text-slate-500">Analyzing mineral sequence...</p>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400 text-xs font-medium">{error}</p>
          </div>
        ) : result && summary ? (
          <>
            {/* ── Classification Rate ─────────────────── */}
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-lg ${classColor.bg} border ${classColor.border}`}
            >
              <span className={`material-symbols-outlined text-sm ${classColor.text}`}>
                {classRate >= 40 ? "check_circle" : classRate >= 15 ? "warning" : "error"}
              </span>
              <span className={`text-[11px] font-bold ${classColor.text}`}>
                {classRate.toFixed(1)}% Classified
              </span>
              <span className="text-[10px] text-slate-500 ml-auto">
                {summary.classified_points} / {summary.total_transect_points} points
              </span>
            </div>

            {/* ── Matched Environments ────────────────── */}
            {result.sequence_matches.length > 0 ? (
              <div className="space-y-2">
                <h3 className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">
                  Paleo-Environments
                </h3>
                {result.sequence_matches.map((match) => (
                  <div
                    key={match.environment}
                    className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20"
                  >
                    <span className="material-symbols-outlined text-base text-amber-400">
                      {ENV_ICONS[match.environment] || "public"}
                    </span>
                    <div className="flex-1">
                      <p className="text-[11px] font-bold text-amber-400">{match.environment}</p>
                      <p className="text-[9px] text-slate-500">
                        Confidence: {(match.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="px-3 py-2 rounded-lg bg-slate-800/40 border border-slate-700/50">
                <p className="text-[10px] text-slate-500">No canonical paleo-environment matched</p>
              </div>
            )}

            {/* ── Summary Stats ───────────────────────── */}
            <div className="grid grid-cols-2 gap-2">
              <StatCard
                label="Transitions"
                value={summary.n_transitions}
                icon="swap_horiz"
              />
              <StatCard
                label="Groups Present"
                value={summary.n_groups_present}
                icon="category"
              />
            </div>

            {/* Dominant group badge */}
            {summary.dominant_group && (
              <div className="flex items-center gap-2 flex-wrap">
                {(() => {
                  const style = GROUP_STYLES[summary.dominant_group!];
                  return (
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-[10px] font-bold ${style?.bg ?? "bg-slate-800/40"} border ${style?.border ?? "border-slate-700/50"} ${style?.text ?? "text-slate-400"}`}>
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: GROUP_COLORS[summary.dominant_group!] || "#6b7c9c" }}
                      />
                      {summary.dominant_group}
                    </span>
                  );
                })()}
                {summary.mean_confidence != null && (
                  <span className="inline-flex items-center gap-1 bg-sky-500/10 border border-sky-500/20 px-2.5 py-1 rounded-full text-[10px] text-sky-400">
                    <span className="material-symbols-outlined text-[12px]">verified</span>
                    Conf: {(summary.mean_confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            )}

            {/* ── Group Histogram ─────────────────────── */}
            {histogramData.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Group Distribution
                </h3>
                <div className="bg-[#101622] rounded-lg border border-slate-700/50 p-2">
                  <ResponsiveContainer width="100%" height={Math.max(100, histogramData.length * 30 + 20)}>
                    <BarChart
                      data={histogramData}
                      layout="vertical"
                      margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e2a40" />
                      <XAxis
                        type="number"
                        stroke="#4a5568"
                        tick={{ fontSize: 9, fill: "#6b7c9c" }}
                      />
                      <YAxis
                        type="category"
                        dataKey="group"
                        width={120}
                        stroke="#4a5568"
                        tick={{ fontSize: 8, fill: "#6b7c9c" }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#101622",
                          border: "1px solid #232f48",
                          fontSize: 11,
                          borderRadius: 6,
                        }}
                      />
                      <Bar dataKey="count" radius={[0, 3, 3, 0]}>
                        {histogramData.map((entry, idx) => (
                          <Cell key={idx} fill={entry.color} fillOpacity={0.7} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* ── Transitions List ────────────────────── */}
            {result.transitions.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Transitions ({result.transitions.length})
                </h3>
                <div className="space-y-1.5 max-h-48 overflow-y-auto scrollbar-dark">
                  {result.transitions.map((tr, idx) => (
                    <div
                      key={idx}
                      className="flex items-center gap-2 px-2.5 py-1.5 bg-slate-800/40 rounded-md border border-slate-700/50 text-[10px]"
                    >
                      <span className="text-slate-500 font-mono">#{tr.position_idx}</span>
                      <span
                        className="px-1.5 py-0.5 rounded text-[9px] font-bold"
                        style={{
                          backgroundColor: `${GROUP_COLORS[tr.from_group] || "#6b7c9c"}30`,
                          color: GROUP_COLORS[tr.from_group] || "#6b7c9c",
                        }}
                      >
                        {tr.from_mineral}
                      </span>
                      <span className="material-symbols-outlined text-slate-600 text-[12px]">arrow_forward</span>
                      <span
                        className="px-1.5 py-0.5 rounded text-[9px] font-bold"
                        style={{
                          backgroundColor: `${GROUP_COLORS[tr.to_group] || "#6b7c9c"}30`,
                          color: GROUP_COLORS[tr.to_group] || "#6b7c9c",
                        }}
                      >
                        {tr.to_mineral}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Parameters / Controls ───────────────── */}
            <div>
              <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                Parameters
              </h3>
              <div className="space-y-3">
                {/* Direction toggle */}
                <div>
                  <label className="text-[10px] text-slate-400 mb-1 block">Transect Direction</label>
                  <div className="flex gap-2">
                    {(["NS", "EW"] as const).map((d) => (
                      <button
                        key={d}
                        onClick={() => setDirection(d)}
                        className={`flex-1 py-1.5 rounded text-xs font-bold transition-colors ${
                          direction === d
                            ? "bg-primary/20 border border-primary/40 text-primary"
                            : "bg-slate-800/40 border border-slate-700/50 text-slate-500 hover:text-slate-300"
                        }`}
                      >
                        {d === "NS" ? "N \u2194 S" : "E \u2194 W"}
                      </button>
                    ))}
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
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="text-[10px] text-slate-400">Transect Offset</label>
                        <span className="text-xs font-mono text-white">{offset.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={offset}
                        onChange={(e) => setOffset(+e.target.value)}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                        <span>0.0 (edge)</span>
                        <span>1.0 (edge)</span>
                      </div>
                    </div>
                  </div>
                )}

                {paramsChanged && (
                  <button
                    onClick={() => runAnalysis(direction, offset)}
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
              const url = getMineralSequenceCsvUrl(obsId, direction, offset);
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
  icon,
}: {
  label: string;
  value?: number | null;
  icon?: string;
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
        {value != null ? value : "\u2014"}
      </p>
    </div>
  );
}
