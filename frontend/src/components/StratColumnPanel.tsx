import { useEffect, useState, useCallback, useRef } from "react";
import {
  fetchStratColumn,
  getStratColumnCsvUrl,
  type StratColumnResult,
  type ColumnLayer,
} from "../api/stratColumn";

/* =========================================================
 * DetectedFeature type (matches CraterDetectPanel)
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

/* =========================================================
 * Instrument icons
 * =======================================================*/
const INSTRUMENT_ICON: Record<string, string> = {
  HiRISE: "satellite_alt",
  MOLA: "terrain",
  SHARAD: "radar",
  CRISM: "science",
};

/* =========================================================
 * Props
 * =======================================================*/
export interface StratColumnPanelProps {
  craterFeature: DetectedFeature;
  onClose: () => void;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function StratColumnPanel({
  craterFeature,
  onClose,
}: StratColumnPanelProps) {
  const [result, setResult] = useState<StratColumnResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [bufferKm, setBufferKm] = useState(30);
  const [includeCrism, setIncludeCrism] = useState(true);
  const [includeSharad, setIncludeSharad] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [panelWidth, setPanelWidth] = useState(420);

  const lastRunRef = useRef({ bufferKm: 30, includeCrism: true, includeSharad: true });
  const paramsChanged =
    bufferKm !== lastRunRef.current.bufferKm ||
    includeCrism !== lastRunRef.current.includeCrism ||
    includeSharad !== lastRunRef.current.includeSharad;

  const { lat, lon, diameter_km } = craterFeature;

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
    async (buf: number, crism: boolean, sharad: boolean) => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchStratColumn(lat, lon, diameter_km || 0, buf, crism, sharad);
        setResult(data);
        lastRunRef.current = { bufferKm: buf, includeCrism: crism, includeSharad: sharad };
      } catch (e: any) {
        setError(e.message || "Analysis failed");
      } finally {
        setLoading(false);
      }
    },
    [lat, lon, diameter_km],
  );

  useEffect(() => {
    runAnalysis(bufferKm, includeCrism, includeSharad);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [craterFeature.id]);

  const summary = result?.summary;
  const layers = result?.layers ?? [];

  /* ── Column height scaling ──────────────────────────── */
  const maxDepth = Math.max(...layers.map((l) => l.depth_bottom_m), 1);

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
        <span className="material-symbols-outlined text-primary text-lg">view_column</span>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-bold text-white truncate">Stratigraphic Column</h2>
          <p className="text-[10px] text-slate-500 font-mono truncate">
            {lat.toFixed(3)}, {lon.toFixed(3)} | {diameter_km?.toFixed(1) ?? "?"} km
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
            <p className="text-xs text-slate-500">Building stratigraphic column...</p>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400 text-xs font-medium">{error}</p>
          </div>
        ) : result && summary ? (
          <>
            {/* ── Summary ─────────────────────────────── */}
            <div className="grid grid-cols-2 gap-2">
              <StatCard label="Layers" value={summary.n_layers} icon="layers" />
              <StatCard label="Total Depth" value={`${summary.total_depth_m} m`} icon="straighten" />
            </div>

            {/* Instrument badges */}
            <div className="flex items-center gap-2 flex-wrap">
              {summary.instruments_used.map((inst) => (
                <span
                  key={inst}
                  className="inline-flex items-center gap-1 bg-slate-800/60 border border-slate-700/50 px-2.5 py-1 rounded-full text-[10px] text-slate-300"
                >
                  <span className="material-symbols-outlined text-[12px]">
                    {INSTRUMENT_ICON[inst] || "devices"}
                  </span>
                  {inst}
                </span>
              ))}
              {summary.has_crism && (
                <span className="inline-flex items-center gap-1 bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-full text-[10px] text-amber-400">
                  CRISM minerals
                </span>
              )}
              {summary.has_sharad_subsurface && (
                <span className="inline-flex items-center gap-1 bg-cyan-500/10 border border-cyan-500/20 px-2.5 py-1 rounded-full text-[10px] text-cyan-400">
                  SHARAD subsurface
                </span>
              )}
            </div>

            {/* ── Vertical Column Visualization ────────── */}
            {layers.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Vertical Column
                </h3>
                <div className="bg-[#101622] rounded-lg border border-slate-700/50 p-3">
                  {/* Depth axis + column */}
                  <div className="flex gap-2">
                    {/* Depth labels */}
                    <div className="flex flex-col justify-between text-[9px] text-slate-600 font-mono w-10 flex-shrink-0">
                      <span>0 m</span>
                      <span>{(maxDepth / 2).toFixed(0)} m</span>
                      <span>{maxDepth.toFixed(0)} m</span>
                    </div>
                    {/* Column bars */}
                    <div className="flex-1 flex flex-col gap-0.5" style={{ minHeight: 200 }}>
                      {layers.map((layer) => (
                        <LayerBar
                          key={layer.layer_idx}
                          layer={layer}
                          maxDepth={maxDepth}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── Layer Details ────────────────────────── */}
            {layers.length > 0 && (
              <div>
                <h3 className="text-[10px] font-bold uppercase text-slate-500 mb-2 tracking-wider">
                  Layer Details ({layers.length})
                </h3>
                <div className="space-y-1.5 max-h-64 overflow-y-auto scrollbar-dark">
                  {layers.map((layer) => (
                    <LayerCard key={layer.layer_idx} layer={layer} />
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
                {/* Toggles */}
                <div className="flex gap-3">
                  <label className="flex items-center gap-2 text-[10px] text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeCrism}
                      onChange={(e) => setIncludeCrism(e.target.checked)}
                      className="accent-primary"
                    />
                    Include CRISM
                  </label>
                  <label className="flex items-center gap-2 text-[10px] text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeSharad}
                      onChange={(e) => setIncludeSharad(e.target.checked)}
                      className="accent-primary"
                    />
                    Include SHARAD
                  </label>
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
                        <label className="text-[10px] text-slate-400">Buffer (km)</label>
                        <span className="text-xs font-mono text-white">{bufferKm}</span>
                      </div>
                      <input
                        type="range"
                        min={5}
                        max={200}
                        step={5}
                        value={bufferKm}
                        onChange={(e) => setBufferKm(+e.target.value)}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                    </div>
                  </div>
                )}

                {paramsChanged && (
                  <button
                    onClick={() => runAnalysis(bufferKm, includeCrism, includeSharad)}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary/20 border border-primary/30 rounded-lg text-primary text-xs font-bold uppercase tracking-wider hover:bg-primary/30 transition-colors disabled:opacity-50"
                  >
                    <span className="material-symbols-outlined text-sm">refresh</span>
                    Re-build Column
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
              const url = getStratColumnCsvUrl(
                lat, lon, diameter_km || 0, bufferKm, includeCrism, includeSharad,
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
function LayerBar({ layer, maxDepth }: { layer: ColumnLayer; maxDepth: number }) {
  const heightPct = Math.max((layer.thickness_m / maxDepth) * 100, 4);
  const [r, g, b, a] = layer.color;
  const bgColor = `rgba(${r}, ${g}, ${b}, ${(a / 255) * 0.7})`;
  const borderColor = `rgba(${r}, ${g}, ${b}, ${(a / 255) * 0.9})`;

  return (
    <div
      className="relative rounded-sm flex items-center px-2 overflow-hidden"
      style={{
        height: `${heightPct}%`,
        minHeight: 20,
        backgroundColor: bgColor,
        borderLeft: `3px solid ${borderColor}`,
      }}
    >
      <div className="flex items-center gap-1.5 text-[9px] text-white/80 truncate">
        <span className="material-symbols-outlined text-[10px] opacity-60">
          {INSTRUMENT_ICON[layer.instrument] || "layers"}
        </span>
        <span className="font-medium truncate">
          {layer.mineral_name || layer.material_class || layer.source}
        </span>
        <span className="text-white/40 ml-auto flex-shrink-0">
          {layer.thickness_m.toFixed(0)}m
        </span>
      </div>
    </div>
  );
}

function LayerCard({ layer }: { layer: ColumnLayer }) {
  const [r, g, b, a] = layer.color;
  const dotColor = `rgba(${r}, ${g}, ${b}, ${a / 255})`;

  return (
    <div className="flex items-start gap-2 px-2.5 py-2 bg-slate-800/40 rounded-md border border-slate-700/50">
      <span className="w-3 h-3 rounded-sm flex-shrink-0 mt-0.5" style={{ backgroundColor: dotColor }} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-bold text-white truncate">
            {layer.mineral_name || layer.material_class || layer.source}
          </span>
          {layer.geochem_group && (
            <span className="text-[8px] text-slate-500 truncate">({layer.geochem_group})</span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[9px] text-slate-500 mt-0.5">
          <span>{layer.depth_top_m}–{layer.depth_bottom_m} m</span>
          <span className="text-slate-600">|</span>
          <span className="flex items-center gap-0.5">
            <span className="material-symbols-outlined text-[10px]">
              {INSTRUMENT_ICON[layer.instrument] || "layers"}
            </span>
            {layer.instrument}
          </span>
          {layer.epsilon_r != null && (
            <>
              <span className="text-slate-600">|</span>
              <span>εr={layer.epsilon_r}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string | number; icon?: string }) {
  return (
    <div className="bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/50">
      <p className="text-slate-500 text-[10px] uppercase font-bold flex items-center gap-1">
        {icon && <span className="material-symbols-outlined text-[12px]">{icon}</span>}
        {label}
      </p>
      <p className="text-white font-mono text-sm">{value}</p>
    </div>
  );
}
