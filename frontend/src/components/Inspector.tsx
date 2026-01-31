import { useEffect, useState } from "react";
import { fetchHiRISEWindow } from "../api/hirise";

/* =========================================================
 * Types
 * =======================================================*/
export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD";

export type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  // CRISM pixel coordinates for spectrum (optional)
  pixelLine?: number;
  pixelSample?: number;
  // HiRISE observation title (e.g., "Gullies in Arcadia Region")
  title?: string;
};

export type RGBWavelengths = {
  r: number;
  g: number;
  b: number;
};

type HiRISETabKey = "Metadata" | "Pixel";
type CRISMTabKey = "Metadata" | "Spectrum" | "Bands";

type SpectrumData = {
  wavelengths: number[];
  reflectance: (number | null)[];
  validBands: number;
};

type WindowStats = {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  sum: number;
  histogram: number[];
  binEdges: number[];
};

// Default CRISM wavelengths (in micrometers)
const DEFAULT_RGB: RGBWavelengths = {
  r: 2.53,  // Red channel
  g: 1.51,  // Green channel
  b: 1.08,  // Blue channel
};

/* =========================================================
 * Inspector Component
 * =======================================================*/
export default function Inspector({
  selected,
  onClose,
  isHighResActive = false,
  onToggleHighRes,
  rgbWavelengths,
  onRGBChange,
  hasHighResData = true,
}: {
  selected: InspectorContext | null;
  onClose: () => void;
  isHighResActive?: boolean;
  onToggleHighRes?: () => void;
  rgbWavelengths?: RGBWavelengths;
  onRGBChange?: (rgb: RGBWavelengths) => void;
  hasHighResData?: boolean;  // Whether high-res data (.img/.tif) is available
}) {
  const [hiriseTab, setHiriseTab] = useState<HiRISETabKey>("Metadata");
  const [crismTab, setCrismTab] = useState<CRISMTabKey>("Metadata");
  const [windowSize, setWindowSize] = useState(5);
  const [stats, setStats] = useState<WindowStats | null>(null);
  const [loading, setLoading] = useState(false);

  // CRISM spectrum state
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [spectrumLoading, setSpectrumLoading] = useState(false);

  // Local RGB state for sliders
  const [localRGB, setLocalRGB] = useState<RGBWavelengths>(rgbWavelengths || DEFAULT_RGB);

  // Sync local state with props
  useEffect(() => {
    if (rgbWavelengths) {
      setLocalRGB(rgbWavelengths);
    }
  }, [rgbWavelengths]);

  // Fetch HiRISE pixel stats when selection or window size changes
  useEffect(() => {
    if (!selected || selected.instrument !== "HIRISE") {
      setStats(null);
      return;
    }

    // Need pixel coordinates to fetch stats
    if (selected.pixelLine === undefined || selected.pixelSample === undefined) {
      setStats(null);
      return;
    }

    const controller = new AbortController();

    async function fetchStats() {
      setLoading(true);
      try {
        const halfSize = Math.floor(windowSize / 2);
        // Use pixel coordinates (sample = x, line = y)
        const data = await fetchHiRISEWindow(
          selected.productId,
          selected.pixelSample!,  // x = sample (column)
          selected.pixelLine!,    // y = line (row)
          halfSize
        );

        const dn = data.dn.flat() as number[];
        const mean = dn.reduce((a, b) => a + b, 0) / dn.length;
        const sorted = [...dn].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const std = Math.sqrt(
          dn.reduce((acc, v) => acc + (v - mean) ** 2, 0) / dn.length
        );
        const min = Math.min(...dn);
        const max = Math.max(...dn);
        const sum = dn.reduce((a, b) => a + b, 0);

        const bins = 10;
        const binWidth = (max - min) / bins || 1;
        const histogram = new Array(bins).fill(0);
        const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + i * binWidth);

        dn.forEach((v) => {
          const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
          histogram[idx]++;
        });

        setStats({ mean, median, std, min, max, sum, histogram, binEdges });
      } catch (e) {
        console.error("Failed to fetch stats:", e);
        setStats(null);
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
    return () => controller.abort();
  }, [selected?.productId, selected?.pixelLine, selected?.pixelSample, windowSize]);

  // Fetch CRISM spectrum when pixel coordinates change
  useEffect(() => {
    if (!selected || selected.instrument !== "CRISM") {
      setSpectrumData(null);
      return;
    }

    if (selected.pixelLine === undefined || selected.pixelSample === undefined) {
      setSpectrumData(null);
      return;
    }

    async function fetchSpectrum() {
      setSpectrumLoading(true);
      try {
        const response = await fetch(`/crism/${selected.productId}/spectrum`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            line: selected.pixelLine,
            sample: selected.pixelSample,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch spectrum: ${response.status}`);
        }

        const data = await response.json();
        setSpectrumData({
          wavelengths: data.wavelengths,
          reflectance: data.reflectance,
          validBands: data.valid_bands,
        });
      } catch (e) {
        console.error("Failed to fetch spectrum:", e);
        setSpectrumData(null);
      } finally {
        setSpectrumLoading(false);
      }
    }

    fetchSpectrum();
  }, [selected?.productId, selected?.pixelLine, selected?.pixelSample]);

  if (!selected) return null;

  const isHiRISE = selected.instrument === "HIRISE";
  const isCRISM = selected.instrument === "CRISM";

  const handleRGBSliderChange = (channel: "r" | "g" | "b", value: number) => {
    const newRGB = { ...localRGB, [channel]: value };
    setLocalRGB(newRGB);
  };

  const handleApplyRGB = () => {
    onRGBChange?.(localRGB);
  };

  // Define tabs based on instrument
  const hiriseTabs: HiRISETabKey[] = ["Metadata", "Pixel"];
  const crismTabs: CRISMTabKey[] = ["Metadata", "Spectrum", "Bands"];

  return (
    <aside className="flex h-full w-96 flex-col border-l border-border-dark bg-surface-dark/40">
      {/* Tabs */}
      <div className="flex border-b border-border-dark">
        {isHiRISE &&
          hiriseTabs.map((t) => (
            <button
              key={t}
              onClick={() => setHiriseTab(t)}
              className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                hiriseTab === t
                  ? "border-b-2 border-primary bg-primary/5 text-white"
                  : "border-b-2 border-transparent text-slate-500 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}

        {isCRISM &&
          crismTabs.map((t) => (
            <button
              key={t}
              onClick={() => setCrismTab(t)}
              className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                crismTab === t
                  ? "border-b-2 border-primary bg-primary/5 text-white"
                  : "border-b-2 border-transparent text-slate-500 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}

        <button
          onClick={onClose}
          className="flex items-center justify-center px-3 text-slate-500 hover:text-red-400 transition-colors"
        >
          <span className="material-symbols-outlined text-lg">close</span>
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-dark">
        {/* HiRISE Tabs */}
        {isHiRISE && hiriseTab === "Metadata" && <MetadataTab selected={selected} />}

        {isHiRISE && hiriseTab === "Pixel" && (
          <PixelTab
            selected={selected}
            stats={stats}
            loading={loading}
            windowSize={windowSize}
            onWindowSizeChange={setWindowSize}
          />
        )}

        {/* CRISM Tabs */}
        {isCRISM && crismTab === "Metadata" && <MetadataTab selected={selected} />}

        {isCRISM && crismTab === "Spectrum" && (
          <SpectrumTab
            selected={selected}
            spectrumData={spectrumData}
            loading={spectrumLoading}
          />
        )}

        {isCRISM && crismTab === "Bands" && (
          <BandsTab
            rgb={localRGB}
            onChange={handleRGBSliderChange}
            onApply={handleApplyRGB}
            isOverlayActive={isHighResActive}
          />
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border-dark bg-bg-dark p-4 space-y-2">
        {onToggleHighRes && (
          <button
            onClick={hasHighResData ? onToggleHighRes : undefined}
            disabled={!hasHighResData}
            className={`flex w-full items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-bold uppercase tracking-widest transition-colors ${
              !hasHighResData
                ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                : isHighResActive
                ? "bg-green-600 text-white hover:bg-green-700"
                : "bg-surface-dark text-slate-300 hover:bg-surface-dark/80"
            }`}
            title={!hasHighResData ? "No high-res data available (browse only)" : undefined}
          >
            <span className="material-symbols-outlined text-sm">
              {!hasHighResData ? "block" : isHighResActive ? "check_circle" : isCRISM ? "palette" : "hd"}
            </span>
            {!hasHighResData
              ? "No High-Res Data (Browse Only)"
              : isHighResActive
              ? isCRISM
                ? "RGB Overlay Active"
                : "High-Res Overlay Active"
              : isCRISM
              ? "Overlay RGB on Map"
              : "Overlay High-Resolution on Map"}
          </button>
        )}
        <button className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary py-3 text-xs font-bold uppercase tracking-widest text-white shadow-lg shadow-primary/20 hover:brightness-110 active:scale-[0.98] transition-all">
          <span className="material-symbols-outlined text-sm">ios_share</span>
          Export Statistics
        </button>
      </div>
    </aside>
  );
}

/* =========================================================
 * Bands Tab (CRISM RGB Selection)
 * =======================================================*/
function BandsTab({
  rgb,
  onChange,
  onApply,
  isOverlayActive,
}: {
  rgb: RGBWavelengths;
  onChange: (channel: "r" | "g" | "b", value: number) => void;
  onApply: () => void;
  isOverlayActive: boolean;
}) {
  // CRISM wavelength range (approximately 1.0 - 4.0 micrometers for VNIR+IR)
  const minWavelength = 1.0;
  const maxWavelength = 4.0;
  const step = 0.01;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">palette</span>
        <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
          RGB Band Selection
        </h3>
      </div>

      <p className="text-[11px] text-slate-400">
        Select wavelengths (in micrometers) to create an RGB composite from CRISM spectral data.
      </p>

      {/* Red Channel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="inline-block w-3 h-3 rounded-full bg-red-500"></span>
            Red Channel
          </label>
          <span className="font-mono text-xs text-red-400">{rgb.r.toFixed(2)} μm</span>
        </div>
        <input
          type="range"
          min={minWavelength}
          max={maxWavelength}
          step={step}
          value={rgb.r}
          onChange={(e) => onChange("r", parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-red-500 bg-slate-700"
        />
        <div className="flex justify-between text-[9px] text-slate-600">
          <span>{minWavelength}</span>
          <span>{maxWavelength}</span>
        </div>
      </div>

      {/* Green Channel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="inline-block w-3 h-3 rounded-full bg-green-500"></span>
            Green Channel
          </label>
          <span className="font-mono text-xs text-green-400">{rgb.g.toFixed(2)} μm</span>
        </div>
        <input
          type="range"
          min={minWavelength}
          max={maxWavelength}
          step={step}
          value={rgb.g}
          onChange={(e) => onChange("g", parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-green-500 bg-slate-700"
        />
        <div className="flex justify-between text-[9px] text-slate-600">
          <span>{minWavelength}</span>
          <span>{maxWavelength}</span>
        </div>
      </div>

      {/* Blue Channel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="inline-block w-3 h-3 rounded-full bg-blue-500"></span>
            Blue Channel
          </label>
          <span className="font-mono text-xs text-blue-400">{rgb.b.toFixed(2)} μm</span>
        </div>
        <input
          type="range"
          min={minWavelength}
          max={maxWavelength}
          step={step}
          value={rgb.b}
          onChange={(e) => onChange("b", parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500 bg-slate-700"
        />
        <div className="flex justify-between text-[9px] text-slate-600">
          <span>{minWavelength}</span>
          <span>{maxWavelength}</span>
        </div>
      </div>

      {/* Presets */}
      <div className="space-y-2">
        <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
          Presets
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => {
              onChange("r", 2.53);
              onChange("g", 1.51);
              onChange("b", 1.08);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            True Color
          </button>
          <button
            onClick={() => {
              onChange("r", 2.3);
              onChange("g", 1.93);
              onChange("b", 1.08);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            Mineralogy
          </button>
          <button
            onClick={() => {
              onChange("r", 1.08);
              onChange("g", 1.51);
              onChange("b", 2.53);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            Inverted
          </button>
          <button
            onClick={() => {
              onChange("r", 2.53);
              onChange("g", 1.93);
              onChange("b", 1.51);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            Hydration
          </button>
        </div>
      </div>

      {/* Apply Button */}
      {isOverlayActive && (
        <button
          onClick={onApply}
          className="w-full py-2.5 rounded-lg bg-primary/20 border border-primary/30 text-primary text-xs font-bold uppercase tracking-widest hover:bg-primary/30 transition-colors"
        >
          <span className="material-symbols-outlined text-sm align-middle mr-1">refresh</span>
          Apply RGB Changes
        </button>
      )}
    </div>
  );
}

/* =========================================================
 * Spectrum Tab (CRISM)
 * =======================================================*/
function SpectrumTab({
  selected,
  spectrumData,
  loading,
}: {
  selected: InspectorContext;
  spectrumData: SpectrumData | null;
  loading: boolean;
}) {
  const hasPixel = selected.pixelLine !== undefined && selected.pixelSample !== undefined;

  if (!hasPixel) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">show_chart</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
            Pixel Spectrum
          </h3>
        </div>

        <div className="flex flex-col items-center justify-center h-64 text-center">
          <span className="material-symbols-outlined text-4xl text-slate-600 mb-3">
            touch_app
          </span>
          <p className="text-sm text-slate-400 mb-2">
            Click on the CRISM overlay to select a pixel
          </p>
          <p className="text-[11px] text-slate-500">
            The spectral profile will be displayed here
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <span className="material-symbols-outlined animate-spin text-2xl text-primary">
          progress_activity
        </span>
      </div>
    );
  }

  if (!spectrumData) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500 text-sm">
        Failed to load spectrum data
      </div>
    );
  }

  // Filter valid data points for the chart
  const validPoints = spectrumData.wavelengths
    .map((wl, i) => ({
      wavelength: wl,
      reflectance: spectrumData.reflectance[i],
    }))
    .filter((p) => p.reflectance !== null) as { wavelength: number; reflectance: number }[];

  // Calculate stats
  const reflValues = validPoints.map((p) => p.reflectance);
  const minRefl = Math.min(...reflValues);
  const maxRefl = Math.max(...reflValues);
  const meanRefl = reflValues.reduce((a, b) => a + b, 0) / reflValues.length;

  // Chart dimensions
  const chartWidth = 340;
  const chartHeight = 180;
  const padding = { top: 10, right: 10, bottom: 25, left: 45 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Scales
  const wlMin = Math.min(...validPoints.map((p) => p.wavelength));
  const wlMax = Math.max(...validPoints.map((p) => p.wavelength));
  const yMin = Math.max(0, minRefl - (maxRefl - minRefl) * 0.1);
  const yMax = maxRefl + (maxRefl - minRefl) * 0.1;

  const xScale = (wl: number) => padding.left + ((wl - wlMin) / (wlMax - wlMin)) * innerWidth;
  const yScale = (r: number) => padding.top + innerHeight - ((r - yMin) / (yMax - yMin)) * innerHeight;

  // Generate path
  const pathD = validPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.wavelength)} ${yScale(p.reflectance)}`)
    .join(" ");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">show_chart</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
            Pixel Spectrum
          </h3>
        </div>
        <span className="text-[10px] text-slate-500">
          {spectrumData.validBands} bands
        </span>
      </div>

      {/* Pixel coordinates */}
      <div className="flex gap-4 text-[11px]">
        <div>
          <span className="text-slate-500">Line: </span>
          <span className="font-mono text-white">{selected.pixelLine}</span>
        </div>
        <div>
          <span className="text-slate-500">Sample: </span>
          <span className="font-mono text-white">{selected.pixelSample}</span>
        </div>
      </div>

      {/* Spectrum Chart */}
      <div className="rounded-lg border border-border-dark bg-bg-dark/60 p-3">
        <svg width={chartWidth} height={chartHeight} className="overflow-visible">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padding.top + innerHeight * (1 - t);
            const val = yMin + (yMax - yMin) * t;
            return (
              <g key={t}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + innerWidth}
                  y2={y}
                  stroke="#334155"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
                <text
                  x={padding.left - 5}
                  y={y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-slate-500 text-[8px]"
                >
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* X axis labels */}
          {[wlMin, (wlMin + wlMax) / 2, wlMax].map((wl, i) => (
            <text
              key={i}
              x={xScale(wl)}
              y={chartHeight - 5}
              textAnchor="middle"
              className="fill-slate-500 text-[8px]"
            >
              {wl.toFixed(1)}
            </text>
          ))}

          {/* Axis labels */}
          <text
            x={chartWidth / 2}
            y={chartHeight}
            textAnchor="middle"
            className="fill-slate-400 text-[9px]"
          >
            Wavelength (μm)
          </text>
          <text
            x={10}
            y={chartHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90, 10, ${chartHeight / 2})`}
            className="fill-slate-400 text-[9px]"
          >
            I/F
          </text>

          {/* Spectrum line */}
          <path d={pathD} fill="none" stroke="#3b82f6" strokeWidth="1.5" />
        </svg>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
          <div className="text-[9px] uppercase text-slate-500">Min</div>
          <div className="font-mono text-xs text-white">{minRefl.toFixed(4)}</div>
        </div>
        <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
          <div className="text-[9px] uppercase text-slate-500">Max</div>
          <div className="font-mono text-xs text-white">{maxRefl.toFixed(4)}</div>
        </div>
        <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
          <div className="text-[9px] uppercase text-slate-500">Mean</div>
          <div className="font-mono text-xs text-white">{meanRefl.toFixed(4)}</div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
 * Metadata Tab
 * =======================================================*/
function MetadataTab({ selected }: { selected: InspectorContext }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">
          {selected.instrument === "CRISM" ? "spectrum" : "satellite_alt"}
        </span>
        <span className="text-sm font-bold">{selected.instrument}</span>
      </div>

      {/* Title (if available) */}
      {selected.title && (
        <div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
          <span className="text-[10px] uppercase text-primary/70 block mb-1">Title</span>
          <span className="text-sm text-white font-medium">{selected.title}</span>
        </div>
      )}

      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Product ID</span>
          <span className="font-mono text-xs">{selected.productId}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Longitude</span>
          <span className="font-mono text-xs">{selected.lon.toFixed(6)}°</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Latitude</span>
          <span className="font-mono text-xs">{selected.lat.toFixed(6)}°</span>
        </div>
      </div>

      <div>
        <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-primary">
          Quickview
        </h4>
        <div className="overflow-hidden rounded-lg border border-border-dark">
          <img
            src={
              selected.instrument === "HIRISE"
                ? `/hirise/quickview/${selected.productId}.jpg`
                : selected.productId.includes("_brcarj_")
                  ? `/crism/quickview/${selected.productId.split("_")[0]}_VNIR.png`
                  : `/crism/quickview/${selected.productId.replace(/_if[0-9a-z]+_mtr3$/i, "_brvnaj_mtr3.png")}`
            }
            className="w-full"
            alt="Quickview"
          />
        </div>
      </div>
    </div>
  );
}

/* =========================================================
 * Pixel Tab (HiRISE Analysis)
 * =======================================================*/
function PixelTab({
  selected,
  stats,
  loading,
  windowSize,
  onWindowSizeChange,
}: {
  selected: InspectorContext;
  stats: WindowStats | null;
  loading: boolean;
  windowSize: number;
  onWindowSizeChange: (size: number) => void;
}) {
  const hasPixel = selected.pixelLine !== undefined && selected.pixelSample !== undefined;

  if (!hasPixel) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">analytics</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
            Pixel Statistics
          </h3>
        </div>

        <div className="flex flex-col items-center justify-center h-64 text-center">
          <span className="material-symbols-outlined text-4xl text-slate-600 mb-3">
            touch_app
          </span>
          <p className="text-sm text-slate-400 mb-2">
            Click on the HiRISE overlay to select a pixel
          </p>
          <p className="text-[11px] text-slate-500">
            Enable the high-resolution overlay first, then click on it
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-primary">
          <span className="material-symbols-outlined text-sm">analytics</span>
          Neighborhood Statistics
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-[9px] uppercase text-slate-500">Window</span>
          <select
            value={windowSize}
            onChange={(e) => onWindowSizeChange(Number(e.target.value))}
            className="rounded border-border-dark bg-bg-dark py-0.5 pl-2 pr-6 text-[10px] focus:border-primary focus:ring-primary"
          >
            <option value={3}>3×3</option>
            <option value={5}>5×5</option>
            <option value={7}>7×7</option>
            <option value={11}>11×11</option>
            <option value={21}>21×21</option>
          </select>
        </div>
      </div>

      {/* Pixel coordinates */}
      <div className="flex gap-4 text-[11px]">
        <div>
          <span className="text-slate-500">Line: </span>
          <span className="font-mono text-white">{selected.pixelLine}</span>
        </div>
        <div>
          <span className="text-slate-500">Sample: </span>
          <span className="font-mono text-white">{selected.pixelSample}</span>
        </div>
      </div>

      {loading ? (
        <div className="flex h-40 items-center justify-center text-slate-500">
          <span className="material-symbols-outlined animate-spin">progress_activity</span>
        </div>
      ) : stats ? (
        <>
          <Histogram histogram={stats.histogram} binEdges={stats.binEdges} />

          <div className="grid grid-cols-2 gap-2">
            <StatCard label="Mean" value={stats.mean.toFixed(2)} />
            <StatCard label="Median" value={stats.median.toFixed(2)} />
            <StatCard label="Std Dev" value={`±${stats.std.toFixed(1)}`} highlight />
            <StatCard label="Sum" value={stats.sum.toLocaleString()} />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center justify-between rounded border border-border-dark/30 bg-bg-dark/20 p-2">
              <span className="text-[9px] uppercase text-slate-500">Min</span>
              <span className="font-mono text-xs text-slate-300">{stats.min.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between rounded border border-border-dark/30 bg-bg-dark/20 p-2">
              <span className="text-[9px] uppercase text-slate-500">Max</span>
              <span className="font-mono text-xs text-slate-300">{stats.max.toLocaleString()}</span>
            </div>
          </div>
        </>
      ) : (
        <div className="flex h-40 items-center justify-center text-slate-500 text-sm">
          Failed to load pixel data
        </div>
      )}
    </div>
  );
}

/* =========================================================
 * Histogram Component
 * =======================================================*/
function Histogram({
  histogram,
  binEdges,
}: {
  histogram: number[];
  binEdges: number[];
}) {
  const maxCount = Math.max(...histogram);

  return (
    <div className="rounded-lg border border-border-dark bg-bg-dark/60 p-3">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-[10px] font-bold uppercase tracking-tight text-slate-500">
          DN Distribution (Sampled Area)
        </span>
        <span className="font-mono text-[9px] text-slate-500">
          n={histogram.reduce((a, b) => a + b, 0)}
        </span>
      </div>

      <div className="flex h-32 items-end gap-1 px-1">
        {histogram.map((count, i) => {
          const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
          const opacity = 0.4 + (height / 100) * 0.6;
          return (
            <div
              key={i}
              className="flex-1 rounded-t-sm bg-gradient-to-t from-primary to-blue-400"
              style={{ height: `${height}%`, opacity }}
            />
          );
        })}
      </div>

      <div className="mt-2 flex justify-between font-mono text-[8px] text-slate-600">
        <span>{Math.round(binEdges[0])}</span>
        <span>{Math.round(binEdges[Math.floor(binEdges.length / 4)])}</span>
        <span>{Math.round(binEdges[Math.floor(binEdges.length / 2)])}</span>
        <span>{Math.round(binEdges[Math.floor((binEdges.length * 3) / 4)])}</span>
        <span>{Math.round(binEdges[binEdges.length - 1])}</span>
      </div>
    </div>
  );
}

/* =========================================================
 * Stat Card Component
 * =======================================================*/
function StatCard({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2.5">
      <div className="mb-0.5 text-[9px] uppercase text-slate-500">{label}</div>
      <div
        className={`font-mono text-sm font-bold ${
          highlight ? "text-blue-400" : "text-white"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
