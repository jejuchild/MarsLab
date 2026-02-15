import { useMemo } from "react";

/* =========================================================
 * Types
 * =======================================================*/
export interface PinnedSpectrum {
  id: string;
  productId: string;
  lat: number;
  lon: number;
  wavelengths: number[];
  reflectance: (number | null)[];
  color: string;
}

interface SpectralComparisonProps {
  spectra: PinnedSpectrum[];
  onRemove: (id: string) => void;
  onClear: () => void;
  onClose: () => void;
}

/* =========================================================
 * SpectralComparison Component
 * =======================================================*/
export default function SpectralComparison({
  spectra,
  onRemove,
  onClear,
  onClose,
}: SpectralComparisonProps) {
  // Chart dimensions
  const chartWidth = 460;
  const chartHeight = 240;
  const padding = { top: 15, right: 15, bottom: 35, left: 55 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Compute global axis bounds from all spectra
  const { wlMin, wlMax, rMin, rMax } = useMemo(() => {
    let wlMinVal = Infinity;
    let wlMaxVal = -Infinity;
    let rMinVal = Infinity;
    let rMaxVal = -Infinity;

    for (const s of spectra) {
      for (let i = 0; i < s.wavelengths.length; i++) {
        const r = s.reflectance[i];
        if (r === null) continue;
        const wl = s.wavelengths[i];
        if (wl < wlMinVal) wlMinVal = wl;
        if (wl > wlMaxVal) wlMaxVal = wl;
        if (r < rMinVal) rMinVal = r;
        if (r > rMaxVal) rMaxVal = r;
      }
    }

    // Fallback if no valid data
    if (!isFinite(wlMinVal)) {
      wlMinVal = 1.0;
      wlMaxVal = 3.5;
      rMinVal = 0;
      rMaxVal = 0.5;
    }

    // Add some padding to reflectance range
    const rRange = rMaxVal - rMinVal || 0.01;
    const yMin = Math.max(0, rMinVal - rRange * 0.08);
    const yMax = rMaxVal + rRange * 0.08;

    return { wlMin: wlMinVal, wlMax: wlMaxVal, rMin: yMin, rMax: yMax };
  }, [spectra]);

  // Scale functions
  const xScale = (wl: number) =>
    padding.left + ((wl - wlMin) / (wlMax - wlMin || 1)) * innerWidth;
  const yScale = (r: number) =>
    padding.top + innerHeight - ((r - rMin) / (rMax - rMin || 1)) * innerHeight;

  // Generate SVG path for a spectrum, skipping null values (creates gaps)
  const buildPath = (s: PinnedSpectrum): string => {
    const segments: string[] = [];
    let drawing = false;

    for (let i = 0; i < s.wavelengths.length; i++) {
      const r = s.reflectance[i];
      if (r === null) {
        drawing = false;
        continue;
      }
      const x = xScale(s.wavelengths[i]);
      const y = yScale(r);
      if (!drawing) {
        segments.push(`M ${x.toFixed(1)} ${y.toFixed(1)}`);
        drawing = true;
      } else {
        segments.push(`L ${x.toFixed(1)} ${y.toFixed(1)}`);
      }
    }

    return segments.join(" ");
  };

  // Generate Y-axis ticks (5 ticks)
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = (rMax - rMin) / 4;
    for (let i = 0; i <= 4; i++) {
      ticks.push(rMin + step * i);
    }
    return ticks;
  }, [rMin, rMax]);

  // Generate X-axis ticks (5 ticks)
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = (wlMax - wlMin) / 4;
    for (let i = 0; i <= 4; i++) {
      ticks.push(wlMin + step * i);
    }
    return ticks;
  }, [wlMin, wlMax]);

  return (
    <div
      className="fixed bottom-4 right-4 z-40 flex flex-col rounded-xl border border-[#232f48] bg-[#101622]/95 shadow-2xl backdrop-blur-sm"
      style={{ width: 500 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#232f48] px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-lg">
            compare
          </span>
          <h3 className="text-sm font-bold text-white">Spectral Comparison</h3>
          <span className="rounded-full bg-primary/20 px-2 py-0.5 text-[10px] font-bold text-primary">
            {spectra.length}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onClear}
            className="rounded px-2 py-1 text-[10px] font-medium text-red-400 hover:bg-red-500/10 transition-colors"
            title="Clear all pinned spectra"
          >
            Clear All
          </button>
          <button
            onClick={onClose}
            className="rounded p-1 text-slate-400 hover:text-white hover:bg-[#232f48] transition-colors"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* Chart Area */}
      <div className="px-4 pt-3 pb-1">
        <svg
          width={chartWidth}
          height={chartHeight}
          className="overflow-visible"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        >
          {/* Background */}
          <rect
            x={padding.left}
            y={padding.top}
            width={innerWidth}
            height={innerHeight}
            fill="#0a0f18"
            rx="2"
          />

          {/* Horizontal grid lines + Y-axis labels */}
          {yTicks.map((val, i) => {
            const y = yScale(val);
            return (
              <g key={`y-${i}`}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + innerWidth}
                  y2={y}
                  stroke="#1e293b"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />
                <text
                  x={padding.left - 6}
                  y={y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-slate-500"
                  fontSize="9"
                >
                  {val.toFixed(3)}
                </text>
              </g>
            );
          })}

          {/* Vertical grid lines + X-axis labels */}
          {xTicks.map((val, i) => {
            const x = xScale(val);
            return (
              <g key={`x-${i}`}>
                <line
                  x1={x}
                  y1={padding.top}
                  x2={x}
                  y2={padding.top + innerHeight}
                  stroke="#1e293b"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />
                <text
                  x={x}
                  y={chartHeight - 10}
                  textAnchor="middle"
                  className="fill-slate-500"
                  fontSize="9"
                >
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Axis labels */}
          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 1}
            textAnchor="middle"
            className="fill-slate-400"
            fontSize="10"
          >
            Wavelength (um)
          </text>
          <text
            x={12}
            y={padding.top + innerHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90, 12, ${padding.top + innerHeight / 2})`}
            className="fill-slate-400"
            fontSize="10"
          >
            I/F Reflectance
          </text>

          {/* Spectrum lines */}
          {spectra.map((s) => (
            <path
              key={s.id}
              d={buildPath(s)}
              fill="none"
              stroke={s.color}
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.9"
            />
          ))}

          {/* Axes border */}
          <rect
            x={padding.left}
            y={padding.top}
            width={innerWidth}
            height={innerHeight}
            fill="none"
            stroke="#334155"
            strokeWidth="1"
          />
        </svg>
      </div>

      {/* Legend */}
      <div className="border-t border-[#232f48] px-4 py-3 space-y-1.5 max-h-40 overflow-y-auto scrollbar-dark">
        {spectra.map((s) => (
          <div
            key={s.id}
            className="flex items-center gap-2 group"
          >
            {/* Color dot */}
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: s.color }}
            />
            {/* Product info */}
            <span className="text-[11px] font-mono text-white flex-1 truncate">
              {s.productId}
            </span>
            <span className="text-[10px] text-slate-500 flex-shrink-0">
              {s.lat.toFixed(2)}, {s.lon.toFixed(2)}
            </span>
            {/* Remove button */}
            <button
              onClick={() => onRemove(s.id)}
              className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
              title="Remove spectrum"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                close
              </span>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
