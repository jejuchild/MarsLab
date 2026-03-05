import { useState } from "react";
import type { InspectorContext, SpectrumData, DustAssessment } from "../types";
import type { OverlayType } from "../../../pages/MainPage";
import BandRatioCalculator from "../../BandRatioCalculator";

type CRISMSpectrumTabProps = {
  selected: InspectorContext;
  spectrumData: SpectrumData | null;
  dustAssessment: DustAssessment | null;
  loading: boolean;
  onPinSpectrum?: (spectrum: {
    productId: string;
    lat: number;
    lon: number;
    wavelengths: number[];
    reflectance: (number | null)[];
  }) => void;
  // UX improvement: direct CTA to enable overlay
  activeOverlayType: string | null;
  onSetOverlay: (type: OverlayType | null) => void;
};

export default function CRISMSpectrumTab({
  selected,
  spectrumData,
  dustAssessment,
  loading,
  onPinSpectrum,
  activeOverlayType,
  onSetOverlay,
}: CRISMSpectrumTabProps) {
  const [showBandRatios, setShowBandRatios] = useState(false);
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
          {!activeOverlayType ? (
            <button
              onClick={() => onSetOverlay("quickview")}
              className="mt-2 flex items-center gap-1.5 rounded-lg px-4 py-2 text-[11px] font-medium bg-primary/20 border border-primary/40 text-primary hover:bg-primary/30 transition-colors"
            >
              <span className="material-symbols-outlined text-sm">visibility</span>
              Enable Quickview First
            </button>
          ) : (
            <p className="text-[11px] text-slate-500">
              The spectral profile will be displayed here
            </p>
          )}
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

  // Valid data points
  const validPoints = spectrumData.wavelengths
    .map((wl, i) => ({ wavelength: wl, reflectance: spectrumData.reflectance[i] }))
    .filter((p): p is { wavelength: number; reflectance: number } => p.reflectance !== null);

  const reflValues = validPoints.map((p) => p.reflectance);
  const minRefl = Math.min(...reflValues);
  const maxRefl = Math.max(...reflValues);
  const meanRefl = reflValues.reduce((a, b) => a + b, 0) / reflValues.length;

  // Chart
  const chartWidth = 340;
  const chartHeight = 180;
  const pad = { top: 10, right: 10, bottom: 25, left: 45 };
  const iw = chartWidth - pad.left - pad.right;
  const ih = chartHeight - pad.top - pad.bottom;

  const wlMin = Math.min(...validPoints.map((p) => p.wavelength));
  const wlMax = Math.max(...validPoints.map((p) => p.wavelength));
  const yMin = Math.max(0, minRefl - (maxRefl - minRefl) * 0.1);
  const yMax = maxRefl + (maxRefl - minRefl) * 0.1;

  const xScale = (wl: number) => pad.left + ((wl - wlMin) / (wlMax - wlMin)) * iw;
  const yScale = (r: number) => pad.top + ih - ((r - yMin) / (yMax - yMin)) * ih;

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
        <div className="flex items-center gap-2">
          {onPinSpectrum && (
            <button
              onClick={() =>
                onPinSpectrum({
                  productId: selected.productId,
                  lat: selected.lat,
                  lon: selected.lon,
                  wavelengths: spectrumData.wavelengths,
                  reflectance: spectrumData.reflectance,
                })
              }
              className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
              title="Pin spectrum for comparison"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>push_pin</span>
              Pin
            </button>
          )}
          <span className="text-[10px] text-slate-500">{spectrumData.validBands} bands</span>
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

      {/* Dust assessment */}
      {dustAssessment && dustAssessment.risk_level !== "LOW" && (
        <div
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-[10px] leading-relaxed ${
            dustAssessment.risk_level === "HIGH"
              ? "bg-red-500/10 border border-red-500/30 text-red-400"
              : "bg-amber-500/10 border border-amber-500/30 text-amber-400"
          }`}
        >
          <span className="material-symbols-outlined text-sm flex-shrink-0">
            {dustAssessment.risk_level === "HIGH" ? "warning" : "info"}
          </span>
          <div>
            <span className="font-semibold">
              {dustAssessment.risk_level === "HIGH" ? "High" : "Moderate"} Dust Risk
            </span>
            {" · "}tau~{dustAssessment.tau_estimated.toFixed(1)}
            {dustAssessment.band_depth_suppression_pct > 0 && (
              <> · Band depths suppressed ~{dustAssessment.band_depth_suppression_pct.toFixed(0)}%</>
            )}
            {dustAssessment.warning_message && (
              <div className="mt-0.5 opacity-80">{dustAssessment.warning_message}</div>
            )}
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="rounded-lg border border-border-dark bg-bg-dark/60 p-3">
        <svg width={chartWidth} height={chartHeight} className="overflow-visible">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = pad.top + ih * (1 - t);
            const val = yMin + (yMax - yMin) * t;
            return (
              <g key={t}>
                <line x1={pad.left} y1={y} x2={pad.left + iw} y2={y} stroke="#334155" strokeWidth="1" strokeDasharray="2,2" />
                <text x={pad.left - 5} y={y} textAnchor="end" dominantBaseline="middle" className="fill-slate-500 text-[8px]">
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}
          {/* X axis labels */}
          {[wlMin, (wlMin + wlMax) / 2, wlMax].map((wl, i) => (
            <text key={i} x={xScale(wl)} y={chartHeight - 5} textAnchor="middle" className="fill-slate-500 text-[8px]">
              {wl.toFixed(1)}
            </text>
          ))}
          <text x={chartWidth / 2} y={chartHeight} textAnchor="middle" className="fill-slate-400 text-[9px]">
            Wavelength (μm)
          </text>
          <text x={10} y={chartHeight / 2} textAnchor="middle" transform={`rotate(-90, 10, ${chartHeight / 2})`} className="fill-slate-400 text-[9px]">
            I/F
          </text>
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

      {/* Band Ratios */}
      {!showBandRatios ? (
        <button
          onClick={() => setShowBandRatios(true)}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg py-2 text-[10px] font-bold uppercase tracking-widest bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20 transition-colors"
        >
          <span className="material-symbols-outlined text-sm">calculate</span>
          Band Ratios
        </button>
      ) : (
        <BandRatioCalculator
          wavelengths={spectrumData.wavelengths}
          reflectance={spectrumData.reflectance}
          onClose={() => setShowBandRatios(false)}
        />
      )}
    </div>
  );
}
