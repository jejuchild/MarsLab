import type { InspectorContext, WindowStats } from "../types";
import type { OverlayType } from "../../../pages/MainPage";
import StatCard from "../widgets/StatCard";
import Histogram from "../widgets/Histogram";

type HiRISEPixelTabProps = {
  selected: InspectorContext;
  stats: WindowStats | null;
  loading: boolean;
  windowSize: number;
  onWindowSizeChange: (size: number) => void;
  // UX improvement: direct CTA to enable overlay
  activeOverlayType: string | null;
  onSetOverlay: (type: OverlayType | null) => void;
};

export default function HiRISEPixelTab({
  selected,
  stats,
  loading,
  windowSize,
  onWindowSizeChange,
  activeOverlayType,
  onSetOverlay,
}: HiRISEPixelTabProps) {
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
              Enable the high-resolution overlay first, then click on it
            </p>
          )}
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
