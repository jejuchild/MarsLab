import { useState, useCallback } from "react";
import type React from "react";
import * as Cesium from "cesium";
import DTMHoverReadout, { type DTMHoverReadoutHandle } from "../DTMHoverReadout";
import MeasurementTools from "../MeasurementTools";
import ScaleBar from "./ScaleBar";

type LatLon = { lat: number; lon: number };
type ScoreProductType = "score_ice" | "score_hyd";

type MapToolbarProps = {
  hover: LatLon | null;
  scoreOverlays: Map<string, Set<ScoreProductType>>;
  dtmHoverReadoutRef: React.RefObject<DTMHoverReadoutHandle | null>;
  dtmHoverMode: "hover" | "click";
  onDTMHoverModeChange: (mode: "hover" | "click") => void;
  viewer: Cesium.Viewer | null;
  showMeasurementTools: boolean;
  onMeasurementPinNote?: (lat: number, lon: number, text: string) => void;
};

export default function MapToolbar({
  hover,
  scoreOverlays,
  dtmHoverReadoutRef,
  dtmHoverMode,
  onDTMHoverModeChange,
  viewer,
  showMeasurementTools,
  onMeasurementPinNote,
}: MapToolbarProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyCoords = useCallback(() => {
    if (!hover) return;
    const text = `${hover.lat.toFixed(4)}, ${hover.lon.toFixed(4)}`;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [hover]);

  // Compute active score types once
  const activeScoreTypes = (() => {
    const types = new Set<ScoreProductType>();
    scoreOverlays.forEach((s) => s.forEach((t) => types.add(t)));
    return types;
  })();
  const hasScores = activeScoreTypes.size > 0;

  return (
    <>
      {/* ── Bottom-left info panel: Scale → Coordinates → Score legend ── */}
      <div className="absolute bottom-6 left-6 z-20 flex flex-col gap-2">
        {/* Scale bar */}
        <div className="rounded-lg border border-border-dark bg-bg-dark/90 px-3 py-2 backdrop-blur-md">
          <ScaleBar viewer={viewer} />
        </div>

        {/* Coordinate readout */}
        {hover && (
          <div className="rounded-lg border border-border-dark bg-bg-dark/90 p-3 backdrop-blur-md">
            <button
              type="button"
              onClick={handleCopyCoords}
              className="group relative flex items-center gap-3 w-full text-left cursor-pointer"
              title="Click to copy coordinates"
            >
              <div className="flex items-center gap-4">
                <div className="space-y-1">
                  <div className="text-[9px] uppercase tracking-tighter text-slate-500">Longitude</div>
                  <div className="font-mono text-xs">{hover.lon.toFixed(4)}°</div>
                </div>
                <div className="h-6 w-px bg-border-dark" />
                <div className="space-y-1">
                  <div className="text-[9px] uppercase tracking-tighter text-slate-500">Latitude</div>
                  <div className="font-mono text-xs">{hover.lat.toFixed(4)}°</div>
                </div>
              </div>
              <span className="material-symbols-outlined text-[14px] text-slate-500 group-hover:text-slate-300 transition-colors ml-auto">
                content_copy
              </span>
              {/* Copied toast */}
              {copied && (
                <span className="absolute -top-7 left-1/2 -translate-x-1/2 rounded bg-primary/90 px-2 py-0.5 text-[10px] font-medium text-white whitespace-nowrap animate-fade-in">
                  Copied!
                </span>
              )}
            </button>
          </div>
        )}

        {/* Score legend (inline below coordinates) */}
        {hasScores && (
          <div className="rounded-lg border border-border-dark bg-bg-dark/90 p-3 backdrop-blur-md">
            <div className="space-y-2">
              <div className="text-[9px] uppercase tracking-widest text-slate-500 font-bold">
                {activeScoreTypes.has("score_ice") && activeScoreTypes.has("score_hyd")
                  ? "Ice / Hydration Score"
                  : activeScoreTypes.has("score_ice")
                  ? "Ice Score"
                  : "Hydration Score"}
              </div>
              <div
                className="h-2.5 w-40 rounded-sm border border-white/10"
                style={{
                  background: "linear-gradient(to right, #ffffff 0%, #808080 25%, #000000 50%, #5a0000 75%, #b40000 100%)",
                }}
              />
              <div className="flex justify-between text-[8px] font-mono text-slate-400 w-40">
                <span>0</span>
                <span>0.5</span>
                <span>1.0</span>
                <span>1.5</span>
                <span>2.0</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── DTM Hover Readout (centered bottom) ── */}
      <DTMHoverReadout
        ref={dtmHoverReadoutRef}
        mode={dtmHoverMode}
        onModeChange={onDTMHoverModeChange}
      />

      {/* ── Measurement tools (always-visible hint or full panel) ── */}
      {showMeasurementTools ? (
        <MeasurementTools
          viewer={viewer}
          isVisible={showMeasurementTools}
          onPinNote={onMeasurementPinNote}
        />
      ) : (
        <div
          className="absolute top-20 right-4 z-[500] flex flex-col items-center rounded-xl border border-border-dark bg-surface-dark/95 p-1.5 backdrop-blur-md shadow-lg"
          style={{ pointerEvents: "auto" }}
          title="Use Measurement Mode in the toolbar to enable"
        >
          <div className="group relative flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 cursor-default">
            <span className="material-symbols-outlined text-[20px]">straighten</span>
            <span className="pointer-events-none absolute right-full mr-2 whitespace-nowrap rounded-md bg-bg-dark/95 px-2 py-1 text-[10px] text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100 border border-border-dark">
              Use Measurement Mode in the toolbar to enable
            </span>
          </div>
        </div>
      )}
    </>
  );
}
