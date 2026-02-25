import type React from "react";
import * as Cesium from "cesium";
import DTMHoverReadout, { type DTMHoverReadoutHandle } from "../DTMHoverReadout";
import MeasurementTools from "../MeasurementTools";

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
  return (
    <>
      {hover && (
        <div className="absolute bottom-6 left-6 rounded-lg border border-border-dark bg-bg-dark/90 p-3 backdrop-blur-md">
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
        </div>
      )}

      {scoreOverlays.size > 0 && (
        <div className="absolute bottom-6 right-6 rounded-lg border border-border-dark bg-bg-dark/90 p-3 backdrop-blur-md">
          {(() => {
            const activeTypes = new Set<ScoreProductType>();
            scoreOverlays.forEach((types) => types.forEach((t) => activeTypes.add(t)));
            return (
              <div className="space-y-2">
                <div className="text-[9px] uppercase tracking-widest text-slate-500 font-bold">
                  {activeTypes.has("score_ice") && activeTypes.has("score_hyd")
                    ? "Ice / Hydration Score"
                    : activeTypes.has("score_ice")
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
            );
          })()}
        </div>
      )}

      <DTMHoverReadout
        ref={dtmHoverReadoutRef}
        mode={dtmHoverMode}
        onModeChange={onDTMHoverModeChange}
      />

      <MeasurementTools
        viewer={viewer}
        isVisible={showMeasurementTools}
        onPinNote={onMeasurementPinNote}
      />
    </>
  );
}
