import type { ViewModeSectionProps } from "../types";
import { lp } from "../tokens";

const BASE_LAYERS: { id: "MOLA" | "HRSC" | "THEMIS"; label: string; hint: string }[] = [
  { id: "MOLA", label: "MGS MOLA ColorShade", hint: "Colour-coded elevation (463 m)" },
  { id: "THEMIS", label: "THEMIS Day IR (greyscale)", hint: "Neutral backdrop for comparing overlays (100 m)" },
  // id kept as "HRSC" for saved links; the tiles are Viking MDIM 2.1 colour.
  { id: "HRSC", label: "Viking MDIM 2.1 colour", hint: "Global colour image mosaic (232 m)" },
];

export default function ViewModeSection({
  mapMode,
  onMapModeChange,
  baseLayer,
  onBaseLayerChange,
}: ViewModeSectionProps) {
  return (
    <div className={lp.section}>
      {/* Map Mode (2D / 3D) */}
      <h3 className={`${lp.h3} mb-3`}>View Mode</h3>
      <div className="flex gap-2">
        <button
          onClick={() => onMapModeChange("2D")}
          aria-label="Switch to 2D map"
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded transition-colors ${
            mapMode === "2D"
              ? lp.toggleActive
              : lp.toggleInactive
          }`}
        >
          <span className="material-symbols-outlined text-base">map</span>
          <span className={`${lp.body}`}>2D Map</span>
        </button>
        <button
          onClick={() => onMapModeChange("3D")}
          aria-label="Switch to 3D globe"
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded transition-colors ${
            mapMode === "3D"
              ? lp.toggleActive
              : lp.toggleInactive
          }`}
        >
          <span className="material-symbols-outlined text-base">globe</span>
          <span className={`${lp.body}`}>3D Globe</span>
        </button>
      </div>

      {/* Base Map */}
      <h3 className={`${lp.h3} mb-3 mt-4`}>Base Map</h3>
      <div className="space-y-2">
        {BASE_LAYERS.map((b) => (
          <label
            key={b.id}
            className={`flex items-start gap-2 p-2 rounded cursor-pointer transition-colors ${
              baseLayer === b.id
                ? "bg-primary/20 border border-primary/50"
                : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
            }`}
            title={b.hint}
          >
            <input
              type="radio"
              name="baseLayer"
              checked={baseLayer === b.id}
              onChange={() => onBaseLayerChange(b.id)}
              className="mt-0.5 rounded-full bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
            />
            <span className="flex flex-col">
              <span className={lp.body}>{b.label}</span>
              <span className="text-[10px] text-slate-500">{b.hint}</span>
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
