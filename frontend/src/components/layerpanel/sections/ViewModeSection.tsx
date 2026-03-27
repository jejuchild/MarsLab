import type { ViewModeSectionProps } from "../types";
import { lp } from "../tokens";

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

      {/* Base Map (MOLA / HRSC) */}
      <h3 className={`${lp.h3} mb-3 mt-4`}>Base Map</h3>
      <div className="space-y-2">
        <label
          className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
            baseLayer === "MOLA"
              ? "bg-primary/20 border border-primary/50"
              : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
          }`}
        >
          <input
            type="radio"
            name="baseLayer"
            checked={baseLayer === "MOLA"}
            onChange={() => onBaseLayerChange("MOLA")}
            className="rounded-full bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
          />
          <span className={lp.body}>MGS MOLA ColorShade</span>
        </label>
        <label
          className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
            baseLayer === "HRSC"
              ? "bg-primary/20 border border-primary/50"
              : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
          }`}
        >
          <input
            type="radio"
            name="baseLayer"
            checked={baseLayer === "HRSC"}
            onChange={() => onBaseLayerChange("HRSC")}
            className="rounded-full bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
          />
          <span className={lp.body}>Mars Express HRSC</span>
        </label>
        <label
          className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
            baseLayer === "CTX"
              ? "bg-primary/20 border border-primary/50"
              : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
          }`}
        >
          <input
            type="radio"
            name="baseLayer"
            checked={baseLayer === "CTX"}
            onChange={() => onBaseLayerChange("CTX")}
            className="rounded-full bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
          />
          <span className={lp.body}>MRO CTX 5m (Arcadia)</span>
        </label>
      </div>
    </div>
  );
}
