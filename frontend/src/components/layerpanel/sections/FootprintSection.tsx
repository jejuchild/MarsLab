import type { FootprintSectionProps } from "../types";
import CollapsibleSection from "../shared/CollapsibleSection";

/**
 * Side-panel part of the footprint controls. Instrument footprints moved to
 * the on-map InstrumentBar (the primary action); this keeps user-uploaded
 * custom datasets.
 */
export default function FootprintSection({
  showCustomData,
  onToggleCustomData,
  onLoadCustomData,
  customDataLoading,
  customDatasets = [],
}: FootprintSectionProps) {
  return (
    <CollapsibleSection title="Custom Data" icon="upload_file" defaultOpen={false} storageKey="custom-data">
      <div className="space-y-1">
        {/* Custom Data Row */}
        <div className="flex items-center gap-1.5 ">
          <label
            className={`flex items-center gap-2 py-1.5 px-2 rounded cursor-pointer transition-colors flex-1 ${
              showCustomData
                ? "bg-fuchsia-500/20 border border-fuchsia-500/50"
                : "bg-transparent border border-transparent hover:bg-[#1a2333]"
            }`}
          >
            <input
              type="checkbox"
              checked={showCustomData}
              onChange={(e) => onToggleCustomData(e.target.checked)}
              className="rounded bg-[#0a0f18] border-[#232f48] text-fuchsia-500 focus:ring-0 focus:ring-offset-0"
            />
            <span className="text-[10px] font-medium text-fuchsia-400">Custom Data</span>
            {customDatasets.length > 0 && (
              <span className="text-[8px] text-fuchsia-400/60 ml-auto">
                {customDatasets.length}
              </span>
            )}
          </label>
          <button
            aria-label="Load custom data"
            onClick={() => onLoadCustomData?.()}
            disabled={customDataLoading}
            className={`px-1.5 py-1 rounded text-[9px] font-medium transition-colors whitespace-nowrap ${
              customDataLoading
                ? "bg-fuchsia-500/10 text-fuchsia-400/50 border border-fuchsia-500/20 cursor-wait"
                : "bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 hover:bg-fuchsia-500/30"
            }`}
          >
            {customDataLoading ? (
              <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
            ) : (
              "Load"
            )}
          </button>
        </div>
      </div>

    </CollapsibleSection>
  );
}
