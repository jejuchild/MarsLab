import type { FootprintSectionProps, FootprintCount, InstrumentId } from "../types";
import { lp, INST_STYLES } from "../tokens";
import { INSTRUMENTS, INSTRUMENT_GROUPS } from "../../../config/instrumentRegistry";

export default function FootprintSection({
  instrumentVisibility,
  onToggleInstrument,
  onLoadFootprints,
  footprintsLoading = {},
  footprintCounts = {},
  showCustomData,
  onToggleCustomData,
  onLoadCustomData,
  customDataLoading,
  customDatasets = [],
  onCustomDatasetToggle,
}: FootprintSectionProps) {
  return (
    <div className={lp.section}>
      <h3 className={`${lp.h3} mb-3`}>Footprints</h3>

      <div className="space-y-1">
        {INSTRUMENT_GROUPS.map((group) => {
          const activeCount = group.instruments.filter((id) => instrumentVisibility[id]).length;
          const allActive = activeCount === group.instruments.length;
          const someActive = activeCount > 0 && !allActive;

          return (
            <div key={group.id} className="rounded overflow-hidden">
              {/* Group header */}
              <button
                aria-label={`Toggle all ${group.displayName} instruments`}
                onClick={() => {
                  const newVal = !allActive;
                  for (const instId of group.instruments) {
                    onToggleInstrument(instId, newVal);
                  }
                }}
                className={`flex items-center gap-2 w-full px-2.5 py-2 text-left transition-colors ${
                  someActive || allActive
                    ? "bg-[#1a2333]"
                    : "bg-[#101622] hover:bg-[#1a2333]"
                }`}
              >
                <span className="material-symbols-outlined text-sm text-[#6b7c9c]">{group.icon}</span>
                <span className="text-[11px] font-bold text-[#92a4c9] flex-1">{group.displayName}</span>
                {activeCount > 0 && (
                  <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-primary/20 text-primary font-bold">
                    {activeCount}
                  </span>
                )}
                <span className="material-symbols-outlined text-xs text-[#6b7c9c]">
                  {someActive || allActive ? "toggle_on" : "toggle_off"}
                </span>
              </button>

              {/* Child instruments */}
              <div className="pl-4 space-y-1 py-1 bg-[#0d1219]">
                {group.instruments.map((instId) => {
                  const inst = INSTRUMENTS[instId];
                  const isVisible = instrumentVisibility[instId];
                  const isLoading = footprintsLoading[instId] ?? false;
                  const count = footprintCounts[instId] as FootprintCount;
                  const style = INST_STYLES[instId];

                  return (
                    <div key={instId} className="flex items-center gap-1.5 pr-2">
                      <label
                        className={`flex items-center gap-2 py-1.5 px-2 rounded cursor-pointer transition-colors flex-1 ${
                          isVisible ? style.bgActive : "bg-transparent hover:bg-[#1a2333]"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isVisible}
                          onChange={(e) => onToggleInstrument(instId, e.target.checked)}
                          className={`rounded bg-[#0a0f18] border-[#232f48] focus:ring-0 focus:ring-offset-0 ${style.checkbox}`}
                        />
                        <span className={`text-[10px] font-medium ${style.text}`}>{inst.subLabel}</span>
                        {count && (
                          <span className={`text-[8px] ${style.text} opacity-60 ml-auto`}>
                            {count.count}
                            {count.truncated && `/${count.total}`}
                          </span>
                        )}
                      </label>
                      <button
                        aria-label={`Load ${inst.displayName} footprints`}
                        onClick={() =>
                          onLoadFootprints?.(
                            inst.name as
                              | "CRISM"
                              | "HIRISE"
                              | "SHARAD"
                              | "SHARAD_HIGHRES"
                              | "CTX"
                              | "HIRISE_DTM"
                              | "CRISM_TRR3",
                          )
                        }
                        disabled={isLoading}
                        className={`px-1.5 py-1 rounded text-[9px] font-medium transition-colors whitespace-nowrap ${
                          isLoading ? style.btnLoading : style.btn
                        }`}
                      >
                        {isLoading ? (
                          <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                        ) : (
                          "Load"
                        )}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Custom Data Row */}
        <div className="flex items-center gap-1.5 mt-2 pt-2 border-t border-[#232f48]">
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
    </div>
  );
}
