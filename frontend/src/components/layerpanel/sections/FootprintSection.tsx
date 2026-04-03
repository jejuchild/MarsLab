import type { FootprintSectionProps, FootprintCount } from "../types";
import { INST_STYLES } from "../tokens";
import { INSTRUMENTS, INSTRUMENT_GROUPS } from "../../../config/instrumentRegistry";
import CollapsibleSection from "../shared/CollapsibleSection";

/** Known data-rich regions for the empty-state helper */
const SUGGESTED_REGIONS = [
  { name: "Jezero Crater", lat: 18.44, lon: 77.45 },
  { name: "Valles Marineris", lat: -13.9, lon: -59.2 },
  { name: "Olympus Mons", lat: 18.65, lon: -133.8 },
];

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
  highResOnly = false,
  onHighResOnlyChange,
  onCustomDatasetToggle: _onCustomDatasetToggle,
}: FootprintSectionProps) {
  // Count how many instruments have loaded data
  const loadedCount = Object.values(footprintCounts).filter(Boolean).length;
  const totalVisible = Object.values(footprintCounts).reduce(
    (sum, c) => sum + ((c as FootprintCount)?.count ?? 0),
    0,
  );

  // Check if any instrument has been loaded but shows 0 products
  const hasLoaded = loadedCount > 0;
  const hasZeroProducts = hasLoaded && totalVisible === 0;

  const trailing = loadedCount > 0 ? (
    <span className="text-primary text-[10px] font-mono">{totalVisible}</span>
  ) : undefined;

  return (
    <CollapsibleSection
      title="Footprints"
      icon="hexagon"
      defaultOpen={false}
      storageKey="footprints"
      trailing={trailing}
    >
      <div className="space-y-1">
        {/* High-Res Only filter toggle */}
        <button
          onClick={() => {
            onHighResOnlyChange?.(!highResOnly);
            // Auto-reload all loaded & visible instruments after state propagates
            setTimeout(() => {
              for (const group of INSTRUMENT_GROUPS) {
                for (const instId of group.instruments) {
                  const count = footprintCounts[instId] as FootprintCount;
                  if (count && instrumentVisibility[instId]) {
                    const inst = INSTRUMENTS[instId];
                    onLoadFootprints?.(inst.name as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3");
                  }
                }
              }
            }, 100);
          }}
          className={`flex items-center gap-2 w-full px-2.5 py-1.5 rounded transition-colors text-left ${
            highResOnly
              ? "bg-purple-500/20 border border-purple-500/40"
              : "bg-[#101622] border border-transparent hover:bg-[#1a2333]"
          }`}
        >
          <span className={`material-symbols-outlined text-sm ${highResOnly ? "text-purple-400" : "text-[#6b7c9c]"}`}>
            hd
          </span>
          <span className={`text-[10px] font-bold flex-1 ${highResOnly ? "text-purple-300" : "text-[#92a4c9]"}`}>
            High-Res Only
          </span>
          <span className={`material-symbols-outlined text-xs ${highResOnly ? "text-purple-400" : "text-[#6b7c9c]"}`}>
            {highResOnly ? "toggle_on" : "toggle_off"}
          </span>
        </button>

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

              {/* Child instruments — single "Load & Show" toggle per instrument */}
              <div className="pl-4 space-y-1 py-1 bg-[#0d1219]">
                {group.instruments.map((instId) => {
                  const inst = INSTRUMENTS[instId];
                  const isVisible = instrumentVisibility[instId];
                  const isLoading = footprintsLoading[instId] ?? false;
                  const count = footprintCounts[instId] as FootprintCount;
                  const style = INST_STYLES[instId];
                  const isLoaded = !!count;

                  // U2: Simplified controls
                  // Not loaded → click row to load & show
                  // Loaded → click row to toggle visibility, small reload button to re-fetch
                  const handleClick = () => {
                    if (isLoading) return;
                    if (!isLoaded) {
                      onToggleInstrument(instId, true);
                      onLoadFootprints?.(inst.name as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3");
                    } else {
                      onToggleInstrument(instId, !isVisible);
                    }
                  };

                  const handleReload = (e: React.MouseEvent) => {
                    e.stopPropagation();
                    if (isLoading) return;
                    onToggleInstrument(instId, true);
                    onLoadFootprints?.(inst.name as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3");
                  };

                  return (
                    <div key={instId} className="flex items-center gap-0.5">
                      <button
                        onClick={handleClick}
                        disabled={isLoading}
                        className={`flex items-center gap-2 flex-1 py-1.5 px-2 rounded-l transition-colors text-left ${
                          isLoading
                            ? "opacity-60 cursor-wait"
                            : isLoaded && isVisible
                              ? style.bgActive
                              : isLoaded
                                ? "bg-transparent opacity-50 hover:opacity-80"
                                : "bg-transparent hover:bg-[#1a2333]"
                        }`}
                      >
                        <span className={`material-symbols-outlined text-xs ${style.text}`}>
                          {isLoading
                            ? "progress_activity"
                            : isLoaded && isVisible
                              ? "visibility"
                              : isLoaded
                                ? "visibility_off"
                                : "download"}
                        </span>
                        <span className={`text-[10px] font-medium flex-1 ${style.text}`}>
                          {inst.subLabel}
                        </span>
                        {count && (
                          <span className={`text-[8px] ${style.text} opacity-60`}>
                            {count.count}
                            {count.truncated && `/${count.total}`}
                          </span>
                        )}
                        {!isLoaded && !isLoading && (
                          <span className={`text-[8px] font-medium ${style.text} opacity-60`}>Load</span>
                        )}
                        {isLoading && (
                          <span className="material-symbols-outlined text-xs animate-spin text-[#6b7c9c]">
                            progress_activity
                          </span>
                        )}
                      </button>
                      {isLoaded && !isLoading && (
                        <button
                          onClick={handleReload}
                          className={`p-1.5 rounded-r transition-colors ${style.text} opacity-40 hover:opacity-80 hover:bg-[#1a2333]`}
                          title="Reload for current view"
                        >
                          <span className="material-symbols-outlined text-xs">refresh</span>
                        </button>
                      )}
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

      {/* U3: Contextual empty state when instruments loaded but 0 products */}
      {hasZeroProducts && (
        <div className="mt-3 p-3 rounded-lg border border-[#232f48] bg-[#0d1219] text-center">
          <span className="material-symbols-outlined text-lg text-[#6b7c9c] mb-1 block">
            search_off
          </span>
          <p className="text-[10px] text-[#6b7c9c] leading-relaxed mb-2">
            No products found in the current view. Try zooming out or panning to a different area.
          </p>
          <p className="text-[9px] text-[#4a5a7a] font-medium mb-1">Data-rich regions:</p>
          <div className="flex flex-wrap gap-1 justify-center">
            {SUGGESTED_REGIONS.map((r) => (
              <span
                key={r.name}
                className="text-[8px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary/70 border border-primary/20"
              >
                {r.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </CollapsibleSection>
  );
}
