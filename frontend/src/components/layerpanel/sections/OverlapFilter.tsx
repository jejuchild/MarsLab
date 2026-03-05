import type { OverlapFilter, OverlapStats } from "../types";
import { lp } from "../tokens";

interface OverlapFilterProps {
  filter?: OverlapFilter;
  onChange?: (filter: OverlapFilter) => void;
  stats?: OverlapStats | null;
}

export default function OverlapFilterSection({
  filter,
  onChange,
  stats,
}: OverlapFilterProps) {
  const currentFilter = filter ?? { enabled: false, instruments: [] };

  const handleToggle = () => {
    onChange?.({ ...currentFilter, enabled: !currentFilter.enabled });
  };

  return (
    <div className={lp.section}>
      <div className="flex items-center justify-between">
        <h3 className={`${lp.h3} flex items-center gap-1`}>
          <span className="material-symbols-outlined text-xs">filter_alt</span>
          Overlap Filter
        </h3>
        <div className="flex items-center gap-2">
          {currentFilter.enabled && stats && stats.totalChecked > 0 && (
            <span className="text-sky-400 text-[10px] font-mono">
              {stats.totalPassing}/{stats.totalChecked}
            </span>
          )}
          <button
            onClick={handleToggle}
            aria-pressed={currentFilter.enabled}
            aria-label={currentFilter.enabled ? "Disable overlap filter" : "Enable overlap filter"}
            className={`text-[8px] px-1.5 py-0.5 rounded font-bold uppercase transition-colors ${
              currentFilter.enabled
                ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48] hover:border-[#3a4a68]"
            }`}
          >
            {currentFilter.enabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      {currentFilter.enabled && (
        <div className="mt-2 space-y-1.5">
          <p className={lp.tiny + " leading-relaxed"}>
            Showing only products that overlap with at least one product from another instrument.
          </p>

          {/* Per-instrument stats */}
          {stats && stats.totalChecked > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Array.from(stats.perInstrument.entries()).map(([inst, s]) =>
                s.checked > 0 && (
                  <span
                    key={inst}
                    className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${lp.caption} bg-[#1a2333] border-[#232f48]`}
                  >
                    {inst}: {s.passing}/{s.checked}
                  </span>
                )
              )}
            </div>
          )}

          {/* No overlap message */}
          {stats && stats.totalPassing === 0 && stats.totalChecked > 0 && (
            <p className="text-[9px] text-orange-400/80 flex items-center gap-1">
              <span className="material-symbols-outlined text-xs">warning</span>
              No overlapping regions found
            </p>
          )}

          {/* No footprints loaded */}
          {(!stats || stats.totalChecked === 0) && (
            <p className={`text-[9px] ${lp.caption} flex items-center gap-1`}>
              <span className="material-symbols-outlined text-xs">info</span>
              Load footprints for at least 2 instruments
            </p>
          )}
        </div>
      )}
    </div>
  );
}
