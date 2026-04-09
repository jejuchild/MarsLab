import type { Lane } from "../../../api/inspector";

export interface LaneTabConfig {
  lane: Lane;
  label: string;
  accent: string;
  count: number;
}

export interface LaneTabsProps {
  tabs: LaneTabConfig[];
  active: Lane;
  onChange: (lane: Lane) => void;
}

/**
 * Horizontal tab switcher for the 4 instrument lanes.
 *
 * Lanes with zero products are dimmed but still clickable.
 */
export default function LaneTabs({ tabs, active, onChange }: LaneTabsProps) {
  return (
    <div
      className="flex items-stretch border-b border-border-dark bg-bg-dark/60"
      role="tablist"
      aria-label="Instrument lanes"
    >
      {tabs.map((tab) => {
        const isActive = tab.lane === active;
        const isEmpty = tab.count === 0;
        return (
          <button
            key={tab.lane}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab.lane)}
            className={`flex-1 flex flex-col items-center gap-0.5 px-2 py-2 text-[10px] font-bold uppercase tracking-wider transition-colors border-b-2 ${
              isActive
                ? "text-white"
                : isEmpty
                ? "text-slate-700 hover:text-slate-500"
                : "text-slate-400 hover:text-white"
            }`}
            style={{
              borderBottomColor: isActive ? tab.accent : "transparent",
            }}
          >
            <span>{tab.label}</span>
            <span
              className={`text-[9px] font-mono ${
                isActive ? "text-white" : "text-slate-500"
              }`}
            >
              {tab.count}
            </span>
          </button>
        );
      })}
    </div>
  );
}
