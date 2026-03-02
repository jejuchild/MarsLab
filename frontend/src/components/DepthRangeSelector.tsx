import type { DepthRange } from "../api/swim_ice";

/* =========================================================
 * Props
 * =======================================================*/
export interface DepthRangeSelectorProps {
  value: DepthRange;
  onChange: (depth: DepthRange) => void;
}

/* =========================================================
 * Depth options
 * =======================================================*/
const DEPTH_OPTIONS: { value: DepthRange; label: string }[] = [
  { value: "0-1m", label: "0–1 m" },
  { value: "1-5m", label: "1–5 m" },
  { value: "5m-plus", label: ">5 m" },
];

/* =========================================================
 * Component
 * =======================================================*/
export default function DepthRangeSelector({ value, onChange }: DepthRangeSelectorProps) {
  return (
    <div className="flex items-center gap-1">
      <span className="mr-1 text-[10px] uppercase tracking-wider text-[#92a4c9]">
        Depth
      </span>
      <div className="flex overflow-hidden rounded border border-[#232f48]">
        {DEPTH_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`px-2 py-0.5 text-[11px] font-medium transition-colors ${
              value === opt.value
                ? "bg-blue-600/30 text-blue-300"
                : "bg-[#111b2a] text-[#92a4c9] hover:bg-[#1a2744]"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
