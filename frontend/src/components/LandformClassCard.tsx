import { LANDFORM_TYPES } from "../api/hirise_landforms";

/* =========================================================
 * Props
 * =======================================================*/
export interface LandformClassCardProps {
  classCode: string;
  className_: string;
  probability: number;
  isTopClass?: boolean;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function LandformClassCard({
  classCode,
  className_,
  probability,
  isTopClass = false,
}: LandformClassCardProps) {
  const fallbackMeta = { label: "Other Terrain", icon: "landscape", color: "bg-slate-500" };
  const meta = LANDFORM_TYPES[classCode] ?? LANDFORM_TYPES["OTHER"] ?? fallbackMeta;
  const pct = Math.round(probability * 100);

  // Color intensity based on probability
  const barColor =
    pct >= 70
      ? "bg-green-500"
      : pct >= 40
        ? "bg-yellow-500"
        : pct >= 15
          ? "bg-orange-500"
          : "bg-slate-500";

  return (
    <div
      className={`flex flex-col gap-1 rounded border px-2 py-1.5 ${
        isTopClass
          ? "border-blue-500/40 bg-blue-500/5"
          : "border-[#232f48] bg-[#111b2a]"
      }`}
    >
      {/* Header: icon + name + code */}
      <div className="flex items-center gap-1.5">
        <span className="material-symbols-outlined text-[14px] text-[#92a4c9]">
          {meta.icon}
        </span>
        <span className="text-[11px] font-medium text-slate-200">
          {className_}
        </span>
        <span className="ml-auto rounded bg-[#1a2744] px-1 py-0.5 font-mono text-[9px] text-[#92a4c9]">
          {classCode}
        </span>
        {isTopClass && (
          <span className="rounded bg-blue-600/20 px-1 py-0.5 text-[8px] font-semibold uppercase text-blue-400">
            Top
          </span>
        )}
      </div>

      {/* Probability bar */}
      <div className="flex items-center gap-1.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#0d1520]">
          <div
            className={`h-full rounded-full transition-all ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="min-w-[30px] text-right font-mono text-[10px] text-slate-200">
          {pct}%
        </span>
      </div>
    </div>
  );
}
