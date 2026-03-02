/* =========================================================
 * IceConsistencyLegend — Color legend for -1 to +1 consistency scale
 * =======================================================*/

const LEGEND_STOPS = [
  { pos: "0%", color: "#b71c1c" },    // -1.0 strong against ice
  { pos: "15%", color: "#ef9a9a" },   // -0.7 moderate against
  { pos: "35%", color: "#9e9e9e" },   // -0.3 ambiguous
  { pos: "65%", color: "#9e9e9e" },   // +0.3 ambiguous
  { pos: "85%", color: "#42a5f5" },   // +0.7 moderate ice
  { pos: "100%", color: "#1a237e" },  // +1.0 strong ice
];

const GRADIENT_ID = "swim-consistency-gradient";

export default function IceConsistencyLegend() {
  return (
    <div className="flex flex-col gap-1 px-1 py-1.5">
      <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-[#92a4c9]">
        <span>Against Ice</span>
        <span>Ambiguous</span>
        <span>Ice Consistent</span>
      </div>

      {/* Gradient bar */}
      <svg width="100%" height="10" className="rounded-sm">
        <defs>
          <linearGradient id={GRADIENT_ID} x1="0%" y1="0%" x2="100%" y2="0%">
            {LEGEND_STOPS.map((stop, i) => (
              <stop key={i} offset={stop.pos} stopColor={stop.color} />
            ))}
          </linearGradient>
        </defs>
        <rect width="100%" height="10" rx="2" fill={`url(#${GRADIENT_ID})`} />
      </svg>

      {/* Numeric labels */}
      <div className="flex items-center justify-between text-[9px] font-mono text-slate-400">
        <span>-1.0</span>
        <span>0</span>
        <span>+1.0</span>
      </div>
    </div>
  );
}
