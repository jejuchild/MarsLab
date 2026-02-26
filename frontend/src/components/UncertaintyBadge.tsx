/**
 * UncertaintyBadge — Reusable uncertainty range visualization.
 * Shows a value with its uncertainty range and a mini visual bar.
 */

interface UncertaintyBadgeProps {
  label: string;
  value: number;
  low: number;
  high: number;
  unit?: string;
  /** 0-1 scale for coloring (higher = more certain = green) */
  confidence?: number;
  compact?: boolean;
}

export default function UncertaintyBadge({
  label,
  value,
  low,
  high,
  unit = "",
  confidence,
  compact = false,
}: UncertaintyBadgeProps) {
  const range = high - low;
  const relativeUncertainty = value > 0 ? range / value : 0;

  // Color based on relative uncertainty: tight = green, wide = amber, very wide = red
  const uncertaintyColor =
    relativeUncertainty < 0.3
      ? { bg: "bg-green-500/10", border: "border-green-500/20", text: "text-green-400", bar: "bg-green-400" }
      : relativeUncertainty < 0.6
        ? { bg: "bg-yellow-500/10", border: "border-yellow-500/20", text: "text-yellow-400", bar: "bg-yellow-400" }
        : { bg: "bg-red-500/10", border: "border-red-500/20", text: "text-red-400", bar: "bg-red-400" };

  if (compact) {
    return (
      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] ${uncertaintyColor.bg} border ${uncertaintyColor.border}`}>
        <span className={`font-mono font-bold ${uncertaintyColor.text}`}>
          {value.toFixed(1)}{unit}
        </span>
        <span className="text-slate-500">
          ±{(range / 2).toFixed(1)}
        </span>
      </span>
    );
  }

  // Full visual bar showing the range
  const totalSpan = high - low;
  const valuePosition = totalSpan > 0 ? ((value - low) / totalSpan) * 100 : 50;

  return (
    <div className={`p-2.5 rounded-lg ${uncertaintyColor.bg} border ${uncertaintyColor.border}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-slate-500 uppercase font-bold">{label}</span>
        {confidence !== undefined && (
          <span className={`text-[9px] font-bold ${
            confidence >= 0.8 ? "text-green-400" : confidence >= 0.5 ? "text-yellow-400" : "text-red-400"
          }`}>
            {(confidence * 100).toFixed(0)}% conf
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`text-sm font-mono font-bold ${uncertaintyColor.text}`}>
          {value.toFixed(1)}
        </span>
        <span className="text-[10px] text-slate-400">{unit}</span>
        <span className="text-[10px] text-slate-500 ml-1">
          [{low.toFixed(1)} – {high.toFixed(1)}]
        </span>
      </div>
      {/* Visual range bar */}
      <div className="mt-1.5 relative h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
        {/* Full range background */}
        <div className={`absolute inset-y-0 ${uncertaintyColor.bar} opacity-20 rounded-full`}
          style={{ left: "0%", right: "0%" }}
        />
        {/* Value position marker */}
        <div
          className={`absolute top-0 bottom-0 w-1 ${uncertaintyColor.bar} rounded-full`}
          style={{ left: `${Math.max(2, Math.min(98, valuePosition))}%` }}
        />
      </div>
      <div className="flex justify-between mt-0.5 text-[8px] text-slate-600 font-mono">
        <span>{low.toFixed(1)}{unit}</span>
        <span>{high.toFixed(1)}{unit}</span>
      </div>
    </div>
  );
}
