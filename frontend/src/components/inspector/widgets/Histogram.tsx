import { useMemo } from "react";

type HistogramProps = {
  histogram: number[];
  binEdges: number[];
};

export default function Histogram({ histogram, binEdges }: HistogramProps) {
  const total = useMemo(
    () => histogram.reduce((sum, v) => sum + v, 0),
    [histogram]
  );

  const maxVal = useMemo(
    () => Math.max(...histogram, 1),
    [histogram]
  );

  const bottomLabels = useMemo(() => {
    if (binEdges.length < 2) return [];
    const step = Math.max(1, Math.floor(binEdges.length / 5));
    const labels: { value: string; pct: number }[] = [];
    for (let i = 0; i < binEdges.length; i += step) {
      labels.push({
        value: binEdges[i].toFixed(0),
        pct: (i / (binEdges.length - 1)) * 100,
      });
    }
    return labels;
  }, [binEdges]);

  return (
    <div className="rounded-lg border border-border-dark bg-bg-dark/60 p-3">
      {/* Header */}
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-slate-500">
          DN Distribution (Sampled Area)
        </span>
        <span className="font-mono text-[10px] text-slate-500">
          n={total.toLocaleString()}
        </span>
      </div>

      {/* Bar chart */}
      <div className="flex h-32 items-end gap-px">
        {histogram.map((count, i) => {
          const heightPct = (count / maxVal) * 100;
          return (
            <div
              key={i}
              className="flex-1 rounded-t-sm bg-gradient-to-t from-primary to-blue-400"
              style={{ height: `${heightPct}%` }}
              title={`Bin ${i}: ${count} pixels`}
            />
          );
        })}
      </div>

      {/* Bottom labels */}
      <div className="relative mt-1 h-3">
        {bottomLabels.map((label, i) => (
          <span
            key={i}
            className="absolute font-mono text-[8px] text-slate-500"
            style={{ left: `${label.pct}%`, transform: "translateX(-50%)" }}
          >
            {label.value}
          </span>
        ))}
      </div>
    </div>
  );
}
