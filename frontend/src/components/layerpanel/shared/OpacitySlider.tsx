interface OpacitySliderProps {
  /** Current opacity 0–1 */
  value: number;
  onChange: (v: number) => void;
  /** Tailwind accent color name, e.g. "emerald", "violet" */
  color?: string;
  label?: string;
}

export default function OpacitySlider({
  value,
  onChange,
  color = "primary",
  label = "Opacity",
}: OpacitySliderProps) {
  const pct = Math.round(value * 100);

  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] text-slate-500">{label}</span>
      <input
        type="range"
        min={0}
        max={100}
        value={pct}
        onChange={(e) => onChange(parseInt(e.target.value) / 100)}
        className={`flex-1 h-1 accent-${color}-500 cursor-pointer`}
        aria-label={`${label} ${pct}%`}
      />
      <span className="text-[9px] text-slate-400 tabular-nums w-7 text-right font-mono">
        {pct}%
      </span>
    </div>
  );
}
