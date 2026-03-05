import type { RGBWavelengths } from "../types";

type CRISMBandsTabProps = {
  rgb: RGBWavelengths;
  onChange: (channel: "r" | "g" | "b", value: number) => void;
  onApply: () => void;
  isOverlayActive: boolean;
};

const MIN_WL = 1.0;
const MAX_WL = 4.0;
const STEP = 0.01;

const PRESETS: { label: string; r: number; g: number; b: number }[] = [
  { label: "True Color", r: 2.53, g: 1.51, b: 1.08 },
  { label: "Mineralogy", r: 2.3, g: 1.93, b: 1.08 },
  { label: "Inverted", r: 1.08, g: 1.51, b: 2.53 },
  { label: "Hydration", r: 2.53, g: 1.93, b: 1.51 },
];

export default function CRISMBandsTab({ rgb, onChange, onApply, isOverlayActive }: CRISMBandsTabProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">palette</span>
        <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
          RGB Band Selection
        </h3>
      </div>

      <p className="text-[11px] text-slate-400">
        Select wavelengths (in micrometers) to create an RGB composite from CRISM spectral data.
      </p>

      {/* Channels */}
      <ChannelSlider label="Red" color="red" value={rgb.r} onChange={(v) => onChange("r", v)} />
      <ChannelSlider label="Green" color="green" value={rgb.g} onChange={(v) => onChange("g", v)} />
      <ChannelSlider label="Blue" color="blue" value={rgb.b} onChange={(v) => onChange("b", v)} />

      {/* Presets */}
      <div className="space-y-2">
        <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Presets</h4>
        <div className="grid grid-cols-2 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => { onChange("r", p.r); onChange("g", p.g); onChange("b", p.b); }}
              className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Apply */}
      <button
        onClick={onApply}
        className="w-full py-2.5 rounded-lg bg-primary/20 border border-primary/30 text-primary text-xs font-bold uppercase tracking-widest hover:bg-primary/30 transition-colors"
      >
        <span className="material-symbols-outlined text-sm align-middle mr-1">
          {isOverlayActive ? "refresh" : "add_photo_alternate"}
        </span>
        {isOverlayActive ? "Apply RGB Changes" : "Create RGB Overlay"}
      </button>
    </div>
  );
}

/* ── Channel Slider ── */
function ChannelSlider({
  label,
  color,
  value,
  onChange,
}: {
  label: string;
  color: "red" | "green" | "blue";
  value: number;
  onChange: (v: number) => void;
}) {
  const dotColor = color === "red" ? "bg-red-500" : color === "green" ? "bg-green-500" : "bg-blue-500";
  const textColor = color === "red" ? "text-red-400" : color === "green" ? "text-green-400" : "text-blue-400";
  const accent = color === "red" ? "accent-red-500" : color === "green" ? "accent-green-500" : "accent-blue-500";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-xs font-medium">
          <span className={`inline-block w-3 h-3 rounded-full ${dotColor}`} />
          {label} Channel
        </label>
        <span className={`font-mono text-xs ${textColor}`}>{value.toFixed(2)} μm</span>
      </div>
      <input
        type="range"
        min={MIN_WL}
        max={MAX_WL}
        step={STEP}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`w-full h-2 rounded-lg appearance-none cursor-pointer ${accent} bg-slate-700`}
      />
      <div className="flex justify-between text-[9px] text-slate-600">
        <span>{MIN_WL}</span>
        <span>{MAX_WL}</span>
      </div>
    </div>
  );
}
