import { useState, useMemo, useCallback } from "react";

/* =========================================================
 * Types
 * =======================================================*/
interface BandRatioCalculatorProps {
  wavelengths: number[];
  reflectance: (number | null)[];
  onClose: () => void;
}

/* =========================================================
 * Preset Band Ratio definitions
 * =======================================================*/
interface BandRatioPreset {
  id: string;
  name: string;
  formula: string;
  description: string;
  compute: (getR: (wl: number) => number | null) => number | null;
  interpret: (val: number) => { text: string; color: string };
}

const PRESETS: BandRatioPreset[] = [
  {
    id: "BD1900",
    name: "BD1900 (H2O)",
    formula: "1 - R[1.93] / ((R[1.85] + R[2.07]) / 2)",
    description: "1.9um band depth - indicates hydrated minerals",
    compute: (getR) => {
      const r193 = getR(1.93);
      const r185 = getR(1.85);
      const r207 = getR(2.07);
      if (r193 === null || r185 === null || r207 === null) return null;
      const continuum = (r185 + r207) / 2;
      if (continuum === 0) return null;
      return 1 - r193 / continuum;
    },
    interpret: (val) => {
      if (val > 0.05) return { text: "Strong hydration signature", color: "#4ADE80" };
      if (val > 0.02) return { text: "Moderate hydration signature", color: "#FBBF24" };
      if (val > 0.01) return { text: "Weak hydration signal", color: "#F97316" };
      return { text: "No significant hydration", color: "#94A3B8" };
    },
  },
  {
    id: "BD2100",
    name: "BD2100 (H2O ice)",
    formula: "1 - R[2.12] / ((R[1.93] + R[2.25]) / 2)",
    description: "2.1um band depth - indicates water ice",
    compute: (getR) => {
      const r212 = getR(2.12);
      const r193 = getR(1.93);
      const r225 = getR(2.25);
      if (r212 === null || r193 === null || r225 === null) return null;
      const continuum = (r193 + r225) / 2;
      if (continuum === 0) return null;
      return 1 - r212 / continuum;
    },
    interpret: (val) => {
      if (val > 0.05) return { text: "Strong water ice signature", color: "#00FFFF" };
      if (val > 0.02) return { text: "Moderate water ice signal", color: "#38BDF8" };
      if (val > 0.01) return { text: "Weak water ice hint", color: "#F97316" };
      return { text: "No water ice detected", color: "#94A3B8" };
    },
  },
  {
    id: "OLINDEX",
    name: "OLINDEX (Olivine)",
    formula: "R[1.695] / (0.1*R[1.05] + 0.1*R[1.21] + 0.4*R[1.33] + 0.4*R[1.47])",
    description: "Olivine index - detects olivine-bearing materials",
    compute: (getR) => {
      const r1695 = getR(1.695);
      const r105 = getR(1.05);
      const r121 = getR(1.21);
      const r133 = getR(1.33);
      const r147 = getR(1.47);
      if (r1695 === null || r105 === null || r121 === null || r133 === null || r147 === null) return null;
      const denom = 0.1 * r105 + 0.1 * r121 + 0.4 * r133 + 0.4 * r147;
      if (denom === 0) return null;
      return r1695 / denom;
    },
    interpret: (val) => {
      if (val > 1.15) return { text: "Strong olivine signature", color: "#4ADE80" };
      if (val > 1.05) return { text: "Moderate olivine signal", color: "#FBBF24" };
      if (val > 1.0) return { text: "Marginal olivine hint", color: "#F97316" };
      return { text: "No olivine detected", color: "#94A3B8" };
    },
  },
  {
    id: "SINDEX",
    name: "SINDEX (Sulfate)",
    formula: "1 - (R[2.12] + R[2.40]) / (2 * R[2.29])",
    description: "Sulfate index - detects sulfate minerals",
    compute: (getR) => {
      const r212 = getR(2.12);
      const r240 = getR(2.40);
      const r229 = getR(2.29);
      if (r212 === null || r240 === null || r229 === null) return null;
      if (r229 === 0) return null;
      return 1 - (r212 + r240) / (2 * r229);
    },
    interpret: (val) => {
      if (val > 0.04) return { text: "Strong sulfate signature", color: "#FBBF24" };
      if (val > 0.02) return { text: "Moderate sulfate signal", color: "#F97316" };
      if (val > 0.01) return { text: "Weak sulfate hint", color: "#A78BFA" };
      return { text: "No sulfate detected", color: "#94A3B8" };
    },
  },
  {
    id: "BD1435",
    name: "BD1435 (CO2 ice)",
    formula: "1 - R[1.43] / ((R[1.37] + R[1.47]) / 2)",
    description: "1.435um band depth - indicates CO2 ice",
    compute: (getR) => {
      const r143 = getR(1.43);
      const r137 = getR(1.37);
      const r147 = getR(1.47);
      if (r143 === null || r137 === null || r147 === null) return null;
      const continuum = (r137 + r147) / 2;
      if (continuum === 0) return null;
      return 1 - r143 / continuum;
    },
    interpret: (val) => {
      if (val > 0.03) return { text: "Strong CO2 ice signature", color: "#A78BFA" };
      if (val > 0.015) return { text: "Moderate CO2 ice signal", color: "#38BDF8" };
      if (val > 0.005) return { text: "Weak CO2 ice hint", color: "#F97316" };
      return { text: "No CO2 ice detected", color: "#94A3B8" };
    },
  },
];

/* =========================================================
 * BandRatioCalculator Component
 * =======================================================*/
export default function BandRatioCalculator({
  wavelengths,
  reflectance,
  onClose,
}: BandRatioCalculatorProps) {
  const [mode, setMode] = useState<"preset" | "custom">("preset");
  const [selectedPresetId, setSelectedPresetId] = useState<string>("BD1900");
  const [customNum, setCustomNum] = useState<string>("1.93");
  const [customDen, setCustomDen] = useState<string>("2.30");

  // Binary search to find the nearest wavelength index
  const findNearestIndex = useCallback(
    (targetWl: number): number => {
      if (wavelengths.length === 0) return -1;

      let lo = 0;
      let hi = wavelengths.length - 1;

      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (wavelengths[mid] < targetWl) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }

      // Check if lo-1 is closer
      if (lo > 0 && Math.abs(wavelengths[lo - 1] - targetWl) < Math.abs(wavelengths[lo] - targetWl)) {
        return lo - 1;
      }
      return lo;
    },
    [wavelengths]
  );

  // Get reflectance at a given target wavelength (snaps to nearest band)
  const getReflectanceAt = useCallback(
    (targetWl: number): number | null => {
      const idx = findNearestIndex(targetWl);
      if (idx < 0 || idx >= reflectance.length) return null;
      return reflectance[idx];
    },
    [findNearestIndex, reflectance]
  );

  // Get actual wavelength for a target wavelength (for display)
  const getNearestWavelength = useCallback(
    (targetWl: number): number | null => {
      const idx = findNearestIndex(targetWl);
      if (idx < 0 || idx >= wavelengths.length) return null;
      return wavelengths[idx];
    },
    [findNearestIndex, wavelengths]
  );

  // Compute preset result
  const presetResult = useMemo(() => {
    const preset = PRESETS.find((p) => p.id === selectedPresetId);
    if (!preset) return null;
    const val = preset.compute(getReflectanceAt);
    if (val === null) return null;
    const interp = preset.interpret(val);
    return { value: val, ...interp, preset };
  }, [selectedPresetId, getReflectanceAt]);

  // Compute custom result
  const customResult = useMemo(() => {
    const numWl = parseFloat(customNum);
    const denWl = parseFloat(customDen);
    if (isNaN(numWl) || isNaN(denWl)) return null;

    const numR = getReflectanceAt(numWl);
    const denR = getReflectanceAt(denWl);
    if (numR === null || denR === null || denR === 0) return null;

    const actualNumWl = getNearestWavelength(numWl);
    const actualDenWl = getNearestWavelength(denWl);

    return {
      value: numR / denR,
      numR,
      denR,
      actualNumWl,
      actualDenWl,
    };
  }, [customNum, customDen, getReflectanceAt, getNearestWavelength]);

  const selectedPreset = PRESETS.find((p) => p.id === selectedPresetId);

  return (
    <div className="space-y-3 rounded-lg border border-[#232f48] bg-[#0a0f18] p-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="material-symbols-outlined text-amber-400 text-sm">
            calculate
          </span>
          <h4 className="text-[11px] font-bold uppercase tracking-wider text-amber-400">
            Band Ratios
          </h4>
        </div>
        <button
          onClick={onClose}
          className="p-0.5 rounded text-slate-500 hover:text-white hover:bg-[#232f48] transition-colors"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
            close
          </span>
        </button>
      </div>

      {/* Mode tabs */}
      <div className="flex rounded-md border border-[#232f48] overflow-hidden">
        <button
          onClick={() => setMode("preset")}
          className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-colors ${
            mode === "preset"
              ? "bg-amber-500/20 text-amber-400"
              : "bg-transparent text-slate-500 hover:text-slate-300"
          }`}
        >
          Presets
        </button>
        <button
          onClick={() => setMode("custom")}
          className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-colors border-l border-[#232f48] ${
            mode === "custom"
              ? "bg-amber-500/20 text-amber-400"
              : "bg-transparent text-slate-500 hover:text-slate-300"
          }`}
        >
          Custom
        </button>
      </div>

      {/* Preset mode */}
      {mode === "preset" && (
        <div className="space-y-3">
          {/* Preset selector */}
          <select
            value={selectedPresetId}
            onChange={(e) => setSelectedPresetId(e.target.value)}
            className="w-full rounded border border-[#232f48] bg-[#101622] px-2 py-1.5 text-[11px] text-white focus:border-amber-500/50 focus:outline-none focus:ring-1 focus:ring-amber-500/30"
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          {/* Formula display */}
          {selectedPreset && (
            <div className="rounded border border-[#232f48]/50 bg-[#101622]/60 px-2 py-1.5">
              <div className="text-[9px] uppercase text-slate-500 mb-0.5">Formula</div>
              <div className="font-mono text-[10px] text-slate-300 break-all">
                {selectedPreset.formula}
              </div>
              <div className="text-[9px] text-slate-500 mt-1">
                {selectedPreset.description}
              </div>
            </div>
          )}

          {/* Result */}
          {presetResult ? (
            <div className="rounded border border-[#232f48] bg-[#101622] p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase text-slate-500">Result</span>
                <span className="font-mono text-lg font-bold text-white">
                  {presetResult.value.toFixed(4)}
                </span>
              </div>
              <div
                className="flex items-center gap-1.5 rounded px-2 py-1.5 text-[10px] font-medium"
                style={{
                  backgroundColor: presetResult.color + "15",
                  borderLeft: `3px solid ${presetResult.color}`,
                  color: presetResult.color,
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                  info
                </span>
                {presetResult.text}
              </div>
            </div>
          ) : (
            <div className="rounded border border-[#232f48]/50 bg-[#101622]/60 px-3 py-2 text-[10px] text-slate-500 text-center">
              Insufficient spectral data for this index
            </div>
          )}
        </div>
      )}

      {/* Custom mode */}
      {mode === "custom" && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            {/* Numerator */}
            <div className="space-y-1">
              <label className="text-[9px] uppercase text-slate-500">
                Numerator (um)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.5"
                max="4.5"
                value={customNum}
                onChange={(e) => setCustomNum(e.target.value)}
                className="w-full rounded border border-[#232f48] bg-[#101622] px-2 py-1.5 text-[11px] font-mono text-white focus:border-amber-500/50 focus:outline-none focus:ring-1 focus:ring-amber-500/30"
              />
              {customNum && getNearestWavelength(parseFloat(customNum)) !== null && (
                <div className="text-[9px] text-slate-500">
                  Nearest: {getNearestWavelength(parseFloat(customNum))?.toFixed(4)} um
                </div>
              )}
            </div>
            {/* Denominator */}
            <div className="space-y-1">
              <label className="text-[9px] uppercase text-slate-500">
                Denominator (um)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.5"
                max="4.5"
                value={customDen}
                onChange={(e) => setCustomDen(e.target.value)}
                className="w-full rounded border border-[#232f48] bg-[#101622] px-2 py-1.5 text-[11px] font-mono text-white focus:border-amber-500/50 focus:outline-none focus:ring-1 focus:ring-amber-500/30"
              />
              {customDen && getNearestWavelength(parseFloat(customDen)) !== null && (
                <div className="text-[9px] text-slate-500">
                  Nearest: {getNearestWavelength(parseFloat(customDen))?.toFixed(4)} um
                </div>
              )}
            </div>
          </div>

          <div className="text-[9px] text-slate-500 text-center">
            Result = R[numerator] / R[denominator]
          </div>

          {/* Result */}
          {customResult ? (
            <div className="rounded border border-[#232f48] bg-[#101622] p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase text-slate-500">Ratio</span>
                <span className="font-mono text-lg font-bold text-white">
                  {customResult.value.toFixed(4)}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div className="rounded border border-[#232f48]/50 bg-[#0a0f18] p-1.5">
                  <div className="text-[8px] uppercase text-slate-500">
                    R[{customResult.actualNumWl?.toFixed(3)}]
                  </div>
                  <div className="font-mono text-white">
                    {customResult.numR.toFixed(4)}
                  </div>
                </div>
                <div className="rounded border border-[#232f48]/50 bg-[#0a0f18] p-1.5">
                  <div className="text-[8px] uppercase text-slate-500">
                    R[{customResult.actualDenWl?.toFixed(3)}]
                  </div>
                  <div className="font-mono text-white">
                    {customResult.denR.toFixed(4)}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded border border-[#232f48]/50 bg-[#101622]/60 px-3 py-2 text-[10px] text-slate-500 text-center">
              {parseFloat(customNum) && parseFloat(customDen)
                ? "No valid reflectance data at these wavelengths"
                : "Enter wavelengths to compute ratio"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
