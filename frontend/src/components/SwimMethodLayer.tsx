import { useState, useCallback } from "react";
import type { SwimMethod, DepthRange } from "../api/swim_ice";
import { getSwimMethodTileUrl } from "../api/swim_ice";

/* =========================================================
 * Props
 * =======================================================*/
export interface SwimMethodLayerProps {
  method: SwimMethod;
  depth: DepthRange;
  visible: boolean;
  onToggle: (method: SwimMethod, visible: boolean) => void;
  onOpacityChange?: (method: SwimMethod, opacity: number) => void;
}

/* =========================================================
 * Method display metadata
 * =======================================================*/
const METHOD_META: Record<SwimMethod, { label: string; icon: string }> = {
  neutron: { label: "Neutron", icon: "radio_button_checked" },
  thermal: { label: "Thermal", icon: "thermostat" },
  radar_surface: { label: "Radar Surface", icon: "radar" },
  radar_dielectric: { label: "Radar Dielectric", icon: "sensors" },
  geomorphic: { label: "Geomorphic", icon: "terrain" },
};

/* =========================================================
 * Component
 * =======================================================*/
export default function SwimMethodLayer({
  method,
  depth,
  visible,
  onToggle,
  onOpacityChange,
}: SwimMethodLayerProps) {
  const [opacity, setOpacity] = useState(0.7);
  const meta = METHOD_META[method];
  const tileUrl = getSwimMethodTileUrl(method, depth);

  const handleOpacity = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = parseFloat(e.target.value);
      setOpacity(val);
      onOpacityChange?.(method, val);
    },
    [method, onOpacityChange],
  );

  return (
    <div className="flex flex-col gap-1 rounded border border-[#232f48] bg-[#111b2a] px-2 py-1.5">
      {/* Header row: checkbox + label */}
      <label className="flex cursor-pointer items-center gap-1.5">
        <input
          type="checkbox"
          checked={visible}
          onChange={(e) => onToggle(method, e.target.checked)}
          className="h-3 w-3 rounded border-[#232f48] bg-[#0d1520] accent-blue-500"
        />
        <span className="material-symbols-outlined text-[14px] text-[#92a4c9]">
          {meta.icon}
        </span>
        <span className="text-[11px] font-medium text-slate-200">
          {meta.label}
        </span>
      </label>

      {/* Opacity slider — only when visible */}
      {visible && (
        <div className="flex items-center gap-1.5 pl-5">
          <span className="text-[9px] text-[#92a4c9]">Opacity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={handleOpacity}
            className="h-1 w-full cursor-pointer accent-blue-500"
          />
          <span className="min-w-[26px] text-right font-mono text-[9px] text-slate-400">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      {/* Hidden data attribute for tile URL (consumed by map integration) */}
      {visible && <data value={tileUrl} data-swim-tile={method} className="hidden" />}
    </div>
  );
}
