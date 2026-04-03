import { useState, useEffect } from "react";
import type { OverlayType, ProductOverlay, CustomDataset } from "../../../pages/MainPage";
import { OVERLAY_CONFIG } from "../types";

const STORAGE_KEY = "marslab.overlaySection.collapsed";

type OverlayControlsProps = {
  productId: string;
  instrument: string;
  isTRR3: boolean;
  isCRISM: boolean;
  isCustom: boolean;
  activeOverlay: ProductOverlay | null;
  onSetOverlay: (type: OverlayType | null) => void;
  onSetOpacity?: (opacity: number) => void;
  hasHighResData: boolean;
  customDataset?: CustomDataset | null;
  onCustomDatasetOpacity?: (id: string, opacity: number) => void;
};

export default function OverlayControls({
  instrument: _instrument,
  isTRR3,
  isCRISM,
  isCustom,
  activeOverlay,
  onSetOverlay,
  onSetOpacity,
  hasHighResData,
  customDataset,
  onCustomDatasetOpacity,
}: OverlayControlsProps) {
  // Default OPEN (key UX change), but remember user's preference
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  });

  const [showOpacity, setShowOpacity] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(collapsed));
    } catch { /* noop */ }
  }, [collapsed]);

  // Available overlay types for this instrument
  // HiRISE/DTM: no "highres" map overlay — use dedicated "View High-Res Image" viewer instead
  const availableOverlays: OverlayType[] = isTRR3
    ? ["quickview", "mineral_cnn"]
    : isCRISM
      ? ["quickview", "highres", "browse_HYD", "browse_ICE", "browse_IC2", "score_ice", "score_hyd"]
      : ["quickview"];

  const displayOverlays = availableOverlays.filter(
    (type) => type !== "highres" || hasHighResData,
  );

  return (
    <div className="border-t border-border-dark pt-3">
      {/* Section header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center justify-between text-[10px] font-bold uppercase tracking-widest text-slate-400 hover:text-slate-200 transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <span className="material-symbols-outlined text-sm">layers</span>
          Overlays
          {activeOverlay && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[8px] bg-green-500/20 text-green-400 normal-case font-medium">
              {OVERLAY_CONFIG[activeOverlay.type].label}
            </span>
          )}
        </span>
        <span className={`material-symbols-outlined text-xs transition-transform duration-200 ${collapsed ? "" : "rotate-180"}`}>
          expand_more
        </span>
      </button>

      {/* Content */}
      <div className={`overflow-hidden transition-all duration-200 ${collapsed ? "max-h-0 opacity-0" : "max-h-[500px] opacity-100"}`}>
        <div className="mt-3 space-y-3">
          {/* Custom dataset opacity */}
          {isCustom && customDataset && (
            <div className="space-y-2">
              <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                Overlay Opacity
              </h4>
              <OpacitySlider
                value={customDataset.opacity}
                onChange={(v) => onCustomDatasetOpacity?.(customDataset.id, v)}
                color="fuchsia"
              />
            </div>
          )}

          {/* Standard overlay controls */}
          {!isCustom && (
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                  Overlay
                </h4>
                {activeOverlay && (
                  <button
                    onClick={() => setShowOpacity(!showOpacity)}
                    className={`p-1 rounded transition-colors ${
                      showOpacity ? "text-primary bg-primary/20" : "text-slate-500 hover:text-slate-300"
                    }`}
                    title="Adjust opacity"
                  >
                    <span className="material-symbols-outlined text-sm">opacity</span>
                  </button>
                )}
              </div>

              {/* Overlay type list */}
              <div className="space-y-1">
                {displayOverlays.map((type) => {
                  const config = OVERLAY_CONFIG[type];
                  const isActive = activeOverlay?.type === type;
                  const isDisabled = type === "highres" && !hasHighResData;

                  return (
                    <button
                      key={type}
                      onClick={() => {
                        if (isDisabled) return;
                        onSetOverlay(isActive ? null : type);
                      }}
                      disabled={isDisabled}
                      className={`flex w-full items-center gap-2.5 px-3 py-2 rounded-lg text-[11px] font-medium transition-all ${
                        isDisabled
                          ? "bg-slate-800/50 text-slate-600 cursor-not-allowed"
                          : isActive
                            ? config.activeClass
                            : "bg-surface-dark/60 border border-border-dark text-slate-400 hover:text-white hover:border-slate-500"
                      }`}
                      title={isDisabled ? "No high-res data available" : config.description}
                    >
                      <span className="material-symbols-outlined text-sm">
                        {isActive ? "check_circle" : config.icon}
                      </span>
                      <span className="flex-1 text-left">{config.label}</span>
                      <span className="text-[9px] opacity-60 max-w-[120px] truncate hidden sm:inline">
                        {config.description}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Opacity slider */}
              {activeOverlay && showOpacity && (
                <OpacitySlider
                  value={activeOverlay.opacity}
                  onChange={(v) => onSetOpacity?.(v)}
                  color="primary"
                />
              )}

              {/* Active overlay indicator */}
              {activeOverlay && (
                <div className="flex items-center justify-between px-3 py-2 bg-green-500/10 rounded-lg border border-green-500/30">
                  <span className="text-[10px] text-green-400 flex items-center gap-1.5">
                    <span className="material-symbols-outlined text-xs">check_circle</span>
                    {OVERLAY_CONFIG[activeOverlay.type].label} overlay active
                  </span>
                  <button
                    onClick={() => onSetOverlay(null)}
                    className="text-[9px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                  >
                    Turn Off
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Opacity Slider ── */
function OpacitySlider({
  value,
  onChange,
  color,
}: {
  value: number;
  onChange: (v: number) => void;
  color: string;
}) {
  const thumbClass = color === "primary"
    ? "[&::-webkit-slider-thumb]:bg-primary"
    : `[&::-webkit-slider-thumb]:bg-${color}-400`;

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-[#0a0f18] rounded-lg border border-[#232f48]">
      <span className="text-[9px] text-[#6b7c9c] uppercase w-12">Opacity</span>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:h-3
          [&::-webkit-slider-thumb]:w-3
          [&::-webkit-slider-thumb]:rounded-full
          ${thumbClass}
          [&::-webkit-slider-thumb]:cursor-pointer`}
      />
      <span className="text-[10px] text-white font-mono w-8 text-right">{value}%</span>
    </div>
  );
}
