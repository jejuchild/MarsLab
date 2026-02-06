/**
 * DTMHoverReadout - Displays elevation info on hover
 *
 * Performance: Uses refs for position updates to avoid re-renders
 */

import { forwardRef, useImperativeHandle, useRef } from "react";

export interface DTMHoverReadoutHandle {
  update: (lat: number, lon: number, elevation: number | null, productId: string) => void;
  hide: () => void;
  show: () => void;
}

interface DTMHoverReadoutProps {
  mode?: "hover" | "click";
  onModeChange?: (mode: "hover" | "click") => void;
}

const DTMHoverReadout = forwardRef<DTMHoverReadoutHandle, DTMHoverReadoutProps>(
  ({ mode = "hover", onModeChange }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const latRef = useRef<HTMLSpanElement>(null);
    const lonRef = useRef<HTMLSpanElement>(null);
    const elevRef = useRef<HTMLSpanElement>(null);
    const productRef = useRef<HTMLSpanElement>(null);

    useImperativeHandle(ref, () => ({
      update: (lat: number, lon: number, elevation: number | null, productId: string) => {
        if (latRef.current) latRef.current.textContent = lat.toFixed(5) + "°";
        if (lonRef.current) lonRef.current.textContent = lon.toFixed(5) + "°";
        if (elevRef.current) {
          elevRef.current.textContent =
            elevation !== null ? elevation.toFixed(1) + " m" : "—";
        }
        if (productRef.current) productRef.current.textContent = productId;
        if (containerRef.current) containerRef.current.style.display = "flex";
      },
      hide: () => {
        if (containerRef.current) containerRef.current.style.display = "none";
      },
      show: () => {
        if (containerRef.current) containerRef.current.style.display = "flex";
      },
    }));

    return (
      <div
        ref={containerRef}
        className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 pointer-events-auto"
        style={{ display: "none" }}
      >
        <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-[#0a0f18]/90 border border-amber-600/30 backdrop-blur-sm shadow-lg">
          {/* DTM indicator */}
          <div className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-amber-600 text-sm">terrain</span>
            <span
              ref={productRef}
              className="text-[9px] font-mono text-amber-600/80 max-w-[120px] truncate"
            >
              —
            </span>
          </div>

          <div className="w-px h-4 bg-[#232f48]" />

          {/* Coordinates */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-[#6b7c9c] uppercase">Lat</span>
              <span ref={latRef} className="text-[11px] font-mono text-white">
                —
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-[#6b7c9c] uppercase">Lon</span>
              <span ref={lonRef} className="text-[11px] font-mono text-white">
                —
              </span>
            </div>
          </div>

          <div className="w-px h-4 bg-[#232f48]" />

          {/* Elevation */}
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-[#6b7c9c] uppercase">Elev</span>
            <span ref={elevRef} className="text-[11px] font-mono text-amber-400 font-bold">
              —
            </span>
          </div>

          <div className="w-px h-4 bg-[#232f48]" />

          {/* Mode toggle */}
          <button
            onClick={() => onModeChange?.(mode === "hover" ? "click" : "hover")}
            className={`px-2 py-0.5 text-[8px] font-bold uppercase rounded transition-colors ${
              mode === "hover"
                ? "bg-amber-600/20 text-amber-400 border border-amber-600/30"
                : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
            }`}
            title={mode === "hover" ? "Switch to click mode" : "Switch to hover mode"}
          >
            {mode === "hover" ? "Hover" : "Click"}
          </button>
        </div>
      </div>
    );
  }
);

DTMHoverReadout.displayName = "DTMHoverReadout";

export default DTMHoverReadout;
