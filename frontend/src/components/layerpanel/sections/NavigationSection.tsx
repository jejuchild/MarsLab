import { useState, useEffect } from "react";
import type { NavigationSectionProps } from "../types";
import { lp } from "../tokens";
import CollapsibleSection from "../shared/CollapsibleSection";
import {
  normalizeLonForMap,
  clampLatitude,
  parseCoordinate,
} from "../../../utils/coordinates";

// ── FlyToInput (internal) ──
function FlyToInput({
  onFlyToCoords,
}: {
  onFlyToCoords?: (lat: number, lon: number) => void;
}) {
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");

  const handleFlyTo = () => {
    const latNum = parseCoordinate(lat);
    const lonNum = parseCoordinate(lon);

    if (latNum === null || lonNum === null) return;

    // Clamp latitude and normalize longitude for map display
    const clampedLat = clampLatitude(latNum);
    const normalizedLon = normalizeLonForMap(lonNum);

    onFlyToCoords?.(clampedLat, normalizedLon);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleFlyTo();
    }
  };

  return (
    <div className="mt-3 pt-3 border-t border-[#232f48]">
      <h4 className={`${lp.h4} mb-2`}>Fly To Location</h4>
      <div className="flex gap-2 items-end">
        <div className="flex-1">
          <label className={`${lp.caption} block mb-1`}>Lat (°)</label>
          <input
            type="number"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="0"
            aria-label="Latitude"
            className={lp.input}
          />
        </div>
        <div className="flex-1">
          <label className={`${lp.caption} block mb-1`}>Lon (°)</label>
          <input
            type="number"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="0"
            aria-label="Longitude"
            className={lp.input}
          />
        </div>
        <button
          onClick={handleFlyTo}
          aria-label="Fly to entered coordinates"
          className={`${lp.btnPrimary} whitespace-nowrap`}
        >
          Fly To
        </button>
      </div>
      <p className={`${lp.tiny} mt-1.5`}>
        Longitude values beyond ±180° are normalized automatically
      </p>
    </div>
  );
}

// ── ViewBoundsInput (internal) ──
function ViewBoundsInput({
  viewBounds,
  onViewBoundsChange,
  selectionMode,
  onSelectionModeChange,
}: {
  viewBounds: NavigationSectionProps["viewBounds"];
  onViewBoundsChange: NavigationSectionProps["onViewBoundsChange"];
  selectionMode?: boolean;
  onSelectionModeChange?: (active: boolean) => void;
}) {
  const [minLat, setMinLat] = useState(viewBounds?.minLat?.toString() ?? "");
  const [maxLat, setMaxLat] = useState(viewBounds?.maxLat?.toString() ?? "");
  const [westLon, setWestLon] = useState(viewBounds?.westLon?.toString() ?? "");
  const [eastLon, setEastLon] = useState(viewBounds?.eastLon?.toString() ?? "");

  // Sync local state when viewBounds changes (e.g., from map selection)
  useEffect(() => {
    if (viewBounds) {
      setMinLat(viewBounds.minLat.toFixed(2));
      setMaxLat(viewBounds.maxLat.toFixed(2));
      setWestLon(viewBounds.westLon.toFixed(2));
      setEastLon(viewBounds.eastLon.toFixed(2));
    }
  }, [viewBounds]);

  const handleApply = () => {
    const min = parseFloat(minLat);
    const max = parseFloat(maxLat);
    const west = parseFloat(westLon);
    const east = parseFloat(eastLon);

    if (!isNaN(min) && !isNaN(max) && !isNaN(west) && !isNaN(east)) {
      onViewBoundsChange({ minLat: min, maxLat: max, westLon: west, eastLon: east });
    }
  };

  const handleClear = () => {
    setMinLat("");
    setMaxLat("");
    setWestLon("");
    setEastLon("");
    onViewBoundsChange(null);
  };

  const handleSetViewBound = () => {
    onSelectionModeChange?.(!selectionMode);
  };

  return (
    <div className="mt-4 pt-3 border-t border-[#232f48]">
      <div className="flex items-center justify-between mb-2">
        <h4 className={lp.h4}>View Bounds</h4>
        {viewBounds && (
          <span className="text-[8px] px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30 font-bold uppercase">
            SET
          </span>
        )}
      </div>

      {/* Set View Bound Button */}
      <button
        onClick={handleSetViewBound}
        aria-label={selectionMode ? "Cancel view bound selection" : "Set view bound on map"}
        className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded transition-colors mb-3 ${
          selectionMode
            ? "bg-amber-500/20 border border-amber-500/50 text-amber-400"
            : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-primary/30"
        }`}
      >
        <span className="material-symbols-outlined text-sm">
          {selectionMode ? "close" : "select_all"}
        </span>
        <span className={lp.body}>
          {selectionMode ? "Cancel Selection" : "Set View Bound"}
        </span>
      </button>

      {selectionMode && (
        <p className="text-[9px] text-amber-400 mb-3 flex items-center gap-1">
          <span className="material-symbols-outlined text-xs">info</span>
          Click and drag on the map to select a region
        </p>
      )}

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className={`${lp.caption} block mb-1`}>Min Lat</label>
          <input
            type="number"
            value={minLat}
            onChange={(e) => setMinLat(e.target.value)}
            placeholder="-90"
            aria-label="Minimum latitude"
            className={lp.input}
          />
        </div>
        <div>
          <label className={`${lp.caption} block mb-1`}>Max Lat</label>
          <input
            type="number"
            value={maxLat}
            onChange={(e) => setMaxLat(e.target.value)}
            placeholder="90"
            aria-label="Maximum latitude"
            className={lp.input}
          />
        </div>
        <div>
          <label className={`${lp.caption} block mb-1`}>West Lon</label>
          <input
            type="number"
            value={westLon}
            onChange={(e) => setWestLon(e.target.value)}
            placeholder="-180"
            aria-label="Western longitude"
            className={lp.input}
          />
        </div>
        <div>
          <label className={`${lp.caption} block mb-1`}>East Lon</label>
          <input
            type="number"
            value={eastLon}
            onChange={(e) => setEastLon(e.target.value)}
            placeholder="180"
            aria-label="Eastern longitude"
            className={lp.input}
          />
        </div>
      </div>
      <div className="flex gap-2 mt-2">
        <button
          onClick={handleApply}
          aria-label="Apply view bounds"
          className={`flex-1 ${lp.btnPrimary}`}
        >
          Apply
        </button>
        <button
          onClick={handleClear}
          aria-label="Clear view bounds"
          className={`flex-1 ${lp.btnSecondary}`}
        >
          Clear
        </button>
      </div>
      <p className={`${lp.tiny} mt-2`}>
        Tip: For wrap-around (e.g., 160° to -150°), enter West=160, East=-150
      </p>
    </div>
  );
}

// ── NavigationSection (main export) ──
export default function NavigationSection({
  onFlyToCoords,
  viewBounds,
  onViewBoundsChange,
  viewBoundSelectionMode,
  onViewBoundSelectionModeChange,
  showGrid,
  onToggleGrid,
  showRegionLayer,
  onToggleRegionLayer,
}: NavigationSectionProps) {
  return (
    <CollapsibleSection title="Navigation" icon="explore" defaultOpen storageKey="navigation">
      {/* Fly To */}
      <FlyToInput onFlyToCoords={onFlyToCoords} />

      {/* View Bounds */}
      <ViewBoundsInput
        viewBounds={viewBounds}
        onViewBoundsChange={onViewBoundsChange}
        selectionMode={viewBoundSelectionMode}
        onSelectionModeChange={onViewBoundSelectionModeChange}
      />

      {/* Coordinate Grid */}
      <label
        className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
          showGrid
            ? "bg-slate-500/20 border border-slate-400/50"
            : "bg-[#1a2333] border border-[#232f48] hover:border-slate-400/30"
        }`}
      >
        <input
          type="checkbox"
          checked={showGrid}
          onChange={(e) => onToggleGrid?.(e.target.checked)}
          className={lp.checkbox}
        />
        <span className="material-symbols-outlined text-xs text-slate-400">grid_on</span>
        <span className={`${lp.body} ${showGrid ? "text-slate-300" : "text-[#92a4c9]"}`}>
          Coordinate Grid
        </span>
      </label>

      {/* Region Layer */}
      <label
        className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
          showRegionLayer
            ? "bg-amber-500/20 border border-amber-400/50"
            : "bg-[#1a2333] border border-[#232f48] hover:border-amber-400/30"
        }`}
      >
        <input
          type="checkbox"
          checked={showRegionLayer}
          onChange={(e) => onToggleRegionLayer?.(e.target.checked)}
          className="rounded bg-[#0a0f18] border-[#232f48] text-amber-400 focus:ring-0 focus:ring-offset-0"
        />
        <span className="material-symbols-outlined text-xs text-amber-400">map</span>
        <span className={`${lp.body} ${showRegionLayer ? "text-amber-300" : "text-[#92a4c9]"}`}>
          Named Regions
        </span>
      </label>
    </CollapsibleSection>
  );
}
