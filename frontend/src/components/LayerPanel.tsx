import { useState, useMemo, useEffect, useCallback } from "react";
import type { VisibleProduct, BaseLayerType, BoundingBox, MapMode, ActiveOverlays, OverlayType, ProductOverlay, CustomDataset, OverlapFilter } from "../pages/MainPage";
import { normalizeLonForMap, clampLatitude, parseCoordinate } from "../utils/coordinates";
import type { FieldNote } from "../api/fieldnotes";
import type { OverlapStats } from "../utils/overlapFilter";
import { INSTRUMENTS, INSTRUMENT_GROUPS, type InstrumentId } from "../config/instrumentRegistry";

type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";
type InstrumentVisibility = Record<InstrumentId, boolean>;
type FootprintCount = { count: number; truncated: boolean; total: number } | null;

/**
 * Extract CRISM observation ID from full product ID
 * e.g., "frt0001fd76_07_if166j_mtr3" -> "frt0001fd76"
 */
function extractCrismObsId(productId: string): string {
  const match = productId.match(/^([a-z]{3}[0-9a-f]{8})/i);
  return match ? match[1].toLowerCase() : productId.toLowerCase();
}

// Static Tailwind class lookup per instrument (avoids dynamic class issues)
const INST_STYLES: Record<InstrumentId, {
  text: string; bgActive: string; checkbox: string;
  btn: string; btnLoading: string;
}> = {
  crism:         { text: "text-cyan-400",   bgActive: "bg-cyan-500/15 border border-cyan-500/40",   checkbox: "text-cyan-500",   btn: "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30",     btnLoading: "bg-cyan-500/10 text-cyan-400/50 border border-cyan-500/20 cursor-wait" },
  hirise:        { text: "text-yellow-400", bgActive: "bg-yellow-500/15 border border-yellow-500/40", checkbox: "text-yellow-500", btn: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30", btnLoading: "bg-yellow-500/10 text-yellow-400/50 border border-yellow-500/20 cursor-wait" },
  sharad:        { text: "text-orange-400", bgActive: "bg-orange-500/15 border border-orange-500/40", checkbox: "text-orange-500", btn: "bg-orange-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30", btnLoading: "bg-orange-500/10 text-orange-400/50 border border-orange-500/20 cursor-wait" },
  sharad_highres:{ text: "text-amber-400",  bgActive: "bg-amber-500/15 border border-amber-500/40",  checkbox: "text-amber-500",  btn: "bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30",   btnLoading: "bg-amber-500/10 text-amber-400/50 border border-amber-500/20 cursor-wait" },
  ctx:           { text: "text-pink-400",   bgActive: "bg-pink-500/15 border border-pink-500/40",   checkbox: "text-pink-500",   btn: "bg-pink-500/20 text-pink-400 border border-pink-500/30 hover:bg-pink-500/30",     btnLoading: "bg-pink-500/10 text-pink-400/50 border border-pink-500/20 cursor-wait" },
  hirise_dtm:    { text: "text-amber-600",  bgActive: "bg-amber-700/15 border border-amber-700/40",  checkbox: "text-amber-600",  btn: "bg-amber-700/20 text-amber-600 border border-amber-700/30 hover:bg-amber-700/30",   btnLoading: "bg-amber-700/10 text-amber-600/50 border border-amber-700/20 cursor-wait" },
  crism_trr3:    { text: "text-teal-400",   bgActive: "bg-teal-500/15 border border-teal-500/40",   checkbox: "text-teal-500",   btn: "bg-teal-500/20 text-teal-400 border border-teal-500/30 hover:bg-teal-500/30",     btnLoading: "bg-teal-500/10 text-teal-400/50 border border-teal-500/20 cursor-wait" },
};

// localStorage key for panel collapse state
const PANEL_COLLAPSED_KEY = "marslab-layer-panel-collapsed";

// Overlay type display names
const OVERLAY_LABELS: Record<OverlayType, { short: string; full: string; color: string }> = {
  quickview: { short: "QV", full: "Quickview", color: "emerald" },
  highres: { short: "HD", full: "High-Res", color: "purple" },
  browse_HYD: { short: "HYD", full: "Hydrated Minerals", color: "fuchsia" },
  browse_ICE: { short: "ICE", full: "Water Ice", color: "blue" },
  browse_IC2: { short: "IC2", full: "CO2 Ice", color: "cyan" },
  score_ice: { short: "S-ICE", full: "Ice Score", color: "sky" },
  score_hyd: { short: "S-HYD", full: "Hydration Score", color: "rose" },
  mineral_cnn: { short: "MIN", full: "Mineral CNN", color: "amber" },
};

// Multi-Instrument Overlap Filter Section Component
function OverlapFilterSection({
  filter,
  onChange,
  stats,
}: {
  filter?: OverlapFilter;
  onChange?: (filter: OverlapFilter) => void;
  stats?: OverlapStats | null;
}) {
  const currentFilter = filter ?? { enabled: false, instruments: [] };

  const handleToggle = () => {
    onChange?.({ ...currentFilter, enabled: !currentFilter.enabled });
  };

  return (
    <div className="p-4 border-b border-[#232f48]">
      <div className="flex items-center justify-between">
        <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="material-symbols-outlined text-xs">filter_alt</span>
          Overlap Filter
        </h3>
        <div className="flex items-center gap-2">
          {currentFilter.enabled && stats && stats.totalChecked > 0 && (
            <span className="text-sky-400 text-[10px] font-mono">{stats.totalPassing}/{stats.totalChecked}</span>
          )}
          <button
            onClick={handleToggle}
            className={`text-[8px] px-1.5 py-0.5 rounded font-bold uppercase transition-colors ${
              currentFilter.enabled
                ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48] hover:border-[#3a4a68]"
            }`}
          >
            {currentFilter.enabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      {currentFilter.enabled && (
        <div className="mt-2 space-y-1.5">
          <p className="text-[8px] text-[#6b7c9c] leading-relaxed">
            Showing only products that overlap with at least one product from another instrument.
          </p>

          {/* Per-instrument stats */}
          {stats && stats.totalChecked > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Array.from(stats.perInstrument.entries()).map(([inst, s]) => (
                s.checked > 0 && (
                  <span key={inst} className="text-[9px] text-[#6b7c9c] font-mono bg-[#1a2333] px-1.5 py-0.5 rounded border border-[#232f48]">
                    {inst}: {s.passing}/{s.checked}
                  </span>
                )
              ))}
            </div>
          )}

          {/* No overlap message */}
          {stats && stats.totalPassing === 0 && stats.totalChecked > 0 && (
            <p className="text-[9px] text-orange-400/80 flex items-center gap-1">
              <span className="material-symbols-outlined text-xs">warning</span>
              No overlapping regions found
            </p>
          )}

          {/* No footprints loaded */}
          {(!stats || stats.totalChecked === 0) && (
            <p className="text-[9px] text-[#6b7c9c] flex items-center gap-1">
              <span className="material-symbols-outlined text-xs">info</span>
              Load footprints for at least 2 instruments
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// Ice Score Filter state
export interface IceScoreFilter {
  enabled: boolean;
  minScore: number;     // threshold m (0.05 - 1.5)
  minPercent: number;   // percentage n (0 - 100)
}

// Available thresholds (must match backend)
const SCORE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5];

// Ice Score Filter Section Component
function IceScoreFilterSection({
  filter,
  onChange,
  filteredCount,
  totalLoaded,
}: {
  filter?: IceScoreFilter;
  onChange?: (filter: IceScoreFilter) => void;
  filteredCount: number | null;
  totalLoaded: number;  // Total loaded CRISM footprints
}) {
  const [isCollapsed, setIsCollapsed] = useState(true);

  // Default filter values
  const currentFilter = filter ?? { enabled: false, minScore: 0.3, minPercent: 5 };

  const handleToggle = () => {
    onChange?.({ ...currentFilter, enabled: !currentFilter.enabled });
  };

  const handleScoreChange = (value: number) => {
    onChange?.({ ...currentFilter, minScore: value });
  };

  const handlePercentChange = (value: number) => {
    onChange?.({ ...currentFilter, minPercent: value });
  };

  // Find closest threshold index for slider
  const scoreIndex = SCORE_THRESHOLDS.findIndex(t => t >= currentFilter.minScore);
  const currentScoreIndex = scoreIndex === -1 ? SCORE_THRESHOLDS.length - 1 : scoreIndex;

  return (
    <div className="p-4 border-b border-[#232f48]">
      {/* Header */}
      <div
        className="flex items-center justify-between mb-2 cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="material-symbols-outlined text-xs">
            {isCollapsed ? "expand_more" : "expand_less"}
          </span>
          Ice Score Filter
        </h3>
        <div className="flex items-center gap-2">
          {currentFilter.enabled && filteredCount !== null && totalLoaded > 0 && (
            <span className="text-sky-400 text-[10px] font-mono">{filteredCount}/{totalLoaded}</span>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleToggle();
            }}
            className={`text-[8px] px-1.5 py-0.5 rounded font-bold uppercase transition-colors ${
              currentFilter.enabled
                ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48] hover:border-[#3a4a68]"
            }`}
          >
            {currentFilter.enabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      {!isCollapsed && (
        <div className="space-y-4">
          {/* Min Score Threshold */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-[9px] text-[#6b7c9c] uppercase">Min Ice Score</label>
              <span className="text-[11px] text-white font-mono">{currentFilter.minScore.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max={SCORE_THRESHOLDS.length - 1}
              value={currentScoreIndex}
              onChange={(e) => handleScoreChange(SCORE_THRESHOLDS[Number(e.target.value)])}
              disabled={!currentFilter.enabled}
              className={`w-full h-1 rounded-lg appearance-none cursor-pointer
                ${currentFilter.enabled ? "bg-[#232f48]" : "bg-[#1a2333] opacity-50"}
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-3
                [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-sky-400
                [&::-webkit-slider-thumb]:cursor-pointer`}
            />
            <div className="flex justify-between text-[8px] text-[#4a5a7c] mt-0.5">
              <span>0.05</span>
              <span>0.5</span>
              <span>1.0</span>
              <span>1.5</span>
            </div>
          </div>

          {/* Min Percentage */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-[9px] text-[#6b7c9c] uppercase">Min % of Pixels</label>
              <span className="text-[11px] text-white font-mono">{currentFilter.minPercent}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="50"
              value={currentFilter.minPercent}
              onChange={(e) => handlePercentChange(Number(e.target.value))}
              disabled={!currentFilter.enabled}
              className={`w-full h-1 rounded-lg appearance-none cursor-pointer
                ${currentFilter.enabled ? "bg-[#232f48]" : "bg-[#1a2333] opacity-50"}
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-3
                [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-sky-400
                [&::-webkit-slider-thumb]:cursor-pointer`}
            />
            <div className="flex justify-between text-[8px] text-[#4a5a7c] mt-0.5">
              <span>0%</span>
              <span>25%</span>
              <span>50%</span>
            </div>
          </div>

          {/* Description */}
          <p className="text-[8px] text-[#6b7c9c] leading-relaxed">
            Show CRISM products where at least <span className="text-sky-400 font-bold">{currentFilter.minPercent}%</span> of
            pixels have ice score ≥ <span className="text-sky-400 font-bold">{currentFilter.minScore.toFixed(2)}</span>
          </p>
        </div>
      )}
    </div>
  );
}

interface LayerPanelProps {
  // Map mode (2D/3D)
  mapMode: MapMode;
  onMapModeChange: (mode: MapMode) => void;

  // Base layer selection
  baseLayer: BaseLayerType;
  onBaseLayerChange: (layer: BaseLayerType) => void;

  // View bounds restriction
  viewBounds: BoundingBox;
  onViewBoundsChange: (bounds: BoundingBox) => void;

  // Footprint visibility (unified)
  instrumentVisibility: InstrumentVisibility;
  onToggleInstrument: (id: InstrumentId, v: boolean) => void;

  // Explicit footprint loading
  onLoadFootprints?: (instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3") => void;
  footprintsLoading?: Record<string, boolean>;
  footprintCounts?: Record<string, FootprintCount>;

  // Ice Score Filter
  iceScoreFilter?: IceScoreFilter;
  onIceScoreFilterChange?: (filter: IceScoreFilter) => void;
  filteredProductIds?: Set<string> | null;  // null = not filtering, Set = filtered IDs

  // Product data
  visibleProducts?: VisibleProduct[];
  activeOverlays?: ActiveOverlays;

  // Unified overlay handlers
  onSetOverlay?: (productId: string, type: OverlayType | null) => void;
  onSetOpacity?: (productId: string, opacity: number) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
  onFlyToProduct?: (productId: string) => void;
  onDeactivateAll?: () => void;

  // Custom datasets
  showCustomData: boolean;
  onToggleCustomData: (v: boolean) => void;
  onLoadCustomData?: () => void;
  customDataLoading?: boolean;
  customDatasets?: CustomDataset[];
  onCustomDatasetToggle?: (id: string, visible: boolean) => void;

  // Analysis mode
  analysisMode?: "slope" | "hirise_dtm_3d" | "line" | "ai_analysis" | "agentic" | "report" | "guided" | "region_stats" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column" | null;
  onAnalysisModeChange?: (mode: "slope" | "hirise_dtm_3d" | "line" | "ai_analysis" | "agentic" | "report" | "guided" | "region_stats" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column" | null) => void;

  // Fly-To navigation
  onFlyToCoords?: (lat: number, lon: number) => void;

  // View bound selection mode
  viewBoundSelectionMode?: boolean;
  onViewBoundSelectionModeChange?: (active: boolean) => void;

  // Region Dashboard
  onShowRegionDashboard?: () => void;

  // Coordinate grid
  showGrid?: boolean;
  onToggleGrid?: (v: boolean) => void;

  // Region layer
  showRegionLayer?: boolean;
  onToggleRegionLayer?: (v: boolean) => void;

  // Field Notes
  fieldNotes?: FieldNote[];
  showFieldNotesOnMap?: boolean;
  onToggleFieldNotesOnMap?: (v: boolean) => void;
  onFieldNoteClick?: (note: FieldNote) => void;
  onActiveTagChange?: (tag: string | null) => void;

  // Multi-Instrument Overlap Filter
  overlapFilter?: OverlapFilter;
  onOverlapFilterChange?: (filter: OverlapFilter) => void;
  overlapStats?: OverlapStats | null;

  // Measurement Tools
  showMeasurementTools?: boolean;
  onToggleMeasurementTools?: (v: boolean) => void;

  // Mobile mode
  isMobile?: boolean;
}

// Fly-To Navigation Component
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
      <h4 className="text-[#92a4c9] text-[9px] font-bold uppercase tracking-widest mb-2">
        Fly To Location
      </h4>
      <div className="flex gap-2 items-end">
        <div className="flex-1">
          <label className="text-[9px] text-[#6b7c9c] block mb-1">Lat (°)</label>
          <input
            type="number"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="0"
            className="w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none"
          />
        </div>
        <div className="flex-1">
          <label className="text-[9px] text-[#6b7c9c] block mb-1">Lon (°)</label>
          <input
            type="number"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="0"
            className="w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none"
          />
        </div>
        <button
          onClick={handleFlyTo}
          className="px-3 py-1 text-[10px] font-medium bg-primary/20 border border-primary/50 rounded text-primary hover:bg-primary/30 transition-colors whitespace-nowrap"
        >
          Fly To
        </button>
      </div>
      <p className="text-[8px] text-[#6b7c9c] mt-1.5">
        Longitude values beyond ±180° are normalized automatically
      </p>
    </div>
  );
}

// View Bounds Input Component
function ViewBoundsInput({
  viewBounds,
  onViewBoundsChange,
  selectionMode,
  onSelectionModeChange,
}: {
  viewBounds: BoundingBox;
  onViewBoundsChange: (bounds: BoundingBox) => void;
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
        <h4 className="text-[#92a4c9] text-[9px] font-bold uppercase tracking-widest">
          View Bounds
        </h4>
        {viewBounds && (
          <span className="text-[8px] px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30 font-bold uppercase">
            SET
          </span>
        )}
      </div>

      {/* Set View Bound Button */}
      <button
        onClick={handleSetViewBound}
        className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded transition-colors mb-3 ${
          selectionMode
            ? "bg-amber-500/20 border border-amber-500/50 text-amber-400"
            : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-primary/30"
        }`}
      >
        <span className="material-symbols-outlined text-sm">
          {selectionMode ? "close" : "select_all"}
        </span>
        <span className="text-[11px] font-medium">
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
          <label className="text-[9px] text-[#6b7c9c] block mb-1">Min Lat</label>
          <input
            type="number"
            value={minLat}
            onChange={(e) => setMinLat(e.target.value)}
            placeholder="-90"
            className="w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none"
          />
        </div>
        <div>
          <label className="text-[9px] text-[#6b7c9c] block mb-1">Max Lat</label>
          <input
            type="number"
            value={maxLat}
            onChange={(e) => setMaxLat(e.target.value)}
            placeholder="90"
            className="w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none"
          />
        </div>
        <div>
          <label className="text-[9px] text-[#6b7c9c] block mb-1">West Lon</label>
          <input
            type="number"
            value={westLon}
            onChange={(e) => setWestLon(e.target.value)}
            placeholder="-180"
            className="w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none"
          />
        </div>
        <div>
          <label className="text-[9px] text-[#6b7c9c] block mb-1">East Lon</label>
          <input
            type="number"
            value={eastLon}
            onChange={(e) => setEastLon(e.target.value)}
            placeholder="180"
            className="w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none"
          />
        </div>
      </div>
      <div className="flex gap-2 mt-2">
        <button
          onClick={handleApply}
          className="flex-1 px-2 py-1 text-[10px] font-medium bg-primary/20 border border-primary/50 rounded text-primary hover:bg-primary/30 transition-colors"
        >
          Apply
        </button>
        <button
          onClick={handleClear}
          className="flex-1 px-2 py-1 text-[10px] font-medium bg-[#1a2333] border border-[#232f48] rounded text-[#92a4c9] hover:border-[#3a4a68] transition-colors"
        >
          Clear
        </button>
      </div>
      <p className="text-[8px] text-[#6b7c9c] mt-2">
        Tip: For wrap-around (e.g., 160° to -150°), enter West=160, East=-150
      </p>
    </div>
  );
}

export default function LayerPanel({
  // Map mode
  mapMode,
  onMapModeChange,
  // Base layer
  baseLayer,
  onBaseLayerChange,
  // View bounds
  viewBounds,
  onViewBoundsChange,
  // Footprints (unified)
  instrumentVisibility,
  onToggleInstrument,
  // Explicit footprint loading
  onLoadFootprints,
  footprintsLoading = {},
  footprintCounts = {},
  // Ice Score Filter
  iceScoreFilter,
  onIceScoreFilterChange,
  filteredProductIds,
  // Data
  visibleProducts = [],
  activeOverlays = new Map(),
  // Handlers
  onSetOverlay,
  onSetOpacity,
  onSelectProduct,
  onFlyToProduct,
  onDeactivateAll,
  // Custom datasets
  showCustomData,
  onToggleCustomData,
  onLoadCustomData,
  customDataLoading = false,
  customDatasets = [],
  onCustomDatasetToggle,
  // Analysis mode
  analysisMode = null,
  onAnalysisModeChange,
  // Region Dashboard
  onShowRegionDashboard,
  // Fly-To navigation
  onFlyToCoords,
  // View bound selection mode
  viewBoundSelectionMode = false,
  onViewBoundSelectionModeChange,
  // Coordinate grid
  showGrid = false,
  onToggleGrid,
  // Region layer
  showRegionLayer = false,
  onToggleRegionLayer,
  // Field Notes
  fieldNotes = [],
  showFieldNotesOnMap = true,
  onToggleFieldNotesOnMap,
  onFieldNoteClick,
  onActiveTagChange,
  // Overlap Filter
  overlapFilter,
  onOverlapFilterChange,
  overlapStats,
  showMeasurementTools = false,
  onToggleMeasurementTools,
  isMobile = false,
}: LayerPanelProps) {
  // Panel collapse state - initialize from localStorage
  const [isCollapsed, setIsCollapsed] = useState(() => {
    try {
      const stored = localStorage.getItem(PANEL_COLLAPSED_KEY);
      return stored === "true";
    } catch {
      return false;
    }
  });

  // Persist collapse state to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(PANEL_COLLAPSED_KEY, String(isCollapsed));
    } catch {
      // Ignore localStorage errors
    }
  }, [isCollapsed]);

  // Panel width (resizable)
  const [panelWidth, setPanelWidth] = useState(320);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientX - startX;
      const maxW = Math.floor(window.innerWidth * 0.5);
      setPanelWidth(Math.max(200, Math.min(maxW, startW + delta)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [panelWidth]);

  // Count active overlays
  const totalActiveOverlays = activeOverlays.size;

  // Compute count of VISIBLE CRISM products that pass the ice score filter
  // This shows how many loaded footprints pass, not the total in database
  const filteredVisibleCount = useMemo(() => {
    if (!filteredProductIds || !iceScoreFilter?.enabled) return null;

    // Count visible CRISM products that pass the filter
    const crismProducts = visibleProducts.filter(p => p.instrument === "CRISM");
    const passingCount = crismProducts.filter(p => {
      const obsId = extractCrismObsId(p.productId);
      return filteredProductIds.has(obsId);
    }).length;

    return passingCount;
  }, [visibleProducts, filteredProductIds, iceScoreFilter?.enabled]);

  // Collapsed state - show thin bar with toggle button (desktop only)
  if (isCollapsed && !isMobile) {
    return (
      <div className="flex h-full w-12 flex-col border-r border-[#232f48] bg-[#101622] transition-all duration-300 ease-in-out">
        {/* Expand button */}
        <button
          onClick={() => setIsCollapsed(false)}
          className="flex flex-col items-center justify-center gap-2 p-3 border-b border-[#232f48] hover:bg-[#1a2333] transition-colors"
          title="Expand Control Panel"
        >
          <span className="material-symbols-outlined text-[#92a4c9]">chevron_right</span>
        </button>

        {/* Vertical icons indicating panel contents */}
        <div className="flex flex-col items-center gap-3 py-4">
          <span
            className="material-symbols-outlined text-sm text-[#6b7c9c]"
            title="Layers"
          >
            layers
          </span>
          <span
            className="material-symbols-outlined text-sm text-[#6b7c9c]"
            title="Footprints"
          >
            hexagon
          </span>
        </div>

        {/* Active overlays indicator */}
        {totalActiveOverlays > 0 && (
          <div className="mt-auto mb-4 flex flex-col items-center">
            <div className="w-6 h-6 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center">
              <span className="text-[9px] font-bold text-green-400">{totalActiveOverlays}</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Expanded state - full panel
  return (
    <div
      className={`relative flex flex-col bg-[#101622] ${isMobile ? 'w-full' : 'h-full border-r border-[#232f48]'}`}
      style={isMobile ? undefined : { width: panelWidth }}
    >
      {/* Resize handle (right edge) - desktop only */}
      {!isMobile && (
        <div
          className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}
      {/* Header */}
      <div className="p-4 border-b border-[#232f48]">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            {!isMobile && (
              <button
                onClick={() => setIsCollapsed(true)}
                className="p-1 rounded hover:bg-[#1a2333] transition-colors"
                title="Collapse Panel"
              >
                <span className="material-symbols-outlined text-sm text-[#92a4c9]">chevron_left</span>
              </button>
            )}
            <h1 className="text-white text-xs font-bold uppercase tracking-wider">
              Control Center
            </h1>
          </div>
          <span className="text-[10px] text-[#92a4c9] font-mono">
            {visibleProducts.length} products
          </span>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto scrollbar-dark">
        {/* Map Mode Section */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            View Mode
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => onMapModeChange("2D")}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded transition-colors ${
                mapMode === "2D"
                  ? "bg-primary/20 border border-primary/50 text-primary"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-primary/30"
              }`}
            >
              <span className="material-symbols-outlined text-base">map</span>
              <span className="text-[11px] font-medium">2D Map</span>
            </button>
            <button
              onClick={() => onMapModeChange("3D")}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded transition-colors ${
                mapMode === "3D"
                  ? "bg-primary/20 border border-primary/50 text-primary"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-primary/30"
              }`}
            >
              <span className="material-symbols-outlined text-base">globe</span>
              <span className="text-[11px] font-medium">3D Globe</span>
            </button>
          </div>
        </div>

        {/* Base Layer Section */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            Base Map
          </h3>
          <div className="space-y-2">
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                baseLayer === "MOLA"
                  ? "bg-primary/20 border border-primary/50"
                  : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
              }`}
            >
              <input
                type="radio"
                name="baseLayer"
                checked={baseLayer === "MOLA"}
                onChange={() => onBaseLayerChange("MOLA")}
                className="rounded-full bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px] font-medium">MGS MOLA ColorShade</span>
            </label>
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                baseLayer === "HRSC"
                  ? "bg-primary/20 border border-primary/50"
                  : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
              }`}
            >
              <input
                type="radio"
                name="baseLayer"
                checked={baseLayer === "HRSC"}
                onChange={() => onBaseLayerChange("HRSC")}
                className="rounded-full bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px] font-medium">Mars Express HRSC</span>
            </label>
          </div>

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
              className="rounded bg-[#0a0f18] border-[#232f48] text-slate-400 focus:ring-0 focus:ring-offset-0"
            />
            <span className="material-symbols-outlined text-xs text-slate-400">grid_on</span>
            <span className={`text-[11px] font-medium ${showGrid ? "text-slate-300" : "text-[#92a4c9]"}`}>
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
            <span className={`text-[11px] font-medium ${showRegionLayer ? "text-amber-300" : "text-[#92a4c9]"}`}>
              Named Regions
            </span>
          </label>
        </div>

        {/* Footprints Section — Hierarchical instrument tree */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            Footprints
          </h3>
          <div className="space-y-1">
            {INSTRUMENT_GROUPS.map((group) => {
              const activeCount = group.instruments.filter(id => instrumentVisibility[id]).length;
              const allActive = activeCount === group.instruments.length;
              const someActive = activeCount > 0 && !allActive;

              return (
                <div key={group.id} className="rounded overflow-hidden">
                  {/* Group header */}
                  <button
                    onClick={() => {
                      // Toggle all instruments in this group
                      const newVal = !allActive;
                      for (const instId of group.instruments) {
                        onToggleInstrument(instId, newVal);
                      }
                    }}
                    className={`flex items-center gap-2 w-full px-2.5 py-2 text-left transition-colors ${
                      someActive || allActive
                        ? "bg-[#1a2333]"
                        : "bg-[#101622] hover:bg-[#1a2333]"
                    }`}
                  >
                    <span className="material-symbols-outlined text-sm text-[#6b7c9c]">{group.icon}</span>
                    <span className="text-[11px] font-bold text-[#92a4c9] flex-1">{group.displayName}</span>
                    {activeCount > 0 && (
                      <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-primary/20 text-primary font-bold">
                        {activeCount}
                      </span>
                    )}
                    <span className="material-symbols-outlined text-xs text-[#6b7c9c]">
                      {someActive || allActive ? "toggle_on" : "toggle_off"}
                    </span>
                  </button>

                  {/* Child instruments */}
                  <div className="pl-4 space-y-1 py-1 bg-[#0d1219]">
                    {group.instruments.map((instId) => {
                      const inst = INSTRUMENTS[instId];
                      const isVisible = instrumentVisibility[instId];
                      const isLoading = footprintsLoading[instId] ?? false;
                      const count = footprintCounts[instId] as FootprintCount;
                      const style = INST_STYLES[instId];

                      return (
                        <div key={instId} className="flex items-center gap-1.5 pr-2">
                          <label
                            className={`flex items-center gap-2 py-1.5 px-2 rounded cursor-pointer transition-colors flex-1 ${
                              isVisible ? style.bgActive : `bg-transparent hover:bg-[#1a2333]`
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={isVisible}
                              onChange={(e) => onToggleInstrument(instId, e.target.checked)}
                              className={`rounded bg-[#0a0f18] border-[#232f48] focus:ring-0 focus:ring-offset-0 ${style.checkbox}`}
                            />
                            <span className={`text-[10px] font-medium ${style.text}`}>{inst.subLabel}</span>
                            {count && (
                              <span className={`text-[8px] ${style.text} opacity-60 ml-auto`}>
                                {count.count}{count.truncated && `/${count.total}`}
                              </span>
                            )}
                          </label>
                          <button
                            onClick={() => onLoadFootprints?.(inst.name as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3")}
                            disabled={isLoading}
                            className={`px-1.5 py-1 rounded text-[9px] font-medium transition-colors whitespace-nowrap ${
                              isLoading ? style.btnLoading : style.btn
                            }`}
                          >
                            {isLoading ? (
                              <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                            ) : (
                              "Load"
                            )}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {/* Custom Data Row (not part of instrument groups) */}
            <div className="flex items-center gap-1.5 mt-2 pt-2 border-t border-[#232f48]">
              <label
                className={`flex items-center gap-2 py-1.5 px-2 rounded cursor-pointer transition-colors flex-1 ${
                  showCustomData
                    ? "bg-fuchsia-500/20 border border-fuchsia-500/50"
                    : "bg-transparent border border-transparent hover:bg-[#1a2333]"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showCustomData}
                  onChange={(e) => onToggleCustomData(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-fuchsia-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[10px] font-medium text-fuchsia-400">Custom Data</span>
                {customDatasets.length > 0 && (
                  <span className="text-[8px] text-fuchsia-400/60 ml-auto">
                    {customDatasets.length}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadCustomData?.()}
                disabled={customDataLoading}
                className={`px-1.5 py-1 rounded text-[9px] font-medium transition-colors whitespace-nowrap ${
                  customDataLoading
                    ? "bg-fuchsia-500/10 text-fuchsia-400/50 border border-fuchsia-500/20 cursor-wait"
                    : "bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 hover:bg-fuchsia-500/30"
                }`}
              >
                {customDataLoading ? (
                  <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                ) : (
                  "Load"
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Analysis Tools Section */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            Analysis Tools
          </h3>
          <div className="space-y-2">
            {/* Slope Analysis */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "slope" ? null : "slope")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "slope"
                  ? "bg-primary/20 border border-primary/50 text-primary"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-primary/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">analytics</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Slope Analysis</span>
                <p className="text-[9px] text-[#6b7c9c]">Click terrain to analyse</p>
              </div>
              {analysisMode === "slope" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Line Profile */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "line" ? null : "line")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "line"
                  ? "bg-emerald-500/20 border border-emerald-500/50 text-emerald-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-emerald-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">show_chart</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Line Profile</span>
                <p className="text-[9px] text-[#6b7c9c]">Click two points for elevation</p>
              </div>
              {analysisMode === "line" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Agentic AI */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "agentic" ? null : "agentic")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "agentic"
                  ? "bg-fuchsia-500/20 border border-fuchsia-500/50 text-fuchsia-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-fuchsia-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">smart_toy</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Agentic AI <span className="text-[8px] px-1 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 font-bold">BETA</span></span>
                <p className="text-[9px] text-[#6b7c9c]">Autonomous multi-instrument analysis</p>
              </div>
              {analysisMode === "agentic" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Landing Site Report Generator */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "report" ? null : "report")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "report"
                  ? "bg-amber-500/20 border border-amber-500/50 text-amber-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-amber-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">assignment</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">AI Landing Site Report <span className="text-[8px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-bold">BETA</span></span>
                <p className="text-[9px] text-[#6b7c9c]">Compare regions with ground rules</p>
              </div>
              {analysisMode === "report" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Guided Workflows */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "guided" ? null : "guided")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "guided"
                  ? "bg-sky-500/20 border border-sky-500/50 text-sky-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-sky-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">explore</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Guided Workflows <span className="text-[8px] px-1 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30 font-bold">NEW</span></span>
                <p className="text-[9px] text-[#6b7c9c]">Step-by-step investigation guides</p>
              </div>
              {analysisMode === "guided" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Region Dashboard */}
            <button
              onClick={() => onShowRegionDashboard?.()}
              className="flex items-center gap-2 w-full p-2 rounded transition-colors text-left bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-teal-500/30"
            >
              <span className="material-symbols-outlined text-sm">public</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Region Dashboard <span className="text-[8px] px-1 py-0.5 rounded bg-teal-500/20 text-teal-400 border border-teal-500/30 font-bold">NEW</span></span>
                <p className="text-[9px] text-[#6b7c9c]">Browse all 55 regions at a glance</p>
              </div>
            </button>

            {/* Measurement Tools */}
            <button
              onClick={() => onToggleMeasurementTools?.(!showMeasurementTools)}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                showMeasurementTools
                  ? "bg-cyan-900/40 border border-cyan-500/40 text-cyan-300"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-cyan-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">straighten</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Measurement Tools</span>
                <p className="text-[9px] text-[#6b7c9c]">Distance, area, elevation, pins</p>
              </div>
              {showMeasurementTools && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Region Stats */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "region_stats" ? null : "region_stats")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "region_stats"
                  ? "bg-indigo-500/20 border border-indigo-500/50 text-indigo-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-indigo-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">pentagon</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Region Stats</span>
                <p className="text-[9px] text-[#6b7c9c]">Draw polygon for area statistics</p>
              </div>
              {analysisMode === "region_stats" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* Crater/LDA Detection */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "crater_detect" ? null : "crater_detect")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "crater_detect"
                  ? "bg-rose-500/20 border border-rose-500/50 text-rose-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-rose-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">target</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Landform Detect
                  <span className="text-[8px] px-1 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 font-bold ml-1">NEW</span>
                </span>
                <p className="text-[9px] text-[#6b7c9c]">Find craters, channels, LDAs from MOLA DEM</p>
              </div>
              {analysisMode === "crater_detect" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

          </div>
        </div>

        {/* Multi-Instrument Overlap Filter Section */}
        <OverlapFilterSection
          filter={overlapFilter}
          onChange={onOverlapFilterChange}
          stats={overlapStats}
        />

        {/* Ice Score Filter Section */}
        <IceScoreFilterSection
          filter={iceScoreFilter}
          onChange={onIceScoreFilterChange}
          filteredCount={filteredVisibleCount}
          totalLoaded={visibleProducts.filter(p => p.instrument === "CRISM").length}
        />

        {/* Field Notes Section */}
        <FieldNotesSection
          fieldNotes={fieldNotes}
          showOnMap={showFieldNotesOnMap ?? true}
          onToggleShowOnMap={onToggleFieldNotesOnMap}
          onFieldNoteClick={onFieldNoteClick}
          onActiveTagChange={onActiveTagChange}
        />

        {/* Displayed Products Section */}
        <DisplayedProductsSection
          visibleProducts={visibleProducts}
          activeOverlays={activeOverlays}
          onSetOverlay={onSetOverlay}
          onSelectProduct={onSelectProduct}
          onFlyToProduct={onFlyToProduct}
          customDatasets={customDatasets}
          onCustomDatasetToggle={onCustomDatasetToggle}
        />

        {/* Active Products Section */}
        <ActiveProductsSection
          activeOverlays={activeOverlays}
          visibleProducts={visibleProducts}
          onSetOverlay={onSetOverlay}
          onSetOpacity={onSetOpacity}
          onFlyToProduct={onFlyToProduct}
          onDeactivateAll={onDeactivateAll}
        />
      </div>
    </div>
  );
}

// Field Notes Section Component
function FieldNotesSection({
  fieldNotes,
  showOnMap,
  onToggleShowOnMap,
  onFieldNoteClick,
  onActiveTagChange,
}: {
  fieldNotes: FieldNote[];
  showOnMap: boolean;
  onToggleShowOnMap?: (v: boolean) => void;
  onFieldNoteClick?: (note: FieldNote) => void;
  onActiveTagChange?: (tag: string | null) => void;
}) {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  const handleTagSelect = useCallback((tag: string) => {
    const next = selectedTag === tag ? null : tag;
    setSelectedTag(next);
    onActiveTagChange?.(next);
    // Auto-enable "Show on Map" when a tag is selected
    if (next && !showOnMap) {
      onToggleShowOnMap?.(true);
    }
  }, [selectedTag, onActiveTagChange, showOnMap, onToggleShowOnMap]);

  // All unique tags sorted
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    fieldNotes.forEach(n => n.tags.forEach(t => tagSet.add(t)));
    return Array.from(tagSet).sort();
  }, [fieldNotes]);

  // Group notes by tag, or flat list when no tag selected
  const grouped = useMemo(() => {
    if (selectedTag) {
      // Single-tag filter: one group
      return [{ tag: selectedTag, notes: fieldNotes.filter(n => n.tags.includes(selectedTag)) }];
    }
    // Group by every tag; notes without tags go into "(Untagged)"
    const map = new Map<string, FieldNote[]>();
    const untagged: FieldNote[] = [];
    for (const n of fieldNotes) {
      if (n.tags.length === 0) {
        untagged.push(n);
      } else {
        for (const t of n.tags) {
          if (!map.has(t)) map.set(t, []);
          map.get(t)!.push(n);
        }
      }
    }
    const groups = Array.from(map.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([tag, notes]) => ({ tag, notes }));
    if (untagged.length > 0) groups.push({ tag: "", notes: untagged });
    return groups;
  }, [fieldNotes, selectedTag]);

  if (fieldNotes.length === 0) return null;

  return (
    <div className="p-4 border-b border-[#232f48]">
      {/* Header */}
      <div
        className="flex items-center justify-between mb-2 cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="material-symbols-outlined text-xs">
            {isCollapsed ? "expand_more" : "expand_less"}
          </span>
          Field Notes
        </h3>
        <span className="text-amber-400 text-[10px] font-mono">{fieldNotes.length}</span>
      </div>

      {!isCollapsed && (
        <div className="space-y-3">
          {/* Show on Map toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showOnMap}
              onChange={(e) => onToggleShowOnMap?.(e.target.checked)}
              className="accent-amber-400 w-3 h-3"
            />
            <span className="text-[10px] text-[#92a4c9]">Show on Map</span>
          </label>

          {/* Tag filter pills */}
          {allTags.length > 0 && (
            <div className="flex flex-wrap gap-1 max-h-16 overflow-y-auto scrollbar-dark">
              {allTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => handleTagSelect(tag)}
                  className={`px-2 py-0.5 rounded-full text-[9px] border transition-colors ${
                    selectedTag === tag
                      ? "bg-amber-500/20 text-amber-400 border-amber-500/50"
                      : "bg-[#1a2333] text-[#92a4c9] border-[#232f48] hover:border-amber-500/30 hover:text-amber-400"
                  }`}
                >
                  {tag}
                </button>
              ))}
            </div>
          )}

          {/* Grouped notes list */}
          <div className="max-h-64 overflow-y-auto scrollbar-dark space-y-2">
            {grouped.map(({ tag, notes }) => (
              <div key={tag || "__untagged"}>
                {/* Group header */}
                <div className="flex items-center gap-1.5 mb-1 px-1">
                  <span className="material-symbols-outlined text-amber-400 text-[10px]">label</span>
                  <span className="text-[10px] font-bold text-amber-400">
                    {tag || "Untagged"}
                  </span>
                  <span className="text-[9px] text-[#6b7c9c]">({notes.length})</span>
                </div>
                {/* Notes under this group */}
                <div className="space-y-1 pl-1">
                  {notes.map(note => {
                    const instColors = INSTRUMENT_COLORS[note.instrument as InstrumentType] ?? INSTRUMENT_COLORS.CUSTOM;
                    // In grouped view, exclude the group tag from displayed tags
                    const otherTags = tag ? note.tags.filter(t => t !== tag) : note.tags;
                    return (
                      <button
                        key={note.id}
                        onClick={() => onFieldNoteClick?.(note)}
                        className="w-full text-left p-2 rounded bg-[#0a0f18] border border-[#232f48] hover:border-amber-500/30 transition-colors space-y-1"
                      >
                        <div className="flex items-center gap-1.5">
                          <span className={`px-1.5 py-0.5 rounded text-[8px] font-bold ${instColors.bg} ${instColors.text} border ${instColors.border}`}>
                            {note.instrument}
                          </span>
                          <span className="text-[10px] text-white font-mono truncate flex-1">
                            {note.product_id}
                          </span>
                        </div>
                        {note.memo && (
                          <p className="text-[9px] text-[#6b7c9c] truncate">{note.memo}</p>
                        )}
                        {otherTags.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {otherTags.map(t => (
                              <span
                                key={t}
                                className="px-1.5 py-0.5 rounded-full text-[8px] bg-amber-500/10 text-amber-400/70 border border-amber-500/20"
                              >
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Instrument colors for badges
const INSTRUMENT_COLORS: Record<InstrumentType, { bg: string; text: string; border: string }> = {
  CRISM: { bg: "bg-cyan-500/20", text: "text-cyan-400", border: "border-cyan-500/30" },
  HIRISE: { bg: "bg-yellow-500/20", text: "text-yellow-400", border: "border-yellow-500/30" },
  SHARAD: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
  SHARAD_HIGHRES: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
  CTX: { bg: "bg-pink-500/20", text: "text-pink-400", border: "border-pink-500/30" },
  CUSTOM: { bg: "bg-fuchsia-500/20", text: "text-fuchsia-400", border: "border-fuchsia-500/30" },
  HIRISE_DTM: { bg: "bg-amber-700/20", text: "text-amber-600", border: "border-amber-700/30" },
  CRISM_TRR3: { bg: "bg-teal-500/20", text: "text-teal-400", border: "border-teal-500/30" },
};

// Displayed Products Section Component
function DisplayedProductsSection({
  visibleProducts,
  activeOverlays,
  onSetOverlay,
  onSelectProduct,
  onFlyToProduct,
  customDatasets,
  onCustomDatasetToggle,
}: {
  visibleProducts: VisibleProduct[];
  activeOverlays: ActiveOverlays;
  onSetOverlay?: (productId: string, type: OverlayType | null) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
  onFlyToProduct?: (productId: string) => void;
  customDatasets?: CustomDataset[];
  onCustomDatasetToggle?: (id: string, visible: boolean) => void;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Group products by instrument
  const groupedProducts = useMemo(() => {
    const groups: Record<InstrumentType, VisibleProduct[]> = {
      CRISM: [],
      HIRISE: [],
      SHARAD: [],
      SHARAD_HIGHRES: [],
      CTX: [],
      CUSTOM: [],
      HIRISE_DTM: [],
      CRISM_TRR3: [],
    };
    for (const product of visibleProducts) {
      if (product.instrument in groups) {
        groups[product.instrument as InstrumentType].push(product);
      }
    }
    return groups;
  }, [visibleProducts]);

  const totalCount = visibleProducts.length;

  return (
    <div className="p-4 border-b border-[#232f48]">
      {/* Header */}
      <div
        className="flex items-center justify-between mb-2 cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="material-symbols-outlined text-xs">
            {isCollapsed ? "expand_more" : "expand_less"}
          </span>
          Displayed Products
        </h3>
        <span className="text-primary text-[10px] font-mono">{totalCount}</span>
      </div>

      {!isCollapsed && (
        <div className="space-y-3 max-h-64 overflow-y-auto scrollbar-dark">
          {totalCount === 0 ? (
            <div className="text-center py-4">
              <span className="material-symbols-outlined text-2xl text-[#3a4a68] mb-1">layers_clear</span>
              <p className="text-[10px] text-[#6b7c9c]">No products in view</p>
              <p className="text-[9px] text-[#4a5a7c] mt-1">Enable footprints and zoom in</p>
            </div>
          ) : (
            Object.entries(groupedProducts).map(([instrument, products]) => {
              if (products.length === 0) return null;

              const instColors = INSTRUMENT_COLORS[instrument as InstrumentType];

              return (
                <div key={instrument}>
                  {/* Instrument group header */}
                  <div className={`text-[9px] font-bold uppercase tracking-wider mb-1.5 ${instColors.text}`}>
                    {instrument} ({products.length})
                  </div>

                  {/* Products list */}
                  <div className="space-y-1">
                    {products.map((product) => {
                      const overlay = activeOverlays.get(product.productId);
                      const hasOverlay = !!overlay;
                      const isCustom = product.instrument === "CUSTOM";
                      const customDs = isCustom ? customDatasets?.find(d => d.id === product.productId) : null;
                      const customVisible = customDs?.visible ?? true;

                      return (
                        <div
                          key={product.productId}
                          className="flex items-center gap-2 p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50 hover:bg-[#1a2333] transition-colors group"
                        >
                          {/* Instrument badge */}
                          <span
                            className={`text-[8px] px-1 py-0.5 rounded font-bold ${instColors.bg} ${instColors.text} ${instColors.border} border`}
                          >
                            {instrument.slice(0, 3)}
                          </span>

                          {/* Product ID/name - clickable to open Inspector */}
                          <span
                            className="flex-1 text-[10px] font-mono text-[#92a4c9] truncate cursor-pointer hover:text-white"
                            onClick={() => onSelectProduct?.(product)}
                            title={`Click to inspect ${product.title || product.productId}`}
                          >
                            {product.title || product.productId}
                          </span>

                          {/* Fly-to button */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onFlyToProduct?.(product.productId);
                            }}
                            className="p-0.5 rounded transition-colors text-slate-600 hover:text-sky-400 hover:bg-sky-500/20"
                            title={`Fly to ${product.productId}`}
                          >
                            <span className="material-symbols-outlined text-xs">my_location</span>
                          </button>

                          {/* Visibility toggle for CUSTOM datasets */}
                          {isCustom && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onCustomDatasetToggle?.(product.productId, !customVisible);
                              }}
                              className={`p-0.5 rounded transition-colors ${
                                customVisible
                                  ? "text-fuchsia-400 bg-fuchsia-500/20"
                                  : "text-slate-600 hover:text-slate-400"
                              }`}
                              title={customVisible ? "Hide overlay" : "Show overlay"}
                            >
                              <span className="material-symbols-outlined text-xs">
                                {customVisible ? "visibility" : "visibility_off"}
                              </span>
                            </button>
                          )}

                          {/* Simple quickview toggle (not for CUSTOM) */}
                          {!isCustom && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                if (hasOverlay && overlay.type === "quickview") {
                                  onSetOverlay?.(product.productId, null);
                                } else {
                                  onSetOverlay?.(product.productId, "quickview");
                                }
                              }}
                              className={`p-0.5 rounded transition-colors ${
                                hasOverlay && overlay.type === "quickview"
                                  ? "text-emerald-400 bg-emerald-500/20"
                                  : "text-slate-600 hover:text-slate-400"
                              }`}
                              title={hasOverlay && overlay.type === "quickview" ? "Turn off Quickview" : "Show Quickview"}
                            >
                              <span className="material-symbols-outlined text-xs">visibility</span>
                            </button>
                          )}

                          {/* Active overlay indicator */}
                          {hasOverlay && overlay.type !== "quickview" && (
                            <span
                              className={`text-[8px] px-1 py-0.5 rounded font-bold bg-${OVERLAY_LABELS[overlay.type].color}-500/20 text-${OVERLAY_LABELS[overlay.type].color}-400 border border-${OVERLAY_LABELS[overlay.type].color}-500/30`}
                            >
                              {OVERLAY_LABELS[overlay.type].short}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

// Active Products Section Component (with OFF controls and per-product opacity)
function ActiveProductsSection({
  activeOverlays,
  visibleProducts,
  onSetOverlay,
  onSetOpacity,
  onFlyToProduct,
  onDeactivateAll,
}: {
  activeOverlays: ActiveOverlays;
  visibleProducts: VisibleProduct[];
  onSetOverlay?: (productId: string, type: OverlayType | null) => void;
  onSetOpacity?: (productId: string, opacity: number) => void;
  onFlyToProduct?: (productId: string) => void;
  onDeactivateAll?: () => void;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [expandedOpacity, setExpandedOpacity] = useState<string | null>(null);

  // Build active products list with overlay info
  const activeProducts = useMemo(() => {
    const result: Array<{
      productId: string;
      instrument: InstrumentType;
      overlay: ProductOverlay;
    }> = [];

    const productMap = new Map(visibleProducts.map((p) => [p.productId, p]));

    for (const [productId, overlay] of activeOverlays) {
      const product = productMap.get(productId);
      const instrument: InstrumentType = product?.instrument ??
        (productId.startsWith("ESP_") || productId.startsWith("PSP_") ? "HIRISE" : "CRISM");

      result.push({ productId, instrument, overlay });
    }

    return result;
  }, [activeOverlays, visibleProducts]);

  // Group by instrument
  const groupedActive = useMemo(() => {
    const groups: Record<InstrumentType, typeof activeProducts> = {
      CRISM: [],
      HIRISE: [],
      SHARAD: [],
      SHARAD_HIGHRES: [],
      CTX: [],
      CUSTOM: [],
      HIRISE_DTM: [],
      CRISM_TRR3: [],
    };
    for (const product of activeProducts) {
      if (product.instrument in groups) {
        groups[product.instrument].push(product);
      }
    }
    return groups;
  }, [activeProducts]);

  const activeCount = activeProducts.length;

  return (
    <div className="p-4">
      {/* Header */}
      <div
        className="flex items-center justify-between mb-2 cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="material-symbols-outlined text-xs">
            {isCollapsed ? "expand_more" : "expand_less"}
          </span>
          Active Overlays
        </h3>
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-mono ${activeCount > 0 ? "text-green-400" : "text-[#6b7c9c]"}`}>
            {activeCount}
          </span>
          {activeCount > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDeactivateAll?.();
              }}
              className="text-[8px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors border border-red-500/30 font-bold uppercase"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {!isCollapsed && (
        <div className="space-y-3 max-h-64 overflow-y-auto scrollbar-dark">
          {activeCount === 0 ? (
            <div className="text-center py-4">
              <span className="material-symbols-outlined text-2xl text-[#3a4a68] mb-1">check_circle</span>
              <p className="text-[10px] text-[#6b7c9c]">No active overlays</p>
              <p className="text-[9px] text-[#4a5a7c] mt-1">Click a product to activate</p>
            </div>
          ) : (
            Object.entries(groupedActive).map(([instrument, products]) => {
              if (products.length === 0) return null;

              const instColors = INSTRUMENT_COLORS[instrument as InstrumentType];

              return (
                <div key={instrument}>
                  {/* Instrument group header */}
                  <div className={`text-[9px] font-bold uppercase tracking-wider mb-1.5 ${instColors.text}`}>
                    {instrument} ({products.length})
                  </div>

                  {/* Products list */}
                  <div className="space-y-1">
                    {products.map(({ productId, overlay }) => {
                      const labelInfo = OVERLAY_LABELS[overlay.type];
                      const isOpacityExpanded = expandedOpacity === productId;

                      return (
                        <div key={productId} className="space-y-1">
                          <div className="flex items-center gap-2 p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50">
                            {/* Instrument badge */}
                            <span
                              className={`text-[8px] px-1 py-0.5 rounded font-bold ${instColors.bg} ${instColors.text} ${instColors.border} border`}
                            >
                              {instrument.slice(0, 3)}
                            </span>

                            {/* Product ID - clickable to fly to */}
                            <span
                              className="flex-1 text-[10px] font-mono text-[#92a4c9] truncate cursor-pointer hover:text-white"
                              onClick={() => onFlyToProduct?.(productId)}
                              title={`Click to fly to ${productId}`}
                            >
                              {productId}
                            </span>

                            {/* Overlay type badge with close button */}
                            <button
                              onClick={() => onSetOverlay?.(productId, null)}
                              className={`text-[8px] px-1 py-0.5 rounded font-bold flex items-center gap-0.5 transition-colors
                                ${labelInfo.color === "emerald" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30" : ""}
                                ${labelInfo.color === "purple" ? "bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30" : ""}
                                ${labelInfo.color === "fuchsia" ? "bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/30 hover:bg-fuchsia-500/30" : ""}
                                ${labelInfo.color === "blue" ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30" : ""}
                                ${labelInfo.color === "cyan" ? "bg-cyan-400/20 text-cyan-400 border-cyan-400/30 hover:bg-cyan-400/30" : ""}
                                ${labelInfo.color === "sky" ? "bg-sky-500/20 text-sky-400 border-sky-500/30 hover:bg-sky-500/30" : ""}
                                ${labelInfo.color === "rose" ? "bg-rose-500/20 text-rose-400 border-rose-500/30 hover:bg-rose-500/30" : ""}
                                border`}
                              title={`Turn off ${labelInfo.full}`}
                            >
                              {labelInfo.short}
                              <span className="material-symbols-outlined text-[10px]">close</span>
                            </button>

                            {/* Opacity toggle */}
                            <button
                              onClick={() => setExpandedOpacity(isOpacityExpanded ? null : productId)}
                              className={`p-0.5 rounded transition-colors ${
                                isOpacityExpanded
                                  ? "text-primary bg-primary/20"
                                  : "text-slate-600 hover:text-slate-400"
                              }`}
                              title="Adjust opacity"
                            >
                              <span className="material-symbols-outlined text-xs">opacity</span>
                            </button>
                          </div>

                          {/* Opacity slider (expanded) */}
                          {isOpacityExpanded && (
                            <div className="flex items-center gap-2 px-2 py-1.5 bg-[#0a0f18] rounded border border-[#232f48]">
                              <span className="text-[9px] text-[#6b7c9c] uppercase">Opacity</span>
                              <input
                                type="range"
                                min="0"
                                max="100"
                                value={overlay.opacity}
                                onChange={(e) => onSetOpacity?.(productId, Number(e.target.value))}
                                className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                                  [&::-webkit-slider-thumb]:appearance-none
                                  [&::-webkit-slider-thumb]:h-3
                                  [&::-webkit-slider-thumb]:w-3
                                  [&::-webkit-slider-thumb]:rounded-full
                                  [&::-webkit-slider-thumb]:bg-primary
                                  [&::-webkit-slider-thumb]:cursor-pointer"
                              />
                              <span className="text-[10px] text-white font-mono w-8 text-right">
                                {overlay.opacity}%
                              </span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
