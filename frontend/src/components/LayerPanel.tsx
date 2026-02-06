import { useState, useMemo, useEffect, useCallback } from "react";
import type { VisibleProduct, BaseLayerType, BoundingBox, MapMode, ActiveOverlays, OverlayType, ProductOverlay, CustomDataset } from "../pages/MainPage";
import { normalizeLonForMap, clampLatitude, parseCoordinate } from "../utils/coordinates";
import type { FieldNote } from "../api/fieldnotes";

type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM";

/**
 * Extract CRISM observation ID from full product ID
 * e.g., "frt0001fd76_07_if166j_mtr3" -> "frt0001fd76"
 */
function extractCrismObsId(productId: string): string {
  const match = productId.match(/^([a-z]{3}[0-9a-f]{8})/i);
  return match ? match[1].toLowerCase() : productId.toLowerCase();
}

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
};

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

  // Footprint visibility toggles
  showCRISM: boolean;
  showHiRISE: boolean;
  showSHARAD: boolean;
  showSharadHighres: boolean;
  showCTX: boolean;
  showHiRISEDTM?: boolean;
  onToggleCRISM: (v: boolean) => void;
  onToggleHiRISE: (v: boolean) => void;
  onToggleSHARAD: (v: boolean) => void;
  onToggleSharadHighres: (v: boolean) => void;
  onToggleCTX: (v: boolean) => void;
  onToggleHiRISEDTM?: (v: boolean) => void;

  // Explicit footprint loading
  onLoadFootprints?: (instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM") => void;
  footprintsLoading?: { crism: boolean; hirise: boolean; sharad: boolean; sharad_highres: boolean; ctx: boolean; hirise_dtm: boolean };
  footprintCounts?: {
    crism: { count: number; truncated: boolean; total: number } | null;
    hirise: { count: number; truncated: boolean; total: number } | null;
    sharad: { count: number; truncated: boolean; total: number } | null;
    sharad_highres: { count: number; truncated: boolean; total: number } | null;
    ctx: { count: number; truncated: boolean; total: number } | null;
    hirise_dtm: { count: number; truncated: boolean; total: number } | null;
  };

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
  analysisMode?: "slope" | "slope3d" | "hirise_dtm_3d" | "line" | null;
  onAnalysisModeChange?: (mode: "slope" | "slope3d" | "hirise_dtm_3d" | "line" | null) => void;

  // Fly-To navigation
  onFlyToCoords?: (lat: number, lon: number) => void;

  // View bound selection mode
  viewBoundSelectionMode?: boolean;
  onViewBoundSelectionModeChange?: (active: boolean) => void;

  // Field Notes
  fieldNotes?: FieldNote[];
  showFieldNotesOnMap?: boolean;
  onToggleFieldNotesOnMap?: (v: boolean) => void;
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
  // Footprints
  showCRISM,
  showHiRISE,
  showSHARAD,
  showSharadHighres,
  showCTX,
  showHiRISEDTM = false,
  onToggleCRISM,
  onToggleHiRISE,
  onToggleSHARAD,
  onToggleSharadHighres,
  onToggleCTX,
  onToggleHiRISEDTM,
  // Explicit footprint loading
  onLoadFootprints,
  footprintsLoading = { crism: false, hirise: false, sharad: false, sharad_highres: false, ctx: false, hirise_dtm: false },
  footprintCounts = { crism: null, hirise: null, sharad: null, sharad_highres: null, ctx: null, hirise_dtm: null },
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
  // Fly-To navigation
  onFlyToCoords,
  // View bound selection mode
  viewBoundSelectionMode = false,
  onViewBoundSelectionModeChange,
  // Field Notes
  fieldNotes = [],
  showFieldNotesOnMap = true,
  onToggleFieldNotesOnMap,
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

  // Collapsed state - show thin bar with toggle button
  if (isCollapsed) {
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
    <div className="relative flex h-full flex-col border-r border-[#232f48] bg-[#101622]" style={{ width: panelWidth }}>
      {/* Resize handle (right edge) */}
      <div
        className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
        onMouseDown={handleResizeStart}
      />
      {/* Header */}
      <div className="p-4 border-b border-[#232f48]">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsCollapsed(true)}
              className="p-1 rounded hover:bg-[#1a2333] transition-colors"
              title="Collapse Panel"
            >
              <span className="material-symbols-outlined text-sm text-[#92a4c9]">chevron_left</span>
            </button>
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
        </div>

        {/* Footprints Section */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            Footprints
          </h3>
          <div className="space-y-3">
            {/* CRISM Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showCRISM
                    ? "bg-cyan-500/20 border border-cyan-500/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-cyan-500/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showCRISM}
                  onChange={(e) => onToggleCRISM(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-cyan-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-cyan-400">CRISM</span>
                {footprintCounts.crism && (
                  <span className="text-[9px] text-cyan-400/70 ml-auto">
                    {footprintCounts.crism.count}{footprintCounts.crism.truncated && `/${footprintCounts.crism.total}`}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadFootprints?.("CRISM")}
                disabled={footprintsLoading.crism}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  footprintsLoading.crism
                    ? "bg-cyan-500/10 text-cyan-400/50 border border-cyan-500/20 cursor-wait"
                    : "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30"
                }`}
              >
                {footprintsLoading.crism ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
                ) : (
                  "Load"
                )}
              </button>
            </div>

            {/* HiRISE Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showHiRISE
                    ? "bg-yellow-500/20 border border-yellow-500/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-yellow-500/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showHiRISE}
                  onChange={(e) => onToggleHiRISE(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-yellow-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-yellow-400">HiRISE</span>
                {footprintCounts.hirise && (
                  <span className="text-[9px] text-yellow-400/70 ml-auto">
                    {footprintCounts.hirise.count}{footprintCounts.hirise.truncated && `/${footprintCounts.hirise.total}`}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadFootprints?.("HIRISE")}
                disabled={footprintsLoading.hirise}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  footprintsLoading.hirise
                    ? "bg-yellow-500/10 text-yellow-400/50 border border-yellow-500/20 cursor-wait"
                    : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30"
                }`}
              >
                {footprintsLoading.hirise ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
                ) : (
                  "Load"
                )}
              </button>
            </div>

            {/* SHARAD Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showSHARAD
                    ? "bg-orange-500/20 border border-orange-500/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-orange-500/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showSHARAD}
                  onChange={(e) => onToggleSHARAD(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-orange-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-orange-400">SHARAD</span>
                {footprintCounts.sharad && (
                  <span className="text-[9px] text-orange-400/70 ml-auto">
                    {footprintCounts.sharad.count}{footprintCounts.sharad.truncated && `/${footprintCounts.sharad.total}`}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadFootprints?.("SHARAD")}
                disabled={footprintsLoading.sharad}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  footprintsLoading.sharad
                    ? "bg-orange-500/10 text-orange-400/50 border border-orange-500/20 cursor-wait"
                    : "bg-orange-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30"
                }`}
              >
                {footprintsLoading.sharad ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
                ) : (
                  "Load"
                )}
              </button>
            </div>

            {/* SHARAD High-Res Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showSharadHighres
                    ? "bg-amber-500/20 border border-amber-500/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-amber-500/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showSharadHighres}
                  onChange={(e) => onToggleSharadHighres(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-amber-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-amber-400">SHARAD HR</span>
                {footprintCounts.sharad_highres && (
                  <span className="text-[9px] text-amber-400/70 ml-auto">
                    {footprintCounts.sharad_highres.count}{footprintCounts.sharad_highres.truncated && `/${footprintCounts.sharad_highres.total}`}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadFootprints?.("SHARAD_HIGHRES")}
                disabled={footprintsLoading.sharad_highres}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  footprintsLoading.sharad_highres
                    ? "bg-amber-500/10 text-amber-400/50 border border-amber-500/20 cursor-wait"
                    : "bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30"
                }`}
              >
                {footprintsLoading.sharad_highres ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
                ) : (
                  "Load"
                )}
              </button>
            </div>

            {/* CTX Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showCTX
                    ? "bg-pink-500/20 border border-pink-500/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-pink-500/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showCTX}
                  onChange={(e) => onToggleCTX(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-pink-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-pink-400">CTX</span>
                {footprintCounts.ctx && (
                  <span className="text-[9px] text-pink-400/70 ml-auto">
                    {footprintCounts.ctx.count}{footprintCounts.ctx.truncated && `/${footprintCounts.ctx.total}`}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadFootprints?.("CTX")}
                disabled={footprintsLoading.ctx}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  footprintsLoading.ctx
                    ? "bg-pink-500/10 text-pink-400/50 border border-pink-500/20 cursor-wait"
                    : "bg-pink-500/20 text-pink-400 border border-pink-500/30 hover:bg-pink-500/30"
                }`}
              >
                {footprintsLoading.ctx ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
                ) : (
                  "Load"
                )}
              </button>
            </div>

            {/* Custom Data Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showCustomData
                    ? "bg-fuchsia-500/20 border border-fuchsia-500/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-fuchsia-500/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showCustomData}
                  onChange={(e) => onToggleCustomData(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-fuchsia-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-fuchsia-400">Custom Data</span>
                {customDatasets.length > 0 && (
                  <span className="text-[9px] text-fuchsia-400/70 ml-auto">
                    {customDatasets.length}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadCustomData?.()}
                disabled={customDataLoading}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  customDataLoading
                    ? "bg-fuchsia-500/10 text-fuchsia-400/50 border border-fuchsia-500/20 cursor-wait"
                    : "bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 hover:bg-fuchsia-500/30"
                }`}
              >
                {customDataLoading ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
                ) : (
                  "Load"
                )}
              </button>
            </div>

            {/* HiRISE DTM Row */}
            <div className="flex items-center gap-2">
              <label
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors flex-1 ${
                  showHiRISEDTM
                    ? "bg-amber-700/20 border border-amber-700/50"
                    : "bg-[#1a2333] border border-[#232f48] hover:border-amber-700/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={showHiRISEDTM}
                  onChange={(e) => onToggleHiRISEDTM?.(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-amber-600 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[11px] font-medium text-amber-600">HiRISE DTM</span>
                {footprintCounts.hirise_dtm && (
                  <span className="text-[9px] text-amber-600/70 ml-auto">
                    {footprintCounts.hirise_dtm.count}{footprintCounts.hirise_dtm.truncated && `/${footprintCounts.hirise_dtm.total}`}
                  </span>
                )}
              </label>
              <button
                onClick={() => onLoadFootprints?.("HIRISE_DTM")}
                disabled={footprintsLoading.hirise_dtm}
                className={`px-2 py-2 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
                  footprintsLoading.hirise_dtm
                    ? "bg-amber-700/10 text-amber-600/50 border border-amber-700/20 cursor-wait"
                    : "bg-amber-700/20 text-amber-600 border border-amber-700/30 hover:bg-amber-700/30"
                }`}
              >
                {footprintsLoading.hirise_dtm ? (
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  </span>
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

            {/* Slope 3D Analysis */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "slope3d" ? null : "slope3d")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "slope3d"
                  ? "bg-purple-500/20 border border-purple-500/50 text-purple-400"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-purple-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">landscape</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">Slope 3D Analysis</span>
                <p className="text-[9px] text-[#6b7c9c]">View local 3D terrain</p>
              </div>
              {analysisMode === "slope3d" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30 font-bold uppercase">
                  ON
                </span>
              )}
            </button>

            {/* HiRISE DTM 3D View */}
            <button
              onClick={() => onAnalysisModeChange?.(analysisMode === "hirise_dtm_3d" ? null : "hirise_dtm_3d")}
              className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
                analysisMode === "hirise_dtm_3d"
                  ? "bg-amber-600/20 border border-amber-600/50 text-amber-600"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-amber-600/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">terrain</span>
              <div className="flex-1">
                <span className="text-[11px] font-medium">HiRISE DTM 3D View</span>
                <p className="text-[9px] text-[#6b7c9c]">Click DTM footprint for 3D</p>
              </div>
              {analysisMode === "hirise_dtm_3d" && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-amber-600/20 text-amber-600 border border-amber-600/30 font-bold uppercase">
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

          </div>
        </div>

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
          onSelectProduct={onSelectProduct}
          onFlyToCoords={onFlyToCoords}
          visibleProducts={visibleProducts}
        />

        {/* Displayed Products Section */}
        <DisplayedProductsSection
          visibleProducts={visibleProducts}
          activeOverlays={activeOverlays}
          onSetOverlay={onSetOverlay}
          onSelectProduct={onSelectProduct}
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
  onSelectProduct,
  onFlyToCoords,
  visibleProducts,
}: {
  fieldNotes: FieldNote[];
  showOnMap: boolean;
  onToggleShowOnMap?: (v: boolean) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
  onFlyToCoords?: (lat: number, lon: number) => void;
  visibleProducts: VisibleProduct[];
}) {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  // All unique tags sorted
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    fieldNotes.forEach(n => n.tags.forEach(t => tagSet.add(t)));
    return Array.from(tagSet).sort();
  }, [fieldNotes]);

  // Filtered notes
  const filtered = useMemo(
    () => selectedTag ? fieldNotes.filter(n => n.tags.includes(selectedTag)) : fieldNotes,
    [fieldNotes, selectedTag]
  );

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
                  onClick={() => setSelectedTag(selectedTag === tag ? null : tag)}
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

          {/* Notes list */}
          <div className="max-h-64 overflow-y-auto scrollbar-dark space-y-1">
            {filtered.map(note => {
              const instColors = INSTRUMENT_COLORS[note.instrument as InstrumentType] ?? INSTRUMENT_COLORS.CUSTOM;
              return (
                <button
                  key={note.id}
                  onClick={() => {
                    // Fly to location
                    if (onFlyToCoords && (note.lat || note.lon)) {
                      onFlyToCoords(note.lat, note.lon);
                    }
                    // Select product if visible
                    const vp = visibleProducts.find(p => p.productId === note.product_id);
                    if (vp && onSelectProduct) {
                      onSelectProduct(vp);
                    }
                  }}
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
                  {note.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {note.tags.map(tag => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 rounded-full text-[8px] bg-amber-500/10 text-amber-400/70 border border-amber-500/20"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </button>
              );
            })}
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
};

// Displayed Products Section Component
function DisplayedProductsSection({
  visibleProducts,
  activeOverlays,
  onSetOverlay,
  onSelectProduct,
  customDatasets,
  onCustomDatasetToggle,
}: {
  visibleProducts: VisibleProduct[];
  activeOverlays: ActiveOverlays;
  onSetOverlay?: (productId: string, type: OverlayType | null) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
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
