import { useState } from "react";
import type { VisibleProduct, BrowseProductType, BaseLayerType, BoundingBox } from "../pages/MainPage";

interface LayerPanelProps {
  // Base layer selection
  baseLayer: BaseLayerType;
  onBaseLayerChange: (layer: BaseLayerType) => void;

  // View bounds restriction
  viewBounds: BoundingBox;
  onViewBoundsChange: (bounds: BoundingBox) => void;

  // Footprint toggles
  showCRISM: boolean;
  showHiRISE: boolean;
  showSHARAD: boolean;
  onToggleCRISM: (v: boolean) => void;
  onToggleHiRISE: (v: boolean) => void;
  onToggleSHARAD: (v: boolean) => void;

  // Quick View toggles
  showCRISMQuickview: boolean;
  showHiRISEQuickview: boolean;
  onToggleCRISMQuickview: (v: boolean) => void;
  onToggleHiRISEQuickview: (v: boolean) => void;

  // Browse product toggles
  showBrowseHYD: boolean;
  showBrowseICE: boolean;
  showBrowseIC2: boolean;
  onToggleBrowseHYD: (v: boolean) => void;
  onToggleBrowseICE: (v: boolean) => void;
  onToggleBrowseIC2: (v: boolean) => void;

  // Score layer toggles (placeholder)
  showIceScore: boolean;
  showHydratedScore: boolean;
  onToggleIceScore: (v: boolean) => void;
  onToggleHydratedScore: (v: boolean) => void;

  // Global opacity
  overlayOpacity: number;
  onOpacityChange: (v: number) => void;

  // Product data
  visibleProducts?: VisibleProduct[];
  quickviewOverlays?: string[];
  highResOverlays?: string[];
  browseOverlays?: Map<string, Set<BrowseProductType>>;
  productsWithHighRes?: Set<string>;

  // Callbacks
  onToggleQuickview?: (productId: string) => void;
  onToggleHighRes?: (productId: string) => void;
  onToggleBrowseProduct?: (productId: string, browseType: BrowseProductType) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
  onDeactivateAll?: () => void;
  onTurnOnAllBrowse?: () => void;
  onTurnOnAllQuickviews?: () => void;

  // ICE Score Filter
  iceScoreThreshold: number;  // n: ICE score threshold (0-2)
  iceAreaThreshold: number;   // m: area percentage threshold (0-100)
  iceFilterActive: boolean;
  iceFilterLoading: boolean;
  onIceScoreThresholdChange: (v: number) => void;
  onIceAreaThresholdChange: (v: number) => void;
  onApplyIceFilter: () => void;
  onClearIceFilter: () => void;
}

// View Bounds Input Component
function ViewBoundsInput({
  viewBounds,
  onViewBoundsChange,
}: {
  viewBounds: BoundingBox;
  onViewBoundsChange: (bounds: BoundingBox) => void;
}) {
  const [minLat, setMinLat] = useState(viewBounds?.minLat?.toString() ?? "");
  const [maxLat, setMaxLat] = useState(viewBounds?.maxLat?.toString() ?? "");
  const [westLon, setWestLon] = useState(viewBounds?.westLon?.toString() ?? "");
  const [eastLon, setEastLon] = useState(viewBounds?.eastLon?.toString() ?? "");

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

  return (
    <div className="mt-4 pt-3 border-t border-[#232f48]">
      <h4 className="text-[#92a4c9] text-[9px] font-bold uppercase tracking-widest mb-2">
        View Bounds
      </h4>
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
  onToggleCRISM,
  onToggleHiRISE,
  onToggleSHARAD,
  // Quick Views
  showCRISMQuickview,
  showHiRISEQuickview,
  onToggleCRISMQuickview,
  onToggleHiRISEQuickview,
  // Browse Products
  showBrowseHYD,
  showBrowseICE,
  showBrowseIC2,
  onToggleBrowseHYD,
  onToggleBrowseICE,
  onToggleBrowseIC2,
  // Score layers
  showIceScore,
  showHydratedScore,
  onToggleIceScore,
  onToggleHydratedScore,
  // Opacity
  overlayOpacity,
  onOpacityChange,
  // Data
  visibleProducts = [],
  quickviewOverlays = [],
  highResOverlays = [],
  browseOverlays = new Map(),
  productsWithHighRes = new Set(),
  // Callbacks
  onToggleQuickview,
  onToggleHighRes,
  onToggleBrowseProduct,
  onSelectProduct,
  onDeactivateAll,
  onTurnOnAllBrowse,
  onTurnOnAllQuickviews,
  // ICE Filter
  iceScoreThreshold = 1.0,
  iceAreaThreshold = 10,
  iceFilterActive = false,
  iceFilterLoading = false,
  onIceScoreThresholdChange,
  onIceAreaThresholdChange,
  onApplyIceFilter,
  onClearIceFilter,
}: LayerPanelProps) {
  const hiriseProducts = visibleProducts.filter((p) => p.instrument === "HIRISE");
  const crismProducts = visibleProducts.filter((p) => p.instrument === "CRISM");

  // Count active overlays
  const totalActiveOverlays =
    quickviewOverlays.length +
    highResOverlays.length +
    Array.from(browseOverlays.values()).reduce((sum, set) => sum + set.size, 0);

  return (
    <div className="flex h-full w-80 flex-col border-r border-[#232f48] bg-[#101622]">
      {/* Header */}
      <div className="p-4 border-b border-[#232f48]">
        <div className="flex items-center justify-between mb-1">
          <h1 className="text-white text-xs font-bold uppercase tracking-wider">
            Control Center
          </h1>
          <span className="text-[10px] text-[#92a4c9] font-mono">
            {visibleProducts.length} products
          </span>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto scrollbar-dark">
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

          {/* View Bounds */}
          <ViewBoundsInput viewBounds={viewBounds} onViewBoundsChange={onViewBoundsChange} />
        </div>

        {/* Filter Section */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            Filter
          </h3>

          {/* ICE Score Filter */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-sky-400">ICE Score Filter</span>
              {iceFilterActive && (
                <span className="text-[9px] px-1.5 py-0.5 bg-sky-500/20 text-sky-400 rounded border border-sky-500/30">
                  Active
                </span>
              )}
            </div>

            {/* ICE Score Threshold (n) */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-[#92a4c9]">ICE Score Threshold (n)</span>
                <span className="text-[10px] text-white font-mono">{iceScoreThreshold.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={iceScoreThreshold}
                onChange={(e) => onIceScoreThresholdChange?.(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:h-3
                  [&::-webkit-slider-thumb]:w-3
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-sky-400
                  [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <div className="flex justify-between text-[8px] text-[#6b7c9c] mt-1">
                <span>0</span>
                <span>1</span>
                <span>2</span>
              </div>
            </div>

            {/* Area Threshold (m%) */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-[#92a4c9]">Area Threshold (m%)</span>
                <span className="text-[10px] text-white font-mono">{iceAreaThreshold}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={iceAreaThreshold}
                onChange={(e) => onIceAreaThresholdChange?.(parseInt(e.target.value))}
                className="w-full h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:h-3
                  [&::-webkit-slider-thumb]:w-3
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-sky-400
                  [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <div className="flex justify-between text-[8px] text-[#6b7c9c] mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <p className="text-[9px] text-[#6b7c9c]">
              Keep CRISM if ≥{iceAreaThreshold}% pixels have ICE score ≥{iceScoreThreshold.toFixed(1)}
            </p>

            <div className="flex gap-2">
              <button
                onClick={onApplyIceFilter}
                disabled={iceFilterLoading}
                className="flex-1 px-2 py-1.5 text-[10px] font-bold bg-sky-500/20 border border-sky-500/50 rounded text-sky-400 hover:bg-sky-500/30 transition-colors disabled:opacity-50 uppercase"
              >
                {iceFilterLoading ? "Applying..." : "Apply"}
              </button>
              <button
                onClick={onClearIceFilter}
                disabled={!iceFilterActive}
                className="flex-1 px-2 py-1.5 text-[10px] font-medium bg-[#1a2333] border border-[#232f48] rounded text-[#92a4c9] hover:border-[#3a4a68] transition-colors disabled:opacity-50 uppercase"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Footprints Section */}
        <div className="p-4 border-b border-[#232f48]">
          <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3">
            Footprints
          </h3>
          <div className="grid grid-cols-3 gap-2">
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                showCRISM
                  ? "bg-primary/20 border border-primary/50"
                  : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
              }`}
            >
              <input
                type="checkbox"
                checked={showCRISM}
                onChange={(e) => onToggleCRISM(e.target.checked)}
                className="rounded bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px] font-medium">CRISM</span>
            </label>
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                showHiRISE
                  ? "bg-primary/20 border border-primary/50"
                  : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
              }`}
            >
              <input
                type="checkbox"
                checked={showHiRISE}
                onChange={(e) => onToggleHiRISE(e.target.checked)}
                className="rounded bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px] font-medium">HiRISE</span>
            </label>
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
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
            </label>
          </div>
        </div>

        {/* Quick Views Section */}
        <div className="p-4 border-b border-[#232f48]">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest">
              Quick Views
            </h3>
            <button
              onClick={onTurnOnAllQuickviews}
              className="text-[9px] px-2 py-0.5 rounded bg-primary text-white font-bold hover:bg-blue-600 transition-colors uppercase"
            >
              Turn On All
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                showCRISMQuickview
                  ? "bg-primary/20 border border-primary/50"
                  : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
              }`}
            >
              <input
                type="checkbox"
                checked={showCRISMQuickview}
                onChange={(e) => onToggleCRISMQuickview(e.target.checked)}
                className="rounded bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px] font-medium">CRISM</span>
            </label>
            <label
              className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                showHiRISEQuickview
                  ? "bg-primary/20 border border-primary/50"
                  : "bg-[#1a2333] border border-[#232f48] hover:border-primary/30"
              }`}
            >
              <input
                type="checkbox"
                checked={showHiRISEQuickview}
                onChange={(e) => onToggleHiRISEQuickview(e.target.checked)}
                className="rounded bg-[#0a0f18] border-[#232f48] text-primary focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px] font-medium">HiRISE</span>
            </label>
          </div>
        </div>

        {/* Browse Products Section */}
        <div className="p-4 border-b border-[#232f48]">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest">
              Browse Products
            </h3>
            <button
              onClick={onTurnOnAllBrowse}
              className="text-[9px] px-2 py-0.5 rounded bg-primary text-white font-bold hover:bg-blue-600 transition-colors uppercase"
            >
              Turn On All
            </button>
          </div>

          <div className="space-y-1.5">
            {/* HYD */}
            <label className="flex items-center justify-between p-2 rounded hover:bg-[#1a2333] group cursor-pointer">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showBrowseHYD}
                  onChange={(e) => onToggleBrowseHYD(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-fuchsia-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[12px] text-fuchsia-500 font-medium">HYD</span>
              </div>
              <span className="text-[10px] text-[#92a4c9] opacity-0 group-hover:opacity-100 transition-opacity">
                Hydrated Minerals
              </span>
            </label>

            {/* ICE */}
            <label className="flex items-center justify-between p-2 rounded hover:bg-[#1a2333] group cursor-pointer">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showBrowseICE}
                  onChange={(e) => onToggleBrowseICE(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-blue-500 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[12px] text-blue-500 font-medium">ICE</span>
              </div>
              <span className="text-[10px] text-[#92a4c9] opacity-0 group-hover:opacity-100 transition-opacity">
                Water Ice
              </span>
            </label>

            {/* IC2 */}
            <label className="flex items-center justify-between p-2 rounded hover:bg-[#1a2333] group cursor-pointer">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showBrowseIC2}
                  onChange={(e) => onToggleBrowseIC2(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-cyan-400 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[12px] text-cyan-400 font-medium">IC2</span>
              </div>
              <span className="text-[10px] text-[#92a4c9] opacity-0 group-hover:opacity-100 transition-opacity">
                CO2 Ice
              </span>
            </label>

            {/* Ice Score */}
            <label className="flex items-center justify-between p-2 rounded hover:bg-[#1a2333] group cursor-pointer">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showIceScore}
                  onChange={(e) => onToggleIceScore(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-sky-400 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[12px] text-sky-400 font-medium">Ice Score</span>
              </div>
              <span className="text-[10px] text-[#92a4c9] opacity-0 group-hover:opacity-100 transition-opacity">
                CRISM Analysis
              </span>
            </label>

            {/* Hydrated Mineral Score */}
            <label className="flex items-center justify-between p-2 rounded hover:bg-[#1a2333] group cursor-pointer">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showHydratedScore}
                  onChange={(e) => onToggleHydratedScore(e.target.checked)}
                  className="rounded bg-[#0a0f18] border-[#232f48] text-amber-400 focus:ring-0 focus:ring-offset-0"
                />
                <span className="text-[12px] text-amber-400 font-medium">Hydration Score</span>
              </div>
              <span className="text-[10px] text-[#92a4c9] opacity-0 group-hover:opacity-100 transition-opacity">
                CRISM Analysis
              </span>
            </label>
          </div>

          {/* Global Opacity Slider */}
          <div className="mt-4 pt-3 border-t border-[#232f48]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-[#92a4c9] uppercase tracking-wider">
                Overlay Opacity
              </span>
              <span className="text-[10px] text-white font-mono">{overlayOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={overlayOpacity}
              onChange={(e) => onOpacityChange(Number(e.target.value))}
              className="w-full h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-3
                [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-primary
                [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        </div>

        {/* In Current View Section */}
        {(showHiRISE || showCRISM) && visibleProducts.length > 0 && (
          <div className="p-4 border-b border-[#232f48]">
            <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-3 flex items-center justify-between">
              In Current View
              <span className="text-primary text-[10px]">{visibleProducts.length} items</span>
            </h3>

            <div className="space-y-1 max-h-48 overflow-y-auto scrollbar-dark">
              {/* CRISM Products */}
              {showCRISM &&
                crismProducts.slice(0, 10).map((product) => {
                  const hasQuickview = quickviewOverlays.includes(product.productId);
                  const hasHighRes = highResOverlays.includes(product.productId);
                  const productBrowse = browseOverlays.get(product.productId);

                  return (
                    <div
                      key={product.productId}
                      className="flex items-center justify-between p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50 hover:bg-[#1a2333] transition-colors cursor-pointer"
                      onClick={() => onSelectProduct?.(product)}
                    >
                      <span className="text-[11px] font-mono text-[#92a4c9] truncate max-w-[140px]">
                        {product.productId.split("_")[0]}
                      </span>
                      <div className="flex gap-1">
                        {hasHighRes && (
                          <span className="text-[9px] px-1 bg-green-500/20 text-green-400 rounded uppercase">
                            RGB
                          </span>
                        )}
                        {hasQuickview && !hasHighRes && (
                          <span className="text-[9px] px-1 bg-primary/20 text-primary rounded uppercase">
                            QV
                          </span>
                        )}
                        {productBrowse?.has("HYD") && (
                          <span className="text-[9px] px-1 bg-fuchsia-500/20 text-fuchsia-400 rounded uppercase">
                            HYD
                          </span>
                        )}
                        {productBrowse?.has("ICE") && (
                          <span className="text-[9px] px-1 bg-blue-500/20 text-blue-400 rounded uppercase">
                            ICE
                          </span>
                        )}
                        {productBrowse?.has("IC2") && (
                          <span className="text-[9px] px-1 bg-cyan-400/20 text-cyan-400 rounded uppercase">
                            IC2
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}

              {/* HiRISE Products */}
              {showHiRISE &&
                hiriseProducts.slice(0, 10).map((product) => {
                  const hasQuickview = quickviewOverlays.includes(product.productId);
                  const hasHighRes = highResOverlays.includes(product.productId);

                  return (
                    <div
                      key={product.productId}
                      className="flex items-center justify-between p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50 hover:bg-[#1a2333] transition-colors cursor-pointer"
                      onClick={() => onSelectProduct?.(product)}
                    >
                      <span className="text-[11px] font-mono text-[#92a4c9] truncate max-w-[140px]">
                        {product.productId}
                      </span>
                      <div className="flex gap-1">
                        {hasHighRes && (
                          <span className="text-[9px] px-1 bg-green-500/20 text-green-400 rounded uppercase">
                            HD
                          </span>
                        )}
                        {hasQuickview && !hasHighRes && (
                          <span className="text-[9px] px-1 bg-primary/20 text-primary rounded uppercase">
                            QV
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}

              {visibleProducts.length > 10 && (
                <div className="text-center text-[10px] text-[#92a4c9] py-2">
                  +{visibleProducts.length - 10} more...
                </div>
              )}
            </div>
          </div>
        )}

        {/* Active Overlays Section */}
        {totalActiveOverlays > 0 && (
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest">
                Active Overlays
              </h3>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-green-400">{totalActiveOverlays} active</span>
                <button
                  onClick={onDeactivateAll}
                  className="text-[9px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors border border-red-500/30 font-bold uppercase"
                >
                  Clear All
                </button>
              </div>
            </div>

            <div className="space-y-2 max-h-60 overflow-y-auto scrollbar-dark">
              {/* High-res / RGB Overlays */}
              {highResOverlays.map((productId) => {
                const isHiRISE = productId.startsWith("ESP_");
                return (
                  <div
                    key={`highres-${productId}`}
                    className="rounded-lg border border-[#232f48] bg-[#1a2333]/20 p-3"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-mono font-bold text-white truncate max-w-[160px]">
                        {productId.split("_")[0]}
                      </span>
                      <div className="flex gap-1 items-center">
                        <span
                          className={`text-[8px] px-1.5 py-0.5 rounded-sm border font-bold uppercase ${
                            isHiRISE
                              ? "bg-blue-500/20 text-blue-400 border-blue-500/30"
                              : "bg-green-500/20 text-green-400 border-green-500/30"
                          }`}
                        >
                          {isHiRISE ? "HD" : "RGB"}
                        </span>
                        <button
                          onClick={() => onToggleHighRes?.(productId)}
                          className="p-0.5 text-red-400 hover:text-red-300 transition-colors"
                        >
                          <span className="material-symbols-outlined text-sm">close</span>
                        </button>
                      </div>
                    </div>
                    <div className="mt-2 text-[10px] text-[#92a4c9]">
                      Opacity: {overlayOpacity}%
                    </div>
                  </div>
                );
              })}

              {/* Quickview Overlays */}
              {quickviewOverlays
                .filter((id) => !highResOverlays.includes(id))
                .map((productId) => {
                  const isHiRISE = productId.startsWith("ESP_");
                  return (
                    <div
                      key={`quickview-${productId}`}
                      className="rounded-lg border border-[#232f48] bg-[#1a2333]/20 p-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-mono font-bold text-white truncate max-w-[160px]">
                          {productId.split("_")[0]}
                        </span>
                        <div className="flex gap-1 items-center">
                          <span className="text-[8px] px-1.5 py-0.5 bg-white/10 text-[#92a4c9] rounded-sm border border-white/10 font-bold uppercase">
                            Quickview
                          </span>
                          <button
                            onClick={() => onToggleQuickview?.(productId)}
                            className="p-0.5 text-red-400 hover:text-red-300 transition-colors"
                          >
                            <span className="material-symbols-outlined text-sm">close</span>
                          </button>
                        </div>
                      </div>
                      <div className="mt-2 text-[10px] text-[#92a4c9]">
                        Opacity: {overlayOpacity}%
                      </div>
                    </div>
                  );
                })}

              {/* Browse Overlays */}
              {Array.from(browseOverlays.entries()).map(([productId, types]) => (
                <div
                  key={`browse-${productId}`}
                  className="rounded-lg border border-[#232f48] bg-[#1a2333]/20 p-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono font-bold text-white truncate max-w-[120px]">
                      {productId.split("_")[0]}
                    </span>
                    <div className="flex gap-1 items-center">
                      {types.has("HYD") && (
                        <span className="text-[8px] px-1.5 py-0.5 bg-fuchsia-500/20 text-fuchsia-400 rounded-sm border border-fuchsia-500/30 font-bold">
                          HYD
                        </span>
                      )}
                      {types.has("ICE") && (
                        <span className="text-[8px] px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded-sm border border-blue-500/30 font-bold">
                          ICE
                        </span>
                      )}
                      {types.has("IC2") && (
                        <span className="text-[8px] px-1.5 py-0.5 bg-cyan-400/20 text-cyan-400 rounded-sm border border-cyan-400/30 font-bold">
                          IC2
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-2 text-[10px] text-[#92a4c9]">
                    Opacity: {overlayOpacity}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
