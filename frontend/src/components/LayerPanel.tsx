import { useState, useMemo, useEffect } from "react";
import type { VisibleProduct, BrowseProductType, BaseLayerType, BoundingBox } from "../pages/MainPage";

type InstrumentType = "CRISM" | "HIRISE" | "SHARAD";

// localStorage key for panel collapse state
const PANEL_COLLAPSED_KEY = "marslab-layer-panel-collapsed";

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
  onToggleBrowseProduct: _onToggleBrowseProduct,
  onSelectProduct,
  onDeactivateAll,
  onTurnOnAllBrowse,
  onTurnOnAllQuickviews,
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

  // Count active overlays
  const totalActiveOverlays =
    quickviewOverlays.length +
    highResOverlays.length +
    Array.from(browseOverlays.values()).reduce((sum, set) => sum + set.size, 0);

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
          <span
            className="material-symbols-outlined text-sm text-[#6b7c9c]"
            title="Browse Products"
          >
            image
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
    <div className="flex h-full w-80 flex-col border-r border-[#232f48] bg-[#101622] transition-all duration-300 ease-in-out">
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

        {/* Displayed Products Section */}
        <DisplayedProductsSection
          visibleProducts={visibleProducts}
          quickviewOverlays={quickviewOverlays}
          highResOverlays={highResOverlays}
          productsWithHighRes={productsWithHighRes}
          onToggleQuickview={onToggleQuickview}
          onToggleHighRes={onToggleHighRes}
          onSelectProduct={onSelectProduct}
        />

        {/* Active Products Section */}
        <ActiveProductsSection
          quickviewOverlays={quickviewOverlays}
          highResOverlays={highResOverlays}
          visibleProducts={visibleProducts}
          onDeactivateAll={onDeactivateAll}
        />
      </div>
    </div>
  );
}

// Instrument colors for badges
const INSTRUMENT_COLORS: Record<InstrumentType, { bg: string; text: string; border: string }> = {
  CRISM: { bg: "bg-cyan-500/20", text: "text-cyan-400", border: "border-cyan-500/30" },
  HIRISE: { bg: "bg-yellow-500/20", text: "text-yellow-400", border: "border-yellow-500/30" },
  SHARAD: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
};

// Displayed Products Section Component
function DisplayedProductsSection({
  visibleProducts,
  quickviewOverlays,
  highResOverlays,
  productsWithHighRes,
  onToggleQuickview,
  onToggleHighRes,
  onSelectProduct,
}: {
  visibleProducts: VisibleProduct[];
  quickviewOverlays: string[];
  highResOverlays: string[];
  productsWithHighRes: Set<string>;
  onToggleQuickview?: (productId: string) => void;
  onToggleHighRes?: (productId: string) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Group products by instrument
  const groupedProducts = useMemo(() => {
    const groups: Record<InstrumentType, VisibleProduct[]> = {
      CRISM: [],
      HIRISE: [],
      SHARAD: [],
    };
    for (const product of visibleProducts) {
      groups[product.instrument].push(product);
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
                      const hasQuickview = quickviewOverlays.includes(product.productId);
                      const hasHighRes = highResOverlays.includes(product.productId);
                      const canHighRes = productsWithHighRes.has(product.productId);

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

                          {/* Product ID - clickable */}
                          <span
                            className="flex-1 text-[10px] font-mono text-[#92a4c9] truncate cursor-pointer hover:text-white"
                            onClick={() => onSelectProduct?.(product)}
                            title={product.productId}
                          >
                            {product.productId}
                          </span>

                          {/* Toggle controls */}
                          <div className="flex items-center gap-1">
                            {/* Quickview toggle */}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onToggleQuickview?.(product.productId);
                              }}
                              className={`p-0.5 rounded transition-colors ${
                                hasQuickview
                                  ? "text-emerald-400 bg-emerald-500/20"
                                  : "text-slate-600 hover:text-slate-400"
                              }`}
                              title={hasQuickview ? "Hide Quickview" : "Show Quickview"}
                            >
                              <span className="material-symbols-outlined text-xs">visibility</span>
                            </button>

                            {/* High-res toggle */}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onToggleHighRes?.(product.productId);
                              }}
                              disabled={!canHighRes}
                              className={`p-0.5 rounded transition-colors ${
                                hasHighRes
                                  ? "text-purple-400 bg-purple-500/20"
                                  : canHighRes
                                    ? "text-slate-600 hover:text-slate-400"
                                    : "text-slate-700 cursor-not-allowed"
                              }`}
                              title={
                                !canHighRes
                                  ? "High-res not available"
                                  : hasHighRes
                                    ? "Hide High-Res"
                                    : "Show High-Res"
                              }
                            >
                              <span className="material-symbols-outlined text-xs">hd</span>
                            </button>
                          </div>
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

// Active Products Section Component (view-only)
function ActiveProductsSection({
  quickviewOverlays,
  highResOverlays,
  visibleProducts,
  onDeactivateAll,
}: {
  quickviewOverlays: string[];
  highResOverlays: string[];
  visibleProducts: VisibleProduct[];
  onDeactivateAll?: () => void;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Derive active products: those with quickview OR high-res ON
  const activeProducts = useMemo(() => {
    const activeIds = new Set([...quickviewOverlays, ...highResOverlays]);

    // Build list of active products with their active layers
    const result: Array<{
      productId: string;
      instrument: InstrumentType;
      hasQuickview: boolean;
      hasHighRes: boolean;
    }> = [];

    const seen = new Set<string>();

    // First, add products from visibleProducts that are active
    for (const product of visibleProducts) {
      if (activeIds.has(product.productId) && !seen.has(product.productId)) {
        seen.add(product.productId);
        result.push({
          productId: product.productId,
          instrument: product.instrument,
          hasQuickview: quickviewOverlays.includes(product.productId),
          hasHighRes: highResOverlays.includes(product.productId),
        });
      }
    }

    // Then, add any active products not in visibleProducts (outside viewport)
    for (const productId of activeIds) {
      if (!seen.has(productId)) {
        seen.add(productId);
        const instrument: InstrumentType = productId.startsWith("ESP_") ? "HIRISE" : "CRISM";
        result.push({
          productId,
          instrument,
          hasQuickview: quickviewOverlays.includes(productId),
          hasHighRes: highResOverlays.includes(productId),
        });
      }
    }

    return result;
  }, [quickviewOverlays, highResOverlays, visibleProducts]);

  // Group by instrument
  const groupedActive = useMemo(() => {
    const groups: Record<InstrumentType, typeof activeProducts> = {
      CRISM: [],
      HIRISE: [],
      SHARAD: [],
    };
    for (const product of activeProducts) {
      groups[product.instrument].push(product);
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
          Active Products
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
        <div className="space-y-3 max-h-48 overflow-y-auto scrollbar-dark">
          {activeCount === 0 ? (
            <div className="text-center py-4">
              <span className="material-symbols-outlined text-2xl text-[#3a4a68] mb-1">check_circle</span>
              <p className="text-[10px] text-[#6b7c9c]">No active overlays</p>
              <p className="text-[9px] text-[#4a5a7c] mt-1">Toggle QV or HD on products above</p>
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

                  {/* Products list (view-only) */}
                  <div className="space-y-1">
                    {products.map((product) => (
                      <div
                        key={product.productId}
                        className="flex items-center gap-2 p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50"
                      >
                        {/* Instrument badge */}
                        <span
                          className={`text-[8px] px-1 py-0.5 rounded font-bold ${instColors.bg} ${instColors.text} ${instColors.border} border`}
                        >
                          {instrument.slice(0, 3)}
                        </span>

                        {/* Product ID */}
                        <span
                          className="flex-1 text-[10px] font-mono text-[#92a4c9] truncate"
                          title={product.productId}
                        >
                          {product.productId}
                        </span>

                        {/* Active layer badges (view-only) */}
                        <div className="flex items-center gap-1">
                          {product.hasQuickview && (
                            <span className="text-[8px] px-1 py-0.5 bg-emerald-500/20 text-emerald-400 rounded border border-emerald-500/30 font-bold">
                              QV
                            </span>
                          )}
                          {product.hasHighRes && (
                            <span className="text-[8px] px-1 py-0.5 bg-purple-500/20 text-purple-400 rounded border border-purple-500/30 font-bold">
                              {product.instrument === "CRISM" ? "RGB" : "HD"}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
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
