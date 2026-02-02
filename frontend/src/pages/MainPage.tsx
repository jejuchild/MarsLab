import { useState, useCallback, useEffect } from "react";
import MapView from "../components/MapView";
import Inspector from "../components/Inspector";
import type { InspectorContext, RGBWavelengths } from "../components/Inspector";
import TopBar from "../components/TopBar";
import LayerPanel from "../components/LayerPanel";
import AppShell from "../components/layout/AppShell";

// Default CRISM wavelengths (in micrometers)
const DEFAULT_RGB_WAVELENGTHS: RGBWavelengths = {
  r: 2.53,
  g: 1.51,
  b: 1.08,
};

export type VisibleProduct = {
  productId: string;
  instrument: "HIRISE" | "CRISM" | "SHARAD";
  title?: string;  // HiRISE observation title (e.g., "Gullies in Arcadia Region")
};

// SHARAD popup state
export type SHARADPopup = {
  productId: string;
  quickviewUrl: string;
  startLat: number;
  startLon: number;
  stopLat: number;
  stopLon: number;
} | null;

// Browse product types for CRISM
export type BrowseProductType = "HYD" | "ICE" | "IC2";

// Base map layer types
export type BaseLayerType = "MOLA" | "HRSC";

// Bounding box type for view restriction
export type BoundingBox = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;

export default function MainPage() {
  // Selected footprint for Inspector
  const [selected, setSelected] = useState<InspectorContext | null>(null);

  // Base layer selection
  const [baseLayer, setBaseLayer] = useState<BaseLayerType>("MOLA");

  // Bounding box for view restriction
  const [viewBounds, setViewBounds] = useState<BoundingBox>(null);

  // Footprint layer toggles
  const [showCRISM, setShowCRISM] = useState(false);
  const [showHiRISE, setShowHiRISE] = useState(false);
  const [showSHARAD, setShowSHARAD] = useState(false);

  // Quick View global toggles (show all quickviews for instrument)
  const [showCRISMQuickview, setShowCRISMQuickview] = useState(false);
  const [showHiRISEQuickview, setShowHiRISEQuickview] = useState(false);

  // Browse product toggles (CRISM only)
  const [showBrowseHYD, setShowBrowseHYD] = useState(false);
  const [showBrowseICE, setShowBrowseICE] = useState(false);
  const [showBrowseIC2, setShowBrowseIC2] = useState(false);

  // Global overlay opacity (0-100)
  const [overlayOpacity, setOverlayOpacity] = useState(80);

  // Visible products in current map view
  const [visibleProducts, setVisibleProducts] = useState<VisibleProduct[]>([]);

  // Quickview overlays (toggled via eye button in LayerPanel - per-product)
  const [quickviewOverlays, setQuickviewOverlays] = useState<string[]>([]);

  // High-resolution overlays (toggled via HD button in LayerPanel/Inspector)
  const [highResOverlays, setHighResOverlays] = useState<string[]>([]);

  // Browse product overlays (product IDs with active browse overlays)
  const [browseOverlays, setBrowseOverlays] = useState<Map<string, Set<BrowseProductType>>>(new Map());

  // Product to fly to (set when clicking product_id in LayerPanel)
  const [flyToProductId, setFlyToProductId] = useState<string | null>(null);

  // Product to bring to front (for z-ordering high-res overlays)
  const [bringToFrontId, setBringToFrontId] = useState<string | null>(null);

  // CRISM RGB wavelengths
  const [rgbWavelengths, setRGBWavelengths] = useState<RGBWavelengths>(DEFAULT_RGB_WAVELENGTHS);

  // SHARAD popup state
  const [sharadPopup, setSharadPopup] = useState<SHARADPopup>(null);

  // Track which products have high-res data available
  const [productsWithHighRes, setProductsWithHighRes] = useState<Set<string>>(new Set());

  // Handle visible products update from map
  const handleVisibleProductsChange = useCallback((products: VisibleProduct[]) => {
    setVisibleProducts(products);
  }, []);

  // Toggle quickview overlay for a product
  const handleToggleQuickview = useCallback((productId: string) => {
    setQuickviewOverlays((prev) =>
      prev.includes(productId)
        ? prev.filter((id) => id !== productId)
        : [...prev, productId]
    );
  }, []);

  // Toggle high-res overlay for a product
  const handleToggleHighRes = useCallback((productId: string) => {
    setHighResOverlays((prev) =>
      prev.includes(productId)
        ? prev.filter((id) => id !== productId)
        : [...prev, productId]
    );
  }, []);

  // Handle clicking product_id in LayerPanel - fly to it and bring to front
  const handleSelectProduct = useCallback((product: VisibleProduct) => {
    // Open inspector
    setSelected({
      instrument: product.instrument,
      productId: product.productId,
      lat: 0,
      lon: 0,
      title: product.title,  // Pass title for HiRISE observations
    });

    // Fly to the product
    setFlyToProductId(product.productId);

    // Bring high-res overlay to front if it exists
    if (highResOverlays.includes(product.productId)) {
      setBringToFrontId(product.productId);
    }
  }, [highResOverlays]);

  // Clear flyTo after it's processed
  const handleFlyToComplete = useCallback(() => {
    setFlyToProductId(null);
  }, []);

  // Clear bringToFront after it's processed
  const handleBringToFrontComplete = useCallback(() => {
    setBringToFrontId(null);
  }, []);

  // Handle RGB wavelength changes from Inspector
  const handleRGBChange = useCallback((rgb: RGBWavelengths) => {
    setRGBWavelengths(rgb);
  }, []);

  // Handle SHARAD track click - open popup with quickview
  const handleSharadClick = useCallback((popup: SHARADPopup) => {
    setSharadPopup(popup);
  }, []);

  // Deactivate all overlays
  const handleDeactivateAll = useCallback(() => {
    setQuickviewOverlays([]);
    setHighResOverlays([]);
    setBrowseOverlays(new Map());
  }, []);

  // Toggle browse product for a specific product
  const handleToggleBrowseProduct = useCallback((productId: string, browseType: BrowseProductType) => {
    setBrowseOverlays((prev) => {
      const newMap = new Map(prev);
      const existing = newMap.get(productId) || new Set();
      const newSet = new Set(existing);

      if (newSet.has(browseType)) {
        newSet.delete(browseType);
      } else {
        newSet.add(browseType);
      }

      if (newSet.size === 0) {
        newMap.delete(productId);
      } else {
        newMap.set(productId, newSet);
      }

      return newMap;
    });
  }, []);

  // Turn on all browse products for visible CRISM products
  const handleTurnOnAllBrowse = useCallback(() => {
    const crismProducts = visibleProducts.filter(p => p.instrument === "CRISM");
    const newMap = new Map(browseOverlays); // Keep existing overlays

    crismProducts.forEach(p => {
      const existing = newMap.get(p.productId) || new Set<BrowseProductType>();
      if (showBrowseHYD) existing.add("HYD");
      if (showBrowseICE) existing.add("ICE");
      if (showBrowseIC2) existing.add("IC2");
      if (existing.size > 0) {
        newMap.set(p.productId, existing);
      }
    });

    setBrowseOverlays(newMap);
  }, [visibleProducts, showBrowseHYD, showBrowseICE, showBrowseIC2, browseOverlays]);

  // Turn on all quickviews for visible products
  const handleTurnOnAllQuickviews = useCallback(() => {
    const productIds = visibleProducts.map(p => p.productId);
    setQuickviewOverlays(productIds);
  }, [visibleProducts]);

  // Check high-res availability for visible products
  useEffect(() => {
    const checkHighResAvailability = async () => {
      const newHighResSet = new Set<string>();

      for (const product of visibleProducts) {
        try {
          const instrument = product.instrument.toLowerCase();
          const response = await fetch(`/api/exists/${instrument}/${encodeURIComponent(product.productId)}`);
          if (response.ok) {
            const data = await response.json();
            // has_core means .img/.lbl files exist for CRISM, or .jp2/.tif for HiRISE
            if (data.has_core) {
              newHighResSet.add(product.productId);
            }
          }
        } catch (e) {
          // If check fails, assume high-res is available (fail open)
          newHighResSet.add(product.productId);
        }
      }

      setProductsWithHighRes(newHighResSet);
    };

    if (visibleProducts.length > 0) {
      checkHighResAvailability();
    }
  }, [visibleProducts]);

  // Handle search result selection from TopBar
  const handleSearchSelect = useCallback((productId: string) => {
    const product = visibleProducts.find((p) => p.productId === productId);
    if (product) {
      handleSelectProduct(product);
    }
  }, [visibleProducts, handleSelectProduct]);

  // All searchable items (productId and title)
  const searchableItems = visibleProducts.map((p) => ({
    productId: p.productId,
    title: p.title,
  }));

  return (
    <AppShell
      header={
        <TopBar
          searchableItems={searchableItems}
          onSelectResult={handleSearchSelect}
        />
      }
      leftPanel={
        <LayerPanel
          // Base layer selection
          baseLayer={baseLayer}
          onBaseLayerChange={setBaseLayer}
          // View bounds restriction
          viewBounds={viewBounds}
          onViewBoundsChange={setViewBounds}
          // Footprint toggles
          showCRISM={showCRISM}
          showHiRISE={showHiRISE}
          showSHARAD={showSHARAD}
          onToggleCRISM={setShowCRISM}
          onToggleHiRISE={setShowHiRISE}
          onToggleSHARAD={setShowSHARAD}
          // Quick View toggles
          showCRISMQuickview={showCRISMQuickview}
          showHiRISEQuickview={showHiRISEQuickview}
          onToggleCRISMQuickview={setShowCRISMQuickview}
          onToggleHiRISEQuickview={setShowHiRISEQuickview}
          // Browse product toggles
          showBrowseHYD={showBrowseHYD}
          showBrowseICE={showBrowseICE}
          showBrowseIC2={showBrowseIC2}
          onToggleBrowseHYD={setShowBrowseHYD}
          onToggleBrowseICE={setShowBrowseICE}
          onToggleBrowseIC2={setShowBrowseIC2}
          // Global opacity
          overlayOpacity={overlayOpacity}
          onOpacityChange={setOverlayOpacity}
          // Existing props
          visibleProducts={visibleProducts}
          quickviewOverlays={quickviewOverlays}
          highResOverlays={highResOverlays}
          browseOverlays={browseOverlays}
          onToggleQuickview={handleToggleQuickview}
          onToggleHighRes={handleToggleHighRes}
          onToggleBrowseProduct={handleToggleBrowseProduct}
          onSelectProduct={handleSelectProduct}
          onDeactivateAll={handleDeactivateAll}
          onTurnOnAllBrowse={handleTurnOnAllBrowse}
          onTurnOnAllQuickviews={handleTurnOnAllQuickviews}
          productsWithHighRes={productsWithHighRes}
        />
      }
      rightPanel={
        selected ? (
          <Inspector
            selected={selected}
            onClose={() => setSelected(null)}
            isHighResActive={highResOverlays.includes(selected.productId)}
            onToggleHighRes={() => handleToggleHighRes(selected.productId)}
            rgbWavelengths={rgbWavelengths}
            onRGBChange={handleRGBChange}
            hasHighResData={productsWithHighRes.has(selected.productId)}
          />
        ) : null
      }
    >
      {/* Map Canvas */}
      <MapView
        baseLayer={baseLayer}
        viewBounds={viewBounds}
        onSelect={setSelected}
        showCRISM={showCRISM}
        showHiRISE={showHiRISE}
        showSHARAD={showSHARAD}
        onSharadClick={handleSharadClick}
        quickviewOverlays={quickviewOverlays}
        highResOverlays={highResOverlays}
        browseOverlays={browseOverlays}
        overlayOpacity={overlayOpacity / 100}
        onVisibleProductsChange={handleVisibleProductsChange}
        flyToProductId={flyToProductId}
        onFlyToComplete={handleFlyToComplete}
        bringToFrontId={bringToFrontId}
        onBringToFrontComplete={handleBringToFrontComplete}
        rgbWavelengths={rgbWavelengths}
      />

      {/* SHARAD Quickview Popup */}
      {sharadPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          <div className="relative max-w-4xl max-h-[90vh] bg-[#101622] rounded-lg border border-[#232f48] shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48] bg-[#0a0f18]">
              <div>
                <h3 className="text-white font-bold text-sm">{sharadPopup.productId}</h3>
                <p className="text-[#92a4c9] text-[10px] mt-0.5">
                  SHARAD Radargram: ({sharadPopup.startLat.toFixed(2)}°, {sharadPopup.startLon.toFixed(2)}°) → ({sharadPopup.stopLat.toFixed(2)}°, {sharadPopup.stopLon.toFixed(2)}°)
                </p>
              </div>
              <button
                onClick={() => setSharadPopup(null)}
                className="p-1.5 rounded hover:bg-[#232f48] transition-colors text-[#92a4c9] hover:text-white"
              >
                <span className="material-symbols-outlined text-xl">close</span>
              </button>
            </div>

            {/* Image */}
            <div className="p-4 overflow-auto max-h-[calc(90vh-120px)]">
              <img
                src={sharadPopup.quickviewUrl}
                alt={`SHARAD ${sharadPopup.productId}`}
                className="max-w-full h-auto"
                style={{ imageRendering: "crisp-edges" }}
              />
            </div>

            {/* Footer with Activate High-Res button placeholder */}
            <div className="px-4 py-3 border-t border-[#232f48] bg-[#0a0f18] flex justify-end gap-2">
              <button
                disabled
                className="px-3 py-1.5 text-[11px] font-medium bg-[#1a2333] border border-[#232f48] rounded text-[#6b7c9c] cursor-not-allowed"
                title="High-resolution data not available yet"
              >
                Activate High-Res Image (Coming Soon)
              </button>
              <button
                onClick={() => setSharadPopup(null)}
                className="px-3 py-1.5 text-[11px] font-medium bg-primary/20 border border-primary/50 rounded text-primary hover:bg-primary/30 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </AppShell>
  );
}
