import { useState, useCallback, useEffect, useMemo } from "react";
import MapView from "../components/MapView";
import Inspector from "../components/Inspector";
import type { InspectorContext, RGBWavelengths } from "../components/Inspector";
import SlopeAnalysis from "../components/SlopeAnalysis";
import Slope3DViewer from "../components/Slope3DViewer";
import HiRiseDTM3DViewer from "../components/HiRiseDTM3DViewer";
import type { HiRiseDTMPoint } from "../components/HiRiseDTM3DViewer";
import type { TerrainPoint } from "../components/SlopeAnalysis";
import LineProfile from "../components/LineProfile";
import type { ProfilePoint } from "../components/LineProfile";
import TopBar from "../components/TopBar";
import LayerPanel from "../components/LayerPanel";
import type { IceScoreFilter } from "../components/LayerPanel";
import SharadHiresInspector from "../components/SharadHiresInspector";
import FieldNoteModal from "../components/FieldNoteModal";
import AiAnalysisPanel from "../components/AiAnalysisPanel";
import { listFieldNotes } from "../api/fieldnotes";
import type { FieldNote } from "../api/fieldnotes";
import AppShell from "../components/layout/AppShell";

// Default CRISM wavelengths (in micrometers)
const DEFAULT_RGB_WAVELENGTHS: RGBWavelengths = {
  r: 2.53,
  g: 1.51,
  b: 1.08,
};

// Cache for HiRISE DTM index
let hiriseDTMIndexCache: { features: Array<{ properties: { product_id: string; west: number; east: number; south: number; north: number } }> } | null = null;

// Get DTM center coordinates (guaranteed to be within bounds)
async function getDTMCenter(productId: string): Promise<{ lat: number; lon: number } | null> {
  if (!hiriseDTMIndexCache) {
    try {
      const res = await fetch("/hirise_dtm_index.geojson");
      if (res.ok) {
        hiriseDTMIndexCache = await res.json();
      }
    } catch {
      return null;
    }
  }

  if (!hiriseDTMIndexCache?.features) return null;

  for (const feature of hiriseDTMIndexCache.features) {
    if (feature.properties?.product_id === productId) {
      const { west, east, south, north } = feature.properties;
      return {
        lat: (south + north) / 2,
        lon: (west + east) / 2,
      };
    }
  }
  return null;
}

export type VisibleProduct = {
  productId: string;
  instrument: "HIRISE" | "CRISM" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM";
  title?: string;  // HiRISE observation title (e.g., "Gullies in Arcadia Region")
  lat?: number;    // Center latitude (from footprint)
  lon?: number;    // Center longitude (from footprint)
};

// Custom user-uploaded dataset
export interface CustomDataset {
  id: string;
  name: string;
  bounds: { west: number; south: number; east: number; north: number };
  crs: string;
  crs_valid: boolean;
  crs_warning?: string | null;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  nodata: number | null;
  created_at: string;
  original_filename: string;
  visible: boolean;
  opacity: number;  // 0-100
}

// SHARAD popup state
export type SHARADPopup = {
  productId: string;
  quickviewUrl: string;
  startLat: number;
  startLon: number;
  stopLat: number;
  stopLon: number;
} | null;

// Unified overlay types - one per product at a time
export type OverlayType =
  | "quickview"
  | "highres"
  | "browse_HYD"
  | "browse_ICE"
  | "browse_IC2"
  | "score_ice"
  | "score_hyd";

// Product overlay state - unified structure
export type ProductOverlay = {
  type: OverlayType;
  opacity: number;  // 0-100
};

// Active overlays map: productId -> overlay state
export type ActiveOverlays = Map<string, ProductOverlay>;

// Base map layer types
export type BaseLayerType = "MOLA" | "HRSC";

// Map mode types (2D flat view vs 3D globe)
export type MapMode = "2D" | "3D";

// Bounding box type for view restriction
export type BoundingBox = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;

// Helper to check if overlay type is a browse product
export function isBrowseOverlay(type: OverlayType): boolean {
  return type.startsWith("browse_");
}

// Helper to get browse type from overlay type
export function getBrowseType(type: OverlayType): "HYD" | "ICE" | "IC2" | null {
  if (type === "browse_HYD") return "HYD";
  if (type === "browse_ICE") return "ICE";
  if (type === "browse_IC2") return "IC2";
  return null;
}

// Helper to check if overlay type is a score product
export function isScoreOverlay(type: OverlayType): boolean {
  return type === "score_ice" || type === "score_hyd";
}

// Helper to get score type from overlay type
export function getScoreType(type: OverlayType): "score_ice" | "score_hyd" | null {
  if (type === "score_ice") return "score_ice";
  if (type === "score_hyd") return "score_hyd";
  return null;
}

export default function MainPage() {
  // Selected footprint for Inspector
  const [selected, setSelected] = useState<InspectorContext | null>(null);

  // Terrain click point for slope analysis (when clicking empty terrain)
  const [terrainPoint, setTerrainPoint] = useState<TerrainPoint | null>(null);

  // Analysis mode: mutually exclusive slope / slope3d / hirise_dtm_3d / line / ai_analysis
  type AnalysisMode = "slope" | "slope3d" | "hirise_dtm_3d" | "line" | "ai_analysis" | null;
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>(null);

  // Slope 3D analysis point (separate from regular slope terrainPoint)
  const [slope3DPoint, setSlope3DPoint] = useState<TerrainPoint | null>(null);

  // HiRISE DTM 3D analysis point (requires product_id + lat/lon)
  const [hiRiseDTM3DPoint, setHiRiseDTM3DPoint] = useState<HiRiseDTMPoint | null>(null);

  // Active HiRISE DTM product (for terrain clicks in hirise_dtm_3d mode)
  const [activeDTMProduct, setActiveDTMProduct] = useState<string | null>(null);

  // Line profile state: two endpoints
  const [linePoints, setLinePoints] = useState<ProfilePoint[]>([]);
  const [lineProfileData, setLineProfileData] = useState<{ start: ProfilePoint; end: ProfilePoint } | null>(null);

  // Base layer selection
  const [baseLayer, setBaseLayer] = useState<BaseLayerType>("MOLA");

  // Map mode (2D flat view vs 3D globe)
  const [mapMode, setMapMode] = useState<MapMode>("2D");

  // Bounding box for view restriction
  const [viewBounds, setViewBounds] = useState<BoundingBox>(null);

  // View bound selection mode (drag to select region on map)
  const [viewBoundSelectionMode, setViewBoundSelectionMode] = useState(false);

  // Footprint layer toggles (visibility only - does NOT trigger loading)
  const [showCRISM, setShowCRISM] = useState(false);
  const [showHiRISE, setShowHiRISE] = useState(false);
  const [showSHARAD, setShowSHARAD] = useState(false);
  const [showCTX, setShowCTX] = useState(false);
  const [showSharadHighres, setShowSharadHighres] = useState(false);
  const [showHiRISEDTM, setShowHiRISEDTM] = useState(false);

  // Explicit footprint loading state
  type FootprintLoadTrigger = { instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM"; timestamp: number } | null;
  const [loadFootprintsTrigger, setLoadFootprintsTrigger] = useState<FootprintLoadTrigger>(null);
  const [footprintsLoading, setFootprintsLoading] = useState<{ crism: boolean; hirise: boolean; sharad: boolean; sharad_highres: boolean; ctx: boolean; hirise_dtm: boolean }>({
    crism: false,
    hirise: false,
    sharad: false,
    sharad_highres: false,
    ctx: false,
    hirise_dtm: false,
  });
  const [footprintCounts, setFootprintCounts] = useState<{
    crism: { count: number; truncated: boolean; total: number } | null;
    hirise: { count: number; truncated: boolean; total: number } | null;
    sharad: { count: number; truncated: boolean; total: number } | null;
    sharad_highres: { count: number; truncated: boolean; total: number } | null;
    ctx: { count: number; truncated: boolean; total: number } | null;
    hirise_dtm: { count: number; truncated: boolean; total: number } | null;
  }>({ crism: null, hirise: null, sharad: null, sharad_highres: null, ctx: null, hirise_dtm: null });

  // Visible products in current map view
  const [visibleProducts, setVisibleProducts] = useState<VisibleProduct[]>([]);

  // Unified active overlays - ONE overlay per product with per-product opacity
  const [activeOverlays, setActiveOverlays] = useState<ActiveOverlays>(new Map());

  // Product to fly to (set when clicking product_id in LayerPanel)
  const [flyToProductId, setFlyToProductId] = useState<string | null>(null);

  // Product to bring to front (for z-ordering overlays)
  const [bringToFrontId, setBringToFrontId] = useState<string | null>(null);

  // CRISM RGB wavelengths
  const [rgbWavelengths, setRGBWavelengths] = useState<RGBWavelengths>(DEFAULT_RGB_WAVELENGTHS);

  // SHARAD popup state
  const [sharadPopup, setSharadPopup] = useState<SHARADPopup>(null);

  // Track which products have high-res data available
  const [productsWithHighRes, setProductsWithHighRes] = useState<Set<string>>(new Set());

  // Ice score filter state
  const [iceScoreFilter, setIceScoreFilter] = useState<IceScoreFilter>({
    enabled: false,
    minScore: 0.3,
    minPercent: 5,
  });

  // SHARAD High-Res Inspector — opens when a product is selected
  const [sharadHiresProductId, setSharadHiresProductId] = useState<string | null>(null);

  // Custom user-uploaded datasets
  const [showCustomData, setShowCustomData] = useState(false);
  const [customDataLoading, setCustomDataLoading] = useState(false);
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);

  // Filtered product IDs from API (null when not filtering, Set when filtering)
  const [filteredProductIds, setFilteredProductIds] = useState<Set<string> | null>(null);

  // Field Notes state
  const [fieldNotes, setFieldNotes] = useState<FieldNote[]>([]);
  const [showFieldNoteModal, setShowFieldNoteModal] = useState<{
    productId: string; instrument: string; lat: number; lon: number;
  } | null>(null);
  const [showFieldNotesOnMap, setShowFieldNotesOnMap] = useState(false);
  const [fieldNoteActiveTag, setFieldNoteActiveTag] = useState<string | null>(null);

  // Notes to show on map: only those matching the active tag (none if no tag selected)
  const mapFieldNotes = useMemo(
    () => fieldNoteActiveTag
      ? fieldNotes.filter(n => n.tags.includes(fieldNoteActiveTag))
      : [],
    [fieldNotes, fieldNoteActiveTag]
  );

  // Coordinate grid
  const [showGrid, setShowGrid] = useState(false);

  // AI Analysis pin
  const [aiAnalysisPin, setAiAnalysisPin] = useState<TerrainPoint | null>(null);

  // Default opacity for new overlays
  const DEFAULT_OPACITY = 80;

  // Handle visible products update from map
  const handleVisibleProductsChange = useCallback((products: VisibleProduct[]) => {
    setVisibleProducts(products);
  }, []);

  // Explicit footprint loading handlers
  const handleLoadFootprints = useCallback((instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM") => {
    // Auto-enable visibility when loading
    if (instrument === "CRISM") setShowCRISM(true);
    else if (instrument === "HIRISE") setShowHiRISE(true);
    else if (instrument === "SHARAD") setShowSHARAD(true);
    else if (instrument === "SHARAD_HIGHRES") setShowSharadHighres(true);
    else if (instrument === "CTX") setShowCTX(true);
    else if (instrument === "HIRISE_DTM") setShowHiRISEDTM(true);
    setLoadFootprintsTrigger({ instrument, timestamp: Date.now() });
  }, []);

  const handleFootprintsLoading = useCallback((instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM", loading: boolean) => {
    setFootprintsLoading((prev) => ({
      ...prev,
      [instrument.toLowerCase()]: loading,
    }));
  }, []);

  const handleFootprintsLoaded = useCallback((result: {
    instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM";
    count: number;
    truncated: boolean;
    total: number;
  }) => {
    setFootprintCounts((prev) => ({
      ...prev,
      [result.instrument.toLowerCase()]: {
        count: result.count,
        truncated: result.truncated,
        total: result.total,
      },
    }));
  }, []);

  // Set overlay for a product (or clear if type is null)
  // This enforces single-overlay-per-product rule
  const handleSetOverlay = useCallback((productId: string, type: OverlayType | null, opacity?: number) => {
    setActiveOverlays((prev) => {
      const newMap = new Map(prev);

      if (type === null) {
        // Clear overlay for this product
        newMap.delete(productId);
      } else {
        // Set new overlay (replaces any existing overlay for this product)
        const existingOpacity = prev.get(productId)?.opacity ?? DEFAULT_OPACITY;
        newMap.set(productId, {
          type,
          opacity: opacity ?? existingOpacity,
        });
      }

      return newMap;
    });
  }, []);

  // Update opacity for a product's overlay
  const handleSetOpacity = useCallback((productId: string, opacity: number) => {
    setActiveOverlays((prev) => {
      const existing = prev.get(productId);
      if (!existing) return prev;

      const newMap = new Map(prev);
      newMap.set(productId, { ...existing, opacity });
      return newMap;
    });
  }, []);

  // Handle clicking product_id - fly to it, select it, and bring overlay to front
  const handleSelectProduct = useCallback((product: VisibleProduct) => {
    // Open inspector with coordinates from product (if available)
    setSelected({
      instrument: product.instrument,
      productId: product.productId,
      lat: product.lat ?? 0,
      lon: product.lon ?? 0,
      title: product.title,
    });

    // Fly to the product
    setFlyToProductId(product.productId);

    // Bring overlay to front if it exists
    if (activeOverlays.has(product.productId)) {
      setBringToFrontId(product.productId);
    }
  }, [activeOverlays]);

  // Handle fly-to from Active Products section
  const handleFlyToProduct = useCallback((productId: string) => {
    setFlyToProductId(productId);

    // Also bring overlay to front
    if (activeOverlays.has(productId)) {
      setBringToFrontId(productId);
    }
  }, [activeOverlays]);

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

  // Handle SHARAD High-Res footprint click - open radargram inspector
  const handleSharadHiresClick = useCallback((productId: string) => {
    setSharadHiresProductId(productId);
    setSelected(null); // Close regular Inspector
  }, []);

  // Field Notes handlers
  useEffect(() => {
    listFieldNotes().then(setFieldNotes).catch(console.error);
  }, []);

  const refreshFieldNotes = useCallback(() => {
    listFieldNotes().then(setFieldNotes).catch(console.error);
  }, []);

  const handleOpenFieldNote = useCallback((productId: string, instrument: string, lat: number, lon: number) => {
    setShowFieldNoteModal({ productId, instrument, lat, lon });
  }, []);

  // Handle field note marker click from map - fly to and open inspector
  const handleFieldNoteClick = useCallback(async (note: FieldNote) => {
    // For HiRISE DTM, use DTM center (stored note coords may be offset)
    if (note.instrument === "HIRISE_DTM") {
      const center = await getDTMCenter(note.product_id);
      const lat = center?.lat ?? note.lat;
      const lon = center?.lon ?? note.lon;

      // Fly to DTM center
      setFlyToCoords({ lat, lon });

      // Open inspector
      setSelected({
        instrument: "HIRISE_DTM",
        productId: note.product_id,
        lat,
        lon,
      });

      // Enable quickview overlay (same as handleHiRiseDTMClick)
      setActiveOverlays((prev) => {
        const newMap = new Map(prev);
        if (!newMap.has(note.product_id)) {
          newMap.set(note.product_id, { type: "quickview", opacity: 80 });
        }
        return newMap;
      });

      setActiveDTMProduct(note.product_id);
      return;
    }

    // Non-DTM products: use original behavior
    setFlyToCoords({ lat: note.lat, lon: note.lon });
    setSelected({
      instrument: note.instrument as InspectorContext["instrument"],
      productId: note.product_id,
      lat: note.lat,
      lon: note.lon,
    });
  }, []);

  // Handle HiRISE DTM footprint click - One-click inspection flow
  // Automatically: 1) Open inspector, 2) Enable quickview, 3) Activate DTM 3D mode
  // Then user can immediately click on the DTM to open 3D terrain view
  const handleHiRiseDTMClick = useCallback((productId: string, lat: number, lon: number) => {
    // 1. Open the product inspector
    setSelected({
      instrument: "HIRISE_DTM",
      productId,
      lat,
      lon,
    });

    // 2. Automatically enable quickview overlay
    setActiveOverlays((prev) => {
      const newMap = new Map(prev);
      // Only add if not already active
      if (!newMap.has(productId)) {
        newMap.set(productId, { type: "quickview", opacity: 80 });
      }
      return newMap;
    });

    // 3. Store the active DTM product (for "Show 3D View" button in Inspector)
    setActiveDTMProduct(productId);

    // 4. Clear other analysis state
    setTerrainPoint(null);
    setSlope3DPoint(null);
    setHiRiseDTM3DPoint(null);
  }, []);

  // Deactivate all overlays
  const handleDeactivateAll = useCallback(() => {
    setActiveOverlays(new Map());
  }, []);

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

  // Fetch filtered product IDs when ice score filter changes
  useEffect(() => {
    if (!iceScoreFilter.enabled) {
      setFilteredProductIds(null);
      return;
    }

    const fetchFilteredIds = async () => {
      try {
        const params = new URLSearchParams({
          min_score: iceScoreFilter.minScore.toString(),
          min_percent: iceScoreFilter.minPercent.toString(),
        });
        const response = await fetch(`/api/filter/ice?${params}`);
        if (response.ok) {
          const data = await response.json();
          setFilteredProductIds(new Set(data.passing_ids));
        }
      } catch (e) {
        console.error("Failed to fetch filtered IDs:", e);
        setFilteredProductIds(null);
      }
    };

    fetchFilteredIds();
  }, [iceScoreFilter.enabled, iceScoreFilter.minScore, iceScoreFilter.minPercent]);

  // Load custom datasets from server (triggered by Load button)
  const handleLoadCustomData = useCallback(async () => {
    setCustomDataLoading(true);
    try {
      const response = await fetch("/api/custom/datasets");
      if (response.ok) {
        const data = await response.json();
        setCustomDatasets(
          (data.datasets || []).map((d: any) => ({
            ...d,
            visible: true,
            opacity: 80,
          }))
        );
        // Auto-enable visibility when loading
        setShowCustomData(true);
      }
    } catch (e) {
      console.error("Failed to load custom datasets:", e);
    } finally {
      setCustomDataLoading(false);
    }
  }, []);

  // Custom dataset handlers (used by LayerPanel + Inspector)
  const handleCustomDatasetToggle = useCallback((datasetId: string, visible: boolean) => {
    setCustomDatasets((prev) =>
      prev.map((d) => (d.id === datasetId ? { ...d, visible } : d))
    );
  }, []);

  const handleCustomDatasetOpacity = useCallback((datasetId: string, opacity: number) => {
    setCustomDatasets((prev) =>
      prev.map((d) => (d.id === datasetId ? { ...d, opacity } : d))
    );
  }, []);

  // Handle terrain click — behavior depends on analysisMode
  const handleTerrainClick = useCallback((lat: number, lon: number) => {
    if (analysisMode === "line") {
      // Line profile mode: collect up to 2 points, then reset
      setLinePoints((prev) => {
        if (prev.length >= 2) {
          // Third click: reset and start new line
          setLineProfileData(null);
          return [{ lat, lon }];
        }
        const next = [...prev, { lat, lon }];
        if (next.length === 2) {
          // Two points collected — trigger profile computation
          setLineProfileData({ start: next[0], end: next[1] });
        }
        return next;
      });
      return;
    }
    if (analysisMode === "slope") {
      // Slope analysis mode: show slope analysis on terrain click
      setSelected(null);
      setTerrainPoint({ lat, lon });
    }
    if (analysisMode === "slope3d") {
      // Slope 3D analysis mode: show 3D terrain viewer on terrain click
      setSelected(null);
      setSlope3DPoint({ lat, lon });
    }
    if (analysisMode === "ai_analysis") {
      // AI Analysis mode: set pin and open panel
      setSelected(null);
      setAiAnalysisPin({ lat, lon });
    }
  }, [analysisMode]);

  // When a product is selected, clear terrain point
  const handleSelect = useCallback((ctx: InspectorContext | null) => {
    setSelected(ctx);
    if (ctx) setTerrainPoint(null);
  }, []);

  // Analysis mode toggle handler
  const handleAnalysisModeChange = useCallback((mode: AnalysisMode) => {
    setAnalysisMode(mode);
    // Clear state for all modes
    if (mode !== "slope") {
      setTerrainPoint(null);
    }
    if (mode !== "slope3d") {
      setSlope3DPoint(null);
    }
    if (mode !== "hirise_dtm_3d") {
      setHiRiseDTM3DPoint(null);
      setActiveDTMProduct(null);
    }
    if (mode !== "line") {
      setLinePoints([]);
      setLineProfileData(null);
    }
    if (mode !== "ai_analysis") {
      setAiAnalysisPin(null);
    }
  }, []);

  // Fly to lat/lon coordinates (for search results not on map)
  const [flyToCoords, setFlyToCoords] = useState<{ lat: number; lon: number } | null>(null);

  const handleFlyToCoordsComplete = useCallback(() => {
    setFlyToCoords(null);
  }, []);

  // Handle fly-to from LayerPanel (Fly To Location input)
  const handleFlyToCoords = useCallback((lat: number, lon: number) => {
    setFlyToCoords({ lat, lon });
  }, []);

  // Handle view bound selected from map drag selection
  const handleViewBoundSelected = useCallback((bounds: BoundingBox) => {
    setViewBounds(bounds);
    setViewBoundSelectionMode(false); // Exit selection mode after selection
  }, []);

  // Handle search result selection from TopBar
  const handleSearchSelect = useCallback((productId: string, instrument?: string, lat?: number | null, lon?: number | null) => {
    // Handle REGION type - just fly to location, no inspector
    if (instrument === "REGION") {
      if (lat != null && lon != null) {
        setFlyToCoords({ lat, lon });
      }
      return;
    }

    // First try to find the product on the map
    const product = visibleProducts.find((p) => p.productId === productId);
    if (product) {
      handleSelectProduct(product);
      return;
    }

    // Product not on map — fly to its coordinates if available
    if (lat != null && lon != null) {
      setFlyToCoords({ lat, lon });
    }

    // Open appropriate panel based on instrument
    if (instrument === "SHARAD_HIGHRES") {
      setSharadHiresProductId(productId);
    } else if (instrument && instrument !== "CTX") {
      setSelected({
        instrument: instrument as any,
        productId,
        lat: lat ?? 0,
        lon: lon ?? 0,
      });
    }
  }, [visibleProducts, handleSelectProduct]);

  // Derive legacy overlay formats for MapView compatibility
  // These will be replaced when MapView is updated to use unified format
  const derivedOverlays = useMemo(() => {
    const quickviewOverlays: string[] = [];
    const highResOverlays: string[] = [];
    const browseOverlays = new Map<string, Set<"HYD" | "ICE" | "IC2">>();
    const scoreOverlays = new Map<string, Set<"score_ice" | "score_hyd">>();
    const opacities = new Map<string, number>();

    for (const [productId, overlay] of activeOverlays) {
      opacities.set(productId, overlay.opacity / 100);

      if (overlay.type === "quickview") {
        quickviewOverlays.push(productId);
      } else if (overlay.type === "highres") {
        highResOverlays.push(productId);
      } else if (isBrowseOverlay(overlay.type)) {
        const browseType = getBrowseType(overlay.type);
        if (browseType) {
          const existing = browseOverlays.get(productId) || new Set();
          existing.add(browseType);
          browseOverlays.set(productId, existing);
        }
      } else if (isScoreOverlay(overlay.type)) {
        const scoreType = getScoreType(overlay.type);
        if (scoreType) {
          const existing = scoreOverlays.get(productId) || new Set();
          existing.add(scoreType);
          scoreOverlays.set(productId, existing);
        }
      }
    }

    return { quickviewOverlays, highResOverlays, browseOverlays, scoreOverlays, opacities };
  }, [activeOverlays]);

  return (
    <AppShell
      header={
        <TopBar
          onSelectResult={handleSearchSelect}
        />
      }
      leftPanel={
        <LayerPanel
          // Map mode (2D/3D)
          mapMode={mapMode}
          onMapModeChange={setMapMode}
          // Base layer selection
          baseLayer={baseLayer}
          onBaseLayerChange={setBaseLayer}
          // View bounds restriction
          viewBounds={viewBounds}
          onViewBoundsChange={setViewBounds}
          // Footprint toggles (visibility)
          showCRISM={showCRISM}
          showHiRISE={showHiRISE}
          showSHARAD={showSHARAD}
          showCTX={showCTX}
          showSharadHighres={showSharadHighres}
          showHiRISEDTM={showHiRISEDTM}
          onToggleCRISM={setShowCRISM}
          onToggleHiRISE={setShowHiRISE}
          onToggleSHARAD={setShowSHARAD}
          onToggleSharadHighres={setShowSharadHighres}
          onToggleCTX={setShowCTX}
          onToggleHiRISEDTM={setShowHiRISEDTM}
          // Explicit footprint loading
          onLoadFootprints={handleLoadFootprints}
          footprintsLoading={footprintsLoading}
          footprintCounts={footprintCounts}
          // Ice Score Filter
          iceScoreFilter={iceScoreFilter}
          onIceScoreFilterChange={setIceScoreFilter}
          filteredProductIds={filteredProductIds}
          // Product data
          visibleProducts={visibleProducts}
          activeOverlays={activeOverlays}
          // Overlay handlers
          onSetOverlay={handleSetOverlay}
          onSetOpacity={handleSetOpacity}
          onSelectProduct={handleSelectProduct}
          onFlyToProduct={handleFlyToProduct}
          onDeactivateAll={handleDeactivateAll}
          // Custom datasets
          showCustomData={showCustomData}
          onToggleCustomData={setShowCustomData}
          onLoadCustomData={handleLoadCustomData}
          customDataLoading={customDataLoading}
          customDatasets={customDatasets}
          onCustomDatasetToggle={handleCustomDatasetToggle}
          // Analysis mode
          analysisMode={analysisMode}
          onAnalysisModeChange={handleAnalysisModeChange}
          // Fly-To navigation
          onFlyToCoords={handleFlyToCoords}
          // View bound selection mode
          viewBoundSelectionMode={viewBoundSelectionMode}
          onViewBoundSelectionModeChange={setViewBoundSelectionMode}
          // Coordinate grid
          showGrid={showGrid}
          onToggleGrid={setShowGrid}
          // Field Notes
          fieldNotes={fieldNotes}
          showFieldNotesOnMap={showFieldNotesOnMap}
          onToggleFieldNotesOnMap={setShowFieldNotesOnMap}
          onFieldNoteClick={handleFieldNoteClick}
          onActiveTagChange={setFieldNoteActiveTag}
        />
      }
      rightPanel={
        sharadHiresProductId ? (
          <SharadHiresInspector
            productId={sharadHiresProductId}
            onClose={() => setSharadHiresProductId(null)}
            fieldNotes={fieldNotes}
            onOpenFieldNote={(pid) => setShowFieldNoteModal({ productId: pid, instrument: "SHARAD_HIGHRES", lat: 0, lon: 0 })}
          />
        ) : selected ? (
          <Inspector
            selected={selected}
            onClose={() => setSelected(null)}
            activeOverlay={activeOverlays.get(selected.productId) || null}
            onSetOverlay={(type) => handleSetOverlay(selected.productId, type)}
            onSetOpacity={(opacity) => handleSetOpacity(selected.productId, opacity)}
            rgbWavelengths={rgbWavelengths}
            onRGBChange={handleRGBChange}
            hasHighResData={productsWithHighRes.has(selected.productId)}
            customDataset={customDatasets.find((d) => d.id === selected.productId) || null}
            onCustomDatasetOpacity={handleCustomDatasetOpacity}
            fieldNotes={fieldNotes}
            onOpenFieldNote={handleOpenFieldNote}
            onShow3DView={async (productId, lat, lon) => {
              setSelected(null);
              // Use DTM center to ensure point is within bounds
              const center = await getDTMCenter(productId);
              const safeLat = center?.lat ?? lat;
              const safeLon = center?.lon ?? lon;
              setHiRiseDTM3DPoint({ productId, lat: safeLat, lon: safeLon });
            }}
          />
        ) : terrainPoint ? (
          <SlopeAnalysis
            point={terrainPoint}
            onClose={() => setTerrainPoint(null)}
          />
        ) : slope3DPoint ? (
          <Slope3DViewer
            point={slope3DPoint}
            onClose={() => setSlope3DPoint(null)}
          />
        ) : hiRiseDTM3DPoint ? (
          <HiRiseDTM3DViewer
            point={hiRiseDTM3DPoint}
            onClose={() => {
              setHiRiseDTM3DPoint(null);
              setAnalysisMode(null);
              setActiveDTMProduct(null);
            }}
          />
        ) : aiAnalysisPin ? (
          <AiAnalysisPanel
            pin={aiAnalysisPin}
            onClose={() => {
              setAiAnalysisPin(null);
              setAnalysisMode(null);
            }}
          />
        ) : null
      }
    >
      {/* Map Canvas */}
      <MapView
        mapMode={mapMode}
        baseLayer={baseLayer}
        viewBounds={viewBounds}
        onSelect={handleSelect}
        onTerrainClick={handleTerrainClick}
        showCRISM={showCRISM}
        showHiRISE={showHiRISE}
        showSHARAD={showSHARAD}
        showSharadHighres={showSharadHighres}
        showCTX={showCTX}
        showHiRISEDTM={showHiRISEDTM}
        onSharadClick={handleSharadClick}
        onSharadHiresClick={handleSharadHiresClick}
        onHiRiseDTMClick={handleHiRiseDTMClick}
        onToggleOverlay={(productId, type) => handleSetOverlay(productId, type)}
        quickviewOverlays={derivedOverlays.quickviewOverlays}
        highResOverlays={derivedOverlays.highResOverlays}
        browseOverlays={derivedOverlays.browseOverlays}
        scoreOverlays={derivedOverlays.scoreOverlays}
        overlayOpacities={derivedOverlays.opacities}
        onVisibleProductsChange={handleVisibleProductsChange}
        flyToProductId={flyToProductId}
        onFlyToComplete={handleFlyToComplete}
        flyToCoords={flyToCoords}
        onFlyToCoordsComplete={handleFlyToCoordsComplete}
        bringToFrontId={bringToFrontId}
        onBringToFrontComplete={handleBringToFrontComplete}
        rgbWavelengths={rgbWavelengths}
        crismFilteredIds={filteredProductIds}
        loadFootprintsTrigger={loadFootprintsTrigger}
        onFootprintsLoaded={handleFootprintsLoaded}
        onFootprintsLoading={handleFootprintsLoading}
        showCustomData={showCustomData}
        customDatasets={customDatasets}
        analysisMode={analysisMode}
        linePoints={linePoints}
        viewBoundSelectionMode={viewBoundSelectionMode}
        onViewBoundSelected={handleViewBoundSelected}
        fieldNotes={showFieldNotesOnMap ? mapFieldNotes : []}
        onFieldNoteClick={handleFieldNoteClick}
        activeDTMProductId={activeDTMProduct}
        showGrid={showGrid}
        aiAnalysisPin={aiAnalysisPin}
      />

      {/* Line Profile Popup */}
      {lineProfileData && (
        <LineProfile
          startPoint={lineProfileData.start}
          endPoint={lineProfileData.end}
          onClose={() => {
            setLineProfileData(null);
            setLinePoints([]);
          }}
        />
      )}

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

      {/* Field Note Modal */}
      {showFieldNoteModal && (
        <FieldNoteModal
          productId={showFieldNoteModal.productId}
          instrument={showFieldNoteModal.instrument}
          lat={showFieldNoteModal.lat}
          lon={showFieldNoteModal.lon}
          onClose={() => setShowFieldNoteModal(null)}
          onNoteSaved={refreshFieldNotes}
        />
      )}
    </AppShell>
  );
}
