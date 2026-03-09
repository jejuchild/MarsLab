import { useState, useCallback, useEffect, useMemo, useRef, lazy, Suspense, memo } from "react";
import ErrorBoundary from "../components/ErrorBoundary";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import MapViewRaw from "../components/MapView";
import InspectorPanelRaw from "../components/inspector/InspectorPanel";
import type { InspectorContext, RGBWavelengths } from "../components/inspector/types";
import SlopeAnalysis from "../components/SlopeAnalysis";
import type { HiRiseDTMPoint } from "../components/HiRiseDTM3DViewer";
import type { TerrainPoint } from "../components/SlopeAnalysis";
import type { ProfilePoint } from "../components/LineProfile";
import TopBar from "../components/TopBar";
import LayerPanelRaw from "../components/layerpanel/LayerPanel";
import SharadHiresInspector from "../components/SharadHiresInspector";
import FieldNoteModal from "../components/FieldNoteModal";
import AiAnalysisPanelRaw from "../components/AiAnalysisPanel";
import AgenticPanelRaw from "../components/AgenticPanel";
import GuidedWorkflowsRaw from "../components/GuidedWorkflows";
import type { WorkflowAction } from "../components/GuidedWorkflows";
import CopilotFab from "../components/CopilotFab";
import type { FieldNote } from "../api/fieldnotes";
import useFieldNotes from "../hooks/useFieldNotes";
import AppShell from "../components/layout/AppShell";
import Footer from "../components/layout/Footer";
import BottomSheet from "../components/BottomSheet";
import useIsMobile from "../hooks/useIsMobile";
import useUrlState from "../hooks/useUrlState";
import useCommandPalette from "../hooks/useCommandPalette";
import usePanelManager from "../hooks/usePanelManager";
import PanelAttentionWrapper from "../components/PanelAttentionWrapper";
import { useUndoRedo } from "../hooks/useUndoRedo";
import AccessibilityExplainTooltip from "../components/AccessibilityExplainTooltip";
import CommandPalette from "../components/CommandPalette";
import type { CommandAction } from "../components/CommandPalette";
import EmptyState from "../components/EmptyState";
import KeyboardShortcuts from "../components/KeyboardShortcuts";
import OnboardingTour from "../components/OnboardingTour";
import type { InstrumentType } from "../utils/FootprintManager";
import { getInstrumentIds, type InstrumentId, isInstrumentId } from "../config/instrumentRegistry";
import type { OverlapStats } from "../utils/overlapFilter";
import SpectralComparison from "../components/SpectralComparison";
import type { PinnedSpectrum } from "../components/SpectralComparison";
import { CuriositySelfieModal, OlympusMonsPanel, OlympusMonsClimber, TerraformOverlay } from "../components/EasterEggs";
import type { DetectedFeature } from "../components/CraterDetectPanel";
import type { SwimMethod, DepthRange } from "../api/swim_ice";
import { SWIM_METHODS } from "../api/swim_ice";
import type { RouteResult } from "../api/pathfinder";
import type { RoverTelemetry, SpeedOption } from "../hooks/useRoverSimulation";

// Memoize heavy child components to prevent unnecessary re-renders
const MapView = memo(MapViewRaw);
const Inspector = memo(InspectorPanelRaw);
const LayerPanel = memo(LayerPanelRaw);
const AiAnalysisPanel = memo(AiAnalysisPanelRaw);
const AgenticPanel = memo(AgenticPanelRaw);
const GuidedWorkflows = memo(GuidedWorkflowsRaw);

// Lazy-loaded heavy components (Three.js / Recharts)
const HiRiseDTM3DViewer = lazy(() => import("../components/HiRiseDTM3DViewer"));
const LineProfile = lazy(() => import("../components/LineProfile"));
const ReportPanel = lazy(() => import("../components/ReportPanel"));
const RegionDashboard = lazy(() => import("../components/RegionDashboard"));
const SpaceGame = lazy(() => import("../components/SpaceGame"));
const RegionStatsPanel = lazy(() => import("../components/RegionStatsPanel"));
const CraterDetectPanel = lazy(() => import("../components/CraterDetectPanel"));
const TemporalComparison = lazy(() => import("../components/TemporalComparison"));
const RegolithPanel = lazy(() => import("../components/RegolithPanel"));
const StratigraphyPanel = lazy(() => import("../components/StratigraphyPanel"));
const AttenuationPanel = lazy(() => import("../components/AttenuationPanel"));
const MineralSequencePanel = lazy(() => import("../components/MineralSequencePanel"));
const StratColumnPanel = lazy(() => import("../components/StratColumnPanel"));
const PathfinderPanel = lazy(() => import("../components/PathfinderPanel"));

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
  instrument: "HIRISE" | "CRISM" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";
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
  | "score_hyd"
  | "mineral_cnn";

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

// Multi-Instrument Overlap Filter
export type OverlapFilter = {
  enabled: boolean;
  instruments: InstrumentType[];
};

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
  const { urlState, updateUrl } = useUrlState();
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const commandPalette = useCommandPalette();
  const { pushAction, undo, redo, canUndo, canRedo, lastAction } = useUndoRedo();
  const [mobilePanel, setMobilePanel] = useState<'none' | 'layers' | 'inspector'>('none');

  // Field notes hook (extracted from MainPage)
  const {
    fieldNotes, showFieldNoteModal, setShowFieldNoteModal,
    showFieldNotesOnMap, setShowFieldNotesOnMap,
    fieldNoteActiveTag: _fieldNoteActiveTag, setFieldNoteActiveTag,
    mapFieldNotesForView, refreshFieldNotes, handleOpenFieldNote,
  } = useFieldNotes();

  // Selected footprint for Inspector
  const [selected, setSelected] = useState<InspectorContext | null>(null);

  // Terrain click point for slope analysis (when clicking empty terrain)
  const [terrainPoint, setTerrainPoint] = useState<TerrainPoint | null>(null);

  // Analysis mode: mutually exclusive slope / hirise_dtm_3d / line / ai_analysis / agentic
  type AnalysisMode = "slope" | "hirise_dtm_3d" | "line" | "ai_analysis" | "agentic" | "report" | "guided" | "region_stats" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column" | "pathfinder" | null;
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>(null);

  // Guided Workflow: user-selected location
  const [guidedLocation, setGuidedLocation] = useState<{ lat: number; lon: number } | null>(null);

  // Pathfinder: start/goal points and route result
  const [pathfinderStart, setPathfinderStart] = useState<{ lat: number; lon: number } | null>(null);
  const [pathfinderGoal, setPathfinderGoal] = useState<{ lat: number; lon: number } | null>(null);
  const [pathfinderRoute, setPathfinderRoute] = useState<RouteResult | null>(null);

  // Rover simulation state
  const [simPlaying, setSimPlaying] = useState(false);
  const [simSpeed, setSimSpeed] = useState<SpeedOption>(1);
  const [simCameraFollow, setSimCameraFollow] = useState(true);
  const [simSeekTo, setSimSeekTo] = useState<number | null>(null);
  const [simProgress, setSimProgress] = useState(0);
  const [simTelemetry, setSimTelemetry] = useState<RoverTelemetry | null>(null);
  const [simComplete, setSimComplete] = useState(false);

  // HiRISE DTM 3D analysis point (requires product_id + lat/lon)
  const [hiRiseDTM3DPoint, setHiRiseDTM3DPoint] = useState<HiRiseDTMPoint | null>(null);

  // Active HiRISE DTM product (for terrain clicks in hirise_dtm_3d mode)
  const [activeDTMProduct, setActiveDTMProduct] = useState<string | null>(null);

  // Recent products (last 5 inspected) for quick re-access
  type RecentProduct = { productId: string; instrument: InspectorContext["instrument"]; lat: number; lon: number; title?: string };
  const [recentProducts, setRecentProducts] = useState<RecentProduct[]>([]);
  const addRecentProduct = useCallback((ctx: InspectorContext) => {
    setRecentProducts(prev => {
      const filtered = prev.filter(p => p.productId !== ctx.productId);
      return [{ productId: ctx.productId, instrument: ctx.instrument, lat: ctx.lat, lon: ctx.lon, title: ctx.title }, ...filtered].slice(0, 5);
    });
  }, []);
  const removeRecentProduct = useCallback((productId: string) => {
    setRecentProducts(prev => {
      const next = prev.filter(p => p.productId !== productId);
      // If we removed the currently-selected product, switch to the next one or close
      setSelected(cur => {
        if (cur && cur.productId === productId) {
          const fallback = next[0];
          return fallback
            ? { instrument: fallback.instrument, productId: fallback.productId, lat: fallback.lat, lon: fallback.lon, title: fallback.title }
            : null;
        }
        return cur;
      });
      return next;
    });
  }, []);

  // Line profile state: two endpoints
  const [linePoints, setLinePoints] = useState<ProfilePoint[]>([]);
  const [lineProfileData, setLineProfileData] = useState<{ start: ProfilePoint; end: ProfilePoint } | null>(null);

  // Measurement tools visibility toggle
  const [showMeasurementTools, setShowMeasurementTools] = useState(false);
  const handleMeasurementPinNote = useCallback((lat: number, lon: number, text: string) => {
    // Create a field note via the existing field notes system
    import("../api/fieldnotes").then(({ createFieldNote }) => {
      createFieldNote({
        product_id: "ANNOTATION",
        instrument: "PIN",
        lat,
        lon,
        tags: ["annotation", "measurement"],
        memo: text,
      }).then(() => {
        refreshFieldNotes();
      }).catch((err) => {
        console.error("Failed to save pin note:", err);
      });
    });
  }, [refreshFieldNotes]);

  // Base layer selection
  const [baseLayer, setBaseLayer] = useState<BaseLayerType>("MOLA");

  // Map mode (2D flat view vs 3D globe)
  const [mapMode, setMapMode] = useState<MapMode>("2D");

  // Bounding box for view restriction
  const [viewBounds, setViewBounds] = useState<BoundingBox>(null);

  // Camera viewport ref — updated by MapView on camera.moveEnd, read on demand
  const cameraViewportRef = useRef<{ minLat: number; maxLat: number; westLon: number; eastLon: number } | null>(null);

  // View bound selection mode (drag to select region on map)
  const [viewBoundSelectionMode, setViewBoundSelectionMode] = useState(false);

  // Footprint layer toggles (visibility only - does NOT trigger loading)
  type InstrumentVisibility = Record<InstrumentId, boolean>;
  const [instrumentVisibility, setInstrumentVisibility] = useState<InstrumentVisibility>(
    () => Object.fromEntries(getInstrumentIds().map(id => [id, false])) as InstrumentVisibility
  );
  const handleToggleInstrument = useCallback((id: InstrumentId, v: boolean) => {
    setInstrumentVisibility(prev => ({ ...prev, [id]: v }));
  }, []);

  // Derived booleans for MapView backward compatibility
  const showCRISM = instrumentVisibility.crism;
  const showHiRISE = instrumentVisibility.hirise;
  const showSHARAD = instrumentVisibility.sharad;
  const showSharadHighres = instrumentVisibility.sharad_highres;
  const showCTX = instrumentVisibility.ctx;
  const showHiRISEDTM = instrumentVisibility.hirise_dtm;
  const showCRISM_TRR3 = instrumentVisibility.crism_trr3;

  // Explicit footprint loading state
  type FootprintLoadTrigger = { instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3"; timestamp: number } | null;
  const [loadFootprintsTrigger, setLoadFootprintsTrigger] = useState<FootprintLoadTrigger>(null);
  const [footprintsLoading, setFootprintsLoading] = useState<{ crism: boolean; hirise: boolean; sharad: boolean; sharad_highres: boolean; ctx: boolean; hirise_dtm: boolean; crism_trr3: boolean }>({
    crism: false,
    hirise: false,
    sharad: false,
    sharad_highres: false,
    ctx: false,
    hirise_dtm: false,
    crism_trr3: false,
  });
  const [footprintCounts, setFootprintCounts] = useState<{
    crism: { count: number; truncated: boolean; total: number } | null;
    hirise: { count: number; truncated: boolean; total: number } | null;
    sharad: { count: number; truncated: boolean; total: number } | null;
    sharad_highres: { count: number; truncated: boolean; total: number } | null;
    ctx: { count: number; truncated: boolean; total: number } | null;
    hirise_dtm: { count: number; truncated: boolean; total: number } | null;
    crism_trr3: { count: number; truncated: boolean; total: number } | null;
  }>({ crism: null, hirise: null, sharad: null, sharad_highres: null, ctx: null, hirise_dtm: null, crism_trr3: null });

  // Visible products in current map view
  const [visibleProducts, setVisibleProducts] = useState<VisibleProduct[]>([]);

  // Unified active overlays - ONE overlay per product with per-product opacity
  const [activeOverlays, setActiveOverlays] = useState<ActiveOverlays>(new Map());

  // Product to fly to (set when clicking product_id in LayerPanel)
  const [flyToProductId, setFlyToProductId] = useState<string | null>(null);

  // Product to highlight after fly-to (temporary bright highlight)
  const [highlightProductId, setHighlightProductId] = useState<string | null>(null);

  // Product to bring to front (for z-ordering overlays)
  const [bringToFrontId, setBringToFrontId] = useState<string | null>(null);

  // CRISM RGB wavelengths
  const [rgbWavelengths, setRGBWavelengths] = useState<RGBWavelengths>(DEFAULT_RGB_WAVELENGTHS);

  // SHARAD popup state
  const [sharadPopup, setSharadPopup] = useState<SHARADPopup>(null);

  // Track which products have high-res data available
  const [productsWithHighRes, setProductsWithHighRes] = useState<Set<string>>(new Set());


  // Multi-Instrument Overlap Filter
  const [overlapFilter, setOverlapFilter] = useState<OverlapFilter>({
    enabled: false,
    instruments: [],
  });
  const [overlapStats, setOverlapStats] = useState<OverlapStats | null>(null);

  const handleOverlapStatsChange = useCallback((stats: OverlapStats | null) => {
    setOverlapStats(stats);
  }, []);

  // SHARAD High-Res Inspector — opens when a product is selected
  const [sharadHiresProductId, setSharadHiresProductId] = useState<string | null>(null);
  // Pin showing clicked radargram location on the map track
  const [sharadTracePin, setSharadTracePin] = useState<{ lat: number; lon: number } | null>(null);

  // Regolith Thickness Estimator — product ID for analysis
  const [regolithProductId, setRegolithProductId] = useState<string | null>(null);

  // Radar Attenuation Mapper — product ID for analysis
  const [attenuationProductId, setAttenuationProductId] = useState<string | null>(null);

  // Mineral Sequence Mapper — obs ID for analysis
  const [mineralSequenceObsId, setMineralSequenceObsId] = useState<string | null>(null);

  // Custom user-uploaded datasets
  const [showCustomData, setShowCustomData] = useState(false);
  const [customDataLoading, setCustomDataLoading] = useState(false);
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);

  // Filtered product IDs (null = no filtering active)
  const [filteredProductIds] = useState<Set<string> | null>(null);

  // Field Notes state
  // (field notes state moved to useFieldNotes hook)

  // Coordinate grid
  const [showGrid, setShowGrid] = useState(false);
  const [showRegionLayer, setShowRegionLayer] = useState(false);
  const [swimLayer, setSwimLayer] = useState<string | false>(false);
  const [accessibilityVisible, setAccessibilityVisible] = useState(false);
  const [accessibilityOpacity, setAccessibilityOpacity] = useState(0.6);
  const [fusionVisible, setFusionVisible] = useState(false);
  const [fusionOpacity, setFusionOpacity] = useState(0.6);
  const [accessibilityExplainMode, setAccessibilityExplainMode] = useState(false);
  const [accessibilityExplainPoint, setAccessibilityExplainPoint] = useState<{ lat: number; lon: number } | null>(null);
  const [scienceLayerVisibility, setScienceLayerVisibility] = useState<Record<SwimMethod, boolean>>(() => {
    const init: Record<string, boolean> = {};
    for (const m of SWIM_METHODS) init[m] = false;
    return init as Record<SwimMethod, boolean>;
  });
  const [scienceLayerDepth, setScienceLayerDepth] = useState<DepthRange>("1-5m");
  const [scienceLayerOpacities, setScienceLayerOpacities] = useState<Record<SwimMethod, number>>(() => {
    const init: Record<string, number> = {};
    for (const m of SWIM_METHODS) init[m] = 0.7;
    return init as Record<SwimMethod, number>;
  });
  const handleScienceLayerToggle = useCallback((method: SwimMethod, visible: boolean) => {
    setScienceLayerVisibility(prev => ({ ...prev, [method]: visible }));
  }, []);
  const handleScienceLayerOpacity = useCallback((method: SwimMethod, opacity: number) => {
    setScienceLayerOpacities(prev => ({ ...prev, [method]: opacity }));
  }, []);

  // Region Dashboard overlay
  const [showRegionDashboard, setShowRegionDashboard] = useState(false);

  // Easter egg game
  const [showGame, setShowGame] = useState(false);
  const [showTerraform, setShowTerraform] = useState(false);
  const [showCuriosity, setShowCuriosity] = useState(false);
  const [showOlympusMons, setShowOlympusMons] = useState(false);
  const [showOlympusMonsClimber, setShowOlympusMonsClimber] = useState(false);

  // Region Stats polygon vertices
  const [regionVertices, setRegionVertices] = useState<{lat: number; lon: number}[]>([]);

  // Crater/Landform Detection
  const [, setCraterDetectCenter] = useState<{lat: number; lon: number} | null>(null);
  const [craterDetectFeatures, setCraterDetectFeatures] = useState<DetectedFeature[]>([]);
  const [epsilonTarget, setEpsilonTarget] = useState<DetectedFeature | null>(null);
  const [stratColumnTarget, setStratColumnTarget] = useState<DetectedFeature | null>(null);

  // Temporal Change Detection modal
  const [showTemporalComparison, setShowTemporalComparison] = useState<{lat: number; lon: number; instrument?: string} | null>(null);

  // Keyboard shortcuts help modal
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);

  // Right panel (inspector) collapsed state
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);

  // Panel intelligence: auto-open, attention pulse, autonomy toggle
  const panelManager = usePanelManager({
    rightPanelCollapsed,
    setRightPanelCollapsed,
    isMobile,
    setMobilePanel,
  });
  // Stable callbacks for memoized children (avoids breaking memo())
  const handleAgenticPanelAttention = useCallback(() => panelManager.ensurePanelVisible("analysis_complete"), [panelManager.ensurePanelVisible]);

  // Onboarding tour (force re-trigger)
  const [showTourForced, setShowTourForced] = useState(false);

  // Pinned spectra for comparison tool
  const [pinnedSpectra, setPinnedSpectra] = useState<PinnedSpectrum[]>([]);
  const [showSpectralComparison, setShowSpectralComparison] = useState(false);

  const SPECTRUM_COLORS = ['#00FFFF', '#FF6B6B', '#4ADE80', '#FBBF24', '#A78BFA'];

  const handlePinSpectrum = useCallback((spectrum: { productId: string; lat: number; lon: number; wavelengths: number[]; reflectance: (number | null)[] }) => {
    setPinnedSpectra(prev => {
      if (prev.length >= 5) {
        toast.error("Maximum 5 pinned spectra");
        return prev;
      }
      if (prev.some(s => s.productId === spectrum.productId)) {
        toast("Spectrum already pinned");
        return prev;
      }
      const color = SPECTRUM_COLORS[prev.length % SPECTRUM_COLORS.length]!;
      return [...prev, { ...spectrum, id: `${spectrum.productId}-${Date.now()}`, color }];
    });
    setShowSpectralComparison(true);
    toast.success("Spectrum pinned for comparison");
  }, []);

  // Terraform mode: auto-dismiss after 10 seconds
  useEffect(() => {
    if (!showTerraform) return;
    const timer = setTimeout(() => setShowTerraform(false), 10000);
    return () => clearTimeout(timer);
  }, [showTerraform]);

  // Mark Watney: fly to Acidalia Planitia + toast
  const handleMarkWatney = useCallback(() => {
    setFlyToCoords({ lat: 41.715, lon: -19.35 });
    toast(
      "\"I'm going to have to science the s**t out of this.\" -- Mark Watney, Sol 6",
      {
        icon: "\u{1F954}",
        duration: 6000,
        style: {
          background: "#101622",
          color: "#92a4c9",
          border: "1px solid #232f48",
          maxWidth: "420px",
        },
      },
    );
  }, []);

  // Olympus Mons triple-click callback
  const handleOlympusMonsClick = useCallback(() => {
    setShowOlympusMons(true);
  }, []);

  // Olympus Mons 7-click: climber animation
  const handleOlympusMonsClimber = useCallback(() => {
    setShowOlympusMons(false); // close comparison if open
    setShowOlympusMonsClimber(true);
  }, []);

  // Memoized inspected product ID for MapView (avoids re-render on unrelated state changes)
  const inspectedProductId = useMemo(() => selected?.productId ?? null, [selected?.productId]);

  // AI Analysis pin
  const [aiAnalysisPin, setAiAnalysisPin] = useState<TerrainPoint | null>(null);

  // Default opacity for new overlays
  const DEFAULT_OPACITY = 80;

  // Handle visible products update from map
  const handleVisibleProductsChange = useCallback((products: VisibleProduct[]) => {
    setVisibleProducts(products);
  }, []);

  // Explicit footprint loading handlers
  const handleLoadFootprints = useCallback((instrument: string) => {
    // Auto-enable visibility when loading
    const id = instrument.toLowerCase() as InstrumentId;
    setInstrumentVisibility(prev => ({ ...prev, [id]: true }));
    setLoadFootprintsTrigger({ instrument: instrument as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3", timestamp: Date.now() });
  }, []);

  const handleFootprintsLoading = useCallback((instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3", loading: boolean) => {
    setFootprintsLoading((prev) => ({
      ...prev,
      [instrument.toLowerCase()]: loading,
    }));
  }, []);

  const handleFootprintsLoaded = useCallback((result: {
    instrument: "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3";
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
    // Capture previous state for undo
    const prevOverlay = activeOverlaysRef.current.get(productId) ?? null;
    const prevType = prevOverlay?.type ?? null;

    // Apply the change
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

    // Push undo action
    pushAction({
      type: "overlay",
      description: type ? `Set ${type} overlay on ${productId.slice(0, 12)}` : `Remove overlay from ${productId.slice(0, 12)}`,
      undo: () => {
        setActiveOverlays((prev) => {
          const newMap = new Map(prev);
          if (prevType === null) {
            newMap.delete(productId);
          } else {
            newMap.set(productId, prevOverlay!);
          }
          return newMap;
        });
      },
      redo: () => {
        setActiveOverlays((prev) => {
          const newMap = new Map(prev);
          if (type === null) {
            newMap.delete(productId);
          } else {
            const existingOpacity = prev.get(productId)?.opacity ?? DEFAULT_OPACITY;
            newMap.set(productId, { type, opacity: opacity ?? existingOpacity });
          }
          return newMap;
        });
      },
    });
  }, [pushAction]);

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

  // Stable ref for activeOverlays to avoid callback dependency on Map reference
  const activeOverlaysRef = useRef(activeOverlays);
  activeOverlaysRef.current = activeOverlays;

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
    panelManager.ensurePanelVisible("feature_select");

    // Fly to the product
    setFlyToProductId(product.productId);

    // Bring overlay to front if it exists
    if (activeOverlaysRef.current.has(product.productId)) {
      setBringToFrontId(product.productId);
    }
  }, [panelManager.ensurePanelVisible]);

  // Handle fly-to from Active Products section
  const handleFlyToProduct = useCallback((productId: string) => {
    setFlyToProductId(productId);

    // Also bring overlay to front
    if (activeOverlaysRef.current.has(productId)) {
      setBringToFrontId(productId);
    }
  }, []);

  // Stable ref for flyToProductId
  const flyToProductIdRef = useRef(flyToProductId);
  flyToProductIdRef.current = flyToProductId;

  // Clear flyTo after it's processed — then trigger highlight
  const handleFlyToComplete = useCallback(() => {
    const pid = flyToProductIdRef.current;
    setFlyToProductId(null);
    // If this fly-to came from a deep-link, highlight the product after arrival
    if (pid && deepLinkFlyToRef.current) {
      deepLinkFlyToRef.current = false;
      setHighlightProductId(pid);
    }
  }, []);

  // Clear highlight after it's processed
  const handleHighlightComplete = useCallback(() => {
    setHighlightProductId(null);
  }, []);

  // Clear bringToFront after it's processed
  const handleBringToFrontComplete = useCallback(() => {
    setBringToFrontId(null);
  }, []);

  // Handle clicking a recent product chip in Inspector
  const handleSelectRecent = useCallback((p: { productId: string; instrument: InspectorContext["instrument"]; lat: number; lon: number; title?: string }) => {
    setSelected({ instrument: p.instrument, productId: p.productId, lat: p.lat, lon: p.lon, title: p.title });
    setFlyToProductId(p.productId);
  }, []);

  // Handle MARVIS unified search results — auto-load instruments, fly to best, select
  const handleSearchResults = useCallback((results: Array<{
    product_id: string;
    instrument: string;
    lat: number;
    lon: number;
    ice_percent?: number;
    hyd_percent?: number;
    paired_product?: string;
    paired_instrument?: string;
    near_landform_type?: string;
    near_landform_distance_km?: number;
  }>, params: Record<string, unknown>) => {
    if (results.length === 0) return;
    const best = results[0];
    if (!best) return;
    const primaryInst = (best.instrument || params.instrument || "") as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3";

    // Load primary instrument
    handleLoadFootprints(primaryInst);

    // If intersection search, also load secondary instrument (staggered)
    if (best.paired_instrument) {
      const secInst = best.paired_instrument as typeof primaryInst;
      setTimeout(() => handleLoadFootprints(secInst), 200);
    }

    // Fly to the best result
    setFlyToCoords({ lat: best.lat, lon: best.lon });

    // Select the best product in the inspector
    setTimeout(() => {
      setSelected({
        instrument: primaryInst as InspectorContext["instrument"],
        productId: best.product_id,
        lat: best.lat,
        lon: best.lon,
      });
      panelManager.ensurePanelVisible("feature_select");
      setHighlightProductId(best.product_id);
    }, 400);
  }, [handleLoadFootprints, panelManager.ensurePanelVisible]);

  // Handle MARVIS "select product" — pick first visible product of the given instrument
  const handleSelectInstrumentProduct = useCallback((instrument: string) => {
    const instUpper = instrument.toUpperCase();
    const product = visibleProductsRef.current.find(
      (p: VisibleProduct) => p.instrument.toUpperCase() === instUpper
    );
    if (product) {
      setSelected({
        instrument: product.instrument,
        productId: product.productId,
        lat: product.lat ?? 0,
        lon: product.lon ?? 0,
        title: product.title,
      });
      panelManager.ensurePanelVisible("feature_select");
      setFlyToProductId(product.productId);
      setHighlightProductId(product.productId);
    }
  }, [panelManager.ensurePanelVisible]);


  // Handle download from Inspector quick actions
  const handleDownloadProduct = useCallback((productId: string, instrument: string) => {
    window.open(`/download?tab=product&product_id=${encodeURIComponent(productId)}&instrument=${encodeURIComponent(instrument)}&autoDownload=true`, "_self");
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

  // Handle opening Regolith Thickness analysis from SharadHiresInspector
  const handleOpenRegolith = useCallback((productId: string) => {
    setSharadHiresProductId(null);
    setSharadTracePin(null);
    setRegolithProductId(productId);
    setAnalysisMode("regolith");
  }, []);

  // Handle opening Radar Attenuation analysis from SharadHiresInspector
  const handleOpenAttenuation = useCallback((productId: string) => {
    setSharadHiresProductId(null);
    setSharadTracePin(null);
    setAttenuationProductId(productId);
    setAnalysisMode("attenuation");
  }, []);

  // Handle opening Mineral Sequence analysis from Inspector CNN section
  const handleOpenMineralSequence = useCallback((obsId: string) => {
    setSelected(null);
    setMineralSequenceObsId(obsId);
    setAnalysisMode("mineral_sequence");
  }, []);

  // Fly to lat/lon coordinates (for search results not on map)
  const [flyToCoords, setFlyToCoords] = useState<{ lat: number; lon: number } | null>(null);
  // Last navigated position (persists after flyToCoords is cleared to null)
  const [lastNavCoords, setLastNavCoords] = useState<{ lat: number; lon: number } | null>(null);

  // Track last navigated position (persists after flyToCoords is cleared)
  useEffect(() => {
    if (flyToCoords) {
      setLastNavCoords(flyToCoords);
    }
  }, [flyToCoords]);

  const handleFlyToCoordsComplete = useCallback(() => {
    setFlyToCoords(null);
  }, []);

  // Handle fly-to from LayerPanel (Fly To Location input)
  const handleFlyToCoords = useCallback((lat: number, lon: number) => {
    setFlyToCoords({ lat, lon });
  }, []);

  // Track whether the current fly-to was triggered by a deep-link
  const deepLinkFlyToRef = useRef(false);

  // --- URL State: Restore on mount ---
  // Read URL params once and apply them to app state.
  const urlStateAppliedRef = useRef(false);
  useEffect(() => {
    if (urlStateAppliedRef.current) return;
    urlStateAppliedRef.current = true;

    // Restore lat/lon → fly to coordinates
    if (urlState.lat !== undefined && urlState.lon !== undefined) {
      setFlyToCoords({ lat: urlState.lat, lon: urlState.lon });
    }

    // Restore instruments → enable visibility + trigger footprint loading
    if (urlState.instruments && urlState.instruments.length > 0) {
      const newVisibility: Partial<Record<InstrumentId, boolean>> = {};
      for (const id of urlState.instruments) {
        newVisibility[id] = true;
      }
      setInstrumentVisibility(prev => ({ ...prev, ...newVisibility }));
      // Stagger load triggers so each one is processed by MapView's useEffect.
      // loadFootprintsTrigger is a single-value state, so rapid synchronous calls
      // get batched and only the last one takes effect. Using setTimeout(_, i*100)
      // ensures each trigger fires in a separate React commit.
      urlState.instruments.forEach((id, i) => {
        setTimeout(() => {
          setLoadFootprintsTrigger({ instrument: id.toUpperCase() as any, timestamp: Date.now() });
        }, i * 100);
      });
    }

    // Restore product → fly to product (deep-link)
    if (urlState.product) {
      deepLinkFlyToRef.current = true;
      setFlyToProductId(urlState.product);
    }

    // Restore analysis mode — with crash-loop protection.
    // If we're reloading into ?mode=assistant within 3 seconds of the
    // last page load, skip restoring it (likely an infinite-reload crash).
    if (urlState.mode) {
      const now = Date.now();
      const lastLoad = Number(sessionStorage.getItem("_marslab_last_load") || "0");
      sessionStorage.setItem("_marslab_last_load", String(now));
      const isCrashLoop =
        urlState.mode === "agentic" &&
        lastLoad > 0 &&
        now - lastLoad < 3000;
      if (!isCrashLoop) {
        setAnalysisMode(urlState.mode as AnalysisMode);
      } else {
        // Break the loop: clear mode from URL

        updateUrl({ mode: undefined });
      }
    }

    // Restore base layer
    if (urlState.base === "MOLA" || urlState.base === "HRSC") {
      setBaseLayer(urlState.base);
    }

    // Restore view mode (2D / 3D)
    if (urlState.view === "2D" || urlState.view === "3D") {
      setMapMode(urlState.view);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- intentionally run once on mount

  // --- URL State: Write on state change (debounced) ---
  // Skip writes until the initial URL state has been applied to avoid
  // overwriting URL params with default state during the first render.
  const urlWriteReadyRef = useRef(false);
  useEffect(() => {
    // Wait one tick after mount effects have run before enabling writes.
    const id = setTimeout(() => { urlWriteReadyRef.current = true; }, 600);
    return () => clearTimeout(id);
  }, []);

  // Sync flyToCoords → lat/lon in URL.
  // Only write when flyToCoords is set (not when cleared to null after fly-to
  // completes), so the URL retains the last navigated position for bookmarking.
  useEffect(() => {
    if (!urlWriteReadyRef.current) return;
    if (flyToCoords) {
      updateUrl({
        lat: flyToCoords.lat,
        lon: flyToCoords.lon,
      });
    }
  }, [flyToCoords, updateUrl]);

  // Sync instrumentVisibility → instruments in URL
  useEffect(() => {
    if (!urlWriteReadyRef.current) return;
    const active = (Object.entries(instrumentVisibility) as [InstrumentId, boolean][])
      .filter(([, v]) => v)
      .map(([k]) => k);
    updateUrl({ instruments: active.length > 0 ? active : undefined });
  }, [instrumentVisibility, updateUrl]);

  // Sync selected product → product in URL
  useEffect(() => {
    if (!urlWriteReadyRef.current) return;
    updateUrl({ product: selected?.productId ?? undefined });
  }, [selected?.productId, updateUrl]);

  // Sync analysisMode → mode in URL
  useEffect(() => {
    if (!urlWriteReadyRef.current) return;
    updateUrl({ mode: analysisMode ?? undefined });
  }, [analysisMode, updateUrl]);

  // Sync baseLayer → base in URL (only write non-default)
  useEffect(() => {
    if (!urlWriteReadyRef.current) return;
    updateUrl({ base: baseLayer !== "MOLA" ? baseLayer : undefined });
  }, [baseLayer, updateUrl]);

  // Sync mapMode → view in URL (only write non-default)
  useEffect(() => {
    if (!urlWriteReadyRef.current) return;
    updateUrl({ view: mapMode !== "2D" ? mapMode : undefined });
  }, [mapMode, updateUrl]);

  // Track recently inspected products
  useEffect(() => {
    if (selected && selected.instrument !== "CUSTOM") {
      addRecentProduct(selected);
    }
  }, [selected, addRecentProduct]);

  // (field notes handlers moved to useFieldNotes hook)

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
      panelManager.ensurePanelVisible("feature_select");

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

    // SHARAD_HIGHRES: open the dedicated SharadHiresInspector and fly to track
    if (note.instrument === "SHARAD_HIGHRES") {
      setSharadHiresProductId(note.product_id);
      setSelected(null); // Close regular Inspector

      // If note has zero coords, look up real coordinates from the index
      if (note.lat === 0 && note.lon === 0) {
        try {
          const res = await fetch(`/api/footprints?instrument=SHARAD_HIGHRES&bbox=-180,-90,180,90&limit=5000&lod=poly`);
          if (res.ok) {
            const data = await res.json();
            const feat = data.features?.find((f: any) => f.properties?.product_id === note.product_id);
            if (feat?.geometry?.coordinates?.length >= 2) {
              const coords = feat.geometry.coordinates;
              const midIdx = Math.floor(coords.length / 2);
              setFlyToCoords({ lat: coords[midIdx][1], lon: coords[midIdx][0] });
              return;
            }
          }
        } catch { /* fall through */ }
      }
      setFlyToCoords({ lat: note.lat, lon: note.lon });
      return;
    }

    // Other non-DTM products: use original behavior
    // If note has zero coords, try to look them up
    let lat = note.lat;
    let lon = note.lon;
    if (lat === 0 && lon === 0) {
      try {
        const instruments = ["SHARAD", "CTX", "HIRISE", "CRISM"];
        for (const inst of instruments) {
          const res = await fetch(`/api/footprints?instrument=${inst}&bbox=-180,-90,180,90&limit=5000&lod=poly`);
          if (!res.ok) continue;
          const data = await res.json();
          const feat = data.features?.find((f: any) => f.properties?.product_id === note.product_id);
          if (feat?.geometry?.coordinates) {
            const coords = feat.geometry.coordinates;
            if (feat.geometry.type === "LineString" && coords.length >= 2) {
              const midIdx = Math.floor(coords.length / 2);
              const mid = coords[midIdx];
              if (Array.isArray(mid) && mid.length >= 2) {
                lat = mid[1]!;
                lon = mid[0]!;
              }
            } else if (feat.geometry.type === "Polygon" && coords[0]?.length >= 4) {
              const ring = coords[0];
              lat = ring.reduce((s: number, c: number[]) => s + (c[1] ?? 0), 0) / ring.length;
              lon = ring.reduce((s: number, c: number[]) => s + (c[0] ?? 0), 0) / ring.length;
            }
            break;
          }
        }
      } catch { /* use original coords */ }
    }
    setFlyToCoords({ lat, lon });
    setSelected({
      instrument: note.instrument as InspectorContext["instrument"],
      productId: note.product_id,
      lat,
      lon,
    });
    panelManager.ensurePanelVisible("feature_select");
  }, [panelManager.ensurePanelVisible]);

  // Handle HiRISE DTM footprint click - One-click inspection flow
  // Automatically: 1) Open inspector, 2) Enable quickview, 3) Activate DTM 3D mode
  // Then user can immediately click on the DTM to open 3D terrain view
  const handleHiRiseDTMClick = useCallback((productId: string, lat: number, lon: number, title?: string) => {
    // 1. Open the product inspector
    setSelected({
      instrument: "HIRISE_DTM",
      productId,
      lat,
      lon,
      title,
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
    setHiRiseDTM3DPoint(null);
  }, []);

  // Deactivate all overlays
  const handleDeactivateAll = useCallback(() => {
    setActiveOverlays(new Map());
    toast("All overlays cleared", { icon: "🗑" });
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


  // Mobile auto-open is handled by panelManager.ensurePanelVisible() at each trigger point

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
          setLineProfileData({ start: next[0]!, end: next[1]! });
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
    if (analysisMode === "ai_analysis") {
      // AI Analysis mode: set pin and open panel
      setSelected(null);
      setAiAnalysisPin({ lat, lon });
    }
    if (analysisMode === "guided") {
      // Guided workflow mode: update selected location
      setGuidedLocation({ lat, lon });
    }
    if (analysisMode === "region_stats") {
      // Region Stats mode: add vertex to polygon
      setRegionVertices((prev) => [...prev, { lat, lon }]);
    }
    if (analysisMode === "crater_detect") {
      // Crater detection: set scan center point
      setCraterDetectCenter({ lat, lon });
    }
    if (analysisMode === "pathfinder") {
      // Pathfinder mode: first click = start, second click = goal
      if (!pathfinderStart) {
        setPathfinderStart({ lat, lon });
      } else if (!pathfinderGoal) {
        setPathfinderGoal({ lat, lon });
      } else {
        // Reset and start new route
        setPathfinderRoute(null);
        setPathfinderStart({ lat, lon });
        setPathfinderGoal(null);
      }
      return;
    }
    // Accessibility explain mode: show tooltip on any terrain click
    if (accessibilityVisible && accessibilityExplainMode) {
      setAccessibilityExplainPoint({ lat, lon });
    }
  }, [analysisMode, accessibilityVisible, accessibilityExplainMode, pathfinderStart, pathfinderGoal]);

  // ── Simulation Callbacks ─────────────────────────────────
  const handleSimProgress = useCallback((progress: number) => {
    setSimProgress(progress);
    setSimSeekTo(null);
  }, []);

  const handleSimTelemetry = useCallback((telemetry: RoverTelemetry) => {
    setSimTelemetry(telemetry);
  }, []);

  const handleSimComplete = useCallback(() => {
    setSimPlaying(false);
    setSimComplete(true);
  }, []);

  // Reset simulation when a new route is planned
  useEffect(() => {
    setSimPlaying(false);
    setSimProgress(0);
    setSimTelemetry(null);
    setSimComplete(false);
    setSimSeekTo(null);
  }, [pathfinderRoute]);

  const simControls = useMemo(() => ({
    play: () => { setSimComplete(false); setSimPlaying(true); },
    pause: () => setSimPlaying(false),
    togglePlayPause: () => {
      if (simComplete) {
        setSimSeekTo(0);
        setSimComplete(false);
        setSimPlaying(true);
      } else {
        setSimPlaying(p => !p);
      }
    },
    setSpeed: (s: SpeedOption) => setSimSpeed(s),
    seek: (p: number) => {
      setSimSeekTo(p);
      setSimPlaying(false);
      if (p < 1) setSimComplete(false);
    },
    reset: () => {
      setSimSeekTo(0);
      setSimPlaying(false);
      setSimComplete(false);
    },
    toggleCamera: () => setSimCameraFollow(c => !c),
  }), [simComplete]);

  // When a product is selected, clear terrain point
  const handleSelect = useCallback((ctx: InspectorContext | null) => {
    console.log('[INSPECTOR] handleSelect called with:', ctx);
    setSelected(ctx);
    if (ctx) {
      setTerrainPoint(null);
      panelManager.ensurePanelVisible("feature_select");
    }
  }, [panelManager.ensurePanelVisible]);

  // Auto-activate quickview overlay when a product is selected and has no active overlay
  // Respects user preference stored in localStorage
  const autoQuickviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    // Clear any pending timer on rapid product switches
    if (autoQuickviewTimerRef.current) {
      clearTimeout(autoQuickviewTimerRef.current);
      autoQuickviewTimerRef.current = null;
    }

    if (!selected) return;

    // Respect user preference (default: enabled)
    const pref = localStorage.getItem("marslab.autoQuickview");
    if (pref === "false") return;

    // Skip if product already has an overlay
    if (activeOverlaysRef.current.has(selected.productId)) return;

    // Skip for custom datasets and SHARAD (no quickview available)
    if (selected.instrument === "CUSTOM" || selected.instrument === "SHARAD" || selected.instrument === "SHARAD_HIGHRES") return;

    // Debounce to avoid rapid-fire overlay creation during fast product switching
    autoQuickviewTimerRef.current = setTimeout(() => {
      // Re-check conditions after debounce
      if (activeOverlaysRef.current.has(selected.productId)) return;
      handleSetOverlay(selected.productId, "quickview");
    }, 300);

    return () => {
      if (autoQuickviewTimerRef.current) {
        clearTimeout(autoQuickviewTimerRef.current);
        autoQuickviewTimerRef.current = null;
      }
    };
  }, [selected?.productId, handleSetOverlay]);

  // Analysis mode toggle handler
  const handleAnalysisModeChange = useCallback((mode: AnalysisMode) => {
    setAnalysisMode(mode);
    // Clear state for all modes — ensures the selected mode's panel actually renders
    // (rightPanelContent priority chain: selected > terrainPoint > ... > analysisMode)
    if (mode === "agentic") {
      // These modes MUST clear higher-priority panel states to render
      setSelected(null);
      setSharadHiresProductId(null);
    }
    if (mode !== "slope") {
      setTerrainPoint(null);
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
    if (mode !== "guided") {
      setGuidedLocation(null);
    }
    if (mode !== "region_stats") {
      setRegionVertices([]);
    }
    if (mode !== "crater_detect") {
      setCraterDetectCenter(null);
      setCraterDetectFeatures([]);
    }
    if (mode !== "agentic" && mode !== "stratigraphy") {
      setEpsilonTarget(null);
    }
    if (mode !== "regolith") {
      setRegolithProductId(null);
    }
    if (mode !== "attenuation") {
      setAttenuationProductId(null);
    }
    if (mode !== "mineral_sequence") {
      setMineralSequenceObsId(null);
    }
    if (mode !== "strat_column") {
      setStratColumnTarget(null);
    }
    if (mode !== "pathfinder") {
      setPathfinderStart(null);
      setPathfinderGoal(null);
      setPathfinderRoute(null);
      setSimPlaying(false);
      setSimProgress(0);
      setSimTelemetry(null);
      setSimComplete(false);
      setSimSeekTo(null);
    }
  }, []);

  // Handle "Run ε Inversion" from CraterDetectPanel → opens StratigraphyPanel
  const handleRunEpsilon = useCallback((feature: DetectedFeature) => {
    setEpsilonTarget(feature);
    setAnalysisMode("stratigraphy");
  }, []);

  // Handle "Strat Column" from CraterDetectPanel → opens StratColumnPanel
  const handleOpenStratColumn = useCallback((feature: DetectedFeature) => {
    setStratColumnTarget(feature);
    setAnalysisMode("strat_column");
  }, []);

  // Guided Workflow action handler
  const handleWorkflowAction = useCallback((action: WorkflowAction) => {
    switch (action.type) {
      case "fly_to":
        setFlyToCoords({ lat: action.lat, lon: action.lon });
        break;
      case "load_instrument": {
        const inst = action.instrument as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM";
        const id = inst.toLowerCase() as InstrumentId;
        setInstrumentVisibility(prev => ({ ...prev, [id]: true }));
        setLoadFootprintsTrigger({ instrument: inst, timestamp: Date.now() });
        break;
      }
      case "set_analysis_mode":
        // Temporarily switch to the requested analysis mode, but keep guided active
        // For now, just trigger slope if that is what is requested
        if (action.mode === "slope" && guidedLocation) {
          setTerrainPoint({ lat: guidedLocation.lat, lon: guidedLocation.lon });
        }
        break;
      case "run_agentic":
        toast.success("Launching Agentic AI analysis...");
        break;
      case "show_results":
        toast.success("Step completed. Results available in the relevant panel.");
        break;
    }
  }, [guidedLocation]);

  // Command Palette action handler
  const handleCommandAction = useCallback((cmd: CommandAction) => {
    const { action } = cmd;
    switch (action.type) {
      case "fly_to":
        setFlyToCoords({ lat: action.lat, lon: action.lon });
        break;
      case "toggle_instrument":
        if (isInstrumentId(action.instrumentId)) {
          setInstrumentVisibility((prev) => ({
            ...prev,
            [action.instrumentId]: !prev[action.instrumentId as InstrumentId],
          }));
        }
        break;
      case "set_analysis":
        handleAnalysisModeChange(action.mode as AnalysisMode);
        break;
      case "set_map_mode":
        setMapMode(action.mode);
        break;
      case "toggle_grid":
        setShowGrid((prev) => !prev);
        break;
      case "navigate_page":
        navigate(action.path);
        break;
      case "show_keyboard_shortcuts":
        setShowKeyboardHelp(true);
        break;
      case "show_tour":
        setShowTourForced(true);
        break;
    }
  }, [handleAnalysisModeChange, navigate]);

  // Keyboard shortcut handler
  useEffect(() => {
    const INSTRUMENT_KEY_MAP: Record<string, InstrumentId> = {
      "1": "crism",
      "2": "hirise",
      "3": "sharad",
      "4": "sharad_highres",
      "5": "ctx",
      "6": "hirise_dtm",
      "7": "crism_trr3",
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't fire when typing in an input or textarea
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      // Don't fire if Command Palette is open
      if (commandPalette.isOpen) return;

      // Ctrl+Z / Cmd+Z — undo; Ctrl+Shift+Z / Cmd+Shift+Z — redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
        return;
      }

      // ? or Shift+/ — toggle keyboard shortcuts help
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setShowKeyboardHelp((prev) => !prev);
        return;
      }

      // Escape — close current panel/modal
      if (e.key === "Escape") {
        // Easter eggs first (highest z-index)
        if (showOlympusMonsClimber) { setShowOlympusMonsClimber(false); return; }
        if (showCuriosity) { setShowCuriosity(false); return; }
        if (showOlympusMons) { setShowOlympusMons(false); return; }
        if (showTerraform) { setShowTerraform(false); return; }
        if (showKeyboardHelp) {
          setShowKeyboardHelp(false);
          return;
        }
        if (sharadHiresProductId) {
          setSharadHiresProductId(null);
          setSharadTracePin(null);
          return;
        }
        if (selected) {
          setSelected(null);
          return;
        }
        if (analysisMode) {
          setAnalysisMode(null);
          return;
        }
        if (sharadPopup) {
          setSharadPopup(null);
          return;
        }
        return;
      }

      // n — select next product
      if (e.key === "n" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        const products = visibleProductsRef.current;
        if (products.length === 0) return;
        const currentIdx = selected
          ? products.findIndex((p: VisibleProduct) => p.productId === selected.productId)
          : -1;
        const nextIdx = currentIdx < products.length - 1 ? currentIdx + 1 : 0;
        const next = products[nextIdx];
        if (next) {
          handleSelectProduct(next);
        }
        return;
      }

      // p — select previous product
      if (e.key === "p" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        const products = visibleProductsRef.current;
        if (products.length === 0) return;
        const currentIdx = selected
          ? products.findIndex((p: VisibleProduct) => p.productId === selected.productId)
          : -1;
        const prevIdx = currentIdx > 0 ? currentIdx - 1 : products.length - 1;
        const prev = products[prevIdx];
        if (prev) {
          handleSelectProduct(prev);
        }
        return;
      }

      // g — toggle coordinate grid
      if (e.key === "g" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        setShowGrid((prev) => !prev);
        return;
      }

      // 1-7 — toggle instrument visibility
      if (INSTRUMENT_KEY_MAP[e.key] && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        const instId = INSTRUMENT_KEY_MAP[e.key];
        if (!instId) return;
        setInstrumentVisibility((prev) => ({ ...prev, [instId]: !prev[instId] }));
        return;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [commandPalette.isOpen, showKeyboardHelp, sharadHiresProductId, selected, analysisMode, sharadPopup, handleSelectProduct, undo, redo, showCuriosity, showOlympusMons, showOlympusMonsClimber, showTerraform]);

  // (flyToCoords state moved earlier — before URL state effects)

  // Handle view bound selected from map drag selection
  const handleViewBoundSelected = useCallback((bounds: BoundingBox) => {
    setViewBounds(bounds);
    setViewBoundSelectionMode(false); // Exit selection mode after selection
  }, []);

  // Stable ref for visibleProducts
  const visibleProductsRef = useRef(visibleProducts);
  visibleProductsRef.current = visibleProducts;

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
    const product = visibleProductsRef.current.find((p: VisibleProduct) => p.productId === productId);
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
      panelManager.ensurePanelVisible("search_result");
    } else if (instrument && instrument !== "CTX") {
      setSelected({
        instrument: instrument as any,
        productId,
        lat: lat ?? 0,
        lon: lon ?? 0,
      });
      panelManager.ensurePanelVisible("search_result");
    }
  }, [handleSelectProduct, panelManager.ensurePanelVisible]);

  // Derive legacy overlay formats for MapView compatibility
  // These will be replaced when MapView is updated to use unified format
  const derivedOverlays = useMemo(() => {
    const quickviewOverlays: string[] = [];
    const highResOverlays: string[] = [];
    const mineralOverlays: string[] = [];
    const browseOverlays = new Map<string, Set<"HYD" | "ICE" | "IC2">>();
    const scoreOverlays = new Map<string, Set<"score_ice" | "score_hyd">>();
    const opacities = new Map<string, number>();

    for (const [productId, overlay] of activeOverlays) {
      opacities.set(productId, overlay.opacity / 100);

      if (overlay.type === "quickview") {
        quickviewOverlays.push(productId);
      } else if (overlay.type === "highres") {
        highResOverlays.push(productId);
      } else if (overlay.type === "mineral_cnn") {
        mineralOverlays.push(productId);
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

    return { quickviewOverlays, highResOverlays, mineralOverlays, browseOverlays, scoreOverlays, opacities };
  }, [activeOverlays]);

  // --- Panel content (shared between desktop sidebar & mobile bottom sheet) ---
  const layerPanelContent = (
    <LayerPanel
      isMobile={isMobile}
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
      instrumentVisibility={instrumentVisibility}
      onToggleInstrument={handleToggleInstrument}
      // Explicit footprint loading
      onLoadFootprints={handleLoadFootprints}
      footprintsLoading={footprintsLoading}
      footprintCounts={footprintCounts}
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
      // Region Dashboard
      onShowRegionDashboard={() => setShowRegionDashboard(true)}
      // Fly-To navigation
      onFlyToCoords={handleFlyToCoords}
      // View bound selection mode
      viewBoundSelectionMode={viewBoundSelectionMode}
      onViewBoundSelectionModeChange={setViewBoundSelectionMode}
      // Coordinate grid
      showGrid={showGrid}
      onToggleGrid={setShowGrid}
      showRegionLayer={showRegionLayer}
      onToggleRegionLayer={setShowRegionLayer}
      swimLayer={swimLayer}
      onSwimLayerChange={setSwimLayer}
      swimIceLat={terrainPoint?.lat ?? null}
      swimIceLon={terrainPoint?.lon ?? null}
      scienceLayerVisibility={scienceLayerVisibility}
      onScienceLayerToggle={handleScienceLayerToggle}
      scienceLayerDepth={scienceLayerDepth}
      onScienceLayerDepthChange={setScienceLayerDepth}
      scienceLayerOpacities={scienceLayerOpacities}
      onScienceLayerOpacity={handleScienceLayerOpacity}
      // Field Notes
      fieldNotes={fieldNotes}
      showFieldNotesOnMap={showFieldNotesOnMap}
      onToggleFieldNotesOnMap={setShowFieldNotesOnMap}
      onFieldNoteClick={handleFieldNoteClick}
      onActiveTagChange={setFieldNoteActiveTag}
      // Overlap Filter
      overlapFilter={overlapFilter}
      onOverlapFilterChange={setOverlapFilter}
      overlapStats={overlapStats}
      // Measurement Tools
      showMeasurementTools={showMeasurementTools}
      onToggleMeasurementTools={setShowMeasurementTools}
      // Ice Accessibility
      accessibilityVisible={accessibilityVisible}
      onAccessibilityVisibleChange={setAccessibilityVisible}
      accessibilityOpacity={accessibilityOpacity}
      onAccessibilityOpacityChange={setAccessibilityOpacity}
      accessibilityExplainMode={accessibilityExplainMode}
      onAccessibilityExplainModeChange={setAccessibilityExplainMode}
      // Ice Prospecting (Fusion)
      fusionVisible={fusionVisible}
      onFusionVisibleChange={setFusionVisible}
      fusionOpacity={fusionOpacity}
      onFusionOpacityChange={setFusionOpacity}
    />
  );

  const rightPanelContent =
    // Explicit analysis modes take priority (user deliberately opened these)
    analysisMode === "agentic" ? (
      <AgenticPanel
        onClose={() => { setAnalysisMode(null); setEpsilonTarget(null); }}
        initialObjective={epsilonTarget ? `Run terrain εr inversion for terraced crater at (${epsilonTarget.lat.toFixed(3)}, ${epsilonTarget.lon.toFixed(3)}), diameter ${epsilonTarget.diameter_km?.toFixed(1) ?? "?"} km, terrace depth ${epsilonTarget.terrace_depth_m?.toFixed(0) ?? "?"} m. Search for SHARAD and HiRISE DTM data, then compute dielectric constant.` : undefined}
        onPanelAttention={handleAgenticPanelAttention}
      />
    ) : regolithProductId && analysisMode === "regolith" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading regolith analysis...</div>}>
        <RegolithPanel
          productId={regolithProductId}
          onClose={() => { setRegolithProductId(null); setAnalysisMode(null); }}
        />
      </Suspense>
    ) : epsilonTarget && analysisMode === "stratigraphy" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading stratigraphy analysis...</div>}>
        <StratigraphyPanel
          craterFeature={epsilonTarget}
          onClose={() => { setEpsilonTarget(null); setAnalysisMode(null); }}
        />
      </Suspense>
    ) : attenuationProductId && analysisMode === "attenuation" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading attenuation analysis...</div>}>
        <AttenuationPanel
          productId={attenuationProductId}
          onClose={() => { setAttenuationProductId(null); setAnalysisMode(null); }}
        />
      </Suspense>
    ) : mineralSequenceObsId && analysisMode === "mineral_sequence" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading mineral sequence...</div>}>
        <MineralSequencePanel
          obsId={mineralSequenceObsId}
          onClose={() => { setMineralSequenceObsId(null); setAnalysisMode(null); }}
        />
      </Suspense>
    ) : stratColumnTarget && analysisMode === "strat_column" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading stratigraphic column...</div>}>
        <StratColumnPanel
          craterFeature={stratColumnTarget}
          onClose={() => { setStratColumnTarget(null); setAnalysisMode(null); }}
        />
      </Suspense>
    ) : sharadHiresProductId ? (
      <SharadHiresInspector
        productId={sharadHiresProductId}
        onClose={() => { setSharadHiresProductId(null); setSharadTracePin(null); }}
        fieldNotes={fieldNotes}
        onOpenFieldNote={(pid, lat, lon) => setShowFieldNoteModal({ productId: pid, instrument: "SHARAD_HIGHRES", lat, lon })}
        onLocatePoint={(lat, lon) => setSharadTracePin({ lat, lon })}
        onOpenRegolith={handleOpenRegolith}
        onOpenAttenuation={handleOpenAttenuation}
      />
    ) : selected ? (
      <Inspector
        selected={selected}
        onClose={() => setSelected(null)}
        onCollapse={() => { setRightPanelCollapsed(true); panelManager.recordManualCollapse(); }}
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
        recentProducts={recentProducts}
        onSelectRecent={handleSelectRecent}
        onRemoveRecent={removeRecentProduct}
        onShow3DView={async (productId, lat, lon) => {
          setSelected(null);
          const center = await getDTMCenter(productId);
          const safeLat = center?.lat ?? lat;
          const safeLon = center?.lon ?? lon;
          setHiRiseDTM3DPoint({ productId, lat: safeLat, lon: safeLon });
        }}
        onFindRelated={(productId, instrument) => {
          window.open(`/download?tab=product&product_id=${encodeURIComponent(productId)}&instrument=${encodeURIComponent(instrument)}`, "_self");
        }}
        onDownloadProduct={handleDownloadProduct}
        onPinSpectrum={handlePinSpectrum}
        onFindTemporalPairs={(lat, lon, instrument) => setShowTemporalComparison({ lat, lon, instrument })}
        onOpenMineralSequence={handleOpenMineralSequence}
      />
    ) : terrainPoint ? (
      <SlopeAnalysis
        point={terrainPoint}
        onClose={() => setTerrainPoint(null)}
      />
    ) : hiRiseDTM3DPoint ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading 3D viewer…</div>}>
        <HiRiseDTM3DViewer
          point={hiRiseDTM3DPoint}
          onClose={() => {
            setHiRiseDTM3DPoint(null);
            setAnalysisMode(null);
            setActiveDTMProduct(null);
          }}
        />
      </Suspense>
    ) : aiAnalysisPin ? (
      <AiAnalysisPanel
        pin={aiAnalysisPin}
        onClose={() => {
          setAiAnalysisPin(null);
          setAnalysisMode(null);
        }}
      />
    ) : analysisMode === "report" ? (
      <Suspense fallback={<div className="flex items-center justify-center h-full text-[#6b7c9c] text-xs">Loading...</div>}>
        <ReportPanel onClose={() => setAnalysisMode(null)} isMobile={isMobile} />
      </Suspense>
    ) : analysisMode === "guided" ? (
      <GuidedWorkflows
        isOpen={true}
        onClose={() => { setAnalysisMode(null); setGuidedLocation(null); }}
        onAction={handleWorkflowAction}
        currentLocation={guidedLocation}
      />
    ) : analysisMode === "region_stats" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading region analysis...</div>}>
        <RegionStatsPanel
          vertices={regionVertices}
          onClose={() => { setAnalysisMode(null); setRegionVertices([]); }}
          onClearPolygon={() => setRegionVertices([])}
        />
      </Suspense>
    ) : analysisMode === "crater_detect" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading landform detector...</div>}>
        <CraterDetectPanel
          cameraViewportRef={cameraViewportRef}
          onClose={() => { setAnalysisMode(null); setCraterDetectCenter(null); setCraterDetectFeatures([]); }}
          onFlyTo={(lat, lon) => setFlyToCoords({ lat, lon })}
          onSearchHiRISE={(lat, lon) => {
            navigate(`/download?lat=${lat}&lon=${lon}&instrument=HIRISE`);
          }}
          onSearchSHARAD={(lat, lon) => {
            navigate(`/download?lat=${lat}&lon=${lon}&instrument=SHARAD`);
          }}
          onFeaturesChanged={setCraterDetectFeatures}
          onRunEpsilonInversion={handleRunEpsilon}
          onOpenStratColumn={handleOpenStratColumn}
        />
      </Suspense>
    ) : analysisMode === "pathfinder" ? (
      <Suspense fallback={<div className="w-96 bg-[#101622] flex items-center justify-center text-[#6b7c9c] text-sm">Loading pathfinder...</div>}>
        <PathfinderPanel
          startPoint={pathfinderStart}
          goalPoint={pathfinderGoal}
          onRouteReady={(route) => setPathfinderRoute(route)}
          onClear={() => { setPathfinderStart(null); setPathfinderGoal(null); setPathfinderRoute(null); }}
          onSuggestRoute={(start, goal) => {
            setPathfinderStart(start);
            setPathfinderGoal(goal);
            setPathfinderRoute(null);
            setFlyToCoords({ lat: (start.lat + goal.lat) / 2, lon: (start.lon + goal.lon) / 2 });
          }}
          simPlaying={simPlaying}
          simSpeed={simSpeed}
          simProgress={simProgress}
          simTelemetry={simTelemetry}
          simComplete={simComplete}
          simCameraFollow={simCameraFollow}
          simControls={simControls}
        />
      </Suspense>
    ) : (
      <div className="h-full flex items-center justify-center bg-[#101622]">
        <EmptyState
          icon="explore"
          title="No product selected"
          description="Click a footprint on the map or search for a product to inspect it. Load instrument layers from the Layers panel to get started."
          actionLabel="Open Agentic AI"
          onAction={() => handleAnalysisModeChange("agentic")}
        />
      </div>
    );

  return (
    <AppShell
      isMobile={isMobile}
      header={
        <TopBar
          isMobile={isMobile}
          onSelectResult={handleSearchSelect}
          onEasterEgg={() => setShowGame(true)}
          onTerraform={() => setShowTerraform(true)}
          onCuriositySelfie={() => setShowCuriosity(true)}
          onMarkWatney={handleMarkWatney}
          canUndo={canUndo}
          canRedo={canRedo}
          lastActionDescription={lastAction?.description}
          onUndo={undo}
          onRedo={redo}
        />
      }
      footer={<Footer />}
      leftPanel={isMobile ? null : layerPanelContent}
      rightPanel={isMobile ? undefined : (
        rightPanelCollapsed ? (
          /* Collapsed strip — thin vertical bar with expand button */
          <div
            key={panelManager.stripPulseKey}
            className={`h-full w-8 bg-[#0a0f18] border-l border-border-dark flex flex-col items-center pt-2 gap-2${panelManager.stripPulse ? " strip-attention-pulse" : ""}`}
            onAnimationEnd={panelManager.clearStripPulse}
          >
            <button
              onClick={() => setRightPanelCollapsed(false)}
              className="p-1 rounded hover:bg-[#232f48] text-[#6b7c9c] hover:text-white transition-colors"
              title="Expand panel"
              aria-label="Expand side panel"
            >
              <span className="material-symbols-outlined text-base">chevron_left</span>
            </button>
            {/* Vertical label */}
            <span className="text-[8px] text-[#6b7c9c] uppercase tracking-widest font-bold"
              style={{ writingMode: "vertical-lr", textOrientation: "mixed" }}>
              {selected ? "Inspector" : analysisMode ? "Analysis" : "Panel"}
            </span>
          </div>
        ) : (
          /* Expanded panel — wrap content with collapse button in header */
          <div className="h-full flex flex-col relative">
            {/* Collapse button — only for panels that don't have their own collapse
                (Inspector and SharadHiresInspector handle it internally) */}
            {!selected && !sharadHiresProductId && (
              <button
                onClick={() => { setRightPanelCollapsed(true); panelManager.recordManualCollapse(); }}
                className="absolute top-2 right-12 z-30 p-1 rounded hover:bg-[#232f48] text-[#6b7c9c] hover:text-white transition-colors"
                title="Collapse panel"
                aria-label="Collapse side panel"
              >
                <span className="material-symbols-outlined text-base">chevron_right</span>
              </button>
            )}
            <div className="flex-1 overflow-hidden">
              <PanelAttentionWrapper
                isActive={panelManager.attentionPulse}
                pulseKey={panelManager.attentionPulseKey}
                onPulseEnd={panelManager.clearAttentionPulse}
              >
                {rightPanelContent}
              </PanelAttentionWrapper>
            </div>
          </div>
        )
      )}
      mobileNav={isMobile ? (
        <div className="flex items-center justify-around border-t border-border-dark bg-bg-dark px-2 py-1.5">
          <button
            onClick={() => setMobilePanel(p => p === 'layers' ? 'none' : 'layers')}
            className={`flex flex-col items-center gap-0.5 px-4 py-1 rounded-lg transition-colors ${
              mobilePanel === 'layers' ? 'text-primary' : 'text-slate-400'
            }`}
            aria-label="Toggle layers panel"
            aria-expanded={mobilePanel === 'layers'}
          >
            <span className="material-symbols-outlined text-xl">layers</span>
            <span className="text-[10px] font-medium">Layers</span>
          </button>
          <button
            onClick={() => setMobilePanel(p => p === 'inspector' ? 'none' : 'inspector')}
            className={`flex flex-col items-center gap-0.5 px-4 py-1 rounded-lg transition-colors ${
              mobilePanel === 'inspector' ? 'text-primary' : 'text-slate-400'
            }`}
            aria-label="Toggle inspector panel"
            aria-expanded={mobilePanel === 'inspector'}
          >
            <span className="material-symbols-outlined text-xl">info</span>
            <span className="text-[10px] font-medium">Inspector</span>
          </button>
        </div>
      ) : undefined}
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
        showCRISM_TRR3={showCRISM_TRR3}
        onSharadClick={handleSharadClick}
        onSharadHiresClick={handleSharadHiresClick}
        onHiRiseDTMClick={handleHiRiseDTMClick}
        onToggleOverlay={(productId, type) => handleSetOverlay(productId, type)}
        quickviewOverlays={derivedOverlays.quickviewOverlays}
        highResOverlays={derivedOverlays.highResOverlays}
        mineralOverlays={derivedOverlays.mineralOverlays}
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
        fieldNotes={mapFieldNotesForView}
        onFieldNoteClick={handleFieldNoteClick}
        activeDTMProductId={activeDTMProduct}
        showGrid={showGrid}
        showRegionLayer={showRegionLayer}
        swimLayer={swimLayer}
        scienceLayerVisibility={scienceLayerVisibility}
        scienceLayerDepth={scienceLayerDepth}
        scienceLayerOpacities={scienceLayerOpacities}
        aiAnalysisPin={aiAnalysisPin}
        overlapFilter={overlapFilter}
        onOverlapStatsChange={handleOverlapStatsChange}
        highlightProductId={highlightProductId}
        onHighlightComplete={handleHighlightComplete}
        inspectedProductId={inspectedProductId}
        sharadTracePin={sharadTracePin}
        showMeasurementTools={showMeasurementTools}
        onMeasurementPinNote={handleMeasurementPinNote}
        terraformMode={showTerraform}
        onOlympusMonsTripleClick={handleOlympusMonsClick}
        onOlympusMonsClimber={handleOlympusMonsClimber}
        craterDetectFeatures={craterDetectFeatures}
        cameraViewportRef={cameraViewportRef}
        accessibilityVisible={accessibilityVisible}
        accessibilityOpacity={accessibilityOpacity}
        fusionVisible={fusionVisible}
        fusionOpacity={fusionOpacity}
        pathfinderStart={pathfinderStart}
        pathfinderGoal={pathfinderGoal}
        pathfinderRoute={pathfinderRoute}
        simPlaying={simPlaying}
        simSpeed={simSpeed}
        simCameraFollow={simCameraFollow}
        simSeekTo={simSeekTo}
        onSimProgress={handleSimProgress}
        onSimTelemetry={handleSimTelemetry}
        onSimComplete={handleSimComplete}
      />

      {/* Accessibility Explain Tooltip — floating on map */}
      {accessibilityExplainPoint && (
        <AccessibilityExplainTooltip
          lat={accessibilityExplainPoint.lat}
          lon={accessibilityExplainPoint.lon}
          onClose={() => setAccessibilityExplainPoint(null)}
        />
      )}

      {/* MARVIS FAB — floating chat widget with grounded tool calling */}
      {!isMobile && (
        <CopilotFab
          hidden={analysisMode === "agentic"}
          inspectorOpen={selected !== null}
          onFlyTo={(lat, lon) => setFlyToCoords({ lat, lon })}
          onLoadInstrument={(inst) => handleLoadFootprints(inst as "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3")}
          onSelectInstrumentProduct={handleSelectInstrumentProduct}
          onSearchResults={handleSearchResults}
          loadedInstruments={
            (Object.entries(instrumentVisibility) as [string, boolean][])
              .filter(([, v]) => v)
              .map(([k]) => k.toUpperCase())
          }
          visibleProductCount={visibleProducts.length}
          currentLat={lastNavCoords?.lat ?? null}
          currentLon={lastNavCoords?.lon ?? null}
        />
      )}

      {/* Spectral Comparison Panel (floating, bottom-right) */}
      {showSpectralComparison && pinnedSpectra.length > 0 && (
        <SpectralComparison
          spectra={pinnedSpectra}
          onRemove={(id) =>
            setPinnedSpectra((prev) => {
              const next = prev.filter((s) => s.id !== id);
              if (next.length === 0) setShowSpectralComparison(false);
              return next;
            })
          }
          onClear={() => {
            setPinnedSpectra([]);
            setShowSpectralComparison(false);
          }}
          onClose={() => setShowSpectralComparison(false)}
        />
      )}

      {/* Mobile Bottom Sheets */}
      {isMobile && (
        <>
          <BottomSheet
            isOpen={mobilePanel === 'layers'}
            onClose={() => setMobilePanel('none')}
            title="Layers"
          >
            {layerPanelContent}
          </BottomSheet>
          <BottomSheet
            isOpen={mobilePanel === 'inspector'}
            onClose={() => setMobilePanel('none')}
            title="Inspector"
          >
            {rightPanelContent || (
              <div className="p-6 text-center text-slate-500 text-sm">
                Tap a footprint on the map to inspect it
              </div>
            )}
          </BottomSheet>
        </>
      )}

      {/* Line Profile Popup */}
      {lineProfileData && (
        <ErrorBoundary scope="LineProfile">
          <Suspense fallback={null}>
            <LineProfile
              startPoint={lineProfileData.start}
              endPoint={lineProfileData.end}
              onClose={() => {
                setLineProfileData(null);
                setLinePoints([]);
              }}
            />
          </Suspense>
        </ErrorBoundary>
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
                aria-label="Close SHARAD popup"
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
                aria-label="Close SHARAD popup"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Temporal Comparison Modal */}
      {showTemporalComparison && (
        <ErrorBoundary scope="TemporalComparison">
          <Suspense fallback={null}>
            <TemporalComparison
              lat={showTemporalComparison.lat}
              lon={showTemporalComparison.lon}
              initialInstrument={showTemporalComparison.instrument}
              onClose={() => setShowTemporalComparison(null)}
            />
          </Suspense>
        </ErrorBoundary>
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

      {/* Command Palette (Cmd+K) — renders via portal to document.body */}
      <CommandPalette
        isOpen={commandPalette.isOpen}
        onClose={commandPalette.close}
        onAction={handleCommandAction}
      />

      {/* Region Dashboard full-viewport overlay */}
      {showRegionDashboard && (
        <ErrorBoundary scope="RegionDashboard">
          <Suspense fallback={null}>
            <RegionDashboard
              isOpen={showRegionDashboard}
              onClose={() => setShowRegionDashboard(false)}
              onFlyTo={(lat, lon) => {
                setShowRegionDashboard(false);
                setFlyToCoords({ lat, lon });
              }}
              onRunReport={() => {
                setShowRegionDashboard(false);
                setAnalysisMode("report");
              }}
            />
          </Suspense>
        </ErrorBoundary>
      )}

      {/* Easter egg: Space Shooter Game */}
      {showGame && (
        <ErrorBoundary scope="SpaceGame">
          <Suspense fallback={null}>
            <SpaceGame onClose={() => setShowGame(false)} />
          </Suspense>
        </ErrorBoundary>
      )}

      {/* Easter egg: Curiosity Selfie */}
      {showCuriosity && (
        <CuriositySelfieModal onClose={() => setShowCuriosity(false)} />
      )}

      {/* Easter egg: Olympus Mons Height Comparison */}
      {showOlympusMons && (
        <OlympusMonsPanel onClose={() => setShowOlympusMons(false)} />
      )}

      {/* Easter egg: Olympus Mons Climber */}
      {showOlympusMonsClimber && (
        <OlympusMonsClimber onClose={() => setShowOlympusMonsClimber(false)} />
      )}

      {/* Easter egg: Terraform Mode overlay */}
      <TerraformOverlay active={showTerraform} />

      {/* Keyboard Shortcuts Help Modal */}
      <KeyboardShortcuts
        isOpen={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
      />

      {/* Onboarding Tour */}
      <OnboardingTour
        forceOpen={showTourForced}
        onComplete={() => setShowTourForced(false)}
      />
    </AppShell>
  );
}
