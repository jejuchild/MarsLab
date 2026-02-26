// src/MapView.tsx
import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { LRUMap } from "../utils/LRUMap";
import { normalizeLonForMap } from "../utils/coordinates";
import MapToolbar from "./map/MapToolbar";
import {
  throttle,
} from "../utils/dtmHover";
import type { FieldNote } from "../api/fieldnotes";
import type { OverlapStats } from "../utils/overlapFilter";
import { getInstrumentCesiumColor } from "../config/instrumentRegistry";
import useDTMHover from "../hooks/useDTMHover";
import useMapViewer from "../hooks/useMapViewer";
import useFootprints from "../hooks/useFootprints";
import useOverlays from "../hooks/useOverlays";


/* ==================================================
 * Types
 * ==================================================*/
export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

export type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  // CRISM pixel coordinates for spectrum (optional)
  pixelLine?: number;
  pixelSample?: number;
  // Product title (e.g., HiRISE observation title)
  title?: string;
};

type VisibleProduct = {
  productId: string;
  instrument: InstrumentType;
  title?: string;
  lat?: number;
  lon?: number;
};

type RGBWavelengths = {
  r: number;
  g: number;
  b: number;
};

type BrowseProductType = "HYD" | "ICE" | "IC2";
type ScoreProductType = "score_ice" | "score_hyd";

type BaseLayerType = "MOLA" | "HRSC";

type MapMode = "2D" | "3D";

type BoundingBox = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;

type SHARADPopup = {
  productId: string;
  quickviewUrl: string;
  startLat: number;
  startLon: number;
  stopLat: number;
  stopLon: number;
} | null;

// Explicit loading applies to all instruments
type ExplicitLoadInstrument = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "HIRISE_DTM" | "CRISM_TRR3";

// Footprint load result for UI feedback
type FootprintLoadResult = {
  instrument: ExplicitLoadInstrument;
  count: number;
  truncated: boolean;
  total: number;
};

type MapViewProps = {
  mapMode?: MapMode;
  baseLayer?: BaseLayerType;
  viewBounds?: BoundingBox;
  onSelect: (ctx: InspectorContext | null) => void;
  showCRISM: boolean;
  showHiRISE: boolean;
  showSHARAD: boolean;
  showSharadHighres: boolean;
  showCTX: boolean;
  showHiRISEDTM?: boolean;
  showCRISM_TRR3?: boolean;
  onSharadClick?: (popup: SHARADPopup) => void;
  onSharadHiresClick?: (productId: string) => void;
  onHiRiseDTMClick?: (productId: string, lat: number, lon: number, title?: string) => void;
  onToggleOverlay?: (productId: string, type: "quickview" | null) => void;
  quickviewOverlays?: string[];
  highResOverlays?: string[];
  mineralOverlays?: string[];
  browseOverlays?: Map<string, Set<BrowseProductType>>;
  scoreOverlays?: Map<string, Set<ScoreProductType>>;
  overlayOpacities?: Map<string, number>; // Per-product opacity (0-1)
  onVisibleProductsChange?: (products: VisibleProduct[]) => void;
  flyToProductId?: string | null;
  onFlyToComplete?: () => void;
  flyToCoords?: { lat: number; lon: number } | null;
  onFlyToCoordsComplete?: () => void;
  bringToFrontId?: string | null;
  onBringToFrontComplete?: () => void;
  rgbWavelengths?: RGBWavelengths;
  // Bidirectional hover highlight for Active Products Panel
  hoveredProductId?: string | null;
  onHoverProduct?: (productId: string | null) => void;
  // Ice score filter - pass filtered CRISM product IDs (null = no filter)
  crismFilteredIds?: Set<string> | null;
  // Explicit footprint loading - triggered by button click
  loadFootprintsTrigger?: { instrument: ExplicitLoadInstrument; timestamp: number } | null;
  onFootprintsLoaded?: (result: FootprintLoadResult) => void;
  onFootprintsLoading?: (instrument: ExplicitLoadInstrument, loading: boolean) => void;
  // Terrain click (no footprint hit) – for slope analysis
  onTerrainClick?: (lat: number, lon: number) => void;
  // Custom user-uploaded datasets
  showCustomData?: boolean;
  customDatasets?: Array<{
    id: string;
    name: string;
    bounds: { west: number; south: number; east: number; north: number };
    visible: boolean;
    opacity: number;
  }>;
  // Analysis mode
  analysisMode?: "slope" | "hirise_dtm_3d" | "line" | "ai_analysis" | "agentic" | "report" | "guided" | "region_stats" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column" | null;
  // Active HiRISE DTM product for hover elevation probing
  activeDTMProductId?: string | null;
  linePoints?: Array<{ lat: number; lon: number }>;
  // View bound selection mode (drag to select rectangle)
  viewBoundSelectionMode?: boolean;
  onViewBoundSelected?: (bounds: BoundingBox) => void;
  // Field Notes – full notes array for independent rendering
  fieldNotes?: FieldNote[];
  // Callback when a field note marker is clicked (opens inspector)
  onFieldNoteClick?: (note: FieldNote) => void;
  // Coordinate grid overlay
  showGrid?: boolean;
  showRegionLayer?: boolean;
  // AI Analysis pin location
  aiAnalysisPin?: { lat: number; lon: number } | null;
  // Multi-Instrument Overlap Filter
  overlapFilter?: { enabled: boolean; instruments: string[] };
  onOverlapStatsChange?: (stats: OverlapStats | null) => void;
  // Temporary highlight after fly-to (from deep-link)
  highlightProductId?: string | null;
  onHighlightComplete?: () => void;
  // Keep inspected product's footprint visible even when layer is off
  inspectedProductId?: string | null;
  // SHARAD radargram trace pin — show clicked radargram location on map
  sharadTracePin?: { lat: number; lon: number } | null;
  // Measurement & annotation tools
  showMeasurementTools?: boolean;
  onMeasurementPinNote?: (lat: number, lon: number, text: string) => void;
  // Crater/Landform Detection
  craterDetectFeatures?: Array<{
    id: string;
    type: string;
    lat: number;
    lon: number;
    diameter_km?: number;
    area_km2?: number;
    length_km?: number;
    morphology?: string;
    confidence: number;
    description: string;
    path?: [number, number][];
    boundary?: [number, number][];
  }>;

  // Camera viewport ref — updated on moveEnd for on-demand viewport queries
  cameraViewportRef?: React.MutableRefObject<{ minLat: number; maxLat: number; westLon: number; eastLon: number } | null>;

  // Easter eggs
  terraformMode?: boolean;
  /** SWIM layer: "0-1m" | "1-5m" | ">5m" | false */
  swimLayer?: string | false;
  onOlympusMonsTripleClick?: () => void;
  onOlympusMonsClimber?: () => void;
};

/* ==================================================
 * Mars constants
 * ==================================================*/
// IAU-defined Mars ellipsoid parameters (meters)
// Source: IAU Working Group on Cartographic Coordinates and Rotational Elements
// These values are used by NASA PDS, HiRISE, CRISM, and NASA Trek base layers
const MARS_EQUATORIAL_RADIUS = 3396190; // meters (a = b axis)
const MARS_POLAR_RADIUS = 3376200;      // meters (c axis)

// Field note marker icon colors by instrument
const FIELDNOTE_COLORS: Record<string, string> = {
  CRISM: "#22d3ee",      // cyan
  HIRISE: "#facc15",     // yellow
  SHARAD: "#fb923c",     // orange
  SHARAD_HIGHRES: "#fb923c",
  CTX: "#f472b6",        // pink
  CUSTOM: "#e879f9",     // fuchsia
  HIRISE_DTM: "#d97706", // amber
};

// Create a canvas-based icon for field note marker (cached per instrument)
const _fieldNoteIconCache = new LRUMap<string, string>(64);
function createFieldNoteIcon(instrument: string): string {
  const cached = _fieldNoteIconCache.get(instrument);
  if (cached) return cached;

  const canvas = document.createElement("canvas");
  canvas.width = 24;
  canvas.height = 24;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const color = FIELDNOTE_COLORS[instrument] || "#fbbf24";

  // Draw pin shape
  ctx.beginPath();
  ctx.arc(12, 9, 7, Math.PI, 0, false);
  ctx.lineTo(12, 22);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Inner circle (white)
  ctx.beginPath();
  ctx.arc(12, 9, 3, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();

  const dataUrl = canvas.toDataURL();
  _fieldNoteIconCache.set(instrument, dataUrl);
  return dataUrl;
}

// Create a proper oblate Mars ellipsoid for accurate geospatial positioning
// This matches the reference used by PDS products and NASA Trek base layers
const MARS_ELLIPSOID = new Cesium.Ellipsoid(
  MARS_EQUATORIAL_RADIUS,
  MARS_EQUATORIAL_RADIUS,
  MARS_POLAR_RADIUS
);

const MARS_RECT = Cesium.Rectangle.fromDegrees(-180, -90, 180, 90);

// Base layer URLs from Trek API
const BASE_LAYER_URLS: Record<BaseLayerType, string> = {
  MOLA: "https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg",
  HRSC: "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg",
};

/* ==================================================
 * Helpers
 * ==================================================*/
function parseLBLValue(
  block: string | null | undefined,
  key: string
): number | null {
  if (!block) return null;
  const m = block.match(new RegExp(`${key}\\s*=\\s*([-+0-9.eE]+)`, "i"));
  return m ? Number(m[1]) : null;
}

// Using shared coordinate utility
const normalizeLonTo180 = normalizeLonForMap;

/**
 * Extract CRISM observation ID from full product ID
 * e.g., "frt0001fd76_07_if166j_mtr3" -> "frt0001fd76"
 *
 * Pattern: 3-letter prefix + 8 hex characters
 */
function extractCrismObsId(productId: string): string {
  // Match 3 letters followed by 8 hex digits at the start
  const match = productId.match(/^([a-z]{3}[0-9a-f]{8})/i);
  const obsId = match?.[1];
  return obsId ? obsId.toLowerCase() : productId.toLowerCase();
}

/* ==================================================
 * LBL loaders with bounds caching
 * ==================================================*/
const hiriseLBLCache = new Map<string, string>();
const crismLBLCache = new Map<string, string>();
const HIRISE_LBL_BASE = "/hirise_lbl";

// PERFORMANCE: Cache parsed bounds to avoid re-parsing LBL files
interface ProductBounds {
  west: number;
  south: number;
  east: number;
  north: number;
  lines?: number;
  samples?: number;
}
const boundsCache = new LRUMap<string, ProductBounds>(500);

async function loadHiRISELBL(id: string): Promise<string | null> {
  if (hiriseLBLCache.has(id)) return hiriseLBLCache.get(id)!;

  const res = await fetch(`${HIRISE_LBL_BASE}/${id}_RED.lbl`);
  if (!res.ok) return null;

  const text = await res.text();
  if (!text.includes("IMAGE_MAP_PROJECTION")) return null;

  hiriseLBLCache.set(id, text);
  return text;
}

async function loadCRISMLBL(id: string): Promise<string | null> {
  if (crismLBLCache.has(id)) return crismLBLCache.get(id)!;
  const CRISM_LBL_BASE = "/crism_lbl";

  // Extract base_key from product_id
  // frt00008a1e_07_if168j_mtr3 -> frt00008a1e_07
  // frt00008a1e_07_brcarj_mtr3 -> frt00008a1e_07
  const baseKey = id.replace(/_(?:if|br)[0-9a-z]+_mtr3$/i, "");

  // Try multiple possible LBL file patterns:
  // 1. Browse LBL: frt00008a1e_07_brcarj_mtr3.lbl (common for downloaded products)
  // 2. Direct LBL: frt00008a1e_07.lbl (legacy pattern)
  const patterns = [
    `${CRISM_LBL_BASE}/${baseKey}_brcarj_mtr3.lbl`,  // Browse LBL (most common)
    `${CRISM_LBL_BASE}/${baseKey}.lbl`,               // Direct LBL (legacy)
    `${CRISM_LBL_BASE}/${id}.lbl`,                    // Full product ID
  ];

  for (const url of patterns) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;

      const text = await res.text();
      if (!text.includes("IMAGE_MAP_PROJECTION")) continue;

      crismLBLCache.set(id, text);
      return text;
    } catch {
      continue;
    }
  }

  return null;
}

type GeoJsonIndexFeature = {
  properties?: {
    product_id?: string;
    west?: number;
    east?: number;
    south?: number;
    north?: number;
  };
  geometry?: {
    coordinates?: number[][][];
  };
};

type GeoJsonIndex = {
  features?: GeoJsonIndexFeature[];
};

// Cache for HiRISE DTM index
let hiriseDTMIndexCache: GeoJsonIndex | null = null;
// Cache for CRISM TRR3 index
let crismTRR3IndexCache: GeoJsonIndex | null = null;

// Compute bounding box from GeoJSON polygon coordinates
function boundsFromPolygon(coords: number[][]): ProductBounds | null {
  if (!coords || coords.length < 3) return null;
  let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
  for (const pair of coords) {
    const lon = pair[0];
    const lat = pair[1];
    if (lon == null || lat == null) continue;
    if (lon < west) west = lon;
    if (lon > east) east = lon;
    if (lat < south) south = lat;
    if (lat > north) north = lat;
  }
  return { west, east, south, north };
}

// Get cached bounds or parse from LBL
async function getProductBounds(productId: string): Promise<ProductBounds | null> {
  if (boundsCache.has(productId)) {
    return boundsCache.get(productId)!;
  }

  // Check for HiRISE DTM products (start with DTE, e.g. DTEEC_, DTEED_)
  const isHiRISEDTM = productId.startsWith("DTE");

  if (isHiRISEDTM) {
    // Load bounds from HiRISE DTM index
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

    if (hiriseDTMIndexCache?.features) {
      for (const feature of hiriseDTMIndexCache.features) {
        if (feature.properties?.product_id === productId) {
          const props = feature.properties;
          if (
            props.west == null ||
            props.east == null ||
            props.south == null ||
            props.north == null
          ) {
            continue;
          }
          const bounds: ProductBounds = {
            west: props.west,
            east: props.east,
            south: props.south,
            north: props.north,
          };
          boundsCache.set(productId, bounds);
          return bounds;
        }
      }
    }
    return null;
  }

  // Check for CRISM TRR3 products — bounds from index.geojson polygon
  const isTRR3 = /^(frs|msv|frt|hrl|hrs|arl|atl)[0-9a-f]+_\d{2}$/i.test(productId);
  if (isTRR3) {
    if (!crismTRR3IndexCache) {
      try {
        const res = await fetch("/crism_trr3_index.geojson");
        if (res.ok) {
          crismTRR3IndexCache = await res.json();
        }
      } catch {
        return null;
      }
    }

    if (crismTRR3IndexCache?.features) {
      for (const feature of crismTRR3IndexCache.features) {
        if (feature.properties?.product_id === productId) {
          const coords = feature.geometry?.coordinates?.[0];
          if (!coords) continue;
          const bounds = boundsFromPolygon(coords);
          if (bounds) {
            boundsCache.set(productId, bounds);
            return bounds;
          }
        }
      }
    }
    return null;
  }

  const isHiRISE = productId.startsWith("ESP_");
  const lbl = isHiRISE ? await loadHiRISELBL(productId) : await loadCRISMLBL(productId);
  if (!lbl) return null;

  const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
  const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
  const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
  const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");
  const lines = parseLBLValue(lbl, "LINES");
  const samples = parseLBLValue(lbl, "LINE_SAMPLES");

  if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
    return null;
  }

  const bounds: ProductBounds = {
    west: normalizeLonTo180(westLon360),
    east: normalizeLonTo180(eastLon360),
    south: Math.min(minLat, maxLat),
    north: Math.max(minLat, maxLat),
    lines: lines ?? undefined,
    samples: samples ?? undefined,
  };

  boundsCache.set(productId, bounds);
  return bounds;
}

/**
 * Fallback: extract bounds from an existing footprint entity's rectangle coordinates.
 * Tries all instrument prefixes to find the entity.
 */
function getFootprintBounds(viewer: Cesium.Viewer, productId: string): ProductBounds | null {
  const prefixes: string[] = ["CRISM", "HIRISE", "CTX", "HIRISE_DTM", "CRISM_TRR3", "SHARAD", "SHARAD_HIGHRES"];
  for (const prefix of prefixes) {
    const ent = viewer.entities.getById(`${prefix}_FP_${productId}`);
    if (!ent?.rectangle?.coordinates) continue;
    try {
      const coords = ent.rectangle.coordinates;
      const rect = coords instanceof Cesium.ConstantProperty
        ? coords.getValue(Cesium.JulianDate.now())
        : coords;
      if (rect instanceof Cesium.Rectangle) {
        const bounds: ProductBounds = {
          west: Cesium.Math.toDegrees(rect.west),
          east: Cesium.Math.toDegrees(rect.east),
          south: Cesium.Math.toDegrees(rect.south),
          north: Cesium.Math.toDegrees(rect.north),
        };
        boundsCache.set(productId, bounds);
        return bounds;
      }
    } catch {
      // Entity may have been removed between check and access
    }
  }
  return null;
}

const HILITE_RECT_MATERIAL_HIRISE = new Cesium.ColorMaterialProperty(
  Cesium.Color.YELLOW.withAlpha(0.7)
);
const HILITE_RECT_MATERIAL_CRISM = new Cesium.ColorMaterialProperty(
  Cesium.Color.CYAN.withAlpha(0.6)
);
const HILITE_RECT_MATERIAL_CUSTOM = new Cesium.ColorMaterialProperty(
  Cesium.Color.FUCHSIA.withAlpha(0.3)
);
const HILITE_RECT_MATERIAL_CTX = new Cesium.ColorMaterialProperty(
  Cesium.Color.fromCssColorString("#FF69B4").withAlpha(0.6)
);
const HILITE_RECT_MATERIAL_DTM = new Cesium.ColorMaterialProperty(
  Cesium.Color.fromCssColorString("#d97706").withAlpha(0.6)
);
const HILITE_RECT_MATERIAL_TRR3 = new Cesium.ColorMaterialProperty(
  Cesium.Color.fromCssColorString("#00CED1").withAlpha(0.6)
);

function getHiliteMaterial(inst: string): Cesium.ColorMaterialProperty {
  switch (inst) {
    case "HIRISE": return HILITE_RECT_MATERIAL_HIRISE;
    case "CUSTOM": return HILITE_RECT_MATERIAL_CUSTOM;
    case "CTX": return HILITE_RECT_MATERIAL_CTX;
    case "HIRISE_DTM": return HILITE_RECT_MATERIAL_DTM;
    case "CRISM_TRR3": return HILITE_RECT_MATERIAL_TRR3;
    default: return HILITE_RECT_MATERIAL_CRISM;
  }
}

function getEntityInstrument(e: Cesium.Entity): InstrumentType | null {
  const p: any = e.properties;
  const inst = p?.instrument?.getValue?.();
  if (typeof inst === "string" && inst.length > 0) return inst as InstrumentType;
  return null;
}

function getEntityProductId(e: Cesium.Entity): string | null {
  const p: any = e.properties;
  const id = p?.product_id?.getValue?.();
  return typeof id === "string" ? id : null;
}

/* ==================================================
 * Click zoom helper (✅ 덜 과하게)
 * ==================================================*/
function paddedRectangle(rect: Cesium.Rectangle, padRatio = 0.6): Cesium.Rectangle {
  // padRatio: 0.6이면 width/height의 60%만큼 여유 (너무 과하면 줄여)
  const w = rect.east - rect.west;
  const h = rect.north - rect.south;

  const padW = w * padRatio * 0.5;
  const padH = h * padRatio * 0.5;

  const west = rect.west - padW;
  const east = rect.east + padW;
  const south = rect.south - padH;
  const north = rect.north + padH;

  // clamp (radians)
  const clampLon = (x: number) => Math.max(-Math.PI, Math.min(Math.PI, x));
  const clampLat = (x: number) => Math.max(-Math.PI / 2, Math.min(Math.PI / 2, x));

  return new Cesium.Rectangle(
    clampLon(west),
    clampLat(south),
    clampLon(east),
    clampLat(north)
  );
}

/* ==================================================
 * Component
 * ==================================================*/
export default function MapView({
  mapMode = "2D",
  baseLayer = "MOLA",
  viewBounds,
  onSelect,
  showCRISM,
  showHiRISE,
  showSHARAD,
  showSharadHighres,
  showCTX,
  showHiRISEDTM = false,
  showCRISM_TRR3 = false,
  onSharadClick,
  onSharadHiresClick,
  onHiRiseDTMClick,
  onToggleOverlay,
  quickviewOverlays = [],
  highResOverlays = [],
  mineralOverlays = [],
  browseOverlays = new Map(),
  scoreOverlays = new Map(),
  overlayOpacities = new Map(),
  onVisibleProductsChange,
  flyToProductId,
  onFlyToComplete,
  flyToCoords,
  onFlyToCoordsComplete,
  bringToFrontId,
  onBringToFrontComplete,
  rgbWavelengths = { r: 2.53, g: 1.51, b: 1.08 },
  hoveredProductId = null,
  onHoverProduct,
  crismFilteredIds = null,
  loadFootprintsTrigger = null,
  onFootprintsLoaded,
  onFootprintsLoading,
  onTerrainClick,
  showCustomData = false,
  customDatasets = [],
  analysisMode = null,
  activeDTMProductId = null,
  linePoints = [],
  viewBoundSelectionMode = false,
  onViewBoundSelected,
  fieldNotes = [],
  onFieldNoteClick,
  showGrid = false,
  showRegionLayer = false,
  aiAnalysisPin = null,
  overlapFilter,
  onOverlapStatsChange,
  highlightProductId,
  onHighlightComplete,
  inspectedProductId,
  sharadTracePin = null,
  showMeasurementTools = false,
  onMeasurementPinNote,
  craterDetectFeatures,
  cameraViewportRef,
  terraformMode = false,
  swimLayer = false,
  onOlympusMonsTripleClick,
  onOlympusMonsClimber,
}: MapViewProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  // SHARAD entities are now managed by FootprintManager (explicit loading)

  const {
    dtmHoverReadoutRef,
    dtmGridCacheRef,
    activeDTMProductRef,
    dtmHoverMode,
    setDtmGrid,
    handleDTMHoverModeChange,
    initializeDTMHover,
  } = useDTMHover({
    activeDTMProductId,
    marsEllipsoid: MARS_ELLIPSOID,
  });

  const { viewerRef, hover } = useMapViewer({
    containerRef: ref,
    mapMode,
    baseLayer,
    viewBounds,
    marsEllipsoid: MARS_ELLIPSOID,
    marsRect: MARS_RECT,
    baseLayerUrls: BASE_LAYER_URLS,
    quickviewOverlays,
    highResOverlays,
    onSelect,
    onSharadClick,
    onSharadHiresClick,
    onHiRiseDTMClick,
    onToggleOverlay,
    onTerrainClick,
    onFieldNoteClick,
    fieldNotes,
    onHoverProduct,
    activeDTMProductRef,
    setDtmGrid,
    initializeDTMHover,
    onOlympusMonsTripleClick,
    onOlympusMonsClimber,
    cameraViewportRef,
    parseLBLValue,
    normalizeLonTo180,
    loadHiRISELBL,
    loadCRISMLBL,
    getProductBounds,
    paddedRectangle,
    getEntityInstrument,
    getEntityProductId,
    getHiliteMaterial,
  });

  const { footprintManagerRef } = useFootprints({
    viewerRef,
    marsEllipsoid: MARS_ELLIPSOID,
    showCRISM,
    showHiRISE,
    showSHARAD,
    showSharadHighres,
    showCTX,
    showHiRISEDTM,
    showCRISM_TRR3,
    crismFilteredIds,
    loadFootprintsTrigger,
    onFootprintsLoaded,
    onFootprintsLoading,
    overlapFilter,
    onOverlapStatsChange,
    inspectedProductId,
    onVisibleProductsChange,
    showCustomData,
    customDatasets,
    showRegionLayer,
    extractCrismObsId,
  });

  useOverlays({
    viewerRef,
    footprintManagerRef,
    quickviewOverlays,
    highResOverlays,
    mineralOverlays,
    browseOverlays,
    scoreOverlays,
    overlayOpacities,
    rgbWavelengths,
    showHiRISEDTM,
    showHiRISE,
    showCRISM,
    activeDTMProductRef,
    dtmGridCacheRef,
    dtmHoverReadoutRef,
    setDtmGrid,
    marsEllipsoid: MARS_ELLIPSOID,
    getProductBounds,
    getFootprintBounds,
    loadCRISMLBL,
    parseLBLValue,
    normalizeLonTo180,
  });

  // View bound selection refs
  const onViewBoundSelectedRef = useRef(onViewBoundSelected);
  const swimLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  useEffect(() => {
    onViewBoundSelectedRef.current = onViewBoundSelected;
  }, [onViewBoundSelected]);

  // View Bound Selection Mode - drag to draw rectangle
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !viewBoundSelectionMode) return;

    // Store drag state
    let isDragging = false;
    let startCartographic: Cesium.Cartographic | null = null;
    const selectionRectId = "__VIEW_BOUND_SELECTION_RECT__";

    // Disable default camera controls during selection
    const scene = viewer.scene;
    scene.screenSpaceCameraController.enableRotate = false;
    scene.screenSpaceCameraController.enableTranslate = false;
    scene.screenSpaceCameraController.enableZoom = false;
    scene.screenSpaceCameraController.enableTilt = false;
    scene.screenSpaceCameraController.enableLook = false;

    // Change cursor to crosshair
    viewer.canvas.style.cursor = "crosshair";

    const handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);

    // Mouse down - start drag
    handler.setInputAction(
      (click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const cartesian = viewer.camera.pickEllipsoid(click.position, MARS_ELLIPSOID);
        if (!cartesian) return;

        startCartographic = Cesium.Cartographic.fromCartesian(cartesian, MARS_ELLIPSOID);
        isDragging = true;

        // Remove existing selection rect if any
        const existing = viewer.entities.getById(selectionRectId);
        if (existing) viewer.entities.remove(existing);
      },
      Cesium.ScreenSpaceEventType.LEFT_DOWN
    );

    // Mouse move - update rectangle
    handler.setInputAction(
      (movement: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
        if (!isDragging || !startCartographic) return;

        const cartesian = viewer.camera.pickEllipsoid(movement.endPosition, MARS_ELLIPSOID);
        if (!cartesian) return;

        const endCartographic = Cesium.Cartographic.fromCartesian(cartesian, MARS_ELLIPSOID);

        // Compute rectangle bounds
        const west = Math.min(startCartographic.longitude, endCartographic.longitude);
        const east = Math.max(startCartographic.longitude, endCartographic.longitude);
        const south = Math.min(startCartographic.latitude, endCartographic.latitude);
        const north = Math.max(startCartographic.latitude, endCartographic.latitude);

        // Remove existing and add new rectangle
        const existing = viewer.entities.getById(selectionRectId);
        if (existing) viewer.entities.remove(existing);

        viewer.entities.add({
          id: selectionRectId,
          rectangle: {
            coordinates: new Cesium.Rectangle(west, south, east, north),
            material: Cesium.Color.YELLOW.withAlpha(0.3),
            outline: true,
            outlineColor: Cesium.Color.YELLOW,
            outlineWidth: 2,
            height: 0,
          },
        });

        scene.requestRender();
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE
    );

    // Mouse up - finalize selection
    handler.setInputAction(
      (click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        if (!isDragging || !startCartographic) {
          isDragging = false;
          return;
        }

        const cartesian = viewer.camera.pickEllipsoid(click.position, MARS_ELLIPSOID);
        if (!cartesian) {
          isDragging = false;
          return;
        }

        const endCartographic = Cesium.Cartographic.fromCartesian(cartesian, MARS_ELLIPSOID);

        // Compute final bounds
        const westRad = Math.min(startCartographic.longitude, endCartographic.longitude);
        const eastRad = Math.max(startCartographic.longitude, endCartographic.longitude);
        const southRad = Math.min(startCartographic.latitude, endCartographic.latitude);
        const northRad = Math.max(startCartographic.latitude, endCartographic.latitude);

        // Convert to degrees
        const westLon = Cesium.Math.toDegrees(westRad);
        const eastLon = Cesium.Math.toDegrees(eastRad);
        const minLat = Cesium.Math.toDegrees(southRad);
        const maxLat = Cesium.Math.toDegrees(northRad);

        // Remove selection rectangle
        const existing = viewer.entities.getById(selectionRectId);
        if (existing) viewer.entities.remove(existing);

        // Only call callback if the selection is meaningful (not just a click)
        const lonSpan = eastLon - westLon;
        const latSpan = maxLat - minLat;
        if (lonSpan > 0.1 && latSpan > 0.1) {
          onViewBoundSelectedRef.current?.({ minLat, maxLat, westLon, eastLon });
        }

        isDragging = false;
        startCartographic = null;
        scene.requestRender();
      },
      Cesium.ScreenSpaceEventType.LEFT_UP
    );

    // Cleanup
    return () => {
      handler.destroy();

      // Re-enable camera controls
      scene.screenSpaceCameraController.enableRotate = true;
      scene.screenSpaceCameraController.enableTranslate = true;
      scene.screenSpaceCameraController.enableZoom = true;
      scene.screenSpaceCameraController.enableTilt = true;
      scene.screenSpaceCameraController.enableLook = true;

      // Reset cursor
      viewer.canvas.style.cursor = "default";

      // Remove selection rectangle if still exists
      const existing = viewer.entities.getById(selectionRectId);
      if (existing) viewer.entities.remove(existing);

      scene.requestRender();
    };
  }, [viewBoundSelectionMode]);

  // Line Profile markers and polyline
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Entity IDs for line profile
    const LP_MARKER_A = "LINE_PROFILE_MARKER_A";
    const LP_MARKER_B = "LINE_PROFILE_MARKER_B";
    const LP_LABEL_A = "LINE_PROFILE_LABEL_A";
    const LP_LABEL_B = "LINE_PROFILE_LABEL_B";
    const LP_LINE = "LINE_PROFILE_LINE";

    // Clear all line profile entities
    const clearAll = () => {
      for (const id of [LP_MARKER_A, LP_MARKER_B, LP_LABEL_A, LP_LABEL_B, LP_LINE]) {
        const ent = viewer.entities.getById(id);
        if (ent) viewer.entities.remove(ent);
      }
    };

    clearAll();

    if (analysisMode !== "line" || linePoints.length === 0) {
      viewer.scene.requestRender();
      return;
    }

    const fmtLabel = (lat: number, lon: number) =>
      `${Math.abs(lat).toFixed(4)}\u00b0${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}\u00b0${lon >= 0 ? "E" : "W"}`;

    // First point marker
    const p1 = linePoints[0]!;
    viewer.entities.add({
      id: LP_MARKER_A,
      position: Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, 0, MARS_ELLIPSOID),
      point: {
        pixelSize: 8,
        color: Cesium.Color.LIME,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    viewer.entities.add({
      id: LP_LABEL_A,
      position: Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, 0, MARS_ELLIPSOID),
      label: {
        text: fmtLabel(p1.lat, p1.lon),
        font: "11px monospace",
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -12),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // Second point + polyline
    if (linePoints.length >= 2) {
      const p2 = linePoints[1]!;
      viewer.entities.add({
        id: LP_MARKER_B,
        position: Cesium.Cartesian3.fromDegrees(p2.lon, p2.lat, 0, MARS_ELLIPSOID),
        point: {
          pixelSize: 8,
          color: Cesium.Color.RED,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      viewer.entities.add({
        id: LP_LABEL_B,
        position: Cesium.Cartesian3.fromDegrees(p2.lon, p2.lat, 0, MARS_ELLIPSOID),
        label: {
          text: fmtLabel(p2.lat, p2.lon),
          font: "11px monospace",
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -12),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      viewer.entities.add({
        id: LP_LINE,
        polyline: {
          positions: [
            Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, 0, MARS_ELLIPSOID),
            Cesium.Cartesian3.fromDegrees(p2.lon, p2.lat, 0, MARS_ELLIPSOID),
          ],
          width: 2,
          material: Cesium.Color.LIME.withAlpha(0.8),
          clampToGround: true,
        },
      });
    }

    viewer.scene.requestRender();
  }, [analysisMode, linePoints]);

  // Fly to product when flyToProductId changes
  useEffect(() => {
    if (!flyToProductId) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    async function flyTo() {
      const v = viewerRef.current;
      if (!v) return;

      const pid = flyToProductId!;

      // Try to find entity from FootprintManager (works for all instruments)
      // Entity ID format: INSTRUMENT_FP_productId
      const instruments = ["HIRISE", "CRISM", "CTX", "SHARAD", "SHARAD_HIGHRES", "HIRISE_DTM", "CRISM_TRR3", "CUSTOM"];
      let foundEntity: Cesium.Entity | null = null;

      for (const inst of instruments) {
        const entity = v.entities.getById(`${inst}_FP_${pid}`);
        if (entity) {
          foundEntity = entity;
          break;
        }
      }

      // If found entity with rectangle, fly to its bounds
      if (foundEntity?.rectangle?.coordinates) {
        const rectCoords = foundEntity.rectangle.coordinates.getValue(Cesium.JulianDate.now());
        if (rectCoords) {
          const padded = paddedRectangle(rectCoords, 0.3);
          v.camera.flyTo({
            destination: padded,
            duration: 0.8,
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // If found entity with position (point), fly above it
      if (foundEntity?.position) {
        const pos = foundEntity.position.getValue(Cesium.JulianDate.now());
        if (pos) {
          const carto = Cesium.Cartographic.fromCartesian(pos, MARS_ELLIPSOID);
          v.camera.flyTo({
            destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, 50000, MARS_ELLIPSOID),
            orientation: {
              heading: 0,
              pitch: Cesium.Math.toRadians(-90),
              roll: 0,
            },
            duration: 0.8,
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // If found entity with polyline, fly above its center
      if (foundEntity?.polyline?.positions) {
        const positions = foundEntity.polyline.positions.getValue(Cesium.JulianDate.now());
        if (positions && positions.length > 0) {
          const midIdx = Math.floor(positions.length / 2);
          const carto = Cesium.Cartographic.fromCartesian(positions[midIdx], MARS_ELLIPSOID);
          v.camera.flyTo({
            destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, 80000, MARS_ELLIPSOID),
            orientation: {
              heading: 0,
              pitch: Cesium.Math.toRadians(-90),
              roll: 0,
            },
            duration: 0.8,
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // Fallback: try LBL-based fly-to for HiRISE/CRISM
      const isHiRISE = pid.startsWith("ESP_") || pid.startsWith("PSP_");
      const isCRISM = pid.toLowerCase().match(/^(frt|hrl|hrs|frs)/);

      if (isHiRISE || isCRISM) {
        const lbl = isHiRISE
          ? await loadHiRISELBL(pid)
          : await loadCRISMLBL(pid);

        if (!lbl) {
          onFlyToComplete?.();
          return;
        }

        const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
        const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
        const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
        const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

        if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
          onFlyToComplete?.();
          return;
        }

        const west = normalizeLonTo180(westLon360);
        const east = normalizeLonTo180(eastLon360);
        const south = Math.min(minLat, maxLat);
        const north = Math.max(minLat, maxLat);

        const rect = Cesium.Rectangle.fromDegrees(west, south, east, north);
        const padded = paddedRectangle(rect, 0.3);

        const v = viewerRef.current;
        if (!v) return;

        v.camera.flyTo({
          destination: padded,
          duration: 0.8,
          complete: () => onFlyToComplete?.(),
        });
        return;
      }

      // Fallback: fetch coordinates from backend footprints API for any instrument
      const fallbackInstruments = ["SHARAD_HIGHRES", "SHARAD", "HIRISE_DTM", "CTX", "HIRISE", "CRISM"];
      for (const inst of fallbackInstruments) {
        try {
          const res = await fetch(`/api/footprints?instrument=${inst}&bbox=-180,-90,180,90&limit=5000&lod=poly`);
          if (!res.ok) continue;
          const data = await res.json();
          const feat = data.features?.find((f: any) => f.properties?.product_id === pid);
          if (!feat?.geometry?.coordinates) continue;

          const coords = feat.geometry.coordinates;
          if (feat.geometry.type === "LineString" && coords.length >= 2) {
            // Fly to midpoint of the line
            const midIdx = Math.floor(coords.length / 2);
            const [_lon, _lat] = coords[midIdx];
            const rect = Cesium.Rectangle.fromDegrees(
              Math.min(...coords.map((c: number[]) => c[0])),
              Math.min(...coords.map((c: number[]) => c[1])),
              Math.max(...coords.map((c: number[]) => c[0])),
              Math.max(...coords.map((c: number[]) => c[1]))
            );
            v.camera.flyTo({
              destination: paddedRectangle(rect, 0.3),
              duration: 0.8,
              complete: () => onFlyToComplete?.(),
            });
            return;
          } else if (feat.geometry.type === "Polygon" && coords[0]?.length >= 4) {
            const ring = coords[0];
            const rect = Cesium.Rectangle.fromDegrees(
              Math.min(...ring.map((c: number[]) => c[0])),
              Math.min(...ring.map((c: number[]) => c[1])),
              Math.max(...ring.map((c: number[]) => c[0])),
              Math.max(...ring.map((c: number[]) => c[1]))
            );
            v.camera.flyTo({
              destination: paddedRectangle(rect, 0.3),
              duration: 0.8,
              complete: () => onFlyToComplete?.(),
            });
            return;
          }
        } catch {
          // Try next instrument
        }
      }

      onFlyToComplete?.();
    }

    flyTo();
  }, [flyToProductId, onFlyToComplete]);

  // Fly to lat/lon coordinates (for search results not on map)
  useEffect(() => {
    if (!flyToCoords) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    const { lat, lon } = flyToCoords;
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, 50000, MARS_ELLIPSOID),
      duration: 1.0,
    });

    onFlyToCoordsComplete?.();
  }, [flyToCoords, onFlyToCoordsComplete]);

  // Temporarily highlight a product after fly-to (deep-link from DataDownloadPage)
  useEffect(() => {
    if (!highlightProductId) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    const pid = highlightProductId;
    const instruments = ["HIRISE", "CRISM", "CTX", "SHARAD", "SHARAD_HIGHRES", "HIRISE_DTM", "CRISM_TRR3", "CUSTOM"];
    let entity: Cesium.Entity | null = null;
    let foundInst = "";

    for (const inst of instruments) {
      const e = viewer.entities.getById(`${inst}_FP_${pid}`);
      if (e) { entity = e; foundInst = inst; break; }
    }

    if (!entity) {
      onHighlightComplete?.();
      return;
    }

    // Auto-select the product so the Inspector opens (deep-link UX)
    if (foundInst) {
      let selectLat = 0;
      let selectLon = 0;
      if (entity.rectangle?.coordinates) {
        const rect = entity.rectangle.coordinates.getValue(Cesium.JulianDate.now());
        if (rect) {
          selectLat = Cesium.Math.toDegrees((rect.south + rect.north) / 2);
          selectLon = Cesium.Math.toDegrees((rect.west + rect.east) / 2);
        }
      } else if (entity.position) {
        const pos = entity.position.getValue(Cesium.JulianDate.now());
        if (pos) {
          const carto = Cesium.Cartographic.fromCartesian(pos, MARS_ELLIPSOID);
          selectLat = Cesium.Math.toDegrees(carto.latitude);
          selectLon = Cesium.Math.toDegrees(carto.longitude);
        }
      }
      const title = entity.properties?.title?.getValue?.() as string | undefined;
      onSelect({
        instrument: foundInst as InspectorContext["instrument"],
        productId: pid,
        lat: selectLat,
        lon: selectLon,
        title,
      });

      // Auto-activate quickview overlay for deep-link products
      onToggleOverlay?.(pid, "quickview");
    }

    // Save original material
    const origMaterial = entity.rectangle?.material;
    const origOutline = entity.rectangle?.outlineColor;
    const origOutlineWidth = entity.rectangle?.outlineWidth;

    // Apply bright highlight
    if (entity.rectangle) {
      entity.rectangle.material = new Cesium.ColorMaterialProperty(
        Cesium.Color.MAGENTA.withAlpha(0.7)
      );
      entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE);
      entity.rectangle.outlineWidth = new Cesium.ConstantProperty(3);
    }
    // Also handle polyline entities (SHARAD)
    if (entity.polyline) {
      const origPolyMaterial = entity.polyline.material;
      const origPolyWidth = entity.polyline.width;
      entity.polyline.material = new Cesium.ColorMaterialProperty(Cesium.Color.MAGENTA);
      entity.polyline.width = new Cesium.ConstantProperty(5);

      viewer.scene.requestRender();

      const timer = setTimeout(() => {
        if (entity?.polyline) {
          entity.polyline.material = origPolyMaterial;
          entity.polyline.width = origPolyWidth;
        }
        viewer.scene.requestRender();
        onHighlightComplete?.();
      }, 3000);
      return () => clearTimeout(timer);
    }

    viewer.scene.requestRender();

    // Restore after 3 seconds
    const timer = setTimeout(() => {
      if (entity?.rectangle) {
        if (origMaterial) entity.rectangle.material = origMaterial;
        if (origOutline) entity.rectangle.outlineColor = origOutline;
        if (origOutlineWidth) entity.rectangle.outlineWidth = origOutlineWidth;
      }
      viewer.scene.requestRender();
      onHighlightComplete?.();
    }, 3000);

    return () => clearTimeout(timer);
  }, [highlightProductId, onHighlightComplete, onSelect, onToggleOverlay]);

  // Bring high-res overlay to front when bringToFrontId changes
  useEffect(() => {
    if (!bringToFrontId) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    const entityId = `HIGHRES_OVERLAY_${bringToFrontId}`;
    const entity = viewer.entities.getById(entityId);

    if (entity) {
      // Remove and re-add to bring to front
      const savedProps = {
        id: entity.id,
        rectangle: entity.rectangle,
        properties: entity.properties,
      };

      viewer.entities.remove(entity);
      viewer.entities.add({
        id: savedProps.id,
        rectangle: savedProps.rectangle,
        properties: savedProps.properties,
      });

      viewer.scene.requestRender();
    }

    onBringToFrontComplete?.();
  }, [bringToFrontId, onBringToFrontComplete]);

  // Landform detection entities
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove old detection entities
    const toRemove = viewer.entities.values.filter(
      (e: Cesium.Entity) => e.id?.startsWith("DETECT_")
    );
    for (const e of toRemove) viewer.entities.remove(e);

    if (!craterDetectFeatures || craterDetectFeatures.length === 0) {
      viewer.scene.requestRender();
      return;
    }

    const COLORS: Record<string, string> = {
      crater: "#fb923c",
      terraced_crater: "#f43f5e",
      volcanic: "#ef4444",
      graben: "#a855f7",
      channel: "#3b82f6",
      wrinkle_ridge: "#eab308",
      lda: "#22d3ee",
    };

    viewer.entities.suspendEvents();

    for (const f of craterDetectFeatures) {
      const color = COLORS[f.type] || "#6b7c9c";
      const cesiumColor = Cesium.Color.fromCssColorString(color);

      if (f.type === "lda" && f.boundary && f.boundary.length > 2) {
        // LDA: polygon — convert each point using Mars ellipsoid
        const positions = f.boundary.map(([lat, lon]: [number, number]) =>
          Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID)
        );
        viewer.entities.add({
          id: `DETECT_${f.id}`,
          polygon: {
            hierarchy: positions,
            material: cesiumColor.withAlpha(0.25),
            outline: true,
            outlineColor: cesiumColor.withAlpha(0.8),
          },
        });
        viewer.entities.add({
          id: `DETECT_L_${f.id}`,
          position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat, 0, MARS_ELLIPSOID),
          label: {
            text: `LDA ${f.area_km2?.toFixed(0) || ""} km\u00b2`,
            font: "bold 11px sans-serif",
            fillColor: cesiumColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 2e6, 0.3),
          },
        });
      } else if ((f.type === "channel" || f.type === "wrinkle_ridge" || f.type === "graben") && f.path && f.path.length > 1) {
        // Polyline features — convert each point using Mars ellipsoid
        const positions = f.path.map(([lat, lon]: [number, number]) =>
          Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID)
        );
        viewer.entities.add({
          id: `DETECT_${f.id}`,
          polyline: {
            positions,
            width: f.type === "graben" ? 3 : 2,
            material: cesiumColor.withAlpha(0.8),
            clampToGround: true,
          },
        });
        const midIdx = Math.floor(f.path.length / 2);
        const midPt = f.path[midIdx];
        if (!midPt) continue;
        const sizeLabel = f.length_km ? `${f.length_km.toFixed(0)} km` : "";
        viewer.entities.add({
          id: `DETECT_L_${f.id}`,
          position: Cesium.Cartesian3.fromDegrees(midPt[1], midPt[0], 0, MARS_ELLIPSOID),
          label: {
            text: `${f.type === "channel" ? "Ch" : f.type === "graben" ? "Gr" : "WR"} ${sizeLabel}`,
            font: "10px sans-serif",
            fillColor: cesiumColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 2e6, 0.3),
          },
        });
      } else {
        // Craters, volcanics — ellipse + visible point marker
        const radiusM = (f.diameter_km || 5) * 500;
        viewer.entities.add({
          id: `DETECT_${f.id}`,
          position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat, 0, MARS_ELLIPSOID),
          ellipse: {
            semiMajorAxis: radiusM,
            semiMinorAxis: radiusM,
            material: cesiumColor.withAlpha(0.2),
            outline: true,
            outlineColor: cesiumColor.withAlpha(0.8),
          },
          point: {
            pixelSize: 7,
            color: cesiumColor,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 1,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label: {
            text: `${f.morphology || f.type}\n${f.diameter_km?.toFixed(1) || ""} km`,
            font: "10px sans-serif",
            fillColor: cesiumColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            pixelOffset: new Cesium.Cartesian2(0, -12),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 2e6, 0.3),
          },
        });
      }
    }

    viewer.entities.resumeEvents();
    viewer.scene.requestRender();

    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      const ents = viewer.entities.values.filter(
        (e: Cesium.Entity) => e.id?.startsWith("DETECT_")
      );
      for (const e of ents) viewer.entities.remove(e);
    };
  }, [craterDetectFeatures]);

  // Easter egg: Terraform mode — tint globe blue/green
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const TERRAFORM_ID = "__TERRAFORM_TINT__";

    if (terraformMode) {
      viewer.entities.add({
        id: TERRAFORM_ID,
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
          material: new Cesium.ColorMaterialProperty(
            new Cesium.Color(0.1, 0.5, 0.8, 0.3)
          ),
          height: 0,
        },
      });
      viewer.scene.requestRender();
    } else {
      const ent = viewer.entities.getById(TERRAFORM_ID);
      if (ent) {
        viewer.entities.remove(ent);
        viewer.scene.requestRender();
      }
    }

    return () => {
      const ent = viewer.entities.getById(TERRAFORM_ID);
      if (ent) viewer.entities.remove(ent);
    };
  }, [terraformMode]);

  // SWIM real data imagery overlay
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove previous SWIM imagery layer
    if (swimLayerRef.current) {
      viewer.imageryLayers.remove(swimLayerRef.current, false);
      swimLayerRef.current = null;
    }

    // Also clean up old entity-based SWIM (migration cleanup)
    const oldEntities = viewer.entities.values.filter(
      (e: Cesium.Entity) => e.id?.startsWith("SWIM_")
    );
    for (const e of oldEntities) viewer.entities.remove(e);

    if (!swimLayer) {
      viewer.scene.requestRender();
      return;
    }

    // Map layer ID to backend tile URL
    const layerMap: Record<string, string> = {
      "0-1m": "/api/swim/tile/0-1m",
      "1-5m": "/api/swim/tile/1-5m",
      ">5m": "/api/swim/tile/%3E5m",
    };

    const tileUrl = layerMap[swimLayer];
    if (!tileUrl) return;

    // Use async fromUrl (constructor deprecated in Cesium 1.104+)
    let cancelled = false;
    Cesium.SingleTileImageryProvider.fromUrl(tileUrl, {
      rectangle: Cesium.Rectangle.fromDegrees(-180, -60, 180, 60),
    }).then((provider) => {
      if (cancelled || !viewer || viewer.isDestroyed()) return;
      const layer = viewer.imageryLayers.addImageryProvider(provider);
      layer.alpha = 0.75;
      swimLayerRef.current = layer;
      viewer.scene.requestRender();
    }).catch((err) => {
      console.warn("Failed to load SWIM overlay:", err);
    });

    return () => {
      cancelled = true;
      if (!viewer || viewer.isDestroyed()) return;
      if (swimLayerRef.current) {
        viewer.imageryLayers.remove(swimLayerRef.current, false);
        swimLayerRef.current = null;
      }
    };
  }, [swimLayer]);

  // Store onHoverProduct in ref to access in hover handler
  const onHoverProductRef = useRef(onHoverProduct);
  useEffect(() => {
    onHoverProductRef.current = onHoverProduct;
  }, [onHoverProduct]);

  // Bidirectional highlight: highlight footprint when hovering in ActiveProductsPanel
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Detect instrument by trying _FP_ entity lookup across all instruments
    const detectInstrument = (pid: string): InstrumentType => {
      const candidates: InstrumentType[] = ["HIRISE", "CRISM", "CTX", "HIRISE_DTM", "CRISM_TRR3", "SHARAD", "SHARAD_HIGHRES"];
      for (const inst of candidates) {
        if (viewer.entities.getById(`${inst}_FP_${pid}`)) return inst;
      }
      // Fallback heuristics
      if (pid.startsWith("ESP_")) return "HIRISE";
      if (pid.startsWith("DTE")) return "HIRISE_DTM";
      if (/^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(pid)) return "CRISM_TRR3";
      return "CRISM";
    };

    // Helper to apply/remove highlight to an entity
    const setEntityHighlight = (entity: Cesium.Entity | undefined, highlighted: boolean, instrument: InstrumentType) => {
      if (!entity?.rectangle) return;

      if (highlighted) {
        entity.rectangle.material = getHiliteMaterial(instrument);
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE);
      } else {
        // Restore to original light fill using instrument color
        const rgb = getInstrumentCesiumColor(instrument.toLowerCase());
        const baseColor = new Cesium.Color(rgb.r, rgb.g, rgb.b, 1.0);
        entity.rectangle.material = new Cesium.ColorMaterialProperty(baseColor.withAlpha(0.10));
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(baseColor);
      }
    };

    // Clear previous highlight if any
    if (!hoveredProductId) {
      viewer.scene.requestRender();
      return;
    }

    // Find and highlight the hovered product
    const instrument = detectInstrument(hoveredProductId);

    // Try FootprintManager entity IDs first (_FP_), then legacy _VP_ IDs
    const entityIds: string[] = [];
    const fpId = `${instrument}_FP_${hoveredProductId}`;
    const fpEntity = viewer.entities.getById(fpId);
    if (fpEntity) {
      setEntityHighlight(fpEntity, true, instrument);
      entityIds.push(fpId);
    }

    // Fallback to legacy VP IDs if no FP entity found
    if (entityIds.length === 0) {
      const vpPrefix = `${instrument}_VP_${hoveredProductId}`;
      for (const id of [vpPrefix, `${vpPrefix}_1`, `${vpPrefix}_2`, `${vpPrefix}_3`]) {
        const entity = viewer.entities.getById(id);
        if (entity) {
          setEntityHighlight(entity, true, instrument);
          entityIds.push(id);
        }
      }
    }

    // Also highlight label if it exists (try _LBL_ first, then legacy)
    const labelEnt = viewer.entities.getById(`${instrument}_LBL_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_VP_LABEL_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_LABEL_${hoveredProductId}`);
    if (labelEnt) {
      labelEnt.show = true;
      if (labelEnt.label) labelEnt.label.scale = new Cesium.ConstantProperty(1.3);
    }

    const pointEnt = viewer.entities.getById(`${instrument}_VP_POINT_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_POINT_${hoveredProductId}`);
    if (pointEnt?.point) {
      pointEnt.point.pixelSize = new Cesium.ConstantProperty(10);
    }

    viewer.scene.requestRender();

    // Cleanup function to restore original appearance
    return () => {
      for (const id of entityIds) {
        const entity = viewer.entities.getById(id);
        if (entity) {
          setEntityHighlight(entity, false, instrument);
        }
      }

      if (labelEnt) {
        labelEnt.show = false;
        if (labelEnt.label) labelEnt.label.scale = new Cesium.ConstantProperty(1.0);
      }

      if (pointEnt?.point) {
        pointEnt.point.pixelSize = new Cesium.ConstantProperty(6);
      }

      viewer.scene.requestRender();
    };
  }, [hoveredProductId]);

  // ====================================================
  // Custom dataset overlay rendering
  // ====================================================
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Track which custom entity IDs should exist
    const desiredIds = new Set<string>();

    for (const dataset of customDatasets) {
      const outlineId = `CUSTOM_FP_${dataset.id}`;
      desiredIds.add(outlineId);

      if (showCustomData && dataset.visible) {
        const rect = Cesium.Rectangle.fromDegrees(
          dataset.bounds.west,
          dataset.bounds.south,
          dataset.bounds.east,
          dataset.bounds.north
        );

        // Footprint rectangle: fully transparent fill (alpha=0) so hover pick works,
        // outline only visible. Highlight fill shown on hover via mouse handler.
        let outlineEnt = viewer.entities.getById(outlineId);
        if (!outlineEnt) {
          viewer.entities.add({
            id: outlineId,
            rectangle: {
              coordinates: rect,
              material: Cesium.Color.FUCHSIA.withAlpha(0.0),
              outline: true,
              outlineColor: Cesium.Color.FUCHSIA,
              outlineWidth: 2,
              height: 0,
            },
            properties: {
              product_id: dataset.id,
              instrument: "CUSTOM",
              kind: "FOOTPRINT",
              dataset_name: dataset.name,
            },
          });
        } else if (outlineEnt.rectangle) {
          // Force transparent fill on existing entity
          outlineEnt.rectangle.material = new Cesium.ColorMaterialProperty(
            Cesium.Color.FUCHSIA.withAlpha(0.0)
          );
        }

        // Add or update label entity
        const labelId = `CUSTOM_LABEL_${dataset.id}`;
        desiredIds.add(labelId);
        let labelEnt = viewer.entities.getById(labelId);
        if (!labelEnt) {
          const centerLon = (dataset.bounds.west + dataset.bounds.east) / 2;
          const centerLat = (dataset.bounds.south + dataset.bounds.north) / 2;
          const carto = Cesium.Cartographic.fromDegrees(centerLon, centerLat, 100);
          const pos = viewer.scene.globe.ellipsoid.cartographicToCartesian(carto);

          viewer.entities.add({
            id: labelId,
            position: pos,
            label: {
              text: dataset.name,
              font: "12px sans-serif",
              fillColor: Cesium.Color.FUCHSIA,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 3,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
            properties: {
              product_id: dataset.id,
              instrument: "CUSTOM",
              kind: "LABEL",
            },
          });
        }
      } else {
        // Not visible - remove if exists
        const outlineEnt = viewer.entities.getById(outlineId);
        if (outlineEnt) viewer.entities.remove(outlineEnt);
        const labelEnt = viewer.entities.getById(`CUSTOM_LABEL_${dataset.id}`);
        if (labelEnt) viewer.entities.remove(labelEnt);
      }
    }

    // Remove entities for datasets that no longer exist
    const toRemove: Cesium.Entity[] = [];
    for (let i = 0; i < viewer.entities.values.length; i++) {
      const ent = viewer.entities.values[i]!;
      if (ent.id.startsWith("CUSTOM_OVERLAY_") || ent.id.startsWith("CUSTOM_FP_") || ent.id.startsWith("CUSTOM_LABEL_")) {
        if (!desiredIds.has(ent.id)) {
          toRemove.push(ent);
        }
      }
    }
    for (const ent of toRemove) {
      viewer.entities.remove(ent);
    }

    viewer.scene.requestRender();
  }, [showCustomData, customDatasets]);

  // ──────────── Field Note indicators (independent of layer state) ────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const PREFIX = "FIELDNOTE_";
    let cancelled = false;

    // Helper to add a field note marker
    const addFieldNoteMarker = (
      v: Cesium.Viewer,
      note: { id: string; product_id: string; instrument: string },
      lat: number,
      lon: number
    ) => {
      // Check if already exists
      if (v.entities.getById(`${PREFIX}${note.id}`)) return;

      v.entities.add({
        id: `${PREFIX}${note.id}`,
        position: Cesium.Cartesian3.fromDegrees(lon, lat, 0),
        billboard: {
          image: createFieldNoteIcon(note.instrument),
          width: 24,
          height: 24,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: "\u2605", // star character
          font: "12px sans-serif",
          fillColor: Cesium.Color.GOLD,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          pixelOffset: new Cesium.Cartesian2(0, -28),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        properties: {
          type: "fieldnote",
          noteId: note.id,
          productId: note.product_id,
          instrument: note.instrument,
        },
      });
    };

    // Remove existing field note markers
    const toRemove: Cesium.Entity[] = [];
    for (let i = 0; i < viewer.entities.values.length; i++) {
      const ent = viewer.entities.values[i]!;
      if (ent.id.startsWith(PREFIX)) toRemove.push(ent);
    }
    for (const ent of toRemove) viewer.entities.remove(ent);

    if (!fieldNotes || fieldNotes.length === 0) {
      viewer.scene.requestRender();
      return;
    }

    // Collect notes that need coordinate lookup
    const notesNeedingCoords: typeof fieldNotes = [];

    // Create markers at each field note's lat/lon (works regardless of layer state)
    for (const note of fieldNotes) {
      const lat = note.lat;
      const lon = note.lon;

      // If coordinates are 0,0, we need to look them up
      if (lat === 0 && lon === 0) {
        notesNeedingCoords.push(note);
        continue;
      }

      addFieldNoteMarker(viewer, note, lat, lon);
    }

    // Fetch coordinates for notes with 0,0 coords (async)
    if (notesNeedingCoords.length > 0) {
      // Group by instrument to minimize API calls
      const byInstrument = new Map<string, typeof fieldNotes>();
      for (const note of notesNeedingCoords) {
        const inst = note.instrument;
        if (!byInstrument.has(inst)) byInstrument.set(inst, []);
        byInstrument.get(inst)!.push(note);
      }

      // Fetch footprints for each instrument
      for (const [instrument, notes] of byInstrument) {
        (async () => {
          try {
            const res = await fetch(
              `/api/footprints?instrument=${instrument}&bbox=-180,-90,180,90&limit=5000&lod=poly`
            );
            if (!res.ok || cancelled) return;

            const data = await res.json();
            if (cancelled) return;

            const v = viewerRef.current;
            if (!v) return;

            for (const note of notes) {
              const feat = data.features?.find(
                (f: any) => f.properties?.product_id === note.product_id
              );
              if (!feat?.geometry?.coordinates) continue;

              let lat = 0, lon = 0;
              const geom = feat.geometry;

              if (geom.type === "LineString" && geom.coordinates.length >= 2) {
                // For LineStrings (SHARAD tracks), use midpoint
                const midIdx = Math.floor(geom.coordinates.length / 2);
                lon = geom.coordinates[midIdx][0];
                lat = geom.coordinates[midIdx][1];
              } else if (geom.type === "Polygon" && geom.coordinates[0]?.length > 0) {
                // For Polygons, compute centroid
                const ring = geom.coordinates[0];
                let sumLat = 0, sumLon = 0;
                for (const [plon, plat] of ring) {
                  sumLon += plon;
                  sumLat += plat;
                }
                lon = sumLon / ring.length;
                lat = sumLat / ring.length;
              } else if (geom.type === "Point") {
                lon = geom.coordinates[0];
                lat = geom.coordinates[1];
              }

              if (lat !== 0 || lon !== 0) {
                addFieldNoteMarker(v, note, lat, lon);
                v.scene.requestRender();
              }
            }
          } catch (err) {
            console.error(`[FieldNotes] Failed to fetch coords for ${instrument}:`, err);
          }
        })();
      }
    }

    viewer.scene.requestRender();

    return () => {
      cancelled = true;
    };
  }, [fieldNotes]);

  // ──────────── AI Analysis Pin + Radius Circle ────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const PIN_ID = "AI_ANALYSIS_PIN";
    const RADIUS_ID = "AI_ANALYSIS_RADIUS";

    // Remove existing entities
    const oldPin = viewer.entities.getById(PIN_ID);
    if (oldPin) viewer.entities.remove(oldPin);
    const oldRadius = viewer.entities.getById(RADIUS_ID);
    if (oldRadius) viewer.entities.remove(oldRadius);

    if (!aiAnalysisPin) {
      viewer.scene.requestRender();
      return;
    }

    const { lat, lon } = aiAnalysisPin;

    // Pin point
    viewer.entities.add({
      id: PIN_ID,
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
      point: {
        pixelSize: 10,
        color: Cesium.Color.fromCssColorString("#8b5cf6"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `AI Analysis\n${lat.toFixed(3)}°, ${lon.toFixed(3)}°`,
        font: "11px sans-serif",
        fillColor: Cesium.Color.fromCssColorString("#8b5cf6"),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -14),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // Radius circle (10 km default; visual only)
    viewer.entities.add({
      id: RADIUS_ID,
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
      ellipse: {
        semiMajorAxis: 10000,  // 10 km
        semiMinorAxis: 10000,
        material: Cesium.Color.fromCssColorString("#8b5cf6").withAlpha(0.12),
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString("#8b5cf6").withAlpha(0.5),
        outlineWidth: 1,
      },
    });

    viewer.scene.requestRender();
  }, [aiAnalysisPin]);

  // Clean up AI Analysis entities when mode changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (analysisMode !== "ai_analysis") {
      const pin = viewer.entities.getById("AI_ANALYSIS_PIN");
      if (pin) viewer.entities.remove(pin);
      const radius = viewer.entities.getById("AI_ANALYSIS_RADIUS");
      if (radius) viewer.entities.remove(radius);
      viewer.scene.requestRender();
    }
  }, [analysisMode]);

  // ──────────── SHARAD Trace Pin ────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const TRACE_PIN_ID = "SHARAD_TRACE_PIN";
    const old = viewer.entities.getById(TRACE_PIN_ID);
    if (old) viewer.entities.remove(old);

    if (!sharadTracePin) {
      viewer.scene.requestRender();
      return;
    }

    const { lat, lon } = sharadTracePin;
    viewer.entities.add({
      id: TRACE_PIN_ID,
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
      point: {
        pixelSize: 10,
        color: Cesium.Color.fromCssColorString("#f59e0b"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `${lat.toFixed(3)}°, ${lon.toFixed(3)}°`,
        font: "bold 11px monospace",
        fillColor: Cesium.Color.fromCssColorString("#f59e0b"),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -12),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    viewer.scene.requestRender();
  }, [sharadTracePin]);

  // ──────────── Coordinate Grid Overlay ────────────
  const gridSpacingRef = useRef<number | null>(null);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const GRID_PREFIX = "GRID_";

    function removeAllGrid() {
      const toRemove: Cesium.Entity[] = [];
      for (let i = 0; i < viewer!.entities.values.length; i++) {
        const ent = viewer!.entities.values[i]!;
        if (ent.id.startsWith(GRID_PREFIX)) toRemove.push(ent);
      }
      for (const ent of toRemove) viewer!.entities.remove(ent);
    }

    if (!showGrid) {
      removeAllGrid();
      gridSpacingRef.current = null;
      viewer.scene.requestRender();
      return;
    }

    function getSpacing(): number {
      const height = viewer!.camera.positionCartographic.height / 1000; // km
      if (height > 5000) return 30;
      if (height > 2000) return 10;
      if (height > 500) return 5;
      if (height > 100) return 1;
      return 0.5;
    }

    function getViewportRect(): { west: number; south: number; east: number; north: number } | null {
      const canvas = viewer!.scene.canvas;
      const cam = viewer!.camera;
      const corners = [
        cam.pickEllipsoid(new Cesium.Cartesian2(0, 0), MARS_ELLIPSOID),
        cam.pickEllipsoid(new Cesium.Cartesian2(canvas.width, 0), MARS_ELLIPSOID),
        cam.pickEllipsoid(new Cesium.Cartesian2(0, canvas.height), MARS_ELLIPSOID),
        cam.pickEllipsoid(new Cesium.Cartesian2(canvas.width, canvas.height), MARS_ELLIPSOID),
      ];
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      let valid = 0;
      for (const c of corners) {
        if (!c) continue;
        const carto = Cesium.Cartographic.fromCartesian(c, MARS_ELLIPSOID);
        const lat = Cesium.Math.toDegrees(carto.latitude);
        const lon = Cesium.Math.toDegrees(carto.longitude);
        minLat = Math.min(minLat, lat);
        maxLat = Math.max(maxLat, lat);
        minLon = Math.min(minLon, lon);
        maxLon = Math.max(maxLon, lon);
        valid++;
      }
      if (valid < 2) return null;
      return { west: minLon, south: minLat, east: maxLon, north: maxLat };
    }

    function rebuildGrid() {
      const spacing = getSpacing();
      if (spacing === gridSpacingRef.current) return;
      gridSpacingRef.current = spacing;

      removeAllGrid();

      // Determine bounds
      let south = -90, north = 90, west = -180, east = 180;
      const pad = spacing * 2;
      if (spacing <= 1) {
        const vp = getViewportRect();
        if (vp) {
          south = Math.max(-90, Math.floor((vp.south - pad) / spacing) * spacing);
          north = Math.min(90, Math.ceil((vp.north + pad) / spacing) * spacing);
          west = Math.max(-180, Math.floor((vp.west - pad) / spacing) * spacing);
          east = Math.min(180, Math.ceil((vp.east + pad) / spacing) * spacing);
        }
      }

      const lineColor = Cesium.Color.WHITE.withAlpha(0.2);
      const labelColor = Cesium.Color.WHITE.withAlpha(0.45);
      const showLabels = spacing <= 5;
      // Label interval: show a label every N grid lines to avoid clutter
      const labelEvery = spacing <= 0.5 ? 2 : 1;

      // Latitude lines (horizontal)
      for (let lat = south; lat <= north; lat = Math.round((lat + spacing) * 1000) / 1000) {
        const pts: number[] = [];
        const lineWest = Math.max(west, -180);
        const lineEast = Math.min(east, 180);
        for (let lon = lineWest; lon <= lineEast; lon += Math.min(spacing, 5)) {
          pts.push(lon, lat);
        }
        // Ensure we reach the east edge
        if (pts.length >= 2 && pts[pts.length - 2]! < lineEast) {
          pts.push(lineEast, lat);
        }
        if (pts.length < 4) continue;

        const positions = Cesium.Cartesian3.fromDegreesArray(pts, MARS_ELLIPSOID);
        viewer!.entities.add({
          id: `${GRID_PREFIX}LAT_${lat}`,
          polyline: {
            positions,
            material: lineColor,
            width: 1,
          },
        });

        // Label
        if (showLabels && Math.round(lat / spacing) % labelEvery === 0) {
          viewer!.entities.add({
            id: `${GRID_PREFIX}LABEL_LAT_${lat}`,
            position: Cesium.Cartesian3.fromDegrees(
              Math.max(west, -179), lat, 0, MARS_ELLIPSOID
            ),
            label: {
              text: `${lat.toFixed(spacing < 1 ? 1 : 0)}°`,
              font: "11px monospace",
              fillColor: labelColor,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(4, 0),
              horizontalOrigin: Cesium.HorizontalOrigin.LEFT,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
              scale: 1.0,
            },
          });
        }
      }

      // Longitude lines (vertical)
      for (let lon = west; lon <= east; lon = Math.round((lon + spacing) * 1000) / 1000) {
        const pts: number[] = [];
        const lineSouth = Math.max(south, -90);
        const lineNorth = Math.min(north, 90);
        for (let lat = lineSouth; lat <= lineNorth; lat += Math.min(spacing, 5)) {
          pts.push(lon, lat);
        }
        if (pts.length >= 2 && pts[pts.length - 1]! < lineNorth) {
          pts.push(lon, lineNorth);
        }
        if (pts.length < 4) continue;

        const positions = Cesium.Cartesian3.fromDegreesArray(pts, MARS_ELLIPSOID);
        viewer!.entities.add({
          id: `${GRID_PREFIX}LON_${lon}`,
          polyline: {
            positions,
            material: lineColor,
            width: 1,
          },
        });

        // Label
        if (showLabels && Math.round(lon / spacing) % labelEvery === 0) {
          viewer!.entities.add({
            id: `${GRID_PREFIX}LABEL_LON_${lon}`,
            position: Cesium.Cartesian3.fromDegrees(
              lon, Math.max(south, -89), 0, MARS_ELLIPSOID
            ),
            label: {
              text: `${lon.toFixed(spacing < 1 ? 1 : 0)}°`,
              font: "11px monospace",
              fillColor: labelColor,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -4),
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
              scale: 1.0,
            },
          });
        }
      }

      viewer!.scene.requestRender();
    }

    // Build immediately
    rebuildGrid();

    // Rebuild on camera move — throttled to avoid excessive entity churn
    const throttledRebuild = throttle(() => {
      const newSpacing = getSpacing();
      if (newSpacing !== gridSpacingRef.current || newSpacing <= 1) {
        gridSpacingRef.current = null; // force rebuild
        rebuildGrid();
      }
    }, 300);

    const removeListener = viewer.camera.moveEnd.addEventListener(throttledRebuild);

    return () => {
      removeListener();
      removeAllGrid();
      gridSpacingRef.current = null;
      viewer.scene.requestRender();
    };
  }, [showGrid]);

  return (
    <>
      <div ref={ref} className="absolute inset-0" role="application" aria-label="Mars 3D Globe" />

      {/* Footprint loading indicators moved to LayerPanel */}
      <MapToolbar
        hover={hover}
        scoreOverlays={scoreOverlays}
        dtmHoverReadoutRef={dtmHoverReadoutRef}
        dtmHoverMode={dtmHoverMode}
        onDTMHoverModeChange={handleDTMHoverModeChange}
        viewer={viewerRef.current}
        showMeasurementTools={showMeasurementTools}
        onMeasurementPinNote={onMeasurementPinNote}
      />
    </>
  );
}
