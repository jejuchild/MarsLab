// src/MapView.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import toast from "react-hot-toast";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import FootprintManager from "../utils/FootprintManager";
import { normalizeLonForMap } from "../utils/coordinates";
import DTMHoverReadout, { type DTMHoverReadoutHandle } from "./DTMHoverReadout";
import MeasurementTools from "./MeasurementTools";
import {
  loadDTMElevationGrid,
  getElevationFromGrid,
  isWithinDTMBounds,
  throttle,
  type DTMElevationGrid,
} from "../utils/dtmHover";
import type { FieldNote } from "../api/fieldnotes";
import { computeOverlapFilter, type OverlapResult, type OverlapStats } from "../utils/overlapFilter";
import type { InstrumentType as FPInstrumentType } from "../utils/FootprintManager";
import { getInstrumentCesiumColor } from "../config/instrumentRegistry";


/* ==================================================
 * Types
 * ==================================================*/
type LatLon = { lat: number; lon: number };

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
  analysisMode?: "slope" | "slope3d" | "hirise_dtm_3d" | "line" | "ai_analysis" | "agentic" | "report" | null;
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
const _fieldNoteIconCache = new Map<string, string>();
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
  return match ? match[1].toLowerCase() : productId.toLowerCase();
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
const boundsCache = new Map<string, ProductBounds>();

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

// Cache for HiRISE DTM index
let hiriseDTMIndexCache: any = null;
// Cache for CRISM TRR3 index
let crismTRR3IndexCache: any = null;

// Compute bounding box from GeoJSON polygon coordinates
function boundsFromPolygon(coords: number[][]): ProductBounds | null {
  if (!coords || coords.length < 3) return null;
  let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
  for (const [lon, lat] of coords) {
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
      const rect = (ent.rectangle.coordinates as any).getValue?.(Cesium.JulianDate.now()) ??
                   (ent.rectangle.coordinates as any);
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

/* ==================================================
 * Hover highlight state type
 * ==================================================*/
type HighlightState = {
  key: string | null; // ✅ NEW: inst+pid (같은 대상이면 재적용 금지)
  rectEnt: Cesium.Entity | null;
  labelEnt: Cesium.Entity | null;
  pointEnt: Cesium.Entity | null;
  origRectMaterial: any;
  origOutlineColor: Cesium.Color | undefined;
  origLabelScale: number | undefined;
  origPointSize: number | undefined;
};

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
  aiAnalysisPin = null,
  overlapFilter,
  onOverlapStatsChange,
  highlightProductId,
  onHighlightComplete,
  inspectedProductId,
  sharadTracePin = null,
  showMeasurementTools = false,
  onMeasurementPinNote,
}: MapViewProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  // Overlap filter state
  const overlapResultRef = useRef<OverlapResult | null>(null);
  const inspectedProductIdRef = useRef<string | null>(null);
  const [footprintVersion, setFootprintVersion] = useState(0);

  // SHARAD entities are now managed by FootprintManager (explicit loading)

  // Default opacity for overlays
  const DEFAULT_OPACITY = 0.8;

  // Helper to get opacity for a specific product
  const getProductOpacity = (productId: string): number => {
    return overlayOpacities.get(productId) ?? DEFAULT_OPACITY;
  };

  // Helper: make footprint fill transparent when an overlay is active,
  // or restore it when overlay is removed. This prevents the semi-transparent
  // footprint fill from tinting the overlay image.
  const setFootprintTransparent = useCallback((viewer: Cesium.Viewer, productId: string, transparent: boolean) => {
    const isHiRISE = productId.startsWith("ESP_");
    const isHiRISEDTM = productId.startsWith("DTE");
    const isTRR3 = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);
    const instrument = isHiRISEDTM ? "HIRISE_DTM" : isHiRISE ? "HIRISE" : isTRR3 ? "CRISM_TRR3" : "CRISM";
    // Try FootprintManager ID
    const fpEnt = viewer.entities.getById(`${instrument}_FP_${productId}`);
    if (fpEnt?.rectangle) {
      if (transparent) {
        fpEnt.rectangle.material = new Cesium.ColorMaterialProperty(
          Cesium.Color.TRANSPARENT
        );
      } else {
        const color = isHiRISEDTM ? Cesium.Color.fromCssColorString("#d97706")
          : isHiRISE ? Cesium.Color.YELLOW
          : isTRR3 ? Cesium.Color.fromCssColorString("#00CED1")
          : Cesium.Color.CYAN;
        fpEnt.rectangle.material = new Cesium.ColorMaterialProperty(
          color.withAlpha(0.10)
        );
      }
    }
  }, []);

  // Refs to track current overlay lists for click handler
  const quickviewOverlaysRef = useRef<string[]>(quickviewOverlays);
  const highResOverlaysRef = useRef<string[]>(highResOverlays);
  const onSharadClickRef = useRef(onSharadClick);
  const onSharadHiresClickRef = useRef(onSharadHiresClick);
  const onHiRiseDTMClickRef = useRef(onHiRiseDTMClick);
  const onToggleOverlayRef = useRef(onToggleOverlay);
  const onTerrainClickRef = useRef(onTerrainClick);
  const onFootprintsLoadedRef = useRef(onFootprintsLoaded);
  const onFootprintsLoadingRef = useRef(onFootprintsLoading);
  const onFieldNoteClickRef = useRef(onFieldNoteClick);
  const fieldNotesRef = useRef(fieldNotes);

  // DTM Hover System - refs for performance (no re-renders on hover)
  const dtmHoverReadoutRef = useRef<DTMHoverReadoutHandle>(null);
  const dtmHoverMarkerRef = useRef<Cesium.Entity | null>(null);
  const dtmGridCacheRef = useRef<Map<string, DTMElevationGrid>>(new Map());
  const DTM_GRID_CACHE_MAX = 4;
  const setDtmGrid = useCallback((productId: string, grid: DTMElevationGrid) => {
    const cache = dtmGridCacheRef.current;
    cache.set(productId, grid);
    // Evict oldest entries if cache exceeds limit (keep active product)
    if (cache.size > DTM_GRID_CACHE_MAX) {
      const active = activeDTMProductRef.current;
      for (const key of cache.keys()) {
        if (key !== active && key !== productId) {
          cache.delete(key);
          if (cache.size <= DTM_GRID_CACHE_MAX) break;
        }
      }
    }
  }, []);
  const dtmHoverModeRef = useRef<"hover" | "click">("hover");
  const [dtmHoverMode, setDtmHoverMode] = useState<"hover" | "click">("hover");
  const activeDTMProductRef = useRef<string | null>(null);
  const showHiRISEDTMRef = useRef(showHiRISEDTM);

  // Keep refs in sync with props
  useEffect(() => {
    quickviewOverlaysRef.current = quickviewOverlays;
  }, [quickviewOverlays]);

  useEffect(() => {
    highResOverlaysRef.current = highResOverlays;
  }, [highResOverlays]);

  useEffect(() => {
    showHiRISEDTMRef.current = showHiRISEDTM;
  }, [showHiRISEDTM]);

  useEffect(() => {
    onSharadClickRef.current = onSharadClick;
  }, [onSharadClick]);

  useEffect(() => {
    onSharadHiresClickRef.current = onSharadHiresClick;
  }, [onSharadHiresClick]);

  useEffect(() => {
    onHiRiseDTMClickRef.current = onHiRiseDTMClick;
  }, [onHiRiseDTMClick]);

  useEffect(() => {
    onToggleOverlayRef.current = onToggleOverlay;
  }, [onToggleOverlay]);

  useEffect(() => {
    onFieldNoteClickRef.current = onFieldNoteClick;
  }, [onFieldNoteClick]);

  useEffect(() => {
    fieldNotesRef.current = fieldNotes;
  }, [fieldNotes]);

  useEffect(() => {
    onTerrainClickRef.current = onTerrainClick;
  }, [onTerrainClick]);

  useEffect(() => {
    onFootprintsLoadedRef.current = onFootprintsLoaded;
  }, [onFootprintsLoaded]);

  useEffect(() => {
    onFootprintsLoadingRef.current = onFootprintsLoading;
  }, [onFootprintsLoading]);

  // View bound selection refs
  const onViewBoundSelectedRef = useRef(onViewBoundSelected);
  useEffect(() => {
    onViewBoundSelectedRef.current = onViewBoundSelected;
  }, [onViewBoundSelected]);

  const highlightRef = useRef<HighlightState>({
    key: null,
    rectEnt: null,
    labelEnt: null,
    pointEnt: null,
    origRectMaterial: null,
    origOutlineColor: undefined,
    origLabelScale: undefined,
    origPointSize: undefined,
  });

  const [hover, setHover] = useState<LatLon | null>(null);

  // FootprintManager ref for explicit snapshot-based loading
  const footprintManagerRef = useRef<FootprintManager | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    try {
      (Cesium as any).buildModuleUrl?.setBaseUrl?.("/cesium/");
    } catch {}

    const viewer = new Cesium.Viewer(ref.current, {
      sceneMode: Cesium.SceneMode.SCENE2D,
      mapProjection: new Cesium.GeographicProjection(MARS_ELLIPSOID),
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      navigationHelpButton: false,
      sceneModePicker: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      terrainProvider: new Cesium.EllipsoidTerrainProvider({
        ellipsoid: MARS_ELLIPSOID,
      }),
    });
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
    Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK
    );

    viewerRef.current = viewer;

    viewer.scene.globe = new Cesium.Globe(MARS_ELLIPSOID);
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewer.scene.globe.enableLighting = false;
    viewer.scene.backgroundColor = Cesium.Color.BLACK;

    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: BASE_LAYER_URLS[baseLayer],
        rectangle: MARS_RECT,
        tilingScheme: new Cesium.GeographicTilingScheme({
          ellipsoid: MARS_ELLIPSOID,
          numberOfLevelZeroTilesX: 2,
          numberOfLevelZeroTilesY: 1,
        }),
      })
    );

    // Set camera to show full Mars or restricted view bounds
    viewer.camera.flyTo({
      destination: MARS_RECT,
      duration: 0,
      complete: () => {
        viewer.scene.requestRender();
      }
    });

    // Initialize FootprintManager for explicit snapshot-based loading
    // NO automatic camera-based updates - footprints load only on explicit button click
    const footprintManager = new FootprintManager({
      viewer,
      ellipsoid: MARS_ELLIPSOID,
      onLoadStart: (instrument) => {
        onFootprintsLoadingRef.current?.(instrument, true);
      },
      onLoadEnd: (instrument, result) => {
        onFootprintsLoadingRef.current?.(instrument, false);
        onFootprintsLoadedRef.current?.({
          instrument,
          count: result.count,
          truncated: result.truncated,
          total: result.total,
        });
      },
      onError: (instrument, error) => {
        console.error(`[FootprintManager] Error loading ${instrument}:`, error);
        onFootprintsLoadingRef.current?.(instrument, false);
      },
    });
    footprintManagerRef.current = footprintManager;

    // HiRISE footprints are now loaded via FootprintManager (viewport-based)
    // Legacy global loading disabled for performance
    // Footprints are loaded via FootprintManager (viewport-based)

    // SHARAD footprints are now loaded via FootprintManager (explicit loading)

    const hoverHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    // Throttled drillPick + highlight logic (expensive operations at 50ms intervals)
    let lastPickPosition: Cesium.Cartesian2 | null = null;
    const throttledPick = throttle((endPosition: Cesium.Cartesian2, hoverLat: number, hoverLon: number) => {
      // Check if hovering over an overlay and change cursor
      let isOverOverlay = false;
      const allOverlayIds = [...highResOverlaysRef.current, ...quickviewOverlaysRef.current];
      for (const productId of allOverlayIds) {
        const overlayEnt = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`) ||
                           viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
        if (overlayEnt?.rectangle?.coordinates) {
          const rect = overlayEnt.rectangle.coordinates.getValue(Cesium.JulianDate.now()) as Cesium.Rectangle;
          const west = Cesium.Math.toDegrees(rect.west);
          const east = Cesium.Math.toDegrees(rect.east);
          const south = Cesium.Math.toDegrees(rect.south);
          const north = Cesium.Math.toDegrees(rect.north);
          if (hoverLon >= west && hoverLon <= east && hoverLat >= south && hoverLat <= north) {
            isOverOverlay = true;
            break;
          }
        }
      }
      viewer.canvas.style.cursor = isOverOverlay ? "crosshair" : "default";

      const picked = viewer.scene
        .drillPick(endPosition)
        .find((x: any) => x?.id instanceof Cesium.Entity);

      const pickedEnt = picked?.id as Cesium.Entity | undefined;
      const hs = highlightRef.current;

      const clearHighlight = () => {
        if (hs.rectEnt?.rectangle) {
          hs.rectEnt.rectangle.material = hs.origRectMaterial;
          if (hs.origOutlineColor && hs.rectEnt.rectangle.outlineColor) {
            hs.rectEnt.rectangle.outlineColor = new Cesium.ConstantProperty(
              hs.origOutlineColor
            ) as any;
          }
        }
        if (hs.labelEnt?.label && typeof hs.origLabelScale === "number") {
          (hs.labelEnt.label.scale as any) = hs.origLabelScale;
        }
        if (hs.pointEnt?.point && typeof hs.origPointSize === "number") {
          (hs.pointEnt.point.pixelSize as any) = hs.origPointSize;
        }

        hs.key = null;
        hs.rectEnt = null;
        hs.labelEnt = null;
        hs.pointEnt = null;
        hs.origRectMaterial = null;
        hs.origOutlineColor = undefined;
        hs.origLabelScale = undefined;
        hs.origPointSize = undefined;
      };

      if (pickedEnt) {
        const inst = getEntityInstrument(pickedEnt);
        const pid = getEntityProductId(pickedEnt);

        if (inst && pid) {
          const key = `${inst}:${pid}`;
          if (hs.key === key) return;

          let rectFallback: Cesium.Entity | null = null;
          if (inst === "CUSTOM") {
            rectFallback = viewer.entities.getById(`CUSTOM_FP_${pid}`) || null;
          } else {
            rectFallback =
              viewer.entities.getById(`${inst}_FP_${pid}`) ||
              viewer.entities.getById(`${inst}_VP_${pid}`) ||
              viewer.entities.getById(`${inst}_VP_${pid}_1`) ||
              null;
          }

          const rectTarget =
            rectFallback && (rectFallback as any).rectangle ? rectFallback : null;

          const labelEnt = inst === "CUSTOM"
            ? viewer.entities.getById(`CUSTOM_LABEL_${pid}`) || null
            : viewer.entities.getById(`${inst}_LBL_${pid}`) ||
              viewer.entities.getById(`${inst}_VP_LABEL_${pid}`) ||
              viewer.entities.getById(`${inst}_LABEL_${pid}`) || null;
          const pointEnt = inst === "CUSTOM"
            ? null
            : viewer.entities.getById(`${inst}_VP_POINT_${pid}`) ||
              viewer.entities.getById(`${inst}_POINT_${pid}`) || null;

          clearHighlight();

          if (rectTarget?.rectangle) {
            hs.key = key;

            hs.rectEnt = rectTarget;
            hs.origRectMaterial = rectTarget.rectangle.material;

            const ocAny: any = rectTarget.rectangle.outlineColor;
            const ocVal =
              ocAny?.getValue?.(Cesium.JulianDate.now()) ??
              (ocAny instanceof Cesium.Color ? ocAny : undefined);
            if (ocVal instanceof Cesium.Color) {
              hs.origOutlineColor = ocVal;
            }

            rectTarget.rectangle.material = getHiliteMaterial(inst);

            rectTarget.rectangle.outlineColor = new Cesium.ConstantProperty(
              Cesium.Color.WHITE
            ) as any;

            if (labelEnt?.label) {
              hs.labelEnt = labelEnt;
              const cur = (labelEnt.label.scale as any);
              hs.origLabelScale = typeof cur === "number" ? cur : 1.0;
              (labelEnt.label.scale as any) = 1.2;
            }

            if (pointEnt?.point) {
              hs.pointEnt = pointEnt;
              const cur = (pointEnt.point.pixelSize as any);
              hs.origPointSize = typeof cur === "number" ? cur : 6;
              (pointEnt.point.pixelSize as any) = (hs.origPointSize ?? 6) + 2;
            }

            onHoverProductRef.current?.(pid);

            viewer.scene.requestRender();
            return;
          }
        }
      }

      if (hs.rectEnt || hs.labelEnt || hs.pointEnt) {
        clearHighlight();
        onHoverProductRef.current?.(null);
        viewer.scene.requestRender();
      }
    }, 50); // 50ms throttle = max 20 picks/sec (was unlimited)

    hoverHandler.setInputAction(
      (m: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
        const p = viewer.camera.pickEllipsoid(m.endPosition, MARS_ELLIPSOID);
        if (!p) return setHover(null);
        const c = Cesium.Cartographic.fromCartesian(p);
        const hoverLat = Cesium.Math.toDegrees(c.latitude);
        const hoverLon = Cesium.Math.toDegrees(c.longitude);
        setHover({ lat: hoverLat, lon: hoverLon });

        // Throttle the expensive drillPick + highlight operations
        lastPickPosition = m.endPosition.clone();
        throttledPick(lastPickPosition, hoverLat, hoverLon);
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE
    );

    const clickHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    clickHandler.setInputAction(
      async (m: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        // Get click position in lat/lon FIRST
        const clickCart = viewer.camera.pickEllipsoid(m.position, MARS_ELLIPSOID);
        if (!clickCart) return;

        const clickCarto = Cesium.Cartographic.fromCartesian(clickCart);
        const clickLon = Cesium.Math.toDegrees(clickCarto.longitude);
        const clickLat = Cesium.Math.toDegrees(clickCarto.latitude);


        // PRIORITY 1: Check if click is within any active overlay bounds
        // This is more reliable than Cesium picking for image overlays
        let overlayProduct: { productId: string; instrument: InstrumentType } | null = null;

        // Check high-res overlays first (higher priority), then quickview
        const highResIds = highResOverlaysRef.current;
        const quickviewIds = quickviewOverlaysRef.current;
        const allOverlayIds = [...highResIds, ...quickviewIds];

        for (const productId of allOverlayIds) {
          const highResEnt = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
          const quickviewEnt = viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
          const overlayEnt = highResEnt || quickviewEnt;

          if (overlayEnt?.rectangle?.coordinates) {
            const rect = overlayEnt.rectangle.coordinates.getValue(Cesium.JulianDate.now()) as Cesium.Rectangle;
            const west = Cesium.Math.toDegrees(rect.west);
            const east = Cesium.Math.toDegrees(rect.east);
            const south = Cesium.Math.toDegrees(rect.south);
            const north = Cesium.Math.toDegrees(rect.north);

            if (clickLon >= west && clickLon <= east && clickLat >= south && clickLat <= north) {
              const instrument = (overlayEnt.properties as any)?.instrument?.getValue?.() as InstrumentType;
              overlayProduct = { productId, instrument: instrument || (productId.startsWith("ESP_") ? "HIRISE" : "CRISM") };
              break;
            }
          }
        }

        // If we found an overlay, use it
        if (overlayProduct) {
          const { productId, instrument } = overlayProduct;

          // For CRISM, calculate pixel coordinates
          let pixelLine: number | undefined;
          let pixelSample: number | undefined;

          if (instrument === "CRISM") {
            const lbl = await loadCRISMLBL(productId);
            if (lbl) {
              const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
              const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
              const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
              const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");
              const lines = parseLBLValue(lbl, "LINES");
              const samples = parseLBLValue(lbl, "LINE_SAMPLES");

              if (minLat != null && maxLat != null && westLon360 != null && eastLon360 != null && lines && samples) {
                const west = normalizeLonTo180(westLon360);
                const east = normalizeLonTo180(eastLon360);
                const south = Math.min(minLat, maxLat);
                const north = Math.max(minLat, maxLat);

                const latFrac = (north - clickLat) / (north - south);
                const lonFrac = (clickLon - west) / (east - west);

                pixelLine = Math.floor(latFrac * lines);
                pixelSample = Math.floor(lonFrac * samples);

                pixelLine = Math.max(0, Math.min(lines - 1, pixelLine));
                pixelSample = Math.max(0, Math.min(samples - 1, pixelSample));

              }
            }
          }

          // For CRISM TRR3, get shape from backend and compute pixel coords
          if (instrument === "CRISM_TRR3") {
            try {
              const obsId = productId.replace(/_\d{2}$/, "");
              const shapeRes = await fetch(`/api/crism-trr3/${obsId}/shape`);
              if (shapeRes.ok) {
                const shape = await shapeRes.json();
                const bounds = await getProductBounds(productId);
                if (bounds && shape.rows && shape.cols) {
                  const latFrac = (bounds.north - clickLat) / (bounds.north - bounds.south);
                  const lonFrac = (clickLon - bounds.west) / (bounds.east - bounds.west);
                  pixelLine = Math.floor(latFrac * shape.rows);
                  pixelSample = Math.floor(lonFrac * shape.cols);
                  pixelLine = Math.max(0, Math.min(shape.rows - 1, pixelLine));
                  pixelSample = Math.max(0, Math.min(shape.cols - 1, pixelSample));
                }
              }
            } catch (e) {
              console.warn("[TRR3] Failed to get shape for pixel coords:", e);
            }
          }

          // For HiRISE, calculate pixel coordinates
          if (instrument === "HIRISE") {
            const lbl = await loadHiRISELBL(productId);
            if (lbl) {
              const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
              const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
              const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
              const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");
              const lines = parseLBLValue(lbl, "LINES");
              const samples = parseLBLValue(lbl, "LINE_SAMPLES");

              if (minLat != null && maxLat != null && westLon360 != null && eastLon360 != null && lines && samples) {
                const west = normalizeLonTo180(westLon360);
                const east = normalizeLonTo180(eastLon360);
                const south = Math.min(minLat, maxLat);
                const north = Math.max(minLat, maxLat);

                const latFrac = (north - clickLat) / (north - south);
                const lonFrac = (clickLon - west) / (east - west);

                pixelLine = Math.floor(latFrac * lines);
                pixelSample = Math.floor(lonFrac * samples);

                pixelLine = Math.max(0, Math.min(lines - 1, pixelLine));
                pixelSample = Math.max(0, Math.min(samples - 1, pixelSample));

              }
            }
          }


          // Compute footprint center for accurate lat/lon
          let overlayLat = clickLat;
          let overlayLon = clickLon;
          const isTRR3Product = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);
          const fpInst = productId.startsWith("DTE") ? "HIRISE_DTM"
            : productId.startsWith("ESP_") ? "HIRISE"
            : isTRR3Product ? "CRISM_TRR3" : "CRISM";
          const fpEnt = viewer.entities.getById(`${fpInst}_FP_${productId}`);
          if (fpEnt?.rectangle?.coordinates) {
            const fpRect = fpEnt.rectangle.coordinates.getValue(Cesium.JulianDate.now()) as Cesium.Rectangle;
            overlayLat = Cesium.Math.toDegrees((fpRect.south + fpRect.north) / 2);
            overlayLon = Cesium.Math.toDegrees((fpRect.west + fpRect.east) / 2);
          }

          const overlayTitle = fpEnt?.properties?.title?.getValue?.() as string | undefined;
          onSelect({
            instrument,
            productId,
            lat: overlayLat,
            lon: overlayLon,
            pixelLine,
            pixelSample,
            title: overlayTitle,
          });

          return; // Don't process further - overlay click handled
        }

        // PRIORITY 2: No overlay clicked, try Cesium entity picking for footprints
        const pickedList = viewer.scene.drillPick(m.position);

        // PRIORITY 2a: Check for field note markers first
        const pickedFieldNote = pickedList.find((p: any) => {
          if (!(p.id instanceof Cesium.Entity)) return false;
          const type = (p.id as Cesium.Entity).properties?.type?.getValue?.();
          return type === "fieldnote";
        });

        if (pickedFieldNote && pickedFieldNote.id instanceof Cesium.Entity) {
          const fnEnt = pickedFieldNote.id as Cesium.Entity;
          const noteId = fnEnt.properties?.noteId?.getValue?.();
          const fnProductId = fnEnt.properties?.productId?.getValue?.();
          const fnInstrument = fnEnt.properties?.instrument?.getValue?.();


          // Find the full note data from fieldNotesRef
          const note = fieldNotesRef.current.find(n => n.id === noteId);
          if (note && onFieldNoteClickRef.current) {
            onFieldNoteClickRef.current(note);
          }
          return;
        }

        const picked = pickedList.find((p: any) => {
          if (!(p.id instanceof Cesium.Entity)) return false;
          const pid = (p.id as Cesium.Entity).properties?.product_id?.getValue?.();
          return !!pid;
        });

        if (!picked || !(picked.id instanceof Cesium.Entity)) {
          onTerrainClickRef.current?.(clickLat, clickLon);
          return;
        }

        const e = picked.id as Cesium.Entity;
        const p: any = e.properties;

        const productId = p?.product_id?.getValue?.();
        const instrument = p?.instrument?.getValue?.();

        if (!productId || !instrument) return;


        // Handle CUSTOM datasets - fly to bounds
        if (instrument === "CUSTOM") {
          let customLat = clickLat;
          let customLon = clickLon;
          const customRectEnt = viewer.entities.getById(`CUSTOM_FP_${productId}`);
          if (customRectEnt?.rectangle?.coordinates) {
            const cr = customRectEnt.rectangle.coordinates.getValue(Cesium.JulianDate.now()) as Cesium.Rectangle;
            customLat = Cesium.Math.toDegrees((cr.south + cr.north) / 2);
            customLon = Cesium.Math.toDegrees((cr.west + cr.east) / 2);
          }

          onSelect({
            instrument: "CUSTOM",
            productId,
            lat: customLat,
            lon: customLon,
          });

          const rectEnt = customRectEnt || viewer.entities.getById(`CUSTOM_FP_${productId}`);
          if (rectEnt?.rectangle?.coordinates) {
            const rect = rectEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now()
            ) as Cesium.Rectangle;
            viewer.camera.flyTo({ destination: paddedRectangle(rect, 0.3), duration: 0.6 });
          }
          return;
        }

        // Handle SHARAD_HIGHRES - open radargram inspector
        if (instrument === "SHARAD_HIGHRES") {
          onSharadHiresClickRef.current?.(productId);
          return;
        }

        // Handle HIRISE_DTM - open 3D viewer and fly to footprint
        if (instrument === "HIRISE_DTM") {
          // Compute footprint center for accurate coordinates
          let dtmLat = clickLat;
          let dtmLon = clickLon;
          const dtmRectEnt = viewer.entities.getById(`HIRISE_DTM_FP_${productId}`);
          if (dtmRectEnt?.rectangle?.coordinates) {
            const rect = dtmRectEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now()
            ) as Cesium.Rectangle;
            dtmLat = Cesium.Math.toDegrees((rect.south + rect.north) / 2);
            dtmLon = Cesium.Math.toDegrees((rect.west + rect.east) / 2);
            viewer.camera.flyTo({ destination: paddedRectangle(rect, 0.5), duration: 0.6 });
          }

          const dtmTitle = dtmRectEnt?.properties?.title?.getValue?.() as string | undefined;
          onHiRiseDTMClickRef.current?.(productId, dtmLat, dtmLon, dtmTitle);

          // Load elevation grid for hover (async, non-blocking)
          activeDTMProductRef.current = productId;
          loadDTMElevationGrid(productId).then((grid) => {
            if (grid) {
              setDtmGrid(productId, grid);
            }
          });
          return;
        }

        // Handle SHARAD separately - show popup instead of Inspector
        if (instrument === "SHARAD") {
          const startLat = p?.start_lat?.getValue?.() ?? 0;
          const startLon = p?.start_lon?.getValue?.() ?? 0;
          const stopLat = p?.stop_lat?.getValue?.() ?? 0;
          const stopLon = p?.stop_lon?.getValue?.() ?? 0;

          onSharadClickRef.current?.({
            productId,
            quickviewUrl: `/sharad/quickview/${productId.toLowerCase()}.jpg`,
            startLat,
            startLon,
            stopLat,
            stopLon,
          });
          return;
        }

        // Handle CTX - toggle tile overlay directly (no Inspector)
        if (instrument === "CTX") {
          const isActive = quickviewOverlaysRef.current.includes(productId);
          onToggleOverlayRef.current?.(productId, isActive ? null : "quickview");

          // Fly to footprint bounds
          const rectEnt = viewer.entities.getById(`CTX_FP_${productId}`);
          if (rectEnt?.rectangle?.coordinates) {
            const rect = rectEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now()
            ) as Cesium.Rectangle;
            viewer.camera.flyTo({ destination: paddedRectangle(rect, 0.3), duration: 0.6 });
          }
          return;
        }

        // Compute footprint center for accurate lat/lon (click position can be offset)
        const rectEntId = `${instrument}_FP_${productId}`;
        const rectEnt = viewer.entities.getById(rectEntId);

        let selectLat = clickLat;
        let selectLon = clickLon;
        if (rectEnt?.rectangle?.coordinates) {
          const rect = rectEnt.rectangle.coordinates.getValue(Cesium.JulianDate.now()) as Cesium.Rectangle;
          selectLat = Cesium.Math.toDegrees((rect.south + rect.north) / 2);
          selectLon = Cesium.Math.toDegrees((rect.west + rect.east) / 2);
        }

        onSelect({
          instrument,
          productId,
          lat: selectLat,
          lon: selectLon,
        });

        if (rectEnt?.rectangle?.coordinates) {
          const rect = rectEnt.rectangle.coordinates.getValue(
            Cesium.JulianDate.now()
          ) as Cesium.Rectangle;

          const dest = paddedRectangle(rect, 0.6);

          viewer.camera.flyTo({
            destination: dest,
            duration: 0.6,
          });
        } else if (instrument === "HIRISE" || instrument === "CRISM") {
          // Fallback: load LBL directly to get bounds for fly-to (HIRISE/CRISM only)
          const isHiRISE = instrument === "HIRISE";
          const lbl = isHiRISE
            ? await loadHiRISELBL(productId)
            : await loadCRISMLBL(productId);

          if (lbl) {
            const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
            const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
            const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
            const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

            if (minLat != null && maxLat != null && westLon360 != null && eastLon360 != null) {
              const west = normalizeLonTo180(westLon360);
              const east = normalizeLonTo180(eastLon360);
              const south = Math.min(minLat, maxLat);
              const north = Math.max(minLat, maxLat);

              const rect = Cesium.Rectangle.fromDegrees(west, south, east, north);
              const dest = paddedRectangle(rect, 0.6);

              viewer.camera.flyTo({
                destination: dest,
                duration: 0.6,
              });
            }
          }
        }
      },
      Cesium.ScreenSpaceEventType.LEFT_CLICK
    );

    // ========================================
    // DTM Hover System - Fast elevation probe
    // ========================================

    // Create persistent hover marker (billboard) - created ONCE
    const dtmHoverMarker = viewer.entities.add({
      id: "DTM_HOVER_MARKER",
      position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
      billboard: {
        image: (() => {
          // Create a small crosshair canvas
          const canvas = document.createElement("canvas");
          canvas.width = 24;
          canvas.height = 24;
          const ctx = canvas.getContext("2d")!;
          ctx.strokeStyle = "#f59e0b"; // amber-500
          ctx.lineWidth = 2;
          // Outer circle
          ctx.beginPath();
          ctx.arc(12, 12, 8, 0, Math.PI * 2);
          ctx.stroke();
          // Crosshair
          ctx.beginPath();
          ctx.moveTo(12, 2);
          ctx.lineTo(12, 6);
          ctx.moveTo(12, 18);
          ctx.lineTo(12, 22);
          ctx.moveTo(2, 12);
          ctx.lineTo(6, 12);
          ctx.moveTo(18, 12);
          ctx.lineTo(22, 12);
          ctx.stroke();
          // Center dot
          ctx.fillStyle = "#f59e0b";
          ctx.beginPath();
          ctx.arc(12, 12, 2, 0, Math.PI * 2);
          ctx.fill();
          return canvas;
        })(),
        scale: 1.0,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      show: false, // Hidden by default
    });
    dtmHoverMarkerRef.current = dtmHoverMarker;

    // Throttled DTM hover handler (25 Hz = 40ms)
    const dtmHoverHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    const throttledDTMHover = throttle((endPosition: Cesium.Cartesian2) => {
      if (dtmHoverModeRef.current !== "hover") return;

      const cartesian = viewer.camera.pickEllipsoid(endPosition, MARS_ELLIPSOID);
      if (!cartesian) {
        dtmHoverMarker.show = false;
        dtmHoverReadoutRef.current?.hide();
        return;
      }

      const carto = Cesium.Cartographic.fromCartesian(cartesian);
      const lat = Cesium.Math.toDegrees(carto.latitude);
      const lon = Cesium.Math.toDegrees(carto.longitude);

      // Check if we're over any DTM footprint with a cached grid
      const grid = dtmGridCacheRef.current.get(activeDTMProductRef.current || "");
      if (!grid || !isWithinDTMBounds(grid, lat, lon)) {
        dtmHoverMarker.show = false;
        dtmHoverReadoutRef.current?.hide();
        return;
      }

      // O(1) elevation lookup
      const elevation = getElevationFromGrid(grid, lat, lon);

      // Update marker position (no entity recreation)
      dtmHoverMarker.position = Cesium.Cartesian3.fromDegrees(lon, lat, 0) as any;
      dtmHoverMarker.show = true;

      // Update readout (via ref, no React re-render)
      dtmHoverReadoutRef.current?.update(lat, lon, elevation, grid.productId);

      viewer.scene.requestRender();
    }, 40);

    dtmHoverHandler.setInputAction(
      (m: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
        throttledDTMHover(m.endPosition);
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE
    );

    return () => {
      hoverHandler.destroy();
      clickHandler.destroy();
      dtmHoverHandler.destroy();
      // Dispose FootprintManager
      if (footprintManagerRef.current) {
        footprintManagerRef.current.dispose();
        footprintManagerRef.current = null;
      }
    };
  }, []);

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

  // Keep inspected product ref in sync
  inspectedProductIdRef.current = inspectedProductId ?? null;

  // Helper: apply per-entity visibility for an instrument considering overlap filter
  // Always force-shows the inspected product's footprint regardless of visibility/filter state
  const applyInstrumentVisibility = useCallback((instrument: FPInstrumentType, show: boolean) => {
    const fm = footprintManagerRef.current;
    const viewer = viewerRef.current;
    if (!fm || !viewer) return;

    const overlapPassing = overlapResultRef.current?.get(instrument);
    if (show && overlapPassing) {
      // Overlap filter active: per-entity visibility
      const features = fm.getFeatures(instrument);
      viewer.entities.suspendEvents();
      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;
        const visible = overlapPassing.has(pid);
        const fpEntity = viewer.entities.getById(`${instrument}_FP_${pid}`);
        if (fpEntity) fpEntity.show = visible;
        // Labels stay hidden (hover-only)
      }
      viewer.entities.resumeEvents();
    } else {
      fm.setVisible(instrument, show);
    }

    // Force-show inspected product's footprint regardless of layer/filter state
    const ipid = inspectedProductIdRef.current;
    if (ipid) {
      const fpEntity = viewer.entities.getById(`${instrument}_FP_${ipid}`);
      if (fpEntity) fpEntity.show = true;
    }

    viewer.scene.requestRender();
  }, []);

  // Toggle footprint visibility (does NOT load new footprints, just shows/hides existing ones)
  useEffect(() => {
    applyInstrumentVisibility("HIRISE", showHiRISE);
  }, [showHiRISE, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("CRISM", showCRISM);
  }, [showCRISM, applyInstrumentVisibility]);

  // Explicit footprint loading - triggered by loadFootprintsTrigger prop
  // After loading completes, ensure visibility is set correctly (fixes reload visibility bug)
  useEffect(() => {
    if (!loadFootprintsTrigger || !footprintManagerRef.current) return;

    const { instrument } = loadFootprintsTrigger;
    const fm = footprintManagerRef.current;
    let cancelled = false;


    // Load footprints and then ensure visibility is set correctly
    // This fixes the issue where reloading doesn't show footprints because
    // the visibility effect doesn't re-run (show state unchanged)
    (async () => {
      try {
        const result = await fm.loadFootprints(instrument);
        if (cancelled) return;
        if (result && result.count > 0) {
          fm.setVisible(instrument, true);
          setFootprintVersion(v => v + 1);
        }
      } catch (e) {
        if (!cancelled) toast.error(`Failed to load ${instrument} footprints`);
        console.error("Footprint load error:", e);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [loadFootprintsTrigger]);

  // Render loaded footprint bounding boxes as dashed rectangles on the map
  useEffect(() => {
    const viewer = viewerRef.current;
    const fm = footprintManagerRef.current;
    if (!viewer || !fm) return;

    // Remove old bbox entities
    const toRemove: string[] = [];
    for (let i = 0; i < viewer.entities.values.length; i++) {
      const e = viewer.entities.values[i];
      if (e.id && typeof e.id === "string" && e.id.startsWith("BBOX_")) {
        toRemove.push(e.id);
      }
    }
    for (const id of toRemove) {
      viewer.entities.removeById(id);
    }

    // Add bbox rectangles for all loaded instruments
    const bboxes = fm.getAllLoadedBboxes();
    const BBOX_COLORS: Record<string, Cesium.Color> = {
      CRISM: Cesium.Color.CYAN.withAlpha(0.35),
      HIRISE: Cesium.Color.YELLOW.withAlpha(0.35),
      SHARAD: Cesium.Color.ORANGE.withAlpha(0.35),
      SHARAD_HIGHRES: Cesium.Color.fromCssColorString("#FFD700").withAlpha(0.35),
      CTX: Cesium.Color.fromCssColorString("#FF1493").withAlpha(0.35),
      HIRISE_DTM: Cesium.Color.fromCssColorString("#B8860B").withAlpha(0.35),
      CRISM_TRR3: Cesium.Color.TEAL.withAlpha(0.35),
    };

    for (const [instrument, bbox] of bboxes) {
      const [west, south, east, north] = bbox;
      const color = BBOX_COLORS[instrument] || Cesium.Color.WHITE.withAlpha(0.3);
      viewer.entities.add({
        id: `BBOX_${instrument}`,
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
          material: Cesium.Color.TRANSPARENT,
          outline: true,
          outlineColor: color,
          outlineWidth: 2,
          height: 0,
        },
        properties: new Cesium.PropertyBag({ isBbox: true, instrument }),
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [footprintVersion]);

  // Update CRISM footprint visibility when ice score filter changes
  useEffect(() => {
    const viewer = viewerRef.current;
    const footprintManager = footprintManagerRef.current;
    if (!viewer || !footprintManager || !showCRISM) return;
    if (crismFilteredIds === null) return; // No ice filter active — let applyInstrumentVisibility handle it

    const crismFeatures = footprintManager.getFeatures("CRISM");
    const overlapPassingCrism = overlapResultRef.current?.get("CRISM" as FPInstrumentType);

    viewer.entities.suspendEvents();
    for (const feature of crismFeatures) {
      const pid = feature.properties.product_id;
      if (!pid) continue;

      const entity = viewer.entities.getById(`CRISM_FP_${pid}`);

      // Compose filters: ice score AND overlap
      let visible = true;
      const obsId = extractCrismObsId(pid);
      visible = crismFilteredIds.has(obsId);
      if (visible && overlapPassingCrism) {
        visible = overlapPassingCrism.has(pid);
      }

      if (entity) entity.show = visible;
      // Labels stay hidden (hover-only) — never set show=true here
    }

    // Force-show inspected product's CRISM footprint
    const ipid = inspectedProductIdRef.current;
    if (ipid) {
      const fpEnt = viewer.entities.getById(`CRISM_FP_${ipid}`);
      if (fpEnt) fpEnt.show = true;
    }

    viewer.entities.resumeEvents();
    viewer.scene.requestRender();
  }, [crismFilteredIds, showCRISM]);

  useEffect(() => {
    applyInstrumentVisibility("SHARAD", showSHARAD);
  }, [showSHARAD, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("SHARAD_HIGHRES", showSharadHighres);
  }, [showSharadHighres, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("CTX", showCTX);
  }, [showCTX, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("CRISM_TRR3", showCRISM_TRR3);
  }, [showCRISM_TRR3, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("HIRISE_DTM", showHiRISEDTM);
    // Also show/hide quickview overlay entities for DTM products
    const viewer = viewerRef.current;
    if (viewer) {
      for (const id of quickviewOverlayIdsRef.current) {
        if (id.startsWith("DTE")) {
          const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${id}`);
          if (ent) ent.show = showHiRISEDTM;
        }
      }
      viewer.scene.requestRender();
    }
  }, [showHiRISEDTM, applyInstrumentVisibility]);

  // Force-show the inspected product's footprint when inspector selection changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !inspectedProductId) return;

    const instruments = ["HIRISE", "CRISM", "CTX", "SHARAD", "SHARAD_HIGHRES", "HIRISE_DTM", "CRISM_TRR3"];
    for (const inst of instruments) {
      const fpEntity = viewer.entities.getById(`${inst}_FP_${inspectedProductId}`);
      if (fpEntity) {
        fpEntity.show = true;
        const lblEntity = viewer.entities.getById(`${inst}_LBL_${inspectedProductId}`);
        if (lblEntity) lblEntity.show = true;
        break;
      }
    }
    viewer.scene.requestRender();
  }, [inspectedProductId]);

  // Note: Legacy footprint overlay hiding is no longer needed since
  // footprints are now managed by FootprintManager (viewport-based loading)

  // Multi-Instrument Overlap Filter: compute and apply visibility
  const overlapEnabled = overlapFilter?.enabled ?? false;

  useEffect(() => {
    const fm = footprintManagerRef.current;
    const viewer = viewerRef.current;
    if (!fm || !viewer) return;

    // If filter disabled, clear and restore normal visibility
    if (!overlapEnabled) {
      overlapResultRef.current = null;
      onOverlapStatsChange?.(null);
      // Restore visibility for all instruments to match their toggle state
      if (showCRISM) fm.setVisible("CRISM", true);
      if (showHiRISE) fm.setVisible("HIRISE", true);
      if (showSHARAD) fm.setVisible("SHARAD", true);
      if (showSharadHighres) fm.setVisible("SHARAD_HIGHRES", true);
      if (showCTX) fm.setVisible("CTX", true);
      if (showHiRISEDTM) fm.setVisible("HIRISE_DTM", true);
      // Re-apply CRISM ice score filter if active
      if (crismFilteredIds !== null && showCRISM) {
        const crismFeatures = fm.getFeatures("CRISM");
        viewer.entities.suspendEvents();
        for (const feature of crismFeatures) {
          const pid = feature.properties.product_id;
          if (!pid) continue;
          const obsId = extractCrismObsId(pid);
          const visible = crismFilteredIds.has(obsId);
          const fpEnt = viewer.entities.getById(`CRISM_FP_${pid}`);
          if (fpEnt) fpEnt.show = visible;
          // Labels stay hidden (hover-only)
        }
        viewer.entities.resumeEvents();
      }
      viewer.scene.requestRender();
      return;
    }

    // Auto-detect all instruments with loaded features
    const allInstruments: FPInstrumentType[] = ["CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES", "CTX", "HIRISE_DTM", "CRISM_TRR3"];
    const featuresByInstrument = new Map<FPInstrumentType, any[]>();
    for (const inst of allInstruments) {
      const features = fm.getFeatures(inst);
      if (features.length > 0) {
        featuresByInstrument.set(inst, features);
      }
    }
    const selectedInstruments = Array.from(featuresByInstrument.keys());

    // Need at least 2 instruments with loaded features
    if (selectedInstruments.length < 2) {
      overlapResultRef.current = null;
      onOverlapStatsChange?.({ totalChecked: 0, totalPassing: 0, perInstrument: new Map() });
      viewer.scene.requestRender();
      return;
    }

    // Compute overlap
    const { result, stats } = computeOverlapFilter(featuresByInstrument, selectedInstruments);
    overlapResultRef.current = result;
    onOverlapStatsChange?.(stats);

    // Apply visibility for each selected instrument (batched)
    viewer.entities.suspendEvents();
    for (const inst of selectedInstruments) {
      const passingIds = result.get(inst);
      const features = fm.getFeatures(inst);
      // Check if this instrument is toggled on
      const instrumentOn = {
        CRISM: showCRISM,
        HIRISE: showHiRISE,
        SHARAD: showSHARAD,
        SHARAD_HIGHRES: showSharadHighres,
        CTX: showCTX,
        HIRISE_DTM: showHiRISEDTM,
      }[inst] ?? false;

      if (!instrumentOn) continue;

      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;
        let visible = passingIds ? passingIds.has(pid) : false;
        // Also compose with CRISM ice score filter
        if (visible && inst === "CRISM" && crismFilteredIds !== null) {
          const obsId = extractCrismObsId(pid);
          visible = crismFilteredIds.has(obsId);
        }
        const fpEnt = viewer.entities.getById(`${inst}_FP_${pid}`);
        if (fpEnt) fpEnt.show = visible;
        // Labels stay hidden (hover-only)
      }
    }
    viewer.entities.resumeEvents();

    viewer.scene.requestRender();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlapEnabled, footprintVersion, crismFilteredIds,
      showCRISM, showHiRISE, showSHARAD, showSharadHighres, showCTX, showHiRISEDTM]);

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
    const p1 = linePoints[0];
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
      const p2 = linePoints[1];
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

  // Switch between 2D and 3D map modes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const targetMode = mapMode === "3D"
      ? Cesium.SceneMode.SCENE3D
      : Cesium.SceneMode.SCENE2D;

    if (viewer.scene.mode !== targetMode) {
      // Morph to new mode with animation
      if (targetMode === Cesium.SceneMode.SCENE3D) {
        viewer.scene.morphTo3D(1.0);
      } else {
        viewer.scene.morphTo2D(1.0);
      }
    }
  }, [mapMode]);

  // Update base layer when baseLayer changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: BASE_LAYER_URLS[baseLayer],
        rectangle: MARS_RECT,
        tilingScheme: new Cesium.GeographicTilingScheme({
          ellipsoid: MARS_ELLIPSOID,
          numberOfLevelZeroTilesX: 2,
          numberOfLevelZeroTilesY: 1,
        }),
      })
    );

    viewer.scene.requestRender();
  }, [baseLayer]);

  // Update view when viewBounds changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (viewBounds) {
      const { westLon, eastLon, minLat, maxLat } = viewBounds;

      // Convert to radians for Cesium (normalize longitude to -180 to 180 first)
      const west = Cesium.Math.toRadians(normalizeLonTo180(westLon));
      const south = Cesium.Math.toRadians(minLat);
      const east = Cesium.Math.toRadians(normalizeLonTo180(eastLon));
      const north = Cesium.Math.toRadians(maxLat);

      // Create rectangle
      const rect = new Cesium.Rectangle(west, south, east, north);

      // For 2D mode, use setView for immediate positioning
      viewer.camera.setView({
        destination: rect,
      });

      viewer.scene.requestRender();
    } else {
      // Show full Mars
      viewer.camera.setView({
        destination: MARS_RECT,
      });
      viewer.scene.requestRender();
    }
  }, [viewBounds]);

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

      // If found entity with position (point/polyline), fly to its position
      if (foundEntity?.position) {
        const pos = foundEntity.position.getValue(Cesium.JulianDate.now());
        if (pos) {
          v.camera.flyTo({
            destination: pos,
            orientation: {
              heading: 0,
              pitch: Cesium.Math.toRadians(-90),
              roll: 0,
            },
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // If found entity with polyline, fly to its center
      if (foundEntity?.polyline?.positions) {
        const positions = foundEntity.polyline.positions.getValue(Cesium.JulianDate.now());
        if (positions && positions.length > 0) {
          const midIdx = Math.floor(positions.length / 2);
          v.camera.flyTo({
            destination: positions[midIdx],
            orientation: {
              heading: 0,
              pitch: Cesium.Math.toRadians(-90),
              roll: 0,
            },
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
          console.warn("[FlyTo] No LBL found for", pid);
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

      console.warn("[FlyTo] Could not find entity or LBL for", pid);
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
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, 500000),
      duration: 1.5,
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

    for (const inst of instruments) {
      const e = viewer.entities.getById(`${inst}_FP_${pid}`);
      if (e) { entity = e; break; }
    }

    if (!entity) {
      onHighlightComplete?.();
      return;
    }

    // Save original material
    const origMaterial = entity.rectangle?.material;
    const origOutline = entity.rectangle?.outlineColor;
    const origOutlineWidth = entity.rectangle?.outlineWidth;

    // Apply bright highlight
    if (entity.rectangle) {
      entity.rectangle.material = new Cesium.ColorMaterialProperty(
        Cesium.Color.MAGENTA.withAlpha(0.7)
      ) as any;
      entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE) as any;
      entity.rectangle.outlineWidth = new Cesium.ConstantProperty(3) as any;
    }
    // Also handle polyline entities (SHARAD)
    if (entity.polyline) {
      const origPolyMaterial = entity.polyline.material;
      const origPolyWidth = entity.polyline.width;
      entity.polyline.material = new Cesium.ColorMaterialProperty(Cesium.Color.MAGENTA) as any;
      entity.polyline.width = new Cesium.ConstantProperty(5) as any;

      viewer.scene.requestRender();

      const timer = setTimeout(() => {
        if (entity?.polyline) {
          entity.polyline.material = origPolyMaterial as any;
          entity.polyline.width = origPolyWidth as any;
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
        entity.rectangle.material = origMaterial as any;
        entity.rectangle.outlineColor = origOutline as any;
        entity.rectangle.outlineWidth = origOutlineWidth as any;
      }
      viewer.scene.requestRender();
      onHighlightComplete?.();
    }, 3000);

    return () => clearTimeout(timer);
  }, [highlightProductId, onHighlightComplete]);

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

  // Hide footprint boxes when high-res overlay is active
  useEffect(() => {
    const viewer = viewerRef.current;
    const footprintManager = footprintManagerRef.current;
    if (!viewer || !footprintManager) return;

    const highResSet = new Set(highResOverlays);

    // Helper to update footprint visibility for FootprintManager entities
    const updateFootprintVisibility = (instrument: "HIRISE" | "CRISM") => {
      const features = footprintManager.getFeatures(instrument);
      const isVisible = instrument === "HIRISE" ? showHiRISE : showCRISM;

      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;

        // Find the main footprint entity (new ID format)
        const entityId = `${instrument}_FP_${pid}`;
        const entity = viewer.entities.getById(entityId);

        if (entity) {
          if (highResSet.has(pid)) {
            // Hide footprint when high-res is active (so clicks go through to overlay)
            entity.show = false;
          } else {
            entity.show = isVisible;
          }
        }

        // Update label entity (FootprintManager uses _LBL_ for labels)
        const labelId = `${instrument}_LBL_${pid}`;
        const labelEnt = viewer.entities.getById(labelId);

        if (labelEnt) {
          labelEnt.show = highResSet.has(pid) ? false : isVisible;
        }
      }
    };

    // Update HiRISE footprint visibility
    if (footprintManager.hasFootprints("HIRISE") || highResSet.size > 0) {
      updateFootprintVisibility("HIRISE");
    }

    // Update CRISM footprint visibility
    if (footprintManager.hasFootprints("CRISM") || highResSet.size > 0) {
      updateFootprintVisibility("CRISM");
    }

    viewer.scene.requestRender();
  }, [highResOverlays, showHiRISE, showCRISM]);

  // Refs to track current overlays
  const quickviewOverlayIdsRef = useRef<Set<string>>(new Set());
  const quickviewBlobUrlsRef = useRef<Map<string, string>>(new Map());
  const highResOverlayIdsRef = useRef<Set<string>>(new Set());
  const mineralOverlayIdsRef = useRef<Set<string>>(new Set());
  const mineralBlobUrlsRef = useRef<Map<string, string>>(new Map());
  const browseOverlayIdsRef = useRef<Map<string, Set<BrowseProductType>>>(new Map());
  const scoreOverlayIdsRef = useRef<Map<string, Set<ScoreProductType>>>(new Map());

  // Track CTX tile imagery layers for cleanup
  const ctxTileLayersRef = useRef<Map<string, Cesium.ImageryLayer>>(new Map());

  // Track blob URLs for CRISM RGB images to clean up later
  const crismBlobUrlsRef = useRef<Map<string, string>>(new Map());

  // Track previous RGB wavelengths to detect changes
  const prevRgbRef = useRef<RGBWavelengths>(rgbWavelengths);

  // PERFORMANCE OPTIMIZED: Quickview overlays effect
  // Key optimizations:
  // 1. Toggle visibility instead of add/remove for existing entities
  // 2. Use cached bounds (getProductBounds)
  // 3. Batch entity operations
  // 4. Single requestRender at end
  // 5. Process additions in parallel
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(quickviewOverlays);
    const existingIds = quickviewOverlayIdsRef.current;

    let needsRender = false;

    // Helper: check if a product is CTX by looking up its footprint entity
    const isCTXProduct = (productId: string): boolean => {
      const fpEnt = viewer.entities.getById(`CTX_FP_${productId}`);
      return !!fpEnt;
    };

    // Helper: get CTX tile info from footprint entity properties
    const getCTXTileInfo = (productId: string): { tileUrl: string; west: number; south: number; east: number; north: number } | null => {
      const fpEnt = viewer.entities.getById(`CTX_FP_${productId}`);
      if (!fpEnt?.properties) return null;
      const p = fpEnt.properties as any;
      const tileUrl = p.tile_url?.getValue?.();
      const west = p.bbox_west?.getValue?.();
      const south = p.bbox_south?.getValue?.();
      const east = p.bbox_east?.getValue?.();
      const north = p.bbox_north?.getValue?.();
      if (!tileUrl || west == null || south == null || east == null || north == null) return null;
      return { tileUrl, west, south, east, north };
    };

    // STEP 1: Hide/remove overlays that are no longer in the list
    const toHide = Array.from(existingIds).filter((id) => !currentIds.has(id));
    for (const id of toHide) {
      // Remove CTX tile layer if it exists
      const ctxLayer = ctxTileLayersRef.current.get(id);
      if (ctxLayer) {
        viewer.imageryLayers.remove(ctxLayer);
        ctxTileLayersRef.current.delete(id);
        needsRender = true;
      }
      // Hide entity overlay (CRISM/HiRISE/HiRISE DTM)
      const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${id}`);
      if (ent) {
        ent.show = false;
        needsRender = true;
      }
      // Clean up quickview blob URL
      const qvBlobUrl = quickviewBlobUrlsRef.current.get(id);
      if (qvBlobUrl) {
        URL.revokeObjectURL(qvBlobUrl);
        quickviewBlobUrlsRef.current.delete(id);
      }
      // Clear active DTM if this was a DTM product being hidden
      if (id.startsWith("DTE") && activeDTMProductRef.current === id) {
        activeDTMProductRef.current = null;
        dtmHoverReadoutRef.current?.hide();
      }
      // Restore footprint fill when overlay is removed
      setFootprintTransparent(viewer, id, false);
      existingIds.delete(id);
    }

    // STEP 2: Show existing overlays that are back in the list
    const toCreate: string[] = [];

    for (const productId of quickviewOverlays) {
      if (existingIds.has(productId)) continue; // Already tracked and visible

      // Check for existing CTX tile layer
      const ctxLayer = ctxTileLayersRef.current.get(productId);
      if (ctxLayer) {
        ctxLayer.show = true;
        existingIds.add(productId);
        needsRender = true;
        continue;
      }

      const existingEnt = viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
      if (existingEnt) {
        // Entity exists but was hidden - show it (but respect DTM layer toggle)
        const isDTM = productId.startsWith("DTE");
        existingEnt.show = isDTM ? showHiRISEDTM : true;
        setFootprintTransparent(viewer, productId, true);
        existingIds.add(productId);
        needsRender = true;

        // For HiRISE DTM, ensure elevation grid is loaded for hover
        if (isDTM) {
          activeDTMProductRef.current = productId;
          if (!dtmGridCacheRef.current.has(productId)) {
            loadDTMElevationGrid(productId).then((grid) => {
              if (grid) {
                setDtmGrid(productId, grid);
              }
            });
          }
        }
      } else {
        toCreate.push(productId);
      }
    }

    // STEP 3: Create new overlays
    // Separate CTX products (tile layers) from CRISM/HiRISE (single images)
    const ctxToCreate = toCreate.filter((id) => isCTXProduct(id));
    const imageToCreate = toCreate.filter((id) => !isCTXProduct(id));

    // Create CTX tile overlays (synchronous - no network fetch needed)
    for (const productId of ctxToCreate) {
      const info = getCTXTileInfo(productId);
      if (!info) continue;

      const provider = new Cesium.UrlTemplateImageryProvider({
        url: info.tileUrl,
        rectangle: Cesium.Rectangle.fromDegrees(info.west, info.south, info.east, info.north),
        tilingScheme: new Cesium.GeographicTilingScheme({
          ellipsoid: MARS_ELLIPSOID,
          numberOfLevelZeroTilesX: 2,
          numberOfLevelZeroTilesY: 1,
        }),
        minimumLevel: 0,
        maximumLevel: 12,
        credit: "NASA/JPL/MSSS - MRO CTX",
      });

      const layer = viewer.imageryLayers.addImageryProvider(provider);
      layer.alpha = getProductOpacity(productId);
      ctxTileLayersRef.current.set(productId, layer);
      quickviewOverlayIdsRef.current.add(productId);
      needsRender = true;
    }

    // Create CRISM/HiRISE/HiRISE DTM image overlays (async)
    if (imageToCreate.length > 0) {
      // Pre-fetch bounds in parallel for faster creation
      Promise.all(imageToCreate.map(async (productId) => {
        try {
          let bounds = await getProductBounds(productId);
          // Fallback: extract bounds from existing footprint entity
          if (!bounds && viewerRef.current) {
            bounds = getFootprintBounds(viewerRef.current, productId);
            if (bounds) console.warn(`[Overlay] Using footprint bounds fallback for ${productId}`);
          }
          if (!bounds || !viewerRef.current) {
            console.warn(`[Overlay] No bounds for quickview: ${productId}`);
            return null;
          }

          const isHiRISE = productId.startsWith("ESP_");
          const isHiRISEDTM = productId.startsWith("DTE");
          // TRR3 IDs: {type}{hex}_{nn} (no _mtr3 suffix), e.g. frs0005bad3_01
          const isTRR3 = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);

          // Derive quickview URL
          let imageUrl: string;
          let instrument: "HIRISE" | "CRISM" | "HIRISE_DTM" | "CRISM_TRR3" = "CRISM";

          if (isHiRISEDTM) {
            imageUrl = `/hirise_dtm/overlay/${productId}.png`;
            instrument = "HIRISE_DTM";
          } else if (isHiRISE) {
            imageUrl = `/hirise/quickview/${productId}.png`;
            instrument = "HIRISE";
          } else if (isTRR3) {
            const obsId = productId.replace(/_\d{2}$/, "");
            imageUrl = `/api/mineral-cnn/quickview/${obsId}`;
            instrument = "CRISM_TRR3";
          } else {
            // Smart backend endpoint handles all naming patterns
            // Pass obs_id (e.g. frt00008a1e_07) so backend can search crism_quickview/ + crism_data/
            const parts = productId.split("_");
            const base = parts.length >= 2 ? `${parts[0]}_${parts[1]}` : productId;
            imageUrl = `/crism/quickview/${base}.png`;
          }

          // Fetch image first to avoid gray rectangles from 404 URLs
          const imgRes = await fetch(imageUrl);
          if (!imgRes.ok) return null;
          const blob = await imgRes.blob();
          const blobUrl = URL.createObjectURL(blob);

          return { productId, bounds, imageUrl: blobUrl, instrument };
        } catch {
          return null;
        }
      })).then((results) => {
        const v = viewerRef.current;
        if (!v) return;

        // Batch entity creation
        v.entities.suspendEvents();

        for (const result of results) {
          if (!result) continue;
          const { productId, bounds, imageUrl, instrument } = result;

          const isDTM = instrument === "HIRISE_DTM";
          v.entities.add({
            id: `QUICKVIEW_OVERLAY_${productId}`,
            show: isDTM ? showHiRISEDTMRef.current : true,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: instrument,
              kind: "OVERLAY",
            },
          });

          // Track blob URL for cleanup
          if (imageUrl.startsWith("blob:")) {
            quickviewBlobUrlsRef.current.set(productId, imageUrl);
          }

          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(v, productId, true);
          quickviewOverlayIdsRef.current.add(productId);

          // For HiRISE DTM products, load elevation grid for hover readout
          if (instrument === "HIRISE_DTM") {
            activeDTMProductRef.current = productId;
            loadDTMElevationGrid(productId).then((grid) => {
              if (grid) {
                setDtmGrid(productId, grid);
              }
            });
          }
        }

        v.entities.resumeEvents();
        v.scene.requestRender();
      });
    }

    if (needsRender) {
      viewer.scene.requestRender();
    }
  }, [quickviewOverlays]);

  // PERFORMANCE OPTIMIZED: High-resolution overlays effect
  // Uses visibility toggling and batched operations
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(highResOverlays);
    const existingIds = highResOverlayIdsRef.current;

    let needsRender = false;

    // STEP 1: Hide overlays no longer in list (keep entity for potential re-show)
    const toHide = Array.from(existingIds).filter((id) => !currentIds.has(id));
    for (const id of toHide) {
      const ent = viewer.entities.getById(`HIGHRES_OVERLAY_${id}`);
      if (ent) {
        ent.show = false;
        needsRender = true;
      }
      // Restore footprint fill when overlay is removed
      setFootprintTransparent(viewer, id, false);
      existingIds.delete(id);

      // Clean up CRISM blob URLs to free memory
      const blobUrl = crismBlobUrlsRef.current.get(id);
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        crismBlobUrlsRef.current.delete(id);
        // Also remove the hidden entity for CRISM to force reload with potentially new wavelengths
        if (!id.startsWith("ESP_") && ent) {
          viewer.entities.remove(ent);
        }
      }
    }

    // STEP 2: Show existing or create new overlays
    const toCreate: string[] = [];
    for (const productId of highResOverlays) {
      if (existingIds.has(productId)) continue;

      const existingEnt = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
      if (existingEnt && productId.startsWith("ESP_")) {
        // HiRISE entity exists - just show it
        existingEnt.show = true;
        setFootprintTransparent(viewer, productId, true);
        existingIds.add(productId);
        needsRender = true;
      } else {
        toCreate.push(productId);
      }
    }

    // STEP 3: Create new overlays (async)
    if (toCreate.length > 0) {
      Promise.all(toCreate.map(async (productId) => {
        try {
          let bounds = await getProductBounds(productId);
          if (!bounds && viewerRef.current) {
            bounds = getFootprintBounds(viewerRef.current, productId);
            if (bounds) console.warn(`[Overlay] Using footprint bounds fallback for ${productId}`);
          }
          if (!bounds || !viewerRef.current) {
            console.warn(`[Overlay] No bounds for highres: ${productId}`);
            return null;
          }

          const isHiRISE = productId.startsWith("ESP_");
          let imageUrl: string;

          if (isHiRISE) {
            imageUrl = `/hirise/overlay/${productId}.png`;
          } else {
            // CRISM / TRR3 RGB request
            const isTRR3 = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);
            let rgbUrl: string;
            if (isTRR3) {
              const obsId = productId.replace(/_\d{2}$/, "");
              rgbUrl = `/api/crism-trr3/${obsId}/rgb`;
            } else {
              rgbUrl = `/crism/${productId}/rgb`;
            }

            const response = await fetch(rgbUrl, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                r_um: rgbWavelengths.r,
                g_um: rgbWavelengths.g,
                b_um: rgbWavelengths.b,
                vmin: 0.02,
                vmax: 0.25,
              }),
            });

            if (!response.ok) return null;

            const blob = await response.blob();
            imageUrl = URL.createObjectURL(blob);
            crismBlobUrlsRef.current.set(productId, imageUrl);
          }

          return { productId, bounds, imageUrl, isHiRISE };
        } catch {
          return null;
        }
      })).then((results) => {
        const v = viewerRef.current;
        if (!v) return;

        v.entities.suspendEvents();

        for (const result of results) {
          if (!result) continue;
          const { productId, bounds, imageUrl, isHiRISE } = result;

          v.entities.add({
            id: `HIGHRES_OVERLAY_${productId}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: isHiRISE ? "HIRISE" : /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId) ? "CRISM_TRR3" : "CRISM",
              kind: "OVERLAY",
            },
          });

          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(v, productId, true);
          highResOverlayIdsRef.current.add(productId);
        }

        v.entities.resumeEvents();
        v.scene.requestRender();
      });
    }

    if (needsRender) {
      viewer.scene.requestRender();
    }
  }, [highResOverlays, rgbWavelengths]);

  // Effect to refresh CRISM overlays when RGB wavelengths change
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Check if RGB wavelengths actually changed
    const prev = prevRgbRef.current;
    if (prev.r === rgbWavelengths.r && prev.g === rgbWavelengths.g && prev.b === rgbWavelengths.b) {
      return;
    }

    prevRgbRef.current = rgbWavelengths;

    // Find CRISM products in highResOverlays and refresh them
    const crismProducts = highResOverlays.filter((id) => !id.startsWith("ESP_"));

    crismProducts.forEach((productId) => {
      // Remove existing entity
      const ent = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
      if (ent) viewer.entities.remove(ent);

      // Clean up blob URL
      const blobUrl = crismBlobUrlsRef.current.get(productId);
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        crismBlobUrlsRef.current.delete(productId);
      }

      // Remove from tracking so main effect will re-add it
      highResOverlayIdsRef.current.delete(productId);
    });

    viewer.scene.requestRender();
  }, [rgbWavelengths, highResOverlays]);

  // Browse product overlays effect (HYD, ICE, IC2)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const existingOverlays = browseOverlayIdsRef.current;

    // Get all product IDs that should have browse overlays
    const newProductIds = new Set(browseOverlays.keys());
    const existingProductIds = new Set(existingOverlays.keys());

    // Remove overlays for products no longer in the list
    existingProductIds.forEach((productId) => {
      if (!newProductIds.has(productId)) {
        // Remove all browse overlays for this product
        const types = existingOverlays.get(productId);
        types?.forEach((browseType) => {
          const ent = viewer.entities.getById(`BROWSE_OVERLAY_${productId}_${browseType}`);
          if (ent) viewer.entities.remove(ent);
        });
        // Restore footprint fill when all browse overlays are removed
        setFootprintTransparent(viewer, productId, false);
        existingOverlays.delete(productId);
      }
    });

    // Update or add overlays for current products
    browseOverlays.forEach(async (types, productId) => {
      const existingTypes = existingOverlays.get(productId) || new Set();

      // Remove types that are no longer active
      existingTypes.forEach((browseType) => {
        if (!types.has(browseType)) {
          const ent = viewer.entities.getById(`BROWSE_OVERLAY_${productId}_${browseType}`);
          if (ent) viewer.entities.remove(ent);
        }
      });

      // Add new types
      for (const browseType of types) {
        if (existingTypes.has(browseType)) continue;

        try {
          // Try getProductBounds first, then footprint fallback
          let west: number, east: number, south: number, north: number;
          let bounds = await getProductBounds(productId);
          if (!bounds && viewerRef.current) {
            bounds = getFootprintBounds(viewerRef.current, productId);
            if (bounds) console.warn(`[Browse] Using footprint bounds fallback for ${productId}`);
          }

          if (bounds) {
            west = bounds.west;
            east = bounds.east;
            south = bounds.south;
            north = bounds.north;
          } else {
            // Last resort: try LBL directly
            const lbl = await loadCRISMLBL(productId);
            if (!lbl) {
              console.warn("[Browse] No bounds for", productId);
              continue;
            }

            const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
            const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
            const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
            const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

            if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
              console.warn("[Browse] Missing bounds for", productId);
              continue;
            }

            west = normalizeLonTo180(westLon360);
            east = normalizeLonTo180(eastLon360);
            south = Math.min(minLat, maxLat);
            north = Math.max(minLat, maxLat);
          }

          // Construct browse image URL
          // Arcadia products: frt00003156_07_brcarj_mtr3 -> frt00003156_HYD.png
          const baseObsId = productId.split("_")[0];
          const imageUrl = `/crism/browse/${baseObsId}_${browseType}.png`;

          // Pre-validate image exists to avoid gray rectangles
          const headRes = await fetch(imageUrl, { method: "HEAD" });
          if (!headRes.ok) {
            console.warn(`[Browse] Image not found: ${imageUrl}`);
            continue;
          }

          if (!viewerRef.current) return;

          viewer.entities.add({
            id: `BROWSE_OVERLAY_${productId}_${browseType}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: "CRISM",
              kind: "BROWSE_OVERLAY",
              browse_type: browseType,
            },
          });
          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(viewer, productId, true);
        } catch (e) {
          console.error("[Browse] Failed to add overlay:", productId, browseType, e);
        }
      }

      // Update tracking
      existingOverlays.set(productId, new Set(types));
    });

    viewer.scene.requestRender();
  }, [browseOverlays, overlayOpacities]);

  // Score product overlays effect (score_ice, score_hyd)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const existingOverlays = scoreOverlayIdsRef.current;

    // Get all product IDs that should have score overlays
    const newProductIds = new Set(scoreOverlays.keys());
    const existingProductIds = new Set(existingOverlays.keys());

    // Remove overlays for products no longer in the list
    existingProductIds.forEach((productId) => {
      if (!newProductIds.has(productId)) {
        // Remove all score overlays for this product
        const types = existingOverlays.get(productId);
        types?.forEach((scoreType) => {
          const ent = viewer.entities.getById(`SCORE_OVERLAY_${productId}_${scoreType}`);
          if (ent) viewer.entities.remove(ent);
        });
        // Restore footprint fill when all score overlays are removed
        setFootprintTransparent(viewer, productId, false);
        existingOverlays.delete(productId);
      }
    });

    // Update or add overlays for current products
    scoreOverlays.forEach(async (types, productId) => {
      const existingTypes = existingOverlays.get(productId) || new Set();

      // Remove types that are no longer active
      existingTypes.forEach((scoreType) => {
        if (!types.has(scoreType)) {
          const ent = viewer.entities.getById(`SCORE_OVERLAY_${productId}_${scoreType}`);
          if (ent) viewer.entities.remove(ent);
        }
      });

      // Add new types
      for (const scoreType of types) {
        if (existingTypes.has(scoreType)) continue;

        try {
          // Load LBL for bounds
          const lbl = await loadCRISMLBL(productId);
          if (!lbl) {
            console.warn("[Score] No LBL for", productId);
            continue;
          }

          const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
          const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
          const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
          const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

          if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
            console.warn("[Score] Missing bounds for", productId);
            continue;
          }

          const west = normalizeLonTo180(westLon360);
          const east = normalizeLonTo180(eastLon360);
          const south = Math.min(minLat, maxLat);
          const north = Math.max(minLat, maxLat);

          // Construct score image URL
          // Score files: frt00003156_score_ice.png, frt00003156_score_hyd.png
          const baseObsId = productId.split("_")[0];
          const imageUrl = `/crism/browse/${baseObsId}_${scoreType}.png`;

          // Pre-validate image exists to avoid gray rectangles
          const headRes = await fetch(imageUrl, { method: "HEAD" });
          if (!headRes.ok) {
            console.warn(`[Score] Image not found: ${imageUrl}`);
            continue;
          }

          if (!viewerRef.current) return;

          viewer.entities.add({
            id: `SCORE_OVERLAY_${productId}_${scoreType}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: "CRISM",
              kind: "SCORE_OVERLAY",
              score_type: scoreType,
            },
          });
          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(viewer, productId, true);
        } catch (e) {
          console.error("[Score] Failed to add overlay:", productId, scoreType, e);
        }
      }

      // Update tracking
      existingOverlays.set(productId, new Set(types));
    });

    viewer.scene.requestRender();
  }, [scoreOverlays, overlayOpacities]);

  // Mineral classification overlays effect (CNN mineral map on TRR3)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(mineralOverlays);
    const existingIds = mineralOverlayIdsRef.current;

    let needsRender = false;

    // STEP 1: Remove overlays no longer in list
    for (const id of Array.from(existingIds)) {
      if (!currentIds.has(id)) {
        const ent = viewer.entities.getById(`MINERAL_OVERLAY_${id}`);
        if (ent) {
          viewer.entities.remove(ent);
          needsRender = true;
        }
        setFootprintTransparent(viewer, id, false);
        existingIds.delete(id);
        // Clean up blob URL
        const blobUrl = mineralBlobUrlsRef.current.get(id);
        if (blobUrl) {
          URL.revokeObjectURL(blobUrl);
          mineralBlobUrlsRef.current.delete(id);
        }
      }
    }

    // STEP 2: Create new overlays
    const toCreate = mineralOverlays.filter((id) => !existingIds.has(id));
    if (toCreate.length > 0) {
      Promise.all(toCreate.map(async (productId) => {
        try {
          const bounds = await getProductBounds(productId);
          if (!bounds || !viewerRef.current) return null;

          // obs_id: strip _NN suffix (e.g. frt0000c702_07 → frt0000c702)
          const obsId = productId.replace(/_\d{2}$/, "");
          const mapUrl = `/api/mineral-cnn/result/${obsId}/mineral-map.png`;

          const res = await fetch(mapUrl);
          if (!res.ok) return null;

          const blob = await res.blob();
          const blobUrl = URL.createObjectURL(blob);

          return { productId, bounds, blobUrl };
        } catch {
          return null;
        }
      })).then((results) => {
        const v = viewerRef.current;
        if (!v) return;

        v.entities.suspendEvents();
        for (const result of results) {
          if (!result) continue;
          const { productId, bounds, blobUrl } = result;

          v.entities.add({
            id: `MINERAL_OVERLAY_${productId}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
              material: new Cesium.ImageMaterialProperty({
                image: blobUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: "CRISM_TRR3",
              kind: "MINERAL_OVERLAY",
            },
          });

          mineralBlobUrlsRef.current.set(productId, blobUrl);
          setFootprintTransparent(v, productId, true);
          mineralOverlayIdsRef.current.add(productId);
        }
        v.entities.resumeEvents();
        v.scene.requestRender();
      });
    }

    if (needsRender) {
      viewer.scene.requestRender();
    }
  }, [mineralOverlays, overlayOpacities]);

  // PERFORMANCE OPTIMIZED: Track visible products in current view
  // Reduced polling frequency and removed excessive logging
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !onVisibleProductsChange) return;

    // Track last result to avoid unnecessary updates
    let lastResultHash = "";

    const updateVisibleProducts = () => {
      const footprintManager = footprintManagerRef.current;
      if (!footprintManager) return;

      const visible: VisibleProduct[] = [];
      const seen = new Set<string>();

      // Helper to extract center coordinates from feature
      const getFeatureCenter = (feature: any): { lat: number; lon: number } | null => {
        const props = feature.properties || {};
        // HiRISE DTM has explicit bounds in properties
        if (props.west != null && props.east != null && props.south != null && props.north != null) {
          return {
            lat: (props.south + props.north) / 2,
            lon: (props.west + props.east) / 2,
          };
        }
        // Try geometry
        const geom = feature.geometry;
        if (!geom) return null;
        const coords = geom.coordinates;
        if (!coords) return null;

        if (geom.type === "Point") {
          return { lon: coords[0], lat: coords[1] };
        } else if (geom.type === "Polygon" && coords[0]) {
          const ring = coords[0];
          let sumLat = 0, sumLon = 0;
          for (const [lon, lat] of ring) {
            sumLon += lon;
            sumLat += lat;
          }
          return { lat: sumLat / ring.length, lon: sumLon / ring.length };
        } else if (geom.type === "LineString" && coords.length > 0) {
          const midIdx = Math.floor(coords.length / 2);
          return { lon: coords[midIdx][0], lat: coords[midIdx][1] };
        }
        return null;
      };

      // Helper: check if product passes the overlap filter
      const overlapResult = overlapResultRef.current;
      const passesOverlap = (inst: string, pid: string): boolean => {
        if (!overlapResult) return true; // No filter active
        const passing = overlapResult.get(inst as FPInstrumentType);
        return passing ? passing.has(pid) : true; // If instrument not in filter, pass
      };

      // Get products from FootprintManager for HiRISE
      if (showHiRISE && footprintManager.hasFootprints("HIRISE")) {
        const hiriseFeatures = footprintManager.getFeatures("HIRISE");
        for (const feature of hiriseFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid) && passesOverlap("HIRISE", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({ productId: pid, instrument: "HIRISE", title, lat: center?.lat, lon: center?.lon });
          }
        }
      }

      // Get products from FootprintManager for CRISM (with optional filter)
      if (showCRISM && footprintManager.hasFootprints("CRISM")) {
        const crismFeatures = footprintManager.getFeatures("CRISM");
        for (const feature of crismFeatures) {
          const pid = feature.properties.product_id;
          if (pid && !seen.has(pid)) {
            // Ice score filter
            if (crismFilteredIds !== null) {
              const obsId = extractCrismObsId(pid);
              if (!crismFilteredIds.has(obsId)) continue;
            }
            // Overlap filter
            if (!passesOverlap("CRISM", pid)) continue;
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({ productId: pid, instrument: "CRISM", lat: center?.lat, lon: center?.lon });
          }
        }
      }

      // Get products from FootprintManager for CTX
      if (showCTX && footprintManager.hasFootprints("CTX")) {
        const ctxFeatures = footprintManager.getFeatures("CTX");
        for (const feature of ctxFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid) && passesOverlap("CTX", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({ productId: pid, instrument: "CTX", title, lat: center?.lat, lon: center?.lon });
          }
        }
      }

      // Get products from FootprintManager for HiRISE DTM
      if (showHiRISEDTM && footprintManager.hasFootprints("HIRISE_DTM")) {
        const dtmFeatures = footprintManager.getFeatures("HIRISE_DTM");
        for (const feature of dtmFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid) && passesOverlap("HIRISE_DTM", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({ productId: pid, instrument: "HIRISE_DTM", title, lat: center?.lat, lon: center?.lon });
          }
        }
      }

      // Get products from FootprintManager for CRISM TRR3
      if (showCRISM_TRR3 && footprintManager.hasFootprints("CRISM_TRR3")) {
        const trr3Features = footprintManager.getFeatures("CRISM_TRR3");
        for (const feature of trr3Features) {
          const pid = feature.properties.product_id;
          if (pid && !seen.has(pid) && passesOverlap("CRISM_TRR3", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({ productId: pid, instrument: "CRISM_TRR3", lat: center?.lat, lon: center?.lon });
          }
        }
      }

      // Include custom datasets that are loaded and visible
      if (showCustomData) {
        for (const dataset of customDatasets) {
          if (dataset.visible && !seen.has(dataset.id)) {
            seen.add(dataset.id);
            const center = {
              lat: (dataset.bounds.south + dataset.bounds.north) / 2,
              lon: (dataset.bounds.west + dataset.bounds.east) / 2,
            };
            visible.push({ productId: dataset.id, instrument: "CUSTOM", title: dataset.name, lat: center.lat, lon: center.lon });
          }
        }
      }

      // Only update if results changed (avoid unnecessary re-renders)
      const newHash = visible.map(p => p.productId).join(",");
      if (newHash !== lastResultHash) {
        lastResultHash = newHash;
        onVisibleProductsChange(visible);
      }
    };

    // Update on camera move end (main trigger)
    const removeListener = viewer.camera.moveEnd.addEventListener(updateVisibleProducts);

    // Initial update with delay for FootprintManager initialization
    const initTimeout = setTimeout(updateVisibleProducts, 500);

    // Run once immediately when deps change (e.g., footprint load, filter toggle)
    updateVisibleProducts();

    return () => {
      removeListener();
      clearTimeout(initTimeout);
    };
  }, [showHiRISE, showCRISM, showCTX, showHiRISEDTM, showCRISM_TRR3, showCustomData, customDatasets, onVisibleProductsChange, crismFilteredIds, overlapEnabled, footprintVersion]);

  // PERFORMANCE OPTIMIZED: Update overlay opacity when per-product opacities change
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Update material opacity for a specific product
    const updateMaterial = (ent: Cesium.Entity | undefined, productId: string) => {
      if (!ent?.rectangle?.material) return;
      const material = ent.rectangle.material as Cesium.ImageMaterialProperty;
      if (material.color) {
        const opacity = getProductOpacity(productId);
        material.color = new Cesium.ConstantProperty(Cesium.Color.WHITE.withAlpha(opacity));
      }
    };

    // Update quickview overlays (entities)
    for (const productId of quickviewOverlayIdsRef.current) {
      // CTX uses imagery layers, not entities
      const ctxLayer = ctxTileLayersRef.current.get(productId);
      if (ctxLayer) {
        ctxLayer.alpha = getProductOpacity(productId);
        continue;
      }
      updateMaterial(viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`), productId);
    }

    // Update high-res overlays
    for (const productId of highResOverlayIdsRef.current) {
      updateMaterial(viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`), productId);
    }

    // Update browse overlays
    for (const [productId, types] of browseOverlayIdsRef.current) {
      for (const browseType of types) {
        updateMaterial(viewer.entities.getById(`BROWSE_OVERLAY_${productId}_${browseType}`), productId);
      }
    }

    // Update score overlays
    for (const [productId, types] of scoreOverlayIdsRef.current) {
      for (const scoreType of types) {
        updateMaterial(viewer.entities.getById(`SCORE_OVERLAY_${productId}_${scoreType}`), productId);
      }
    }

    // Update mineral classification overlays
    for (const productId of mineralOverlayIdsRef.current) {
      updateMaterial(viewer.entities.getById(`MINERAL_OVERLAY_${productId}`), productId);
    }

    viewer.scene.requestRender();
  }, [overlayOpacities]);

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
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE) as any;
      } else {
        // Restore to original light fill using instrument color
        const rgb = getInstrumentCesiumColor(instrument.toLowerCase() as any);
        const baseColor = new Cesium.Color(rgb.r, rgb.g, rgb.b, 1.0);
        entity.rectangle.material = new Cesium.ColorMaterialProperty(baseColor.withAlpha(0.10));
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(baseColor) as any;
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
      if (labelEnt.label) (labelEnt.label.scale as any) = 1.3;
    }

    const pointEnt = viewer.entities.getById(`${instrument}_VP_POINT_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_POINT_${hoveredProductId}`);
    if (pointEnt?.point) {
      (pointEnt.point.pixelSize as any) = 10;
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
        if (labelEnt.label) (labelEnt.label.scale as any) = 1.0;
      }

      if (pointEnt?.point) {
        (pointEnt.point.pixelSize as any) = 6;
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
      const ent = viewer.entities.values[i];
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
      const ent = viewer.entities.values[i];
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
            console.warn(`[FieldNotes] Failed to fetch coords for ${instrument}:`, err);
          }
        })();
      }
    }

    viewer.scene.requestRender();

    return () => {
      cancelled = true;
    };
  }, [fieldNotes]);

  // Keep DTM hover mode ref in sync with state
  useEffect(() => {
    dtmHoverModeRef.current = dtmHoverMode;
  }, [dtmHoverMode]);

  // Sync activeDTMProductId prop with ref and load elevation grid
  useEffect(() => {
    if (activeDTMProductId) {
      activeDTMProductRef.current = activeDTMProductId;

      // Load elevation grid if not already cached
      if (!dtmGridCacheRef.current.has(activeDTMProductId)) {
        loadDTMElevationGrid(activeDTMProductId).then((grid) => {
          if (grid) {
            setDtmGrid(activeDTMProductId, grid);
          }
        });
      }
    } else {
      activeDTMProductRef.current = null;
    }
  }, [activeDTMProductId]);

  // Handle DTM hover mode change
  const handleDTMHoverModeChange = useCallback((mode: "hover" | "click") => {
    setDtmHoverMode(mode);
    if (mode === "click") {
      // Hide marker in click mode until clicked
      if (dtmHoverMarkerRef.current) {
        dtmHoverMarkerRef.current.show = false;
      }
      dtmHoverReadoutRef.current?.hide();
    }
  }, []);

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
        const ent = viewer!.entities.values[i];
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
        if (pts.length >= 2 && pts[pts.length - 2] < lineEast) {
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
        if (pts.length >= 2 && pts[pts.length - 1] < lineNorth) {
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
      <div ref={ref} className="absolute inset-0" />

      {/* Coordinate Display */}
      {hover && (
        <div className="absolute bottom-6 left-6 rounded-lg border border-border-dark bg-bg-dark/90 p-3 backdrop-blur-md">
          <div className="flex items-center gap-4">
            <div className="space-y-1">
              <div className="text-[9px] uppercase tracking-tighter text-slate-500">Longitude</div>
              <div className="font-mono text-xs">{hover.lon.toFixed(4)}°</div>
            </div>
            <div className="h-6 w-px bg-border-dark" />
            <div className="space-y-1">
              <div className="text-[9px] uppercase tracking-tighter text-slate-500">Latitude</div>
              <div className="font-mono text-xs">{hover.lat.toFixed(4)}°</div>
            </div>
          </div>
        </div>
      )}

      {/* Footprint loading indicators moved to LayerPanel */}

      {/* Score Overlay Legend */}
      {scoreOverlays.size > 0 && (
        <div className="absolute bottom-6 right-6 rounded-lg border border-border-dark bg-bg-dark/90 p-3 backdrop-blur-md">
          {/* Determine which score types are active */}
          {(() => {
            const activeTypes = new Set<ScoreProductType>();
            scoreOverlays.forEach((types) => types.forEach((t) => activeTypes.add(t)));
            return (
              <div className="space-y-2">
                <div className="text-[9px] uppercase tracking-widest text-slate-500 font-bold">
                  {activeTypes.has("score_ice") && activeTypes.has("score_hyd")
                    ? "Ice / Hydration Score"
                    : activeTypes.has("score_ice")
                    ? "Ice Score"
                    : "Hydration Score"}
                </div>
                {/* Gradient bar */}
                <div
                  className="h-2.5 w-40 rounded-sm border border-white/10"
                  style={{
                    background: "linear-gradient(to right, #ffffff 0%, #808080 25%, #000000 50%, #5a0000 75%, #b40000 100%)",
                  }}
                />
                {/* Tick labels */}
                <div className="flex justify-between text-[8px] font-mono text-slate-400 w-40">
                  <span>0</span>
                  <span>0.5</span>
                  <span>1.0</span>
                  <span>1.5</span>
                  <span>2.0</span>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* DTM Hover Readout - shows elevation on hover */}
      <DTMHoverReadout
        ref={dtmHoverReadoutRef}
        mode={dtmHoverMode}
        onModeChange={handleDTMHoverModeChange}
      />

      {/* Measurement & Annotation Tools */}
      <MeasurementTools
        viewer={viewerRef.current}
        isVisible={showMeasurementTools}
        onPinNote={onMeasurementPinNote}
      />
    </>
  );
}
