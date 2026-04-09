// src/MapView.tsx
import { useRef, useState, useEffect, useCallback } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { LRUMap } from "../utils/LRUMap";
import { normalizeLonForMap } from "../utils/coordinates";
import MapToolbar from "./map/MapToolbar";
import type { FieldNote } from "../api/fieldnotes";
import type { OverlapStats } from "../utils/overlapFilter";
import useDTMHover from "../hooks/useDTMHover";
import useMapViewer from "../hooks/useMapViewer";
import useFootprints from "../hooks/useFootprints";
import useOverlays from "../hooks/useOverlays";
import useMapLayers from "../hooks/useMapLayers";
import useFlyTo from "../hooks/useFlyTo";
import useViewBoundSelection from "../hooks/useViewBoundSelection";
import useAnnotations from "../hooks/useAnnotations";
import useHoverHighlight from "../hooks/useHoverHighlight";
import useGridOverlay from "../hooks/useGridOverlay";
import useCustomDatasets from "../hooks/useCustomDatasets";
import usePathfinderOverlay from "../hooks/usePathfinderOverlay";
import useRoverSimulation from "../hooks/useRoverSimulation";
import { useHighContrastMode } from "../hooks/useHighContrastMode";
import useMapKeyboard from "../hooks/useMapKeyboard";
import useBookmarks from "../hooks/useBookmarks";
import type { MapBookmark } from "../hooks/useBookmarks";
import type FootprintManager from "../utils/FootprintManager";
import CesiumLoadingPlaceholder from "./map/CesiumLoadingPlaceholder";
import FootprintLoadingOverlay from "./map/FootprintLoadingOverlay";
// ZoomGuide removed — footprints load on explicit button click
import BookmarkPanel from "./map/BookmarkPanel";
import { getEntityInstrument, getEntityProductId } from "../utils/cesiumEntityUtils";


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
type ExplicitLoadInstrument = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CTX_MOSAIC" | "HIRISE_DTM" | "CRISM_TRR3";

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
  analysisMode?: "slope" | "hirise_dtm_3d" | "line" | "ai_analysis" | "agentic" | "report" | "guided" | "region_stats" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column" | "pathfinder" | null;
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
  scienceLayerVisibility?: Record<string, boolean>;
  scienceLayerDepth?: string;
  scienceLayerOpacities?: Record<string, number>;
  onOlympusMonsTripleClick?: () => void;
  // Ice Accessibility heatmap overlay
  accessibilityVisible?: boolean;
  accessibilityOpacity?: number;
  // Ice Prospecting Fusion overlay
  fusionVisible?: boolean;
  fusionOpacity?: number;
  // CTX Mosaic overlay opacity (Murray Lab 5m, Arcadia Planitia)
  ctxMosaicOpacity?: number;
  // High-Res Only filter
  highResOnly?: boolean;
  onOlympusMonsClimber?: () => void;
  // Pathfinder route overlay
  pathfinderStart?: { lat: number; lon: number } | null;
  pathfinderGoal?: { lat: number; lon: number } | null;
  pathfinderRoute?: import("../api/pathfinder").RouteResult | null;
  // Rover simulation
  simPlaying?: boolean;
  simSpeed?: number;
  simCameraFollow?: boolean;
  simSeekTo?: number | null;
  onSimProgress?: (progress: number) => void;
  onSimTelemetry?: (telemetry: import("../hooks/useRoverSimulation").RoverTelemetry) => void;
  onSimComplete?: () => void;
};

/* ==================================================
 * Mars constants
 * ==================================================*/
// IAU-defined Mars ellipsoid parameters (meters)
// Source: IAU Working Group on Cartographic Coordinates and Rotational Elements
// These values are used by NASA PDS, HiRISE, CRISM, and NASA Trek base layers
const MARS_EQUATORIAL_RADIUS = 3396190; // meters (a = b axis)
const MARS_POLAR_RADIUS = 3376200;      // meters (c axis)

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
  polygon?: [number, number][];  // actual polygon corners [lon, lat][] for precise overlay
}
const boundsCache = new LRUMap<string, ProductBounds>(500);

async function loadHiRISELBL(id: string): Promise<string | null> {
  if (hiriseLBLCache.has(id)) return hiriseLBLCache.get(id)!;

  // Try multiple path patterns: subdirectory (new downloads) and flat (legacy)
  const patterns = [
    `${HIRISE_LBL_BASE}/${id}_RED/${id}_RED.LBL`,
    `${HIRISE_LBL_BASE}/${id}_RED.lbl`,
    `${HIRISE_LBL_BASE}/${id}_RED/${id}_RED.lbl`,
  ];

  for (const url of patterns) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const text = await res.text();
      if (!text.includes("IMAGE_MAP_PROJECTION")) continue;
      hiriseLBLCache.set(id, text);
      return text;
    } catch {
      continue;
    }
  }

  return null;
}

async function loadCRISMLBL(id: string): Promise<string | null> {
  if (crismLBLCache.has(id)) return crismLBLCache.get(id)!;
  const CRISM_LBL_BASE = "/crism_lbl";

  // Extract base_key from product_id
  const baseKey = id.replace(/_(?:if|br)[0-9a-z]+_mtr3$/i, "");

  const patterns = [
    `${CRISM_LBL_BASE}/${baseKey}_brcarj_mtr3.lbl`,
    `${CRISM_LBL_BASE}/${baseKey}.lbl`,
    `${CRISM_LBL_BASE}/${id}.lbl`,
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

// Cache for HiRISE index
let hiriseIndexCache: GeoJsonIndex | null = null;
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
          const coords = feature.geometry?.coordinates?.[0] as [number, number][] | undefined;
          if (!coords || coords.length < 4) continue;
          const bbox = boundsFromPolygon(coords);
          if (bbox) {
            const bounds: ProductBounds = {
              ...bbox,
              polygon: coords.slice(0, -1),  // strip closing vertex
            };
            boundsCache.set(productId, bounds);
            return bounds;
          }
        }
      }
    }
    return null;
  }

  // Check for HiRISE products — bounds from index.geojson polygon
  const isHiRISE = productId.startsWith("ESP_") || productId.startsWith("PSP_");
  if (isHiRISE) {
    if (!hiriseIndexCache) {
      try {
        const res = await fetch("/hirise_index.geojson");
        if (res.ok) {
          hiriseIndexCache = await res.json();
        }
      } catch {
        // fall through to LBL parsing
      }
    }

    if (hiriseIndexCache?.features) {
      for (const feature of hiriseIndexCache.features) {
        if (feature.properties?.product_id === productId) {
          const props = feature.properties;
          const coords = feature.geometry?.coordinates?.[0] as [number, number][] | undefined;
          // Use polygon coordinates for precise overlay placement
          if (coords && coords.length >= 4) {
            const bbox = boundsFromPolygon(coords);
            if (bbox) {
              const bounds: ProductBounds = {
                ...bbox,
                polygon: coords.slice(0, -1),  // strip closing vertex
              };
              boundsCache.set(productId, bounds);
              return bounds;
            }
          }
          // Fallback to LBL bbox properties
          if (props.west != null && props.east != null && props.south != null && props.north != null) {
            const bounds: ProductBounds = {
              west: props.west, east: props.east,
              south: props.south, north: props.north,
            };
            boundsCache.set(productId, bounds);
            return bounds;
          }
          break;
        }
      }
    }
  }

  // Fallback: parse LBL file for precise bounds
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
function getFootprintBounds(
  viewer: Cesium.Viewer,
  productId: string,
  footprintManager?: FootprintManager | null,
): ProductBounds | null {
  const prefixes: string[] = ["CRISM", "HIRISE", "CTX", "CTX_MOSAIC", "HIRISE_DTM", "CRISM_TRR3", "SHARAD", "SHARAD_HIGHRES"];
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

  if (footprintManager) {
    for (const prefix of prefixes) {
      const bounds = footprintManager.getFeatureBounds(`${prefix}_FP_${productId}`);
      if (bounds) {
        boundsCache.set(productId, bounds);
        return bounds;
      }
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
  scienceLayerVisibility = {},
  scienceLayerDepth = "1-5m",
  scienceLayerOpacities = {},
  accessibilityVisible = false,
  accessibilityOpacity = 0.6,
  fusionVisible = false,
  fusionOpacity = 0.6,
  ctxMosaicOpacity: _ctxMosaicOpacity = 1.0,
  highResOnly = false,
  onOlympusMonsTripleClick,
  onOlympusMonsClimber,
  pathfinderStart = null,
  pathfinderGoal = null,
  pathfinderRoute = null,
  simPlaying = false,
  simSpeed = 1,
  simCameraFollow = false,
  simSeekTo = null,
  onSimProgress,
  onSimTelemetry,
  onSimComplete,
}: MapViewProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  // ── Standalone component state ─────────────────
  const [cesiumReady, setCesiumReady] = useState(false);
  const [footprintLoading, setFootprintLoading] = useState(false);
  const [footprintLoadingInstrument, setFootprintLoadingInstrument] = useState<string | undefined>(undefined);
  // cameraHeight state removed — ZoomGuide was removed
  const [bookmarkPanelOpen, setBookmarkPanelOpen] = useState(false);
  const sharedFootprintManagerRef = useRef<FootprintManager | null>(null);

  // ── High contrast mode ──────────────────────────
  const { isActive: highContrastActive, toggle: toggleHighContrast } = useHighContrastMode();

  // ── Bookmarks ──────────────────────────────────
  const { bookmarks, addBookmark, removeBookmark, renameBookmark } = useBookmarks();

  // ── Intercept onFootprintsLoading for local state ──
  const handleFootprintsLoading = useCallback(
    (instrument: ExplicitLoadInstrument, loading: boolean) => {
      setFootprintLoading(loading);
      setFootprintLoadingInstrument(loading ? instrument : undefined);
      onFootprintsLoading?.(instrument, loading);
    },
    [onFootprintsLoading],
  );

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

  const { viewerRef, hover, initError } = useMapViewer({
    containerRef: ref,
    mapMode,
    baseLayer,
    viewBounds,
    marsEllipsoid: MARS_ELLIPSOID,
    marsRect: MARS_RECT,
    baseLayerUrls: BASE_LAYER_URLS,
    quickviewOverlays,
    highResOverlays,
    footprintManagerRef: sharedFootprintManagerRef,
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

  // ── Detect Cesium viewer initialization ────────
  useEffect(() => {
    if (cesiumReady) return;
    const interval = setInterval(() => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        setCesiumReady(true);
        clearInterval(interval);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [cesiumReady, viewerRef]);

  // Camera height tracking removed (ZoomGuide removed)

  const handleAddBookmark = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;
    const cart = viewer.camera.positionCartographic;
    const lat = Cesium.Math.toDegrees(cart.latitude);
    const lon = Cesium.Math.toDegrees(cart.longitude);
    const height = cart.height;
    addBookmark(`Bookmark ${bookmarks.length + 1}`, lat, lon, height);
  }, [viewerRef, addBookmark, bookmarks.length]);

  const handleSelectBookmark = useCallback(
    (bookmark: MapBookmark) => {
      const viewer = viewerRef.current;
      if (!viewer || viewer.isDestroyed()) return;
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(
          bookmark.lon,
          bookmark.lat,
          bookmark.height,
          MARS_ELLIPSOID,
        ),
        duration: 1.5,
      });
      setBookmarkPanelOpen(false);
    },
    [viewerRef],
  );

  // ── Keyboard shortcuts ──────────────────────────
  useMapKeyboard({
    viewer: viewerRef.current,
    onAddBookmark: handleAddBookmark,
    enabled: cesiumReady,
  });

  const { footprintManagerRef } = useFootprints({
    footprintManagerRef: sharedFootprintManagerRef,
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
    onFootprintsLoading: handleFootprintsLoading,
    overlapFilter,
    onOverlapStatsChange,
    inspectedProductId,
    onVisibleProductsChange,
    showCustomData,
    customDatasets,
    showRegionLayer,
    extractCrismObsId,
    highResOnly,
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
    getFootprintBounds: (viewer, productId) =>
      getFootprintBounds(viewer, productId, footprintManagerRef.current),
    loadCRISMLBL,
    parseLBLValue,
    normalizeLonTo180,
  });

  // ── Extracted hooks ──────────────────────────────

  useMapLayers({
    viewerRef,
    swimLayer,
    scienceLayerVisibility,
    scienceLayerDepth,
    scienceLayerOpacities,
    accessibilityVisible,
    accessibilityOpacity,
    fusionVisible,
    fusionOpacity,
  });

  useFlyTo({
    viewerRef,
    marsEllipsoid: MARS_ELLIPSOID,
    flyToProductId: flyToProductId ?? null,
    onFlyToComplete,
    flyToCoords: flyToCoords ?? null,
    onFlyToCoordsComplete,
    bringToFrontId: bringToFrontId ?? null,
    onBringToFrontComplete,
    highlightProductId: highlightProductId ?? null,
    onHighlightComplete,
    onSelect,
    onToggleOverlay,
    paddedRectangle,
    normalizeLonTo180,
    parseLBLValue,
    loadHiRISELBL,
    loadCRISMLBL,
  });

  useViewBoundSelection({
    viewerRef,
    marsEllipsoid: MARS_ELLIPSOID,
    viewBoundSelectionMode,
    onViewBoundSelected,
  });

  useAnnotations({
    viewerRef,
    marsEllipsoid: MARS_ELLIPSOID,
    analysisMode,
    linePoints,
    fieldNotes,
    aiAnalysisPin,
    sharadTracePin,
    terraformMode,
    craterDetectFeatures,
  });

  usePathfinderOverlay({
    viewerRef,
    marsEllipsoid: MARS_ELLIPSOID,
    analysisMode,
    startPoint: pathfinderStart,
    goalPoint: pathfinderGoal,
    routeResult: pathfinderRoute,
  });

  useRoverSimulation({
    viewerRef,
    marsEllipsoid: MARS_ELLIPSOID,
    routeResult: pathfinderRoute,
    vlmAnalysis: pathfinderRoute?.vlm_analysis,
    analysisMode,
    isPlaying: simPlaying,
    speed: (simSpeed as 1 | 2 | 5 | 10) || 1,
    cameraFollow: simCameraFollow,
    seekTo: simSeekTo,
    onProgress: onSimProgress,
    onTelemetry: onSimTelemetry,
    onComplete: onSimComplete,
  });

  useHoverHighlight({
    viewerRef,
    footprintManagerRef,
    hoveredProductId,
    onHoverProduct,
  });

  useGridOverlay({
    viewerRef,
    showGrid,
  });

  useCustomDatasets({
    viewerRef,
    showCustomData,
    customDatasets,
  });

  if (initError) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0a0f18] text-center gap-4 p-8">
        <div className="text-amber-400 text-lg font-medium">Map failed to load</div>
        <p className="text-slate-400 text-sm max-w-md">
          The 3D map viewer could not initialize. This may be due to WebGL not being available in your browser.
        </p>
        <pre className="text-xs text-slate-600 max-w-lg overflow-auto bg-slate-900/50 rounded p-2">{initError}</pre>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 text-sm rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
        >
          Reload page
        </button>
      </div>
    );
  }

  return (
    <>
      {/* P1: Cesium loading placeholder — visible until viewer initializes */}
      <CesiumLoadingPlaceholder visible={!cesiumReady} />

      <div ref={ref} className="absolute inset-0" role="application" aria-label="Mars 3D Globe" />

      {/* D3: Zoom guide — visible when camera is too far out to see footprints */}
      {/* ZoomGuide removed — footprints load on explicit button click */}

      {/* D2: Footprint loading overlay — shows active loading status */}
      <FootprintLoadingOverlay
        loading={footprintLoading}
        instrument={footprintLoadingInstrument}
      />

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

      {/* D4: Mini layer toggle indicators */}
      <div className="absolute top-1/2 -translate-y-1/2 right-4 z-20 flex flex-col gap-1">
        {[
          { key: "crism", label: "CR", show: showCRISM, color: "#e879f9" },
          { key: "hirise", label: "Hi", show: showHiRISE, color: "#4ade80" },
          { key: "ctx", label: "CX", show: showCTX, color: "#facc15" },
          { key: "sharad", label: "SH", show: showSHARAD, color: "#22d3ee" },
        ].map(({ key, label, show, color }) => (
          <div
            key={key}
            className={`flex h-7 w-7 items-center justify-center rounded-md border text-[9px] font-bold transition-all ${
              show
                ? "border-white/20 bg-surface-dark/90 text-white shadow-sm"
                : "border-transparent bg-transparent text-slate-600"
            }`}
            style={show ? { borderLeftColor: color, borderLeftWidth: "2px" } : undefined}
            title={`${key.toUpperCase()} footprints ${show ? "visible" : "hidden"}`}
          >
            {label}
          </div>
        ))}

        {/* D5: High contrast mode toggle */}
        <button
          onClick={toggleHighContrast}
          className={`flex h-7 w-7 items-center justify-center rounded-md border text-[9px] font-bold transition-all mt-2 ${
            highContrastActive
              ? "border-amber-400/50 bg-surface-dark/90 text-amber-400 shadow-sm"
              : "border-transparent bg-transparent text-slate-600 hover:text-slate-400"
          }`}
          title={`High contrast mode ${highContrastActive ? "ON" : "OFF"}`}
        >
          <span className="material-symbols-outlined text-[14px]">contrast</span>
        </button>

        {/* P5: Bookmark button */}
        <button
          onClick={() => setBookmarkPanelOpen((prev) => !prev)}
          className={`flex h-7 w-7 items-center justify-center rounded-md border text-[9px] font-bold transition-all ${
            bookmarkPanelOpen
              ? "border-amber-400/50 bg-surface-dark/90 text-amber-400 shadow-sm"
              : "border-transparent bg-transparent text-slate-600 hover:text-slate-400"
          }`}
          title="Bookmarks (B)"
        >
          <span className="material-symbols-outlined text-[14px]">star</span>
        </button>
      </div>

      {/* P5: Bookmark panel */}
      <BookmarkPanel
        bookmarks={bookmarks}
        onSelect={handleSelectBookmark}
        onRemove={removeBookmark}
        onRename={renameBookmark}
        isOpen={bookmarkPanelOpen}
        onClose={() => setBookmarkPanelOpen(false)}
      />
    </>
  );
}
