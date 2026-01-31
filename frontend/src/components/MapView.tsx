// src/MapView.tsx
import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

/* ==================================================
 * Types
 * ==================================================*/
type LatLon = { lat: number; lon: number };

export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD";

export type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  // CRISM pixel coordinates for spectrum (optional)
  pixelLine?: number;
  pixelSample?: number;
};

type VisibleProduct = {
  productId: string;
  instrument: InstrumentType;
  title?: string;
};

type RGBWavelengths = {
  r: number;
  g: number;
  b: number;
};

type BrowseProductType = "HYD" | "ICE" | "IC2";

type BaseLayerType = "MOLA" | "HRSC";

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

type MapViewProps = {
  baseLayer?: BaseLayerType;
  viewBounds?: BoundingBox;
  onSelect: (ctx: InspectorContext | null) => void;
  showCRISM: boolean;
  showHiRISE: boolean;
  showSHARAD: boolean;
  onSharadClick?: (popup: SHARADPopup) => void;
  quickviewOverlays?: string[];
  highResOverlays?: string[];
  browseOverlays?: Map<string, Set<BrowseProductType>>;
  overlayOpacity?: number; // 0-1
  onVisibleProductsChange?: (products: VisibleProduct[]) => void;
  flyToProductId?: string | null;
  onFlyToComplete?: () => void;
  bringToFrontId?: string | null;
  onBringToFrontComplete?: () => void;
  rgbWavelengths?: RGBWavelengths;
  // Score map toggles
  showIceScore?: boolean;
  showHydratedScore?: boolean;
  // ICE filter - set of obs_ids that pass the filter (null = no filter)
  iceFilterPassingObs?: Set<string> | null;
};

/* ==================================================
 * Mars constants
 * ==================================================*/
const MARS_RADIUS = 3389500;
const MARS_ELLIPSOID = new Cesium.Ellipsoid(
  MARS_RADIUS,
  MARS_RADIUS,
  MARS_RADIUS
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

function normalizeLonTo180(lon360: number) {
  return lon360 > 180 ? lon360 - 360 : lon360;
}

/* ==================================================
 * ID helpers
 * ==================================================*/
function getCRISMId(e: Cesium.Entity): string | null {
  const p: any = e.properties;
  if (!p) return null;

  const candidates = [p.product_id, p.PRODUCT_ID, p.id, p.name];

  for (const c of candidates) {
    const v = c?.getValue?.();
    if (typeof v === "string") return v;
  }
  return null;
}

function getHiRISEId(e: Cesium.Entity): string | null {
  const p: any = e.properties;
  if (!p) return null;

  const candidates = [
    p.product_id,
    p.PRODUCT_ID,
    p._product_id,
    p.id,
    p.OBSERVATION_ID,
    p.name,
  ];

  for (const c of candidates) {
    const v = c?.getValue?.();
    if (typeof v === "string" && v.startsWith("ESP_")) {
      return v;
    }
  }
  return null;
}

/* ==================================================
 * LBL loaders
 * ==================================================*/
const hiriseLBLCache = new Map<string, string>();
const crismLBLCache = new Map<string, string>();
const HIRISE_LBL_BASE = "/hirise_lbl";

async function loadHiRISELBL(id: string): Promise<string | null> {
  if (hiriseLBLCache.has(id)) return hiriseLBLCache.get(id)!;

  const res = await fetch(`${HIRISE_LBL_BASE}/${id}_RED.lbl`);
  if (!res.ok) {
    console.warn("[DEBUG][HIRISE] fetch failed", id, res.status);
    return null;
  }

  const text = await res.text();
  if (!text.includes("IMAGE_MAP_PROJECTION")) {
    console.warn("[DEBUG][HIRISE] no IMAGE_MAP_PROJECTION", id);
    return null;
  }

  hiriseLBLCache.set(id, text);
  return text;
}

async function loadCRISMLBL(id: string): Promise<string | null> {
  if (crismLBLCache.has(id)) return crismLBLCache.get(id)!;
  const CRISM_LBL_BASE = "/crism_lbl";
  const res = await fetch(`${CRISM_LBL_BASE}/${id}.lbl`);
  if (!res.ok) {
    console.warn("[DEBUG][CRISM] fetch failed", id, res.status);
    return null;
  }

  const text = await res.text();
  if (!text.includes("IMAGE_MAP_PROJECTION")) {
    console.warn("[DEBUG][CRISM] no IMAGE_MAP_PROJECTION", id);
    return null;
  }

  crismLBLCache.set(id, text);
  return text;
}

/* ==================================================
 * Footprint from LBL
 * ==================================================*/
function rectanglesFromLBL_BBOX(lbl: string): Cesium.Rectangle[] {
  const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
  const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
  const westLon = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
  const eastLon = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

  if (
    minLat == null ||
    maxLat == null ||
    westLon == null ||
    eastLon == null
  ) {
    console.warn("[DEBUG][BBOX] missing values");
    return [];
  }

  const south = Math.min(minLat, maxLat);
  const north = Math.max(minLat, maxLat);

  // Normalize to -180 to 180 range
  const west = normalizeLonTo180(westLon);
  const east = normalizeLonTo180(eastLon);

  // Calculate width going eastward from west to east
  let width = east - west;
  if (width < 0) width += 360; // Handle wrap-around

  // CRISM footprints are small (typically < 1°). If apparent width > 180°,
  // the footprint crosses the antimeridian and we should go the "short way"
  const crossesAntimeridian = width > 180;

  if (!crossesAntimeridian) {
    // Simple case: footprint doesn't cross antimeridian
    // Just need to handle if west > east after normalization
    if (west <= east) {
      return [Cesium.Rectangle.fromDegrees(west, south, east, north)];
    } else {
      // west > east means it crosses antimeridian going eastward
      return [
        Cesium.Rectangle.fromDegrees(west, south, 180, north),
        Cesium.Rectangle.fromDegrees(-180, south, east, north),
      ];
    }
  }

  // Footprint crosses antimeridian the "short way"
  // Example: west=-179.87, east=179.81
  // The footprint is near the antimeridian, spanning from east (179.81) to west (-179.87)
  // going through 180°/-180°
  // Split into two rectangles:
  // 1. From east (179.81) to 180°
  // 2. From -180° to west (-179.87)
  return [
    Cesium.Rectangle.fromDegrees(east, south, 180, north),
    Cesium.Rectangle.fromDegrees(-180, south, west, north),
  ];
}

/* ==================================================
 * (추가) UI helpers: label/point/hover highlight
 * ==================================================*/
function rectCenterCartesian(rect: Cesium.Rectangle): Cesium.Cartesian3 {
  const c = Cesium.Rectangle.center(rect); // Cartographic (radians)
  return Cesium.Cartesian3.fromRadians(
    c.longitude,
    c.latitude,
    0,
    MARS_ELLIPSOID
  );
}

function addFootprintLabelAndPoint(params: {
  viewer: Cesium.Viewer;
  id: string;
  instrument: InstrumentType;
  rect: Cesium.Rectangle;
  show: boolean;
  color: Cesium.Color;
}): { labelEnt: Cesium.Entity; pointEnt: Cesium.Entity } {
  const { viewer, id, instrument, rect, show, color } = params;

  const pos = rectCenterCartesian(rect);

  const pointEnt = viewer.entities.add({
    id: `${instrument}_POINT_${id}`,
    show,
    position: pos,
    point: {
      pixelSize: 6,
      color,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 1,
      disableDepthTestDistance: Infinity,
    },
    properties: {
      product_id: id,
      instrument,
      kind: "FOOTPRINT_POINT",
    },
  });

  const labelEnt = viewer.entities.add({
    id: `${instrument}_LABEL_${id}`,
    show,
    position: pos,
    label: {
      text: id,
      font: "12px sans-serif",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      pixelOffset: new Cesium.Cartesian2(0, -12),
      verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
      disableDepthTestDistance: Infinity,
      distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, 9.0e7),

      // (선택) label이 거리 따라 너무 커지지 않게 하려면 scaleByDistance도 가능하지만
      // 지금은 "무한 커짐" 원인이 누적 scale이므로 여기서 건드리지 않음.
    },
    properties: {
      product_id: id,
      instrument,
      kind: "FOOTPRINT_LABEL",
    },
  });

  return { labelEnt, pointEnt };
}

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

function getEntityInstrument(e: Cesium.Entity): InstrumentType | null {
  const p: any = e.properties;
  const inst = p?.instrument?.getValue?.();
  return inst === "HIRISE" || inst === "CRISM" ? inst : null;
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
  baseLayer = "MOLA",
  viewBounds,
  onSelect,
  showCRISM,
  showHiRISE,
  showSHARAD,
  onSharadClick,
  quickviewOverlays = [],
  highResOverlays = [],
  browseOverlays = new Map(),
  overlayOpacity = 0.8,
  onVisibleProductsChange,
  flyToProductId,
  onFlyToComplete,
  bringToFrontId,
  onBringToFrontComplete,
  rgbWavelengths = { r: 2.53, g: 1.51, b: 1.08 },
  showIceScore = false,
  showHydratedScore = false,
  iceFilterPassingObs = null,
}: MapViewProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  const hiriseEntitiesRef = useRef<Cesium.Entity[]>([]);
  const crismEntitiesRef = useRef<Cesium.Entity[]>([]);
  const sharadEntitiesRef = useRef<Cesium.Entity[]>([]);
  const overlayLayerRef = useRef<Cesium.ImageryLayer | null>(null);

  // Score map overlay entities
  const iceScoreEntitiesRef = useRef<Cesium.Entity[]>([]);
  const hydrationScoreEntitiesRef = useRef<Cesium.Entity[]>([]);
  // Store CRISM feature data for score maps (use state so useEffects re-run when loaded)
  const [crismScoreFeatures, setCrismScoreFeatures] = useState<any[]>([]);

  // Refs to track current overlay lists for click handler
  const quickviewOverlaysRef = useRef<string[]>(quickviewOverlays);
  const highResOverlaysRef = useRef<string[]>(highResOverlays);
  const onSharadClickRef = useRef(onSharadClick);

  // Keep refs in sync with props
  useEffect(() => {
    quickviewOverlaysRef.current = quickviewOverlays;
  }, [quickviewOverlays]);

  useEffect(() => {
    highResOverlaysRef.current = highResOverlays;
  }, [highResOverlays]);

  useEffect(() => {
    onSharadClickRef.current = onSharadClick;
  }, [onSharadClick]);

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
  const [crismDisclaimer, setCrismDisclaimer] = useState<{ displayed: number; total: number } | null>(null);
  const [viewerReady, setViewerReady] = useState(false);
  const [crismEntitiesLoaded, setCrismEntitiesLoaded] = useState(false);

  useEffect(() => {
    if (!ref.current) return;

    try {
      (Cesium as any).buildModuleUrl?.setBaseUrl?.("/cesium/");
    } catch {}

    const viewer = new Cesium.Viewer(ref.current, {
      sceneMode: Cesium.SceneMode.SCENE2D,
      mapProjection: new Cesium.GeographicProjection(MARS_ELLIPSOID),
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
    setViewerReady(true);

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
    // =================

    // Load HiRISE index - use GeoJSON geometry directly when available
    fetch("/hirise_index.geojson")
      .then((res) => res.json())
      .then(async (geojson: any) => {
        const features = geojson.features || [];
        console.log("[DEBUG][HIRISE] GeoJSON features:", features.length);

        let loadedCount = 0;
        let lblFallbackCount = 0;

        for (const feature of features) {
          const props = feature.properties || {};
          const id = props.product_id;
          const title = props.title;  // HiRISE observation title
          if (!id) continue;

          let rects: Cesium.Rectangle[] = [];

          // Check if GeoJSON has valid polygon geometry
          const geom = feature.geometry;
          if (geom?.type === "Polygon" && geom.coordinates?.[0]?.length >= 4) {
            // Use coordinates directly from GeoJSON
            const coords = geom.coordinates[0];
            // Extract bounds from polygon coordinates (normalize to -180 to 180)
            let minLon = Infinity, maxLon = -Infinity;
            let minLat = Infinity, maxLat = -Infinity;
            for (const [lon, lat] of coords) {
              const normLon = normalizeLonTo180(lon);
              if (normLon < minLon) minLon = normLon;
              if (normLon > maxLon) maxLon = normLon;
              if (lat < minLat) minLat = lat;
              if (lat > maxLat) maxLat = lat;
            }
            // Skip invalid coordinates (like [0,0] placeholders)
            if (minLon !== 0 || maxLon !== 0 || minLat !== 0 || maxLat !== 0) {
              const rect = Cesium.Rectangle.fromDegrees(minLon, minLat, maxLon, maxLat);
              rects = [rect];
            }
          }

          // Fallback to LBL if no valid geometry
          if (rects.length === 0) {
            lblFallbackCount++;
            const lbl = await loadHiRISELBL(id);
            if (!lbl) {
              console.warn("[HIRISE] skip footprint, no LBL:", id);
              continue;
            }
            rects = rectanglesFromLBL_BBOX(lbl);
          }

          if (rects.length === 0) continue;

          loadedCount++;
          rects.forEach((rect, i) => {
            const ent = viewer.entities.add({
              id: `HIRISE_${id}_${i}`,
              show: false, // Will be toggled by effect
              rectangle: {
                coordinates: rect,
                material: Cesium.Color.YELLOW.withAlpha(0.3),
                outline: true,
                outlineColor: Cesium.Color.YELLOW,
                height: 0,
              },
              properties: {
                product_id: id,
                instrument: "HIRISE",
                title: title,  // Store title for inspector
              },
            });
            hiriseEntitiesRef.current.push(ent);

            const { labelEnt, pointEnt } = addFootprintLabelAndPoint({
              viewer,
              id,
              instrument: "HIRISE",
              rect,
              show: false, // Will be toggled by effect
              color: Cesium.Color.YELLOW,
            });
            hiriseEntitiesRef.current.push(labelEnt);
            hiriseEntitiesRef.current.push(pointEnt);
          });
        }

        console.log(`[HIRISE] Loaded ${loadedCount} footprints (${lblFallbackCount} needed LBL fallback)`);
        viewer.scene.requestRender();
      });

    // Load CRISM index and create footprints
    const CRISM_MAX_DISPLAY = 2000;
    // Add cache-busting parameter to ensure fresh data
    fetch(`/crism_index.geojson?_=${Date.now()}`)
      .then((res) => res.json())
      .then((geojson: any) => {
        const totalFeatures = geojson.features?.length || 0;
        console.log("[DEBUG][CRISM] Total GeoJSON features:", totalFeatures);

        // Limit to max display
        const features = (geojson.features || []).slice(0, CRISM_MAX_DISPLAY);
        let crismCount = 0;

        for (const feature of features) {
          const props = feature.properties || {};
          const id = props.product_id;
          if (!id) continue;

          const geom = feature.geometry;
          let rects: Cesium.Rectangle[] = [];

          // Try to get coordinates from GeoJSON geometry directly
          if (geom?.type === "Polygon" && geom.coordinates?.[0]?.length >= 4) {
            const coords = geom.coordinates[0];
            // Normalize longitudes to -180 to 180 range
            const lons = coords.map((c: number[]) => normalizeLonTo180(c[0]));
            const lats = coords.map((c: number[]) => c[1]);
            const west = Math.min(...lons);
            const east = Math.max(...lons);
            const south = Math.min(...lats);
            const north = Math.max(...lats);

            // Check if coordinates are valid (not at 0,0)
            if (west === 0 && east === 0 && south === 0 && north === 0) continue;

            // Check for antimeridian crossing (width > 180° indicates wrap-around)
            const width = east - west;
            if (width > 180) {
              // Split into two rectangles
              rects = [
                Cesium.Rectangle.fromDegrees(east, south, 180, north),
                Cesium.Rectangle.fromDegrees(-180, south, west, north),
              ];
            } else {
              rects = [Cesium.Rectangle.fromDegrees(west, south, east, north)];
            }
          }

          // Skip if no valid geometry
          if (rects.length === 0) continue;

          crismCount++;
          rects.forEach((rect, i) => {
            const ent = viewer.entities.add({
              id: `CRISM_${id}_${i}`,
              show: false, // Will be toggled by effect
              rectangle: {
                coordinates: rect,
                material: Cesium.Color.CYAN.withAlpha(0.35),
                outline: true,
                outlineColor: Cesium.Color.BLACK,
                height: 0,
              },
              properties: {
                product_id: id,
                instrument: "CRISM",
              },
            });
            crismEntitiesRef.current.push(ent);

            const { labelEnt, pointEnt } = addFootprintLabelAndPoint({
              viewer,
              id,
              instrument: "CRISM",
              rect,
              show: false, // Will be toggled by effect
              color: Cesium.Color.CYAN,
            });
            crismEntitiesRef.current.push(labelEnt);
            crismEntitiesRef.current.push(pointEnt);
          });
        }

        // Set disclaimer if there are more features than displayed
        if (totalFeatures > crismCount) {
          setCrismDisclaimer({ displayed: crismCount, total: totalFeatures });
        }

        // Store features for score map overlays (only those with score maps)
        const scoreFeatures = features.filter((f: any) => {
          const props = f.properties || {};
          return props.hydration_score_map || props.ice_score_map;
        });
        setCrismScoreFeatures(scoreFeatures);
        console.log("[DEBUG][CRISM] Features with score maps:", scoreFeatures.length);
        if (scoreFeatures.length > 0) {
          console.log("[DEBUG][CRISM] Sample score feature:", scoreFeatures[0].properties?.product_id,
            "ice:", scoreFeatures[0].properties?.ice_score_map,
            "hyd:", scoreFeatures[0].properties?.hydration_score_map);
        } else {
          console.log("[DEBUG][CRISM] No score features found! First feature props:", features[0]?.properties);
        }

        console.log("[DEBUG][CRISM] Created", crismEntitiesRef.current.length, "entities (displayed:", crismCount, "of", totalFeatures, ")");
        setCrismEntitiesLoaded(true);
        viewer.scene.requestRender();
      });

    // Load SHARAD index and create polylines
    fetch("/sharad_index.geojson")
      .then((res) => res.json())
      .then((geojson: any) => {
        console.log("[DEBUG][SHARAD] GeoJSON features:", geojson.features?.length);

        for (const feature of geojson.features || []) {
          const props = feature.properties || {};
          const id = props.product_id;
          if (!id) continue;

          const geom = feature.geometry;
          if (geom?.type !== "LineString" || !geom.coordinates?.length) continue;

          // Create polyline from LineString coordinates (normalize lon to -180 to 180)
          const positions = geom.coordinates.map((coord: number[]) =>
            Cesium.Cartesian3.fromDegrees(normalizeLonTo180(coord[0]), coord[1], 0, MARS_ELLIPSOID)
          );

          const ent = viewer.entities.add({
            id: `SHARAD_${id}`,
            show: false, // Will be toggled by effect
            polyline: {
              positions,
              width: 3,
              material: Cesium.Color.ORANGE.withAlpha(0.8),
              clampToGround: true,
            },
            properties: {
              product_id: id,
              instrument: "SHARAD",
              start_lat: props.start_lat,
              start_lon: props.start_lon,
              stop_lat: props.stop_lat,
              stop_lon: props.stop_lon,
            },
          });
          sharadEntitiesRef.current.push(ent);

          // Add label at midpoint of line (normalize lon to -180 to 180)
          const midLon = normalizeLonTo180((props.start_lon + props.stop_lon) / 2);
          const midLat = (props.start_lat + props.stop_lat) / 2;
          const labelPos = Cesium.Cartesian3.fromDegrees(midLon, midLat, 0, MARS_ELLIPSOID);

          const labelEnt = viewer.entities.add({
            id: `SHARAD_LABEL_${id}`,
            show: false, // Will be toggled by effect
            position: labelPos,
            label: {
              text: id,
              font: "11px sans-serif",
              fillColor: Cesium.Color.ORANGE,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -10),
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              disableDepthTestDistance: Infinity,
              distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, 5.0e6),
            },
            properties: {
              product_id: id,
              instrument: "SHARAD",
              kind: "FOOTPRINT_LABEL",
            },
          });
          sharadEntitiesRef.current.push(labelEnt);
        }

        console.log("[DEBUG][SHARAD] Created", sharadEntitiesRef.current.length, "entities");
        viewer.scene.requestRender();
      })
      .catch((err) => console.warn("[SHARAD] Failed to load index:", err));

    const hoverHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    hoverHandler.setInputAction(
      (m) => {
        const p = viewer.camera.pickEllipsoid(m.endPosition, MARS_ELLIPSOID);
        if (!p) return setHover(null);
        const c = Cesium.Cartographic.fromCartesian(p);
        const hoverLat = Cesium.Math.toDegrees(c.latitude);
        const hoverLon = Cesium.Math.toDegrees(c.longitude);
        setHover({ lat: hoverLat, lon: hoverLon });

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
          .drillPick(m.endPosition)
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
            hs.labelEnt.label.scale = hs.origLabelScale;
          }
          if (hs.pointEnt?.point && typeof hs.origPointSize === "number") {
            hs.pointEnt.point.pixelSize = hs.origPointSize;
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

            // ✅ 같은 대상이면 재적용 금지 (무한 scale 누적 버그 근본 차단)
            if (hs.key === key) return;

            const rectFallback =
              viewer.entities.getById(`${inst === "HIRISE" ? "HIRISE" : "CRISM"}_${pid}_0`) ||
              viewer.entities.getById(`${inst === "HIRISE" ? "HIRISE" : "CRISM"}_${pid}_1`) ||
              viewer.entities.getById(`${inst === "HIRISE" ? "HIRISE" : "CRISM"}_${pid}_2`) ||
              viewer.entities.getById(`${inst === "HIRISE" ? "HIRISE" : "CRISM"}_${pid}_3`) ||
              null;

            const rectTarget =
              rectFallback && (rectFallback as any).rectangle ? rectFallback : null;

            const labelEnt = viewer.entities.getById(`${inst}_LABEL_${pid}`) || null;
            const pointEnt = viewer.entities.getById(`${inst}_POINT_${pid}`) || null;

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

              rectTarget.rectangle.material =
                inst === "HIRISE"
                  ? HILITE_RECT_MATERIAL_HIRISE
                  : HILITE_RECT_MATERIAL_CRISM;

              rectTarget.rectangle.outlineColor = new Cesium.ConstantProperty(
                Cesium.Color.WHITE
              ) as any;

              // ✅ label scale은 "곱하기"가 아니라 "고정값"으로 (누적 방지)
              if (labelEnt?.label) {
                hs.labelEnt = labelEnt;
                const cur = (labelEnt.label.scale as any);
                hs.origLabelScale = typeof cur === "number" ? cur : 1.0;
                labelEnt.label.scale = 1.2; // <- 고정
              }

              if (pointEnt?.point) {
                hs.pointEnt = pointEnt;
                const cur = (pointEnt.point.pixelSize as any);
                hs.origPointSize = typeof cur === "number" ? cur : 6;
                pointEnt.point.pixelSize = (hs.origPointSize ?? 6) + 2;
              }

              viewer.scene.requestRender();
              return;
            }
          }
        }

        if (hs.rectEnt || hs.labelEnt || hs.pointEnt) {
          clearHighlight();
          viewer.scene.requestRender();
        }
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE
    );

    const clickHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    clickHandler.setInputAction(
      async (m) => {
        // Get click position in lat/lon FIRST
        const clickCart = viewer.camera.pickEllipsoid(m.position, MARS_ELLIPSOID);
        if (!clickCart) return;

        const clickCarto = Cesium.Cartographic.fromCartesian(clickCart);
        const clickLon = Cesium.Math.toDegrees(clickCarto.longitude);
        const clickLat = Cesium.Math.toDegrees(clickCarto.latitude);

        console.log(`[Click] Position: lat=${clickLat.toFixed(4)}, lon=${clickLon.toFixed(4)}`);

        // PRIORITY 1: Check if click is within any active overlay bounds
        // This is more reliable than Cesium picking for image overlays
        let overlayProduct: { productId: string; instrument: InstrumentType } | null = null;

        // Check high-res overlays first (higher priority), then quickview
        const highResIds = highResOverlaysRef.current;
        const quickviewIds = quickviewOverlaysRef.current;
        const allOverlayIds = [...highResIds, ...quickviewIds];
        console.log("[Click] Active overlays:", { highRes: highResIds, quickview: quickviewIds });

        for (const productId of allOverlayIds) {
          const highResEnt = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
          const quickviewEnt = viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
          const overlayEnt = highResEnt || quickviewEnt;
          console.log(`[Click] Checking overlay ${productId}:`, {
            highResEntExists: !!highResEnt,
            quickviewEntExists: !!quickviewEnt,
            hasRectangle: !!overlayEnt?.rectangle,
            hasCoordinates: !!overlayEnt?.rectangle?.coordinates
          });

          if (overlayEnt?.rectangle?.coordinates) {
            const rect = overlayEnt.rectangle.coordinates.getValue(Cesium.JulianDate.now()) as Cesium.Rectangle;
            const west = Cesium.Math.toDegrees(rect.west);
            const east = Cesium.Math.toDegrees(rect.east);
            const south = Cesium.Math.toDegrees(rect.south);
            const north = Cesium.Math.toDegrees(rect.north);

            if (clickLon >= west && clickLon <= east && clickLat >= south && clickLat <= north) {
              const instrument = (overlayEnt.properties as any)?.instrument?.getValue?.() as InstrumentType;
              console.log("[Click] Found overlay via bounds:", productId, instrument);
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

                console.log(`[CRISM Overlay Click] -> line=${pixelLine}, sample=${pixelSample}`);
              }
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

                console.log(`[HiRISE Overlay Click] -> line=${pixelLine}, sample=${pixelSample}`);
              }
            }
          }

          console.log(`[Click] Overlay: ${instrument} ${productId}`);

          onSelect({
            instrument,
            productId,
            lat: clickLat,
            lon: clickLon,
            pixelLine,
            pixelSample,
          });

          return; // Don't process further - overlay click handled
        }

        // PRIORITY 2: No overlay clicked, try Cesium entity picking for footprints
        const pickedList = viewer.scene.drillPick(m.position);
        console.log("[Click] Picked entities:", pickedList.length, pickedList.map((p: any) => p.id?.id || "unknown"));

        const picked = pickedList.find((p: any) => {
          if (!(p.id instanceof Cesium.Entity)) return false;
          const pid = (p.id as Cesium.Entity).properties?.product_id?.getValue?.();
          return !!pid;
        });

        if (!picked || !(picked.id instanceof Cesium.Entity)) {
          console.log("[Click] No valid entity picked");
          return;
        }

        const e = picked.id as Cesium.Entity;
        const p: any = e.properties;

        const productId = p?.product_id?.getValue?.();
        const instrument = p?.instrument?.getValue?.();

        if (!productId || !instrument) return;

        console.log(`[Click] Footprint: ${instrument} ${productId}`);

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

        onSelect({
          instrument,
          productId,
          lat: clickLat,
          lon: clickLon,
        });

        // Fly to footprint
        const prefix = instrument === "HIRISE" ? "HIRISE_" : "CRISM_";
        const rectEntId = `${prefix}${productId}_0`;
        console.log("[Click] Looking for rectangle entity:", rectEntId);
        const rectEnt =
          viewer.entities.getById(`${prefix}${productId}_0`) ||
          viewer.entities.getById(`${prefix}${productId}_1`) ||
          viewer.entities.getById(`${prefix}${productId}_2`) ||
          viewer.entities.getById(`${prefix}${productId}_3`);
        console.log("[Click] Rectangle entity found:", !!rectEnt);

        if (rectEnt?.rectangle?.coordinates) {
          const rect = rectEnt.rectangle.coordinates.getValue(
            Cesium.JulianDate.now()
          ) as Cesium.Rectangle;

          const dest = paddedRectangle(rect, 0.6);

          viewer.camera.flyTo({
            destination: dest,
            duration: 0.6,
          });
        } else {
          // Fallback: load LBL directly to get bounds for fly-to
          console.log("[Click] No rectangle entity found, loading LBL for fly-to:", productId);
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

    return () => {
      hoverHandler.destroy();
      clickHandler.destroy();
    };
  }, []);

  useEffect(() => {
    hiriseEntitiesRef.current.forEach((e) => (e.show = showHiRISE));
    viewerRef.current?.scene.requestRender();
  }, [showHiRISE]);

  useEffect(() => {
    crismEntitiesRef.current.forEach((e) => {
      if (!showCRISM) {
        e.show = false;
        return;
      }
      // If ICE filter is active, only show CRISM entities that pass the filter
      if (iceFilterPassingObs !== null) {
        const productId = e.properties?.getValue(Cesium.JulianDate.now())?.product_id;
        if (productId) {
          const obsId = productId.split("_")[0];
          e.show = iceFilterPassingObs.has(obsId);
        } else {
          e.show = false;
        }
      } else {
        e.show = true;
      }
    });
    viewerRef.current?.scene.requestRender();
  }, [showCRISM, iceFilterPassingObs]);

  useEffect(() => {
    sharadEntitiesRef.current.forEach((e) => (e.show = showSHARAD));
    viewerRef.current?.scene.requestRender();
  }, [showSHARAD]);

  // Hide footprints when they have image overlays
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Collect all product IDs and obs_ids that have overlays
    const productsWithOverlays = new Set<string>();
    const obsIdsWithOverlays = new Set<string>();

    // Quickview overlays
    quickviewOverlays.forEach((pid) => productsWithOverlays.add(pid));

    // High-res overlays
    highResOverlays.forEach((pid) => productsWithOverlays.add(pid));

    // Browse overlays
    browseOverlays.forEach((_, pid) => productsWithOverlays.add(pid));

    // Ice Score overlays (when enabled)
    if (showIceScore) {
      iceScoreEntitiesRef.current.forEach((e) => {
        const obsId = e.properties?.getValue(Cesium.JulianDate.now())?.obs_id;
        if (obsId) obsIdsWithOverlays.add(obsId);
      });
    }

    // Hydration Score overlays (when enabled)
    if (showHydratedScore) {
      hydrationScoreEntitiesRef.current.forEach((e) => {
        const obsId = e.properties?.getValue(Cesium.JulianDate.now())?.obs_id;
        if (obsId) obsIdsWithOverlays.add(obsId);
      });
    }

    // Update HiRISE footprints
    hiriseEntitiesRef.current.forEach((e) => {
      const props = e.properties?.getValue(Cesium.JulianDate.now());
      const kind = props?.kind;
      if (kind === "FOOTPRINT_LABEL" || kind === "FOOTPRINT_POINT") return;

      const pid = props?.product_id;
      if (!pid) return;

      const hasOverlay = productsWithOverlays.has(pid);
      if (e.rectangle?.material) {
        if (hasOverlay) {
          // Make footprint fully transparent
          (e.rectangle.material as any) = Cesium.Color.TRANSPARENT;
        } else {
          // Restore original color
          (e.rectangle.material as any) = Cesium.Color.YELLOW.withAlpha(0.3);
        }
      }
    });

    // Update CRISM footprints
    crismEntitiesRef.current.forEach((e) => {
      const props = e.properties?.getValue(Cesium.JulianDate.now());
      const kind = props?.kind;
      if (kind === "FOOTPRINT_LABEL" || kind === "FOOTPRINT_POINT") return;

      const pid = props?.product_id;
      if (!pid) return;

      const obsId = pid.split("_")[0];
      const hasOverlay = productsWithOverlays.has(pid) || obsIdsWithOverlays.has(obsId);

      if (e.rectangle?.material) {
        if (hasOverlay) {
          // Make footprint fully transparent
          (e.rectangle.material as any) = Cesium.Color.TRANSPARENT;
        } else {
          // Restore original color
          (e.rectangle.material as any) = Cesium.Color.CYAN.withAlpha(0.35);
        }
      }
    });

    viewer.scene.requestRender();
  }, [quickviewOverlays, highResOverlays, browseOverlays, showIceScore, showHydratedScore]);

  // Handle Ice Score layer toggle - uses crismScoreFeatures from GeoJSON
  useEffect(() => {
    console.log("[DEBUG][ICE_SCORE] useEffect triggered:", { showIceScore, viewerReady, featuresCount: crismScoreFeatures.length });
    const viewer = viewerRef.current;
    if (!viewer || !viewerReady) {
      console.log("[DEBUG][ICE_SCORE] Early return - no viewer or not ready");
      return;
    }

    // Remove existing ice score entities
    iceScoreEntitiesRef.current.forEach((e) => viewer.entities.remove(e));
    iceScoreEntitiesRef.current = [];

    if (!showIceScore) {
      console.log("[DEBUG][ICE_SCORE] showIceScore is false, skipping");
      viewer.scene.requestRender();
      return;
    }

    if (crismScoreFeatures.length === 0) {
      console.log("[DEBUG][ICE_SCORE] No score features available yet");
      return;
    }

    console.log("[DEBUG][ICE_SCORE] Creating overlays for", crismScoreFeatures.length, "features");
    if (crismScoreFeatures.length > 0) {
      const sample = crismScoreFeatures[0];
      console.log("[DEBUG][ICE_SCORE] Sample feature:", sample.properties?.product_id, "ice_url:", sample.properties?.ice_score_map);
    }
    let created = 0;
    let skippedNoUrl = 0;
    let skippedNoGeom = 0;

    for (const feature of crismScoreFeatures) {
      const props = feature.properties || {};
      const iceScoreUrl = props.ice_score_map;
      if (!iceScoreUrl) {
        skippedNoUrl++;
        continue;
      }

      const geom = feature.geometry;
      if (geom?.type !== "Polygon" || !geom.coordinates?.[0]?.length) {
        skippedNoGeom++;
        continue;
      }

      const coords = geom.coordinates[0];
      const lons = coords.map((c: number[]) => normalizeLonTo180(c[0]));
      const lats = coords.map((c: number[]) => c[1]);
      const west = Math.min(...lons);
      const east = Math.max(...lons);
      const south = Math.min(...lats);
      const north = Math.max(...lats);

      const obsId = props.product_id?.split("_")[0] || props.product_id;

      try {
        const ent = viewer.entities.add({
          id: `ICE_SCORE_${obsId}`,
          show: true,
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
            material: new Cesium.ImageMaterialProperty({
              image: iceScoreUrl,
              transparent: true,
              color: Cesium.Color.WHITE.withAlpha(overlayOpacity),
            }),
            height: 0,
          },
          properties: {
            obs_id: obsId,
            instrument: "CRISM",
            kind: "ICE_SCORE",
          },
        });
        iceScoreEntitiesRef.current.push(ent);
        created++;
      } catch (e) {
        console.log("[DEBUG][ICE_SCORE] Error adding overlay for:", obsId, e);
      }
    }

    console.log("[DEBUG][ICE_SCORE] Created", created, "overlays, skippedNoUrl:", skippedNoUrl, "skippedNoGeom:", skippedNoGeom);
    viewer.scene.requestRender();
  }, [showIceScore, crismScoreFeatures, viewerReady, overlayOpacity]);

  // Handle Hydration Score layer toggle - uses crismScoreFeatures from GeoJSON
  useEffect(() => {
    console.log("[DEBUG][HYDRATION_SCORE] useEffect triggered:", { showHydratedScore, viewerReady, featuresCount: crismScoreFeatures.length });
    const viewer = viewerRef.current;
    if (!viewer || !viewerReady) return;

    // Remove existing hydration score entities
    hydrationScoreEntitiesRef.current.forEach((e) => viewer.entities.remove(e));
    hydrationScoreEntitiesRef.current = [];

    if (showHydratedScore && crismScoreFeatures.length > 0) {
      console.log("[DEBUG][HYDRATION_SCORE] Creating overlays for features with hydration_score_map");
      let created = 0;

      for (const feature of crismScoreFeatures) {
        const props = feature.properties || {};
        const hydrationScoreUrl = props.hydration_score_map;
        if (!hydrationScoreUrl) continue;

        const geom = feature.geometry;
        if (geom?.type !== "Polygon" || !geom.coordinates?.[0]?.length) continue;

        const coords = geom.coordinates[0];
        const lons = coords.map((c: number[]) => normalizeLonTo180(c[0]));
        const lats = coords.map((c: number[]) => c[1]);
        const west = Math.min(...lons);
        const east = Math.max(...lons);
        const south = Math.min(...lats);
        const north = Math.max(...lats);

        const obsId = props.product_id?.split("_")[0] || props.product_id;

        try {
          const ent = viewer.entities.add({
            id: `HYDRATION_SCORE_${obsId}`,
            show: true,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
              material: new Cesium.ImageMaterialProperty({
                image: hydrationScoreUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(overlayOpacity),
              }),
              height: 0,
            },
            properties: {
              obs_id: obsId,
              instrument: "CRISM",
              kind: "HYDRATION_SCORE",
            },
          });
          hydrationScoreEntitiesRef.current.push(ent);
          created++;
        } catch (e) {
          console.log("[DEBUG][HYDRATION_SCORE] Could not add overlay for:", obsId);
        }
      }
      console.log("[DEBUG][HYDRATION_SCORE] Created", created, "overlays");
    }

    viewer.scene.requestRender();
  }, [showHydratedScore, crismScoreFeatures, viewerReady, overlayOpacity]);

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
      console.log("[DEBUG] viewBounds:", viewBounds);

      // Convert to radians for Cesium (normalize longitude to -180 to 180 first)
      const west = Cesium.Math.toRadians(normalizeLonTo180(westLon));
      const south = Cesium.Math.toRadians(minLat);
      const east = Cesium.Math.toRadians(normalizeLonTo180(eastLon));
      const north = Cesium.Math.toRadians(maxLat);

      // Create rectangle
      const rect = new Cesium.Rectangle(west, south, east, north);
      console.log("[DEBUG] rect:", rect);

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
      // Determine if HiRISE or CRISM
      const isHiRISE = flyToProductId!.startsWith("ESP_");
      const lbl = isHiRISE
        ? await loadHiRISELBL(flyToProductId!)
        : await loadCRISMLBL(flyToProductId!);

      if (!lbl) {
        console.warn("[FlyTo] No LBL found for", flyToProductId);
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

      // Fly to the rectangle with some padding
      const rect = Cesium.Rectangle.fromDegrees(west, south, east, north);
      const padded = paddedRectangle(rect, 0.3);

      viewer.camera.flyTo({
        destination: padded,
        duration: 0.8,
        complete: () => {
          onFlyToComplete?.();
        },
      });
    }

    flyTo();
  }, [flyToProductId, onFlyToComplete]);

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
    if (!viewer) return;

    const highResSet = new Set(highResOverlays);

    // Update HiRISE footprint visibility
    hiriseEntitiesRef.current.forEach((e) => {
      const p: any = e.properties;
      const pid = p?.product_id?.getValue?.();
      const kind = p?.kind?.getValue?.();

      // Only affect rectangle entities (not labels/points)
      if (!pid || kind === "FOOTPRINT_LABEL" || kind === "FOOTPRINT_POINT") return;

      if (e.rectangle) {
        if (highResSet.has(pid)) {
          // Hide footprint completely when high-res is active (so clicks go through to overlay)
          e.show = false;
        } else {
          // Show and restore original appearance
          e.show = showHiRISE;
          e.rectangle.material = Cesium.Color.YELLOW.withAlpha(0.3) as any;
          e.rectangle.outline = true as any;
          e.rectangle.outlineColor = Cesium.Color.YELLOW as any;
        }
      }
    });

    // Update CRISM footprint visibility
    crismEntitiesRef.current.forEach((e) => {
      const p: any = e.properties;
      const pid = p?.product_id?.getValue?.();
      const kind = p?.kind?.getValue?.();

      if (!pid || kind === "FOOTPRINT_LABEL" || kind === "FOOTPRINT_POINT") return;

      if (e.rectangle) {
        if (highResSet.has(pid)) {
          // Hide footprint completely when high-res is active (so clicks go through to overlay)
          e.show = false;
        } else {
          // Show and restore original appearance
          e.show = showCRISM;
          e.rectangle.material = Cesium.Color.CYAN.withAlpha(0.35) as any;
          e.rectangle.outline = true as any;
          e.rectangle.outlineColor = Cesium.Color.BLACK as any;
        }
      }
    });

    viewer.scene.requestRender();
  }, [highResOverlays, showHiRISE, showCRISM]);

  // Refs to track current overlays
  const quickviewOverlayIdsRef = useRef<Set<string>>(new Set());
  const highResOverlayIdsRef = useRef<Set<string>>(new Set());
  const browseOverlayIdsRef = useRef<Map<string, Set<BrowseProductType>>>(new Map());

  // Track blob URLs for CRISM RGB images to clean up later
  const crismBlobUrlsRef = useRef<Map<string, string>>(new Map());

  // Track previous RGB wavelengths to detect changes
  const prevRgbRef = useRef<RGBWavelengths>(rgbWavelengths);

  // Quickview overlays effect
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(quickviewOverlays);
    const existingIds = quickviewOverlayIdsRef.current;

    // Remove overlays that are no longer in the list
    // Convert to array first to avoid issues with modifying Set during iteration
    const toRemove = Array.from(existingIds).filter((id) => !currentIds.has(id));
    toRemove.forEach((id) => {
      console.log("[Quickview] Removing overlay:", id);
      const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${id}`);
      if (ent) viewer.entities.remove(ent);
      existingIds.delete(id);
    });

    if (toRemove.length > 0) {
      viewer.scene.requestRender();
    }

    // Add new overlays
    quickviewOverlays.forEach(async (productId) => {
      if (existingIds.has(productId)) return; // Already exists

      try {
        // Determine if HiRISE or CRISM based on ID pattern
        const isHiRISE = productId.startsWith("ESP_");

        let lbl: string | null = null;
        if (isHiRISE) {
          lbl = await loadHiRISELBL(productId);
        } else {
          lbl = await loadCRISMLBL(productId);
        }

        if (!lbl) {
          console.warn("[Quickview] No LBL found for", productId);
          return;
        }

        const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
        const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
        const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
        const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

        if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
          console.warn("[Quickview] Missing bounds for", productId);
          return;
        }

        const west = normalizeLonTo180(westLon360);
        const east = normalizeLonTo180(eastLon360);
        const south = Math.min(minLat, maxLat);
        const north = Math.max(minLat, maxLat);

        // Derive quickview URL based on product type
        let imageUrl: string;
        if (isHiRISE) {
          imageUrl = `/hirise/quickview/${productId}.png`;
        } else if (productId.includes("_brcarj_")) {
          // Arcadia browse products: frt00003156_07_brcarj_mtr3 -> frt00003156_VNIR.png
          const baseObsId = productId.split("_")[0];
          imageUrl = `/crism/quickview/${baseObsId}_VNIR.png`;
        } else {
          // Standard CRISM: frt0001fd76_07_if166j_mtr3 -> frt0001fd76_07_brvnaj_mtr3.png
          imageUrl = `/crism/quickview/${productId.replace(/_if[0-9a-z]+_mtr3$/i, "_brvnaj_mtr3")}.png`;
        }

        console.log("[Quickview] Adding overlay:", productId, { west, south, east, north, imageUrl });

        // Check if viewer still exists (component might have unmounted)
        if (!viewerRef.current) return;

        viewer.entities.add({
          id: `QUICKVIEW_OVERLAY_${productId}`,
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
            material: new Cesium.ImageMaterialProperty({
              image: imageUrl,
              transparent: true,
              color: Cesium.Color.WHITE.withAlpha(overlayOpacity),
            }),
            height: 0,
          },
          properties: {
            product_id: productId,
            instrument: isHiRISE ? "HIRISE" : "CRISM",
            kind: "OVERLAY",
          },
        });

        existingIds.add(productId);
        viewer.scene.requestRender();
      } catch (e) {
        console.error("[Quickview] Failed to add overlay:", e);
      }
    });
  }, [quickviewOverlays]);

  // High-resolution overlays effect
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(highResOverlays);
    const existingIds = highResOverlayIdsRef.current;

    // Remove overlays that are no longer in the list
    // Convert to array first to avoid issues with modifying Set during iteration
    const toRemove = Array.from(existingIds).filter((id) => !currentIds.has(id));
    toRemove.forEach((id) => {
      console.log("[HighRes] Removing overlay:", id);
      const ent = viewer.entities.getById(`HIGHRES_OVERLAY_${id}`);
      if (ent) {
        viewer.entities.remove(ent);
        console.log("[HighRes] Entity removed:", id);
      }
      existingIds.delete(id);

      // Clean up blob URL if exists
      const blobUrl = crismBlobUrlsRef.current.get(id);
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        crismBlobUrlsRef.current.delete(id);
      }
    });

    if (toRemove.length > 0) {
      viewer.scene.requestRender();
    }

    // Add new overlays
    highResOverlays.forEach(async (productId) => {
      if (existingIds.has(productId)) return; // Already exists

      try {
        // Determine if HiRISE or CRISM based on ID pattern
        const isHiRISE = productId.startsWith("ESP_");

        let lbl: string | null = null;
        if (isHiRISE) {
          lbl = await loadHiRISELBL(productId);
        } else {
          lbl = await loadCRISMLBL(productId);
        }

        if (!lbl) {
          console.warn("[HighRes] No LBL found for", productId);
          return;
        }

        const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
        const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
        const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
        const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

        if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
          console.warn("[HighRes] Missing bounds for", productId);
          return;
        }

        const west = normalizeLonTo180(westLon360);
        const east = normalizeLonTo180(eastLon360);
        const south = Math.min(minLat, maxLat);
        const north = Math.max(minLat, maxLat);

        let imageUrl: string;

        if (isHiRISE) {
          // Use high-res overlay endpoint for HiRISE
          imageUrl = `/hirise/overlay/${productId}.png`;
        } else {
          // For CRISM, make POST request to get RGB image
          console.log("[HighRes] Fetching CRISM RGB for", productId, rgbWavelengths);

          const response = await fetch(`/crism/${productId}/rgb`, {
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

          if (!response.ok) {
            console.error("[HighRes] CRISM RGB request failed:", response.status);
            return;
          }

          const blob = await response.blob();
          imageUrl = URL.createObjectURL(blob);
          crismBlobUrlsRef.current.set(productId, imageUrl);
        }

        console.log("[HighRes] Adding overlay:", productId, { west, south, east, north, imageUrl: isHiRISE ? imageUrl : "(blob)" });

        // Check if viewer still exists
        if (!viewerRef.current) return;

        const newEntity = viewer.entities.add({
          id: `HIGHRES_OVERLAY_${productId}`,
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
            material: new Cesium.ImageMaterialProperty({
              image: imageUrl,
              transparent: true,
              color: Cesium.Color.WHITE.withAlpha(overlayOpacity),
            }),
            height: 0,
          },
          properties: {
            product_id: productId,
            instrument: isHiRISE ? "HIRISE" : "CRISM",
            kind: "OVERLAY",
          },
        });

        console.log("[HighRes] Entity added:", productId, {
          entityId: newEntity.id,
          bounds: { west, south, east, north }
        });

        existingIds.add(productId);
        viewer.scene.requestRender();
      } catch (e) {
        console.error("[HighRes] Failed to add overlay:", e);
      }
    });
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

    console.log("[RGB] Wavelengths changed, refreshing CRISM overlays", rgbWavelengths);
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
          // Load LBL for bounds
          const lbl = await loadCRISMLBL(productId);
          if (!lbl) {
            console.warn("[Browse] No LBL for", productId);
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

          const west = normalizeLonTo180(westLon360);
          const east = normalizeLonTo180(eastLon360);
          const south = Math.min(minLat, maxLat);
          const north = Math.max(minLat, maxLat);

          // Construct browse image URL
          // Arcadia products: frt00003156_07_brcarj_mtr3 -> frt00003156_HYD.png
          const baseObsId = productId.split("_")[0];
          const imageUrl = `/crism/browse/${baseObsId}_${browseType}.png`;

          console.log("[Browse] Adding overlay:", productId, browseType, imageUrl);

          if (!viewerRef.current) return;

          viewer.entities.add({
            id: `BROWSE_OVERLAY_${productId}_${browseType}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(overlayOpacity),
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
        } catch (e) {
          console.error("[Browse] Failed to add overlay:", productId, browseType, e);
        }
      }

      // Update tracking
      existingOverlays.set(productId, new Set(types));
    });

    viewer.scene.requestRender();
  }, [browseOverlays, overlayOpacity]);

  // Track visible products in current view
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !onVisibleProductsChange) return;

    // Helper to check if two rectangles overlap
    const rectanglesOverlap = (r1: Cesium.Rectangle, r2: Cesium.Rectangle): boolean => {
      // Check if one rectangle is completely outside the other
      if (r1.east < r2.west || r2.east < r1.west) return false;
      if (r1.north < r2.south || r2.north < r1.south) return false;
      return true;
    };

    const updateVisibleProducts = () => {
      const camera = viewer.camera;

      // Get view rectangle - try multiple methods for reliability
      let viewRect = camera.computeViewRectangle(viewer.scene.globe.ellipsoid);

      // Fallback: compute from canvas corners
      if (!viewRect) {
        const canvas = viewer.scene.canvas;
        const corners = [
          new Cesium.Cartesian2(0, 0),
          new Cesium.Cartesian2(canvas.width, 0),
          new Cesium.Cartesian2(canvas.width, canvas.height),
          new Cesium.Cartesian2(0, canvas.height),
        ];

        let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;

        corners.forEach((corner) => {
          const cartesian = camera.pickEllipsoid(corner, MARS_ELLIPSOID);
          if (cartesian) {
            const carto = Cesium.Cartographic.fromCartesian(cartesian, MARS_ELLIPSOID);
            west = Math.min(west, carto.longitude);
            east = Math.max(east, carto.longitude);
            south = Math.min(south, carto.latitude);
            north = Math.max(north, carto.latitude);
          }
        });

        if (west !== Infinity) {
          viewRect = new Cesium.Rectangle(west, south, east, north);
        }
      }

      if (!viewRect) {
        console.warn("[VisibleProducts] Could not compute view rectangle");
        return;
      }

      const visible: VisibleProduct[] = [];
      const seen = new Set<string>();

      // Check HiRISE entities
      // Note: Don't check e.show because footprints are hidden when overlay is active
      if (showHiRISE) {
        hiriseEntitiesRef.current.forEach((e) => {
          const p: any = e.properties;
          const pid = p?.product_id?.getValue?.();
          const kind = p?.kind?.getValue?.();
          const title = p?.title?.getValue?.();  // Get title for search

          // Skip labels and points, only check rectangle entities
          if (!pid || seen.has(pid) || kind === "FOOTPRINT_LABEL" || kind === "FOOTPRINT_POINT") return;

          // Check if entity's rectangle intersects view
          const entRect = (e.rectangle as any)?.coordinates?.getValue?.(Cesium.JulianDate.now());
          if (entRect && rectanglesOverlap(viewRect!, entRect)) {
            seen.add(pid);
            visible.push({ productId: pid, instrument: "HIRISE", title });
          }
        });
      }

      // Check CRISM entities
      if (showCRISM) {
        crismEntitiesRef.current.forEach((e) => {
          const p: any = e.properties;
          const pid = p?.product_id?.getValue?.();
          const kind = p?.kind?.getValue?.();

          if (!pid || seen.has(pid) || kind === "FOOTPRINT_LABEL" || kind === "FOOTPRINT_POINT") return;

          const entRect = (e.rectangle as any)?.coordinates?.getValue?.(Cesium.JulianDate.now());
          if (entRect && rectanglesOverlap(viewRect!, entRect)) {
            seen.add(pid);
            visible.push({ productId: pid, instrument: "CRISM" });
          }
        });
      }

      onVisibleProductsChange(visible);
    };

    // Update on camera move end
    const removeListener = viewer.camera.moveEnd.addEventListener(updateVisibleProducts);

    // Initial update (with delay to ensure entities are loaded)
    setTimeout(updateVisibleProducts, 1000);

    // Also update periodically in case moveEnd doesn't fire
    const interval = setInterval(updateVisibleProducts, 2000);

    return () => {
      removeListener();
      clearInterval(interval);
    };
  }, [showHiRISE, showCRISM, onVisibleProductsChange]);

  // Update overlay opacity when overlayOpacity changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Update quickview overlays
    quickviewOverlayIdsRef.current.forEach((productId) => {
      const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
      if (ent?.rectangle?.material) {
        const material = ent.rectangle.material as Cesium.ImageMaterialProperty;
        if (material.color) {
          material.color = new Cesium.ConstantProperty(
            Cesium.Color.WHITE.withAlpha(overlayOpacity)
          );
        }
      }
    });

    // Update high-res overlays
    highResOverlayIdsRef.current.forEach((productId) => {
      const ent = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
      if (ent?.rectangle?.material) {
        const material = ent.rectangle.material as Cesium.ImageMaterialProperty;
        if (material.color) {
          material.color = new Cesium.ConstantProperty(
            Cesium.Color.WHITE.withAlpha(overlayOpacity)
          );
        }
      }
    });

    // Update browse overlays
    browseOverlayIdsRef.current.forEach((types, productId) => {
      types.forEach((browseType) => {
        const ent = viewer.entities.getById(`BROWSE_OVERLAY_${productId}_${browseType}`);
        if (ent?.rectangle?.material) {
          const material = ent.rectangle.material as Cesium.ImageMaterialProperty;
          if (material.color) {
            material.color = new Cesium.ConstantProperty(
              Cesium.Color.WHITE.withAlpha(overlayOpacity)
            );
          }
        }
      });
    });

    viewer.scene.requestRender();
  }, [overlayOpacity]);

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

      {/* CRISM Disclaimer */}
      {showCRISM && crismDisclaimer && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-lg border border-amber-500/50 bg-amber-900/80 px-4 py-2 backdrop-blur-md">
          <div className="flex items-center gap-2 text-amber-200">
            <span className="material-symbols-outlined text-sm">warning</span>
            <span className="text-[11px]">
              Showing {crismDisclaimer.displayed} of {crismDisclaimer.total} CRISM footprints (max 1000)
            </span>
          </div>
        </div>
      )}
    </>
  );
}
