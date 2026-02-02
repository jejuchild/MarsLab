// src/MapView.tsx
import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import FootprintManager from "../utils/FootprintManager";

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
  // Bidirectional hover highlight for Active Products Panel
  hoveredProductId?: string | null;
  onHoverProduct?: (productId: string | null) => void;
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
  const res = await fetch(`${CRISM_LBL_BASE}/${id}.lbl`);
  if (!res.ok) return null;

  const text = await res.text();
  if (!text.includes("IMAGE_MAP_PROJECTION")) return null;

  crismLBLCache.set(id, text);
  return text;
}

// Get cached bounds or parse from LBL
async function getProductBounds(productId: string): Promise<ProductBounds | null> {
  if (boundsCache.has(productId)) {
    return boundsCache.get(productId)!;
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
  hoveredProductId = null,
  onHoverProduct,
}: MapViewProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  const sharadEntitiesRef = useRef<Cesium.Entity[]>([]);

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
  const [hiriseDisclaimer, setHiriseDisclaimer] = useState<{ displayed: number; total: number } | null>(null);
  const [isLoadingFootprints, setIsLoadingFootprints] = useState(false);

  // Current LOD state for UI feedback
  const [currentLOD, setCurrentLOD] = useState<"none" | "point" | "poly">("none");
  const [cameraHeightKm, setCameraHeightKm] = useState<number>(Infinity);

  // FootprintManager ref for viewport-based loading
  const footprintManagerRef = useRef<FootprintManager | null>(null);

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

    // Initialize FootprintManager for viewport-based loading
    const footprintManager = new FootprintManager({
      viewer,
      ellipsoid: MARS_ELLIPSOID,
      debounceMs: 300,
      maxCacheSize: 100,
      onTruncated: (instrument, returned, total) => {
        console.log(`[FootprintManager] ${instrument} truncated: ${returned}/${total}`);
        if (instrument === "CRISM") {
          setCrismDisclaimer({ displayed: returned, total });
        } else if (instrument === "HIRISE") {
          setHiriseDisclaimer({ displayed: returned, total });
        }
      },
      onLoadStart: (instrument) => {
        console.log(`[FootprintManager] Loading ${instrument}...`);
        setIsLoadingFootprints(true);
      },
      onLoadEnd: (instrument, count) => {
        console.log(`[FootprintManager] Loaded ${instrument}: ${count} features`);
        setIsLoadingFootprints(false);
        // Clear disclaimer when data is not truncated
        if (instrument === "CRISM") {
          setCrismDisclaimer(null);
        } else if (instrument === "HIRISE") {
          setHiriseDisclaimer(null);
        }
      },
      onError: (instrument, error) => {
        console.error(`[FootprintManager] Error loading ${instrument}:`, error);
        setIsLoadingFootprints(false);
      },
      onLODChange: (lod, cameraHeight) => {
        console.log(`[FootprintManager] LOD changed: ${lod}, height: ${(cameraHeight / 1000).toFixed(0)} km`);
        setCurrentLOD(lod);
        setCameraHeightKm(cameraHeight / 1000);
        // Clear disclaimers when LOD changes to none (zoomed out)
        if (lod === "none") {
          setCrismDisclaimer(null);
          setHiriseDisclaimer(null);
        }
      },
    });
    footprintManagerRef.current = footprintManager;
    // =================

    // HiRISE footprints are now loaded via FootprintManager (viewport-based)
    // Legacy global loading disabled for performance
    // Footprints are loaded via FootprintManager (viewport-based)
    console.log("[HIRISE] Footprints will be loaded via FootprintManager (viewport-based)");
    console.log("[CRISM] Footprints will be loaded via FootprintManager (viewport-based)");

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
      (m: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
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

            // ✅ 같은 대상이면 재적용 금지 (무한 scale 누적 버그 근본 차단)
            if (hs.key === key) return;

            // Try FootprintManager entity IDs first, then legacy IDs
            const rectFallback =
              viewer.entities.getById(`${inst}_VP_${pid}`) ||  // FootprintManager ID
              viewer.entities.getById(`${inst}_VP_${pid}_1`) ||
              viewer.entities.getById(`${inst}_VP_${pid}_2`) ||
              viewer.entities.getById(`${inst}_VP_${pid}_3`) ||
              viewer.entities.getById(`${inst === "HIRISE" ? "HIRISE" : "CRISM"}_${pid}_0`) ||  // Legacy ID
              viewer.entities.getById(`${inst === "HIRISE" ? "HIRISE" : "CRISM"}_${pid}_1`) ||
              null;

            const rectTarget =
              rectFallback && (rectFallback as any).rectangle ? rectFallback : null;

            // Try FootprintManager entity IDs first, then legacy IDs
            const labelEnt = viewer.entities.getById(`${inst}_VP_LABEL_${pid}`) ||
                             viewer.entities.getById(`${inst}_LABEL_${pid}`) || null;
            const pointEnt = viewer.entities.getById(`${inst}_VP_POINT_${pid}`) ||
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
                (labelEnt.label.scale as any) = 1.2; // <- 고정
              }

              if (pointEnt?.point) {
                hs.pointEnt = pointEnt;
                const cur = (pointEnt.point.pixelSize as any);
                hs.origPointSize = typeof cur === "number" ? cur : 6;
                (pointEnt.point.pixelSize as any) = (hs.origPointSize ?? 6) + 2;
              }

              // Notify parent of hovered product (bidirectional sync)
              onHoverProductRef.current?.(pid);

              viewer.scene.requestRender();
              return;
            }
          }
        }

        if (hs.rectEnt || hs.labelEnt || hs.pointEnt) {
          clearHighlight();
          // Clear hovered product when moving away
          onHoverProductRef.current?.(null);
          viewer.scene.requestRender();
        }
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
      // Dispose FootprintManager
      if (footprintManagerRef.current) {
        footprintManagerRef.current.dispose();
        footprintManagerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    // Toggle FootprintManager viewport-based entities for HiRISE
    if (footprintManagerRef.current) {
      footprintManagerRef.current.setEnabled("HIRISE", showHiRISE);
    }
    viewerRef.current?.scene.requestRender();
  }, [showHiRISE]);

  useEffect(() => {
    // Toggle FootprintManager viewport-based entities for CRISM
    if (footprintManagerRef.current) {
      footprintManagerRef.current.setEnabled("CRISM", showCRISM);
    }
    viewerRef.current?.scene.requestRender();
  }, [showCRISM]);

  useEffect(() => {
    sharadEntitiesRef.current.forEach((e) => (e.show = showSHARAD));
    viewerRef.current?.scene.requestRender();
  }, [showSHARAD]);

  // Note: Legacy footprint overlay hiding is no longer needed since
  // footprints are now managed by FootprintManager (viewport-based loading)

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

      // Re-check viewer after async operations
      const v = viewerRef.current;
      if (!v) return;

      v.camera.flyTo({
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
    const footprintManager = footprintManagerRef.current;
    if (!viewer || !footprintManager) return;

    const highResSet = new Set(highResOverlays);

    // Helper to update footprint visibility for FootprintManager entities
    const updateFootprintVisibility = (instrument: "HIRISE" | "CRISM") => {
      const features = footprintManager.getFeatures(instrument);
      const isEnabled = instrument === "HIRISE" ? showHiRISE : showCRISM;

      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;

        // Find the main footprint entity (could be point or polygon)
        const entityId = `${instrument}_VP_${pid}`;
        const entity = viewer.entities.getById(entityId);

        if (entity) {
          if (highResSet.has(pid)) {
            // Hide footprint when high-res is active (so clicks go through to overlay)
            entity.show = false;
          } else {
            entity.show = isEnabled;
          }
        }

        // Also handle split entities for antimeridian crossing
        for (let i = 1; i < 4; i++) {
          const splitEntity = viewer.entities.getById(`${entityId}_${i}`);
          if (splitEntity) {
            if (highResSet.has(pid)) {
              splitEntity.show = false;
            } else {
              splitEntity.show = isEnabled;
            }
          }
        }

        // Update label and point entities
        const labelId = `${instrument}_VP_LABEL_${pid}`;
        const pointId = `${instrument}_VP_POINT_${pid}`;
        const labelEnt = viewer.entities.getById(labelId);
        const pointEnt = viewer.entities.getById(pointId);

        if (labelEnt) {
          labelEnt.show = highResSet.has(pid) ? false : isEnabled;
        }
        if (pointEnt) {
          pointEnt.show = highResSet.has(pid) ? false : isEnabled;
        }
      }
    };

    // Update HiRISE footprint visibility
    if (footprintManager.isEnabled("HIRISE") || highResSet.size > 0) {
      updateFootprintVisibility("HIRISE");
    }

    // Update CRISM footprint visibility
    if (footprintManager.isEnabled("CRISM") || highResSet.size > 0) {
      updateFootprintVisibility("CRISM");
    }

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

    // STEP 1: Hide overlays that are no longer in the list (instead of removing)
    const toHide = Array.from(existingIds).filter((id) => !currentIds.has(id));
    for (const id of toHide) {
      const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${id}`);
      if (ent) {
        ent.show = false;
        needsRender = true;
      }
      existingIds.delete(id);
    }

    // STEP 2: Show existing overlays that are back in the list
    const toCreate: string[] = [];

    for (const productId of quickviewOverlays) {
      if (existingIds.has(productId)) continue; // Already tracked and visible

      const existingEnt = viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
      if (existingEnt) {
        // Entity exists but was hidden - just show it
        existingEnt.show = true;
        existingIds.add(productId);
        needsRender = true;
      } else {
        toCreate.push(productId);
      }
    }

    // STEP 3: Create new overlays in parallel (async)
    if (toCreate.length > 0) {
      // Pre-fetch bounds in parallel for faster creation
      Promise.all(toCreate.map(async (productId) => {
        try {
          const bounds = await getProductBounds(productId);
          if (!bounds || !viewerRef.current) return null;

          const isHiRISE = productId.startsWith("ESP_");

          // Derive quickview URL
          let imageUrl: string;
          if (isHiRISE) {
            imageUrl = `/hirise/quickview/${productId}.png`;
          } else if (productId.includes("_brcarj_")) {
            const baseObsId = productId.split("_")[0];
            imageUrl = `/crism/quickview/${baseObsId}_VNIR.png`;
          } else {
            imageUrl = `/crism/quickview/${productId.replace(/_if[0-9a-z]+_mtr3$/i, "_brvnaj_mtr3")}.png`;
          }

          return { productId, bounds, imageUrl, isHiRISE };
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
          const { productId, bounds, imageUrl, isHiRISE } = result;

          v.entities.add({
            id: `QUICKVIEW_OVERLAY_${productId}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
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

          quickviewOverlayIdsRef.current.add(productId);
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
          const bounds = await getProductBounds(productId);
          if (!bounds || !viewerRef.current) return null;

          const isHiRISE = productId.startsWith("ESP_");
          let imageUrl: string;

          if (isHiRISE) {
            imageUrl = `/hirise/overlay/${productId}.png`;
          } else {
            // CRISM RGB request
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

      // Get products from FootprintManager for HiRISE
      if (showHiRISE && footprintManager.isEnabled("HIRISE")) {
        const hiriseFeatures = footprintManager.getFeatures("HIRISE");
        for (const feature of hiriseFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid)) {
            seen.add(pid);
            visible.push({ productId: pid, instrument: "HIRISE", title });
          }
        }
      }

      // Get products from FootprintManager for CRISM
      if (showCRISM && footprintManager.isEnabled("CRISM")) {
        const crismFeatures = footprintManager.getFeatures("CRISM");
        for (const feature of crismFeatures) {
          const pid = feature.properties.product_id;
          if (pid && !seen.has(pid)) {
            seen.add(pid);
            visible.push({ productId: pid, instrument: "CRISM" });
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
    const initTimeout = setTimeout(updateVisibleProducts, 1000);

    // Reduced polling frequency (5s instead of 2s) - just a fallback
    const interval = setInterval(updateVisibleProducts, 5000);

    return () => {
      removeListener();
      clearTimeout(initTimeout);
      clearInterval(interval);
    };
  }, [showHiRISE, showCRISM, onVisibleProductsChange]);

  // PERFORMANCE OPTIMIZED: Update overlay opacity when overlayOpacity changes
  // Pre-create color property to avoid repeated object creation
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Pre-create the color value once
    const newColor = Cesium.Color.WHITE.withAlpha(overlayOpacity);
    const newColorProperty = new Cesium.ConstantProperty(newColor);

    // Update all overlays in a single pass
    const updateMaterial = (ent: Cesium.Entity | undefined) => {
      if (!ent?.rectangle?.material) return;
      const material = ent.rectangle.material as Cesium.ImageMaterialProperty;
      if (material.color) {
        material.color = newColorProperty;
      }
    };

    // Update quickview overlays
    for (const productId of quickviewOverlayIdsRef.current) {
      updateMaterial(viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`));
    }

    // Update high-res overlays
    for (const productId of highResOverlayIdsRef.current) {
      updateMaterial(viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`));
    }

    // Update browse overlays
    for (const [productId, types] of browseOverlayIdsRef.current) {
      for (const browseType of types) {
        updateMaterial(viewer.entities.getById(`BROWSE_OVERLAY_${productId}_${browseType}`));
      }
    }

    viewer.scene.requestRender();
  }, [overlayOpacity]);

  // Store onHoverProduct in ref to access in hover handler
  const onHoverProductRef = useRef(onHoverProduct);
  useEffect(() => {
    onHoverProductRef.current = onHoverProduct;
  }, [onHoverProduct]);

  // Bidirectional highlight: highlight footprint when hovering in ActiveProductsPanel
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Helper to apply/remove highlight to an entity
    const setEntityHighlight = (entity: Cesium.Entity | undefined, highlighted: boolean, instrument: InstrumentType) => {
      if (!entity?.rectangle) return;

      if (highlighted) {
        entity.rectangle.material = instrument === "HIRISE"
          ? HILITE_RECT_MATERIAL_HIRISE
          : HILITE_RECT_MATERIAL_CRISM;
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE) as any;
      } else {
        // Restore original appearance
        entity.rectangle.material = (instrument === "HIRISE"
          ? Cesium.Color.YELLOW.withAlpha(0.3)
          : Cesium.Color.CYAN.withAlpha(0.35)) as any;
        entity.rectangle.outlineColor = (instrument === "HIRISE"
          ? Cesium.Color.YELLOW
          : Cesium.Color.BLACK) as any;
      }
    };

    // Clear previous highlight if any
    if (!hoveredProductId) {
      viewer.scene.requestRender();
      return;
    }

    // Find and highlight the hovered product
    const isHiRISE = hoveredProductId.startsWith("ESP_");
    const instrument: InstrumentType = isHiRISE ? "HIRISE" : "CRISM";

    // Try FootprintManager entity IDs first, then legacy IDs
    const entityIds: string[] = [];
    const vpPrefix = `${instrument}_VP_${hoveredProductId}`;
    const legacyPrefix = `${instrument}_${hoveredProductId}`;

    // Try FootprintManager IDs first
    for (const id of [vpPrefix, `${vpPrefix}_1`, `${vpPrefix}_2`, `${vpPrefix}_3`]) {
      const entity = viewer.entities.getById(id);
      if (entity) {
        setEntityHighlight(entity, true, instrument);
        entityIds.push(id);
      }
    }

    // Also try legacy IDs
    if (entityIds.length === 0) {
      for (let i = 0; i < 4; i++) {
        const id = `${legacyPrefix}_${i}`;
        const entity = viewer.entities.getById(id);
        if (entity) {
          setEntityHighlight(entity, true, instrument);
          entityIds.push(id);
        }
      }
    }

    // Also highlight label and point if they exist (try VP IDs first)
    const labelEnt = viewer.entities.getById(`${instrument}_VP_LABEL_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_LABEL_${hoveredProductId}`);
    if (labelEnt?.label) {
      (labelEnt.label.scale as any) = 1.3;
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

      if (labelEnt?.label) {
        (labelEnt.label.scale as any) = 1.0;
      }

      if (pointEnt?.point) {
        (pointEnt.point.pixelSize as any) = 6;
      }

      viewer.scene.requestRender();
    };
  }, [hoveredProductId]);

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

      {/* Zoom Gating Hint - Show when zoomed out and footprints are enabled */}
      {(showCRISM || showHiRISE) && currentLOD === "none" && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-lg border border-sky-500/50 bg-sky-900/80 px-4 py-2 backdrop-blur-md">
          <div className="flex items-center gap-2 text-sky-200">
            <span className="material-symbols-outlined text-sm">zoom_in</span>
            <span className="text-[11px]">
              Zoom in to see footprints (current: {cameraHeightKm.toFixed(0)} km)
            </span>
          </div>
        </div>
      )}

      {/* LOD Indicator - Show current view mode */}
      {(showCRISM || showHiRISE) && currentLOD === "point" && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-lg border border-cyan-500/50 bg-cyan-900/80 px-4 py-2 backdrop-blur-md">
          <div className="flex items-center gap-2 text-cyan-200">
            <span className="material-symbols-outlined text-sm">radio_button_checked</span>
            <span className="text-[11px]">
              Showing centroids only. Zoom in for polygons ({cameraHeightKm.toFixed(0)} km)
            </span>
          </div>
        </div>
      )}

      {/* Footprint Truncation Warnings */}
      {(showCRISM && crismDisclaimer) || (showHiRISE && hiriseDisclaimer) ? (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 rounded-lg border border-amber-500/50 bg-amber-900/80 px-4 py-2 backdrop-blur-md">
          <div className="flex flex-col gap-1">
            {showCRISM && crismDisclaimer && (
              <div className="flex items-center gap-2 text-amber-200">
                <span className="material-symbols-outlined text-sm">warning</span>
                <span className="text-[11px]">
                  Too many footprints — zoom in further ({crismDisclaimer.displayed}/{crismDisclaimer.total} CRISM)
                </span>
              </div>
            )}
            {showHiRISE && hiriseDisclaimer && (
              <div className="flex items-center gap-2 text-amber-200">
                <span className="material-symbols-outlined text-sm">warning</span>
                <span className="text-[11px]">
                  Too many footprints — zoom in further ({hiriseDisclaimer.displayed}/{hiriseDisclaimer.total} HiRISE)
                </span>
              </div>
            )}
          </div>
        </div>
      ) : null}

      {/* Loading Indicator */}
      {isLoadingFootprints && (
        <div className="absolute top-4 right-4 rounded-lg border border-blue-500/50 bg-blue-900/80 px-3 py-1.5 backdrop-blur-md">
          <div className="flex items-center gap-2 text-blue-200">
            <div className="h-3 w-3 animate-spin rounded-full border-2 border-blue-300 border-t-transparent" />
            <span className="text-[11px]">Loading footprints...</span>
          </div>
        </div>
      )}
    </>
  );
}
