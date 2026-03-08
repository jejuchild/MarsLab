import { useCallback, useEffect, useRef, useState } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { loadDTMElevationGrid, throttle, type DTMElevationGrid } from "../utils/dtmHover";
import type { FieldNote } from "../api/fieldnotes";
import type FootprintManager from "../utils/FootprintManager";

type LatLon = { lat: number; lon: number };
type BaseLayerType = "MOLA" | "HRSC";
type MapMode = "2D" | "3D";
type InstrumentType =
  | "CRISM"
  | "HIRISE"
  | "SHARAD"
  | "SHARAD_HIGHRES"
  | "CTX"
  | "CUSTOM"
  | "HIRISE_DTM"
  | "CRISM_TRR3";

type BoundingBox = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;

type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  pixelLine?: number;
  pixelSample?: number;
  title?: string;
};

type SHARADPopup = {
  productId: string;
  quickviewUrl: string;
  startLat: number;
  startLon: number;
  stopLat: number;
  stopLon: number;
} | null;

type ProductBounds = {
  west: number;
  south: number;
  east: number;
  north: number;
  lines?: number;
  samples?: number;
};

type HighlightState = {
  key: string | null;
  rectEnt: Cesium.Entity | null;
  labelEnt: Cesium.Entity | null;
  pointEnt: Cesium.Entity | null;
  origRectMaterial: Cesium.MaterialProperty | undefined;
  origOutlineColor: Cesium.Color | undefined;
  origLabelScale: number | undefined;
  origPointSize: number | undefined;
};

type UseMapViewerParams = {
  containerRef: React.RefObject<HTMLDivElement | null>;
  mapMode: MapMode;
  baseLayer: BaseLayerType;
  viewBounds?: BoundingBox;
  marsEllipsoid: Cesium.Ellipsoid;
  marsRect: Cesium.Rectangle;
  baseLayerUrls: Record<BaseLayerType, string>;
  quickviewOverlays: string[];
  highResOverlays: string[];
  footprintManagerRef: React.MutableRefObject<FootprintManager | null>;
  onSelect: (ctx: InspectorContext | null) => void;
  onSharadClick?: (popup: SHARADPopup) => void;
  onSharadHiresClick?: (productId: string) => void;
  onHiRiseDTMClick?: (productId: string, lat: number, lon: number, title?: string) => void;
  onToggleOverlay?: (productId: string, type: "quickview" | null) => void;
  onTerrainClick?: (lat: number, lon: number) => void;
  onFieldNoteClick?: (note: FieldNote) => void;
  fieldNotes: FieldNote[];
  onHoverProduct?: (productId: string | null) => void;
  activeDTMProductRef: React.MutableRefObject<string | null>;
  setDtmGrid: (productId: string, grid: DTMElevationGrid) => void;
  initializeDTMHover: (viewer: Cesium.Viewer) => () => void;
  onOlympusMonsTripleClick?: () => void;
  onOlympusMonsClimber?: () => void;
  cameraViewportRef?: React.MutableRefObject<
    { minLat: number; maxLat: number; westLon: number; eastLon: number } | null
  >;
  parseLBLValue: (block: string | null | undefined, key: string) => number | null;
  normalizeLonTo180: (lon: number) => number;
  loadHiRISELBL: (id: string) => Promise<string | null>;
  loadCRISMLBL: (id: string) => Promise<string | null>;
  getProductBounds: (productId: string) => Promise<ProductBounds | null>;
  paddedRectangle: (rect: Cesium.Rectangle, padRatio?: number) => Cesium.Rectangle;
  getEntityInstrument: (entity: Cesium.Entity) => InstrumentType | null;
  getEntityProductId: (entity: Cesium.Entity) => string | null;
  getHiliteMaterial: (instrument: string) => Cesium.ColorMaterialProperty;
};

type UseMapViewerResult = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  hover: LatLon | null;
  initError: string | null;
  switchSceneMode: (mode: MapMode) => void;
  switchBaseLayer: (layer: BaseLayerType) => void;
};

export default function useMapViewer({
  containerRef,
  mapMode,
  baseLayer,
  viewBounds,
  marsEllipsoid,
  marsRect,
  baseLayerUrls,
  quickviewOverlays,
  highResOverlays,
  footprintManagerRef,
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
}: UseMapViewerParams): UseMapViewerResult {
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  // Refs to track current overlay lists for click handler
  const quickviewOverlaysRef = useRef<string[]>(quickviewOverlays);
  const highResOverlaysRef = useRef<string[]>(highResOverlays);
  const onSharadClickRef = useRef(onSharadClick);
  const onSharadHiresClickRef = useRef(onSharadHiresClick);
  const onHiRiseDTMClickRef = useRef(onHiRiseDTMClick);
  const onToggleOverlayRef = useRef(onToggleOverlay);
  const onTerrainClickRef = useRef(onTerrainClick);
  const onFieldNoteClickRef = useRef(onFieldNoteClick);
  const fieldNotesRef = useRef(fieldNotes);
  const onOlympusMonsTripleClickRef = useRef(onOlympusMonsTripleClick);
  const onOlympusMonsClimberRef = useRef(onOlympusMonsClimber);
  const onHoverProductRef = useRef(onHoverProduct);

  // Easter egg: Olympus Mons triple-click tracking
  const olympusMonsClickCountRef = useRef(0);
  const olympusMonsClickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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
    onOlympusMonsTripleClickRef.current = onOlympusMonsTripleClick;
  }, [onOlympusMonsTripleClick]);

  useEffect(() => {
    onOlympusMonsClimberRef.current = onOlympusMonsClimber;
  }, [onOlympusMonsClimber]);

  useEffect(() => {
    onTerrainClickRef.current = onTerrainClick;
  }, [onTerrainClick]);

  useEffect(() => {
    onHoverProductRef.current = onHoverProduct;
  }, [onHoverProduct]);

  const highlightRef = useRef<HighlightState>({
    key: null,
    rectEnt: null,
    labelEnt: null,
    pointEnt: null,
    origRectMaterial: undefined,
    origOutlineColor: undefined,
    origLabelScale: undefined,
    origPointSize: undefined,
  });

  const [hover, setHover] = useState<LatLon | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    try {
      const moduleUrl = Cesium as unknown as {
        buildModuleUrl?: { setBaseUrl?: (url: string) => void };
      };
      moduleUrl.buildModuleUrl?.setBaseUrl?.("/cesium/");
    } catch {}

    // P2: Suppress Cesium Ion entirely — MarsLab uses direct tile URLs on Mars,
    // not Ion-hosted Earth imagery. Setting defaultAccessToken to empty string
    // prevents ALL Ion API calls (terrain metadata, credits, assets/2/endpoint).
    // `baseLayer: false` alone only blocks the default Bing Maps layer.
    Cesium.Ion.defaultAccessToken = "";

    // Guard: ensure container has non-zero dimensions before creating viewer.
    // Cesium crashes with `DeveloperError: Expected width > 0` if the container
    // hasn't been laid out yet (e.g. tab hidden, CSS not applied).
    const { clientWidth, clientHeight } = containerRef.current;
    if (clientWidth === 0 || clientHeight === 0) {
      console.warn('[useMapViewer] Container has zero dimensions, deferring init');
      return;
    }
    let viewer: Cesium.Viewer;
    try {
    viewer = new Cesium.Viewer(containerRef.current, {
      sceneMode: Cesium.SceneMode.SCENE2D,
      mapProjection: new Cesium.GeographicProjection(marsEllipsoid),
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      baseLayer: false,  // P2: prevent default Ion/Bing imagery
      geocoder: false,
      homeButton: false,
      navigationHelpButton: false,
      sceneModePicker: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      terrainProvider: new Cesium.EllipsoidTerrainProvider({
        ellipsoid: marsEllipsoid,
      }),
    });
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK,
    );

    viewerRef.current = viewer;

    // P1+P0: Suppress Cesium tile-load and render errors (network CORS failures,
    // trek.nasa.gov opaque responses) so they don't crash the page.
    viewer.scene.renderError.addEventListener(() => {});
    viewer.scene.globe.tileLoadProgressEvent?.addEventListener?.(() => {});
    // Suppress tile request errors (CORS from trek.nasa.gov is expected)
    viewer.scene.imageryLayers?.layerAdded?.addEventListener?.(() => {});

    viewer.scene.globe = new Cesium.Globe(marsEllipsoid);
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewer.scene.globe.enableLighting = false;
    viewer.scene.backgroundColor = Cesium.Color.BLACK;

    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: baseLayerUrls[baseLayer],
        rectangle: marsRect,
        tilingScheme: new Cesium.GeographicTilingScheme({
          ellipsoid: marsEllipsoid,
          numberOfLevelZeroTilesX: 2,
          numberOfLevelZeroTilesY: 1,
        }),
      }),
    );

    // Set camera to show full Mars or restricted view bounds
    viewer.camera.flyTo({
      destination: marsRect,
      duration: 0,
      complete: () => {
        viewer.scene.requestRender();
      },
    });

    const hoverHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    // Throttled drillPick + highlight logic (expensive operations at 50ms intervals)
    let lastPickPosition: Cesium.Cartesian2 | null = null;
    const throttledPick = throttle((endPosition: Cesium.Cartesian2, hoverLat: number, hoverLon: number) => {
      // Check if hovering over an overlay and change cursor
      let isOverOverlay = false;
      const allOverlayIds = [...highResOverlaysRef.current, ...quickviewOverlaysRef.current];
      for (const productId of allOverlayIds) {
        const overlayEnt =
          viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`) ||
          viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
        if (overlayEnt?.rectangle?.coordinates) {
          const rect = overlayEnt.rectangle.coordinates.getValue(
            Cesium.JulianDate.now(),
          ) as Cesium.Rectangle;
          const west = Cesium.Math.toDegrees(rect.west);
          const east = Cesium.Math.toDegrees(rect.east);
          const south = Cesium.Math.toDegrees(rect.south);
          const north = Cesium.Math.toDegrees(rect.north);
          if (
            hoverLon >= west &&
            hoverLon <= east &&
            hoverLat >= south &&
            hoverLat <= north
          ) {
            isOverOverlay = true;
            break;
          }
        }
      }
      viewer.canvas.style.cursor = isOverOverlay ? "crosshair" : "default";

      const pickedList = viewer.scene.drillPick(endPosition);
      const picked = pickedList.find((x: any) => x?.id instanceof Cesium.Entity);
      const pickedPrimitiveId = pickedList.find(
        (x: any) => typeof x?.id === "string" && x.id.includes("_FP_"),
      )?.id as string | undefined;

      const pickedEnt = picked?.id as Cesium.Entity | undefined;
      const hs = highlightRef.current;

      const clearHighlight = () => {
        if (hs.rectEnt?.rectangle) {
          if (hs.origRectMaterial) {
            hs.rectEnt.rectangle.material = hs.origRectMaterial;
          }
          if (hs.origOutlineColor && hs.rectEnt.rectangle.outlineColor) {
            hs.rectEnt.rectangle.outlineColor = new Cesium.ConstantProperty(
              hs.origOutlineColor,
            );
          }
        }
        if (hs.labelEnt?.label && typeof hs.origLabelScale === "number") {
          hs.labelEnt.label.scale = new Cesium.ConstantProperty(hs.origLabelScale);
        }
        if (hs.pointEnt?.point && typeof hs.origPointSize === "number") {
          hs.pointEnt.point.pixelSize = new Cesium.ConstantProperty(hs.origPointSize);
        }

        hs.key = null;
        hs.rectEnt = null;
        hs.labelEnt = null;
        hs.pointEnt = null;
        hs.origRectMaterial = undefined;
        hs.origOutlineColor = undefined;
        hs.origLabelScale = undefined;
        hs.origPointSize = undefined;
        footprintManagerRef.current?.hideHoverLabel();
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

          const rectTarget = rectFallback?.rectangle ? rectFallback : null;

          const labelEnt =
            inst === "CUSTOM"
              ? viewer.entities.getById(`CUSTOM_LABEL_${pid}`) || null
              : viewer.entities.getById(`${inst}_LBL_${pid}`) ||
                viewer.entities.getById(`${inst}_VP_LABEL_${pid}`) ||
                viewer.entities.getById(`${inst}_LABEL_${pid}`) ||
                null;
          const pointEnt =
            inst === "CUSTOM"
              ? null
              : viewer.entities.getById(`${inst}_VP_POINT_${pid}`) ||
                viewer.entities.getById(`${inst}_POINT_${pid}`) ||
                null;

          clearHighlight();

          if (rectTarget?.rectangle) {
            hs.key = key;

            hs.rectEnt = rectTarget;
            hs.origRectMaterial = rectTarget.rectangle.material;

            const ocVal = rectTarget.rectangle.outlineColor?.getValue?.(
              Cesium.JulianDate.now(),
            );
            if (ocVal instanceof Cesium.Color) {
              hs.origOutlineColor = ocVal;
            }

            rectTarget.rectangle.material = getHiliteMaterial(inst);

            rectTarget.rectangle.outlineColor = new Cesium.ConstantProperty(
              Cesium.Color.WHITE,
            );

            if (labelEnt?.label) {
              hs.labelEnt = labelEnt;
              const cur = labelEnt.label.scale?.getValue?.(Cesium.JulianDate.now());
              hs.origLabelScale = typeof cur === "number" ? cur : 1.0;
              labelEnt.label.scale = new Cesium.ConstantProperty(1.2);
            }

            if (pointEnt?.point) {
              hs.pointEnt = pointEnt;
              const cur = pointEnt.point.pixelSize?.getValue?.(Cesium.JulianDate.now());
              hs.origPointSize = typeof cur === "number" ? cur : 6;
              pointEnt.point.pixelSize = new Cesium.ConstantProperty(
                (hs.origPointSize ?? 6) + 2,
              );
            }

            onHoverProductRef.current?.(pid);

            viewer.scene.requestRender();
            return;
          }
        }
      }

      if (pickedPrimitiveId && footprintManagerRef.current?.hasFeature(pickedPrimitiveId)) {
        const metadata = footprintManagerRef.current.getFeatureMetadata(pickedPrimitiveId);
        if (metadata) {
          const key = `${metadata.instrument}:${metadata.productId}`;
          if (hs.key === key) return;
          clearHighlight();
          hs.key = key;
          const pos = Cesium.Cartesian3.fromDegrees(
            (metadata.bounds.west + metadata.bounds.east) / 2,
            (metadata.bounds.south + metadata.bounds.north) / 2,
            0,
            marsEllipsoid,
          );
          footprintManagerRef.current.showHoverLabel(pos, metadata.productId);
          onHoverProductRef.current?.(metadata.productId);
          viewer.scene.requestRender();
          return;
        }
      }

      if (hs.key || hs.rectEnt || hs.labelEnt || hs.pointEnt) {
        clearHighlight();
        onHoverProductRef.current?.(null);
        viewer.scene.requestRender();
      }
    }, 50); // 50ms throttle = max 20 picks/sec (was unlimited)

    hoverHandler.setInputAction(
      (m: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
        const p = viewer.camera.pickEllipsoid(m.endPosition, marsEllipsoid);
        if (!p) return setHover(null);
        const c = Cesium.Cartographic.fromCartesian(p);
        const hoverLat = Cesium.Math.toDegrees(c.latitude);
        const hoverLon = Cesium.Math.toDegrees(c.longitude);
        setHover({ lat: hoverLat, lon: hoverLon });

        // Throttle the expensive drillPick + highlight operations
        lastPickPosition = m.endPosition.clone();
        throttledPick(lastPickPosition, hoverLat, hoverLon);
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE,
    );

    const clickHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    clickHandler.setInputAction(
      async (m: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        // Get click position in lat/lon FIRST
        const clickCart = viewer.camera.pickEllipsoid(m.position, marsEllipsoid);
        if (!clickCart) return;

        const clickCarto = Cesium.Cartographic.fromCartesian(clickCart);
        const clickLon = Cesium.Math.toDegrees(clickCarto.longitude);
        const clickLat = Cesium.Math.toDegrees(clickCarto.latitude);

        // Easter egg: Olympus Mons summit clicks (~18.65°N, -133.5°E)
        // 3 clicks → height comparison, 7 clicks → climber animation
        if (
          Math.abs(clickLat - 18.65) < 1.5 &&
          Math.abs(clickLon - -133.5) < 1.5
        ) {
          olympusMonsClickCountRef.current += 1;
          if (olympusMonsClickTimerRef.current) {
            clearTimeout(olympusMonsClickTimerRef.current);
          }
          olympusMonsClickTimerRef.current = setTimeout(() => {
            olympusMonsClickCountRef.current = 0;
          }, 5000);
          if (olympusMonsClickCountRef.current >= 7) {
            olympusMonsClickCountRef.current = 0;
            if (olympusMonsClickTimerRef.current) {
              clearTimeout(olympusMonsClickTimerRef.current);
            }
            onOlympusMonsClimberRef.current?.();
            return;
          }
          if (olympusMonsClickCountRef.current === 3) {
            onOlympusMonsTripleClickRef.current?.();
            return;
          }
        }

        // PRIORITY 1: Check if click is within any active overlay bounds
        // This is more reliable than Cesium picking for image overlays
        let overlayProduct: { productId: string; instrument: InstrumentType } | null = null;

        // Check high-res overlays first (higher priority), then quickview
        const highResIds = highResOverlaysRef.current;
        const quickviewIds = quickviewOverlaysRef.current;
        const allOverlayIds = [...highResIds, ...quickviewIds];

        for (const productId of allOverlayIds) {
          const highResEnt = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
          const quickviewEnt = viewer.entities.getById(
            `QUICKVIEW_OVERLAY_${productId}`,
          );
          const overlayEnt = highResEnt || quickviewEnt;

          if (overlayEnt?.rectangle?.coordinates) {
            const rect = overlayEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now(),
            ) as Cesium.Rectangle;
            const west = Cesium.Math.toDegrees(rect.west);
            const east = Cesium.Math.toDegrees(rect.east);
            const south = Cesium.Math.toDegrees(rect.south);
            const north = Cesium.Math.toDegrees(rect.north);

            if (
              clickLon >= west &&
              clickLon <= east &&
              clickLat >= south &&
              clickLat <= north
            ) {
              const instrumentProp = (
                overlayEnt.properties as unknown as {
                  instrument?: Cesium.Property;
                }
              )?.instrument;
              const instrument = instrumentProp?.getValue(
                Cesium.JulianDate.now(),
              ) as InstrumentType | undefined;
              overlayProduct = {
                productId,
                instrument: instrument || (productId.startsWith("ESP_") ? "HIRISE" : "CRISM"),
              };
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

              if (
                minLat != null &&
                maxLat != null &&
                westLon360 != null &&
                eastLon360 != null &&
                lines &&
                samples
              ) {
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
                  const latFrac =
                    (bounds.north - clickLat) / (bounds.north - bounds.south);
                  const lonFrac =
                    (clickLon - bounds.west) / (bounds.east - bounds.west);
                  pixelLine = Math.floor(latFrac * shape.rows);
                  pixelSample = Math.floor(lonFrac * shape.cols);
                  pixelLine = Math.max(0, Math.min(shape.rows - 1, pixelLine));
                  pixelSample = Math.max(0, Math.min(shape.cols - 1, pixelSample));
                }
              }
            } catch {
              // Ignore shape fetch failure
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

              if (
                minLat != null &&
                maxLat != null &&
                westLon360 != null &&
                eastLon360 != null &&
                lines &&
                samples
              ) {
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
          const fpInst = productId.startsWith("DTE")
            ? "HIRISE_DTM"
            : productId.startsWith("ESP_")
              ? "HIRISE"
              : isTRR3Product
                ? "CRISM_TRR3"
                : "CRISM";
          const fpEnt = viewer.entities.getById(`${fpInst}_FP_${productId}`);
          const fpMeta = footprintManagerRef.current?.getFeatureMetadata(`${fpInst}_FP_${productId}`) ?? null;
          if (fpEnt?.rectangle?.coordinates) {
            const fpRect = fpEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now(),
            ) as Cesium.Rectangle;
            overlayLat = Cesium.Math.toDegrees((fpRect.south + fpRect.north) / 2);
            overlayLon = Cesium.Math.toDegrees((fpRect.west + fpRect.east) / 2);
          } else if (fpMeta) {
            overlayLat = (fpMeta.bounds.south + fpMeta.bounds.north) / 2;
            overlayLon = (fpMeta.bounds.west + fpMeta.bounds.east) / 2;
          }

          const overlayTitle = fpEnt?.properties?.title?.getValue?.() as
            | string
            | undefined
            ?? (typeof fpMeta?.properties.title === "string" ? fpMeta.properties.title : undefined);
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
          // Find the full note data from fieldNotesRef
          const note = fieldNotesRef.current.find((n) => n.id === noteId);
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

        const pickedPrimitiveId = pickedList.find(
          (p: any) => typeof p?.id === "string" && p.id.includes("_FP_"),
        )?.id as string | undefined;
        const pickedPrimitiveMetadata = pickedPrimitiveId
          ? (footprintManagerRef.current?.getFeatureMetadata(pickedPrimitiveId) ?? null)
          : null;

        if ((!picked || !(picked.id instanceof Cesium.Entity)) && !pickedPrimitiveMetadata) {
          onTerrainClickRef.current?.(clickLat, clickLon);
          return;
        }

        const e = picked?.id instanceof Cesium.Entity ? (picked.id as Cesium.Entity) : null;
        const p = e?.properties;

        const getPickedProperty = (key: string): unknown => {
          if (pickedPrimitiveMetadata) return pickedPrimitiveMetadata.properties[key];
          const prop = p?.[key];
          if (prop && typeof prop.getValue === "function") {
            return prop.getValue(Cesium.JulianDate.now());
          }
          return prop;
        };

        const productId = (getPickedProperty("product_id") as string | undefined)
          ?? pickedPrimitiveMetadata?.productId;
        const instrument = (getPickedProperty("instrument") as InstrumentType | undefined)
          ?? (pickedPrimitiveMetadata?.instrument as InstrumentType | undefined);

        if (!productId || !instrument) return;

        // Handle CUSTOM datasets - fly to bounds
        if (instrument === "CUSTOM") {
          let customLat = clickLat;
          let customLon = clickLon;
          const customRectEnt = viewer.entities.getById(`CUSTOM_FP_${productId}`);
          if (customRectEnt?.rectangle?.coordinates) {
            const cr = customRectEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now(),
            ) as Cesium.Rectangle;
            customLat = Cesium.Math.toDegrees((cr.south + cr.north) / 2);
            customLon = Cesium.Math.toDegrees((cr.west + cr.east) / 2);
          }

          onSelect({
            instrument: "CUSTOM",
            productId,
            lat: customLat,
            lon: customLon,
          });

          const rectEnt =
            customRectEnt || viewer.entities.getById(`CUSTOM_FP_${productId}`);
          if (rectEnt?.rectangle?.coordinates) {
            const rect = rectEnt.rectangle.coordinates.getValue(
              Cesium.JulianDate.now(),
            ) as Cesium.Rectangle;
            viewer.camera.flyTo({
              destination: paddedRectangle(rect, 0.3),
              duration: 0.6,
            });
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
              Cesium.JulianDate.now(),
            ) as Cesium.Rectangle;
            dtmLat = Cesium.Math.toDegrees((rect.south + rect.north) / 2);
            dtmLon = Cesium.Math.toDegrees((rect.west + rect.east) / 2);
            viewer.camera.flyTo({
              destination: paddedRectangle(rect, 0.5),
              duration: 0.6,
            });
          } else {
            const bounds = footprintManagerRef.current?.getFeatureBounds(`HIRISE_DTM_FP_${productId}`);
            if (bounds) {
              dtmLat = (bounds.south + bounds.north) / 2;
              dtmLon = (bounds.west + bounds.east) / 2;
              viewer.camera.flyTo({
                destination: paddedRectangle(
                  Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
                  0.5,
                ),
                duration: 0.6,
              });
            }
          }

          const dtmTitle = dtmRectEnt?.properties?.title?.getValue?.() as
            | string
            | undefined
            ?? (typeof pickedPrimitiveMetadata?.properties.title === "string"
              ? pickedPrimitiveMetadata.properties.title
              : undefined);
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
          const startLat = Number(getPickedProperty("start_lat") ?? 0);
          const startLon = Number(getPickedProperty("start_lon") ?? 0);
          const stopLat = Number(getPickedProperty("stop_lat") ?? 0);
          const stopLon = Number(getPickedProperty("stop_lon") ?? 0);

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
              Cesium.JulianDate.now(),
            ) as Cesium.Rectangle;
            viewer.camera.flyTo({
              destination: paddedRectangle(rect, 0.3),
              duration: 0.6,
            });
          } else {
            const bounds = footprintManagerRef.current?.getFeatureBounds(`CTX_FP_${productId}`);
            if (bounds) {
              viewer.camera.flyTo({
                destination: paddedRectangle(
                  Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
                  0.3,
                ),
                duration: 0.6,
              });
            }
          }
          return;
        }

        // Compute footprint center for accurate lat/lon (click position can be offset)
        const rectEntId = `${instrument}_FP_${productId}`;
        const rectEnt = viewer.entities.getById(rectEntId);

        let selectLat = clickLat;
        let selectLon = clickLon;
        if (rectEnt?.rectangle?.coordinates) {
          const rect = rectEnt.rectangle.coordinates.getValue(
            Cesium.JulianDate.now(),
          ) as Cesium.Rectangle;
          selectLat = Cesium.Math.toDegrees((rect.south + rect.north) / 2);
          selectLon = Cesium.Math.toDegrees((rect.west + rect.east) / 2);
        } else {
          const bounds = footprintManagerRef.current?.getFeatureBounds(rectEntId);
          if (bounds) {
            selectLat = (bounds.south + bounds.north) / 2;
            selectLon = (bounds.west + bounds.east) / 2;
          }
        }

        onSelect({
          instrument,
          productId,
          lat: selectLat,
          lon: selectLon,
        });

        if (rectEnt?.rectangle?.coordinates) {
          const rect = rectEnt.rectangle.coordinates.getValue(
            Cesium.JulianDate.now(),
          ) as Cesium.Rectangle;

          const dest = paddedRectangle(rect, 0.6);

          viewer.camera.flyTo({
            destination: dest,
            duration: 0.6,
          });
        } else {
          const bounds = footprintManagerRef.current?.getFeatureBounds(rectEntId);
          if (bounds) {
            viewer.camera.flyTo({
              destination: paddedRectangle(
                Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
                0.6,
              ),
              duration: 0.6,
            });
          } else if (instrument === "HIRISE" || instrument === "CRISM") {
            const isHiRISE = instrument === "HIRISE";
            const lbl = isHiRISE
              ? await loadHiRISELBL(productId)
              : await loadCRISMLBL(productId);

            if (lbl) {
              const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
              const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
              const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
              const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

              if (
                minLat != null &&
                maxLat != null &&
                westLon360 != null &&
                eastLon360 != null
              ) {
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
        }
      },
      Cesium.ScreenSpaceEventType.LEFT_CLICK,
    );

    const cleanupDTMHover = initializeDTMHover(viewer);

    return () => {
      hoverHandler.destroy();
      clickHandler.destroy();
      cleanupDTMHover();
      if (olympusMonsClickTimerRef.current) {
        clearTimeout(olympusMonsClickTimerRef.current);
      }
      if (!viewer.isDestroyed()) {
        viewer.destroy();
      }
      if (viewerRef.current === viewer) {
        viewerRef.current = null;
      }
    };
    } catch (err) {
      console.error('[useMapViewer] Cesium initialization failed:', err);
      setInitError(err instanceof Error ? err.message : 'Failed to initialize map viewer');
      return;
    }
  }, []);

  // Keep cameraViewportRef updated on every camera moveEnd (for on-demand reads)
  // Uses screen corner-picking on the Mars ellipsoid for accurate viewport bounds
  useEffect(() => {
    const viewportRef = cameraViewportRef;
    if (!viewportRef) return;
    let removeListener: (() => void) | null = null;

    function updateViewport() {
      const v = viewerRef.current;
      if (!v) return;
      try {
        const canvas = v.scene.canvas;
        const cam = v.camera;

        // Sample a grid of screen points and pick positions on the Mars ellipsoid
        const gridSize = 5;
        const lons: number[] = [];
        const lats: number[] = [];

        for (let i = 0; i <= gridSize; i++) {
          for (let j = 0; j <= gridSize; j++) {
            const pt = new Cesium.Cartesian2(
              (canvas.width * i) / gridSize,
              (canvas.height * j) / gridSize,
            );
            const c = cam.pickEllipsoid(pt, marsEllipsoid);
            if (c) {
              const carto = Cesium.Cartographic.fromCartesian(c, marsEllipsoid);
              lons.push(Cesium.Math.toDegrees(carto.longitude));
              lats.push(Cesium.Math.toDegrees(carto.latitude));
            }
          }
        }

        // Need enough points to have a meaningful viewport
        if (lats.length < 4) return;

        const south = Math.min(...lats);
        const north = Math.max(...lats);
        const west = Math.min(...lons);
        const east = Math.max(...lons);

        // Sanity: if viewport spans nearly full globe, the user is zoomed out too far
        // for meaningful landform loading — still store it, but frontend can warn
        if (viewportRef) {
          viewportRef.current = {
            minLat: south,
            maxLat: north,
            westLon: west,
            eastLon: east,
          };
        }
      } catch {
        // viewer not ready yet
      }
    }

    // Poll until viewer is ready, then attach listener
    const poll = setInterval(() => {
      const viewer = viewerRef.current;
      if (!viewer) return;
      clearInterval(poll);
      updateViewport();
      removeListener = viewer.camera.moveEnd.addEventListener(updateViewport);
    }, 150);

    return () => {
      clearInterval(poll);
      removeListener?.();
    };
  }, [cameraViewportRef, marsEllipsoid]);

  const switchSceneMode = useCallback((mode: MapMode) => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const targetMode = mode === "3D" ? Cesium.SceneMode.SCENE3D : Cesium.SceneMode.SCENE2D;

    if (viewer.scene.mode !== targetMode) {
      // Morph to new mode with animation
      if (targetMode === Cesium.SceneMode.SCENE3D) {
        viewer.scene.morphTo3D(1.0);
      } else {
        viewer.scene.morphTo2D(1.0);
      }
    }
  }, []);

  const switchBaseLayer = useCallback(
    (layer: BaseLayerType) => {
      const viewer = viewerRef.current;
      if (!viewer) return;

      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(
        new Cesium.UrlTemplateImageryProvider({
          url: baseLayerUrls[layer],
          rectangle: marsRect,
          tilingScheme: new Cesium.GeographicTilingScheme({
            ellipsoid: marsEllipsoid,
            numberOfLevelZeroTilesX: 2,
            numberOfLevelZeroTilesY: 1,
          }),
        }),
      );

      viewer.scene.requestRender();
    },
    [baseLayerUrls, marsEllipsoid, marsRect],
  );

  // Switch between 2D and 3D map modes
  useEffect(() => {
    switchSceneMode(mapMode);
  }, [mapMode, switchSceneMode]);

  // Update base layer when baseLayer changes
  useEffect(() => {
    switchBaseLayer(baseLayer);
  }, [baseLayer, switchBaseLayer]);

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
        destination: marsRect,
      });
      viewer.scene.requestRender();
    }
  }, [viewBounds, marsRect, normalizeLonTo180]);

  return {
    viewerRef,
    hover,
    initError,
    switchSceneMode,
    switchBaseLayer,
  };
}
