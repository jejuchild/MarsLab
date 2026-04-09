import { useCallback, useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import {
  loadDTMElevationGrid,
  getElevationFromGrid,
  isWithinDTMBounds,
  throttle,
  type DTMElevationGrid,
} from "../utils/dtmHover";
import type { DTMHoverReadoutHandle } from "../components/DTMHoverReadout";

type UseDTMHoverParams = {
  activeDTMProductId: string | null;
  marsEllipsoid: Cesium.Ellipsoid;
};

type UseDTMHoverResult = {
  dtmHoverReadoutRef: React.RefObject<DTMHoverReadoutHandle | null>;
  dtmGridCacheRef: React.MutableRefObject<Map<string, DTMElevationGrid>>;
  activeDTMProductRef: React.MutableRefObject<string | null>;
  dtmHoverMode: "hover" | "click";
  setDtmGrid: (productId: string, grid: DTMElevationGrid) => void;
  handleDTMHoverModeChange: (mode: "hover" | "click") => void;
  initializeDTMHover: (viewer: Cesium.Viewer) => () => void;
};

export default function useDTMHover({
  activeDTMProductId,
  marsEllipsoid,
}: UseDTMHoverParams): UseDTMHoverResult {
  const dtmHoverReadoutRef = useRef<DTMHoverReadoutHandle>(null);
  const dtmHoverMarkerRef = useRef<Cesium.Entity | null>(null);
  const dtmGridCacheRef = useRef<Map<string, DTMElevationGrid>>(new Map());
  const DTM_GRID_CACHE_MAX = 4;
  const activeDTMProductRef = useRef<string | null>(null);
  const dtmHoverModeRef = useRef<"hover" | "click">("hover");
  const [dtmHoverMode, setDtmHoverMode] = useState<"hover" | "click">("hover");

  const setDtmGrid = useCallback((productId: string, grid: DTMElevationGrid) => {
    const cache = dtmGridCacheRef.current;
    cache.set(productId, grid);
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

  useEffect(() => {
    dtmHoverModeRef.current = dtmHoverMode;
  }, [dtmHoverMode]);

  useEffect(() => {
    if (activeDTMProductId) {
      activeDTMProductRef.current = activeDTMProductId;
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
  }, [activeDTMProductId, setDtmGrid]);

  const handleDTMHoverModeChange = useCallback((mode: "hover" | "click") => {
    setDtmHoverMode(mode);
    if (mode === "click") {
      if (dtmHoverMarkerRef.current) {
        dtmHoverMarkerRef.current.show = false;
      }
      dtmHoverReadoutRef.current?.hide();
    }
  }, []);

  const initializeDTMHover = useCallback((viewer: Cesium.Viewer) => {
    const dtmHoverMarker = viewer.entities.add({
      id: "DTM_HOVER_MARKER",
      position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
      billboard: {
        image: (() => {
          const canvas = document.createElement("canvas");
          canvas.width = 24;
          canvas.height = 24;
          const ctx = canvas.getContext("2d")!;
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(12, 12, 8, 0, Math.PI * 2);
          ctx.stroke();
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
      show: false,
    });
    dtmHoverMarkerRef.current = dtmHoverMarker;

    const dtmHoverHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    const throttledDTMHover = throttle((endPosition: Cesium.Cartesian2) => {
      if (dtmHoverModeRef.current !== "hover") return;

      const cartesian = viewer.camera.pickEllipsoid(endPosition, marsEllipsoid);
      if (!cartesian) {
        dtmHoverMarker.show = false;
        dtmHoverReadoutRef.current?.hide();
        return;
      }

      const carto = Cesium.Cartographic.fromCartesian(cartesian);
      const lat = Cesium.Math.toDegrees(carto.latitude);
      const lon = Cesium.Math.toDegrees(carto.longitude);

      const grid = dtmGridCacheRef.current.get(activeDTMProductRef.current || "");
      if (!grid || !isWithinDTMBounds(grid, lat, lon)) {
        dtmHoverMarker.show = false;
        dtmHoverReadoutRef.current?.hide();
        return;
      }

      const elevation = getElevationFromGrid(grid, lat, lon);
      dtmHoverMarker.position = Cesium.Cartesian3.fromDegrees(lon, lat, 0) as unknown as Cesium.PositionProperty;
      dtmHoverMarker.show = true;
      dtmHoverReadoutRef.current?.update(lat, lon, elevation, grid.productId);

      viewer.scene.requestRender();
    }, 40);

    dtmHoverHandler.setInputAction(
      (m: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
        throttledDTMHover(m.endPosition);
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE,
    );

    return () => {
      dtmHoverHandler.destroy();
    };
  }, [marsEllipsoid]);

  return {
    dtmHoverReadoutRef,
    dtmGridCacheRef,
    activeDTMProductRef,
    dtmHoverMode,
    setDtmGrid,
    handleDTMHoverModeChange,
    initializeDTMHover,
  };
}
