import { useEffect, useRef } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { MarsGridImageryProvider } from "../utils/GridImageryProvider";

type UseGridOverlayParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  showGrid: boolean;
};

export default function useGridOverlay({
  viewerRef,
  showGrid,
}: UseGridOverlayParams): void {
  const gridLayerRef = useRef<Cesium.ImageryLayer | null>(null);

  // Coordinate Grid Overlay (imagery-based)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (!showGrid) {
      if (gridLayerRef.current) {
        viewer.imageryLayers.remove(gridLayerRef.current, false);
        gridLayerRef.current = null;
      }
      return;
    }

    // Add grid imagery layer
    const provider = new MarsGridImageryProvider();
    const layer = viewer.imageryLayers.addImageryProvider(provider as unknown as Cesium.ImageryProvider);
    layer.alpha = 1.0;
    gridLayerRef.current = layer;

    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      if (gridLayerRef.current) {
        viewer.imageryLayers.remove(gridLayerRef.current, false);
        gridLayerRef.current = null;
      }
    };
  }, [showGrid, viewerRef]);
}
