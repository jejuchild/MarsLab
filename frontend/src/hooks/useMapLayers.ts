import type React from "react";
import * as Cesium from "cesium";

/**
 * useMapLayers — Imagery layer manager.
 *
 * After Phase 1 (SWIM/Accessibility/Fusion cut), this hook is a no-op stub
 * that retains the param signature for MapView's call site. It will be
 * removed entirely in Phase 2 when MapView's hook composition is reorganized.
 */
type UseMapLayersParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
};

export default function useMapLayers(_params: UseMapLayersParams): void {
  // No-op: SWIM/accessibility/fusion overlays were cut in Phase 1.
}
