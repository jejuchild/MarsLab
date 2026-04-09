import { useCallback, useRef, useState } from "react";
import type { BaseLayerType, MapMode, BoundingBox } from "../pages/MainPage";

/* =========================================================
 * Camera viewport (updated by MapView on moveEnd)
 * =======================================================*/

export type CameraViewport = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;

/* =========================================================
 * Hook
 * =======================================================*/

export interface UseMapNavigationReturn {
  // Base layer
  baseLayer: BaseLayerType;
  setBaseLayer: (l: BaseLayerType) => void;

  // 2D / 3D mode
  mapMode: MapMode;
  setMapMode: (m: MapMode) => void;

  // View bounds (bbox restriction)
  viewBounds: BoundingBox;
  setViewBounds: (b: BoundingBox) => void;

  // Drag-to-select bbox mode
  viewBoundSelectionMode: boolean;
  setViewBoundSelectionMode: (active: boolean) => void;

  // Camera viewport ref — read on demand by panels that need
  // the current map viewport without triggering re-renders.
  cameraViewportRef: React.MutableRefObject<CameraViewport>;
  setCameraViewport: (v: CameraViewport) => void;

  // Fly-to coordinates: set this to trigger MapView to fly to lat/lon
  flyToCoords: { lat: number; lon: number } | null;
  setFlyToCoords: (c: { lat: number; lon: number } | null) => void;

  // Fly-to product: set this to trigger MapView to fly to a specific product footprint
  flyToProductId: string | null;
  setFlyToProductId: (id: string | null) => void;

  // Bring-to-front: set this to raise an overlay above others
  bringToFrontId: string | null;
  setBringToFrontId: (id: string | null) => void;

  // Highlight: set this to flash a feature briefly (deep-link arrival animation)
  highlightProductId: string | null;
  setHighlightProductId: (id: string | null) => void;

  // Convenience: fly to a coordinate (sets flyToCoords)
  flyTo: (lat: number, lon: number) => void;
}

/**
 * useMapNavigation — owns map view state.
 *
 * Absorbs from MainPage: baseLayer, mapMode, viewBounds, cameraViewportRef,
 * viewBoundSelectionMode, flyToCoords, flyToProductId, bringToFrontId,
 * highlightProductId.
 */
export default function useMapNavigation(): UseMapNavigationReturn {
  const [baseLayer, setBaseLayer] = useState<BaseLayerType>("MOLA");
  const [mapMode, setMapMode] = useState<MapMode>("2D");
  const [viewBounds, setViewBounds] = useState<BoundingBox>(null);
  const [viewBoundSelectionMode, setViewBoundSelectionMode] = useState(false);

  const cameraViewportRef = useRef<CameraViewport>(null);
  const setCameraViewport = useCallback((v: CameraViewport) => {
    cameraViewportRef.current = v;
  }, []);

  const [flyToCoords, setFlyToCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [flyToProductId, setFlyToProductId] = useState<string | null>(null);
  const [bringToFrontId, setBringToFrontId] = useState<string | null>(null);
  const [highlightProductId, setHighlightProductId] = useState<string | null>(null);

  const flyTo = useCallback((lat: number, lon: number) => {
    setFlyToCoords({ lat, lon });
  }, []);

  return {
    baseLayer,
    setBaseLayer,
    mapMode,
    setMapMode,
    viewBounds,
    setViewBounds,
    viewBoundSelectionMode,
    setViewBoundSelectionMode,
    cameraViewportRef,
    setCameraViewport,
    flyToCoords,
    setFlyToCoords,
    flyToProductId,
    setFlyToProductId,
    bringToFrontId,
    setBringToFrontId,
    highlightProductId,
    setHighlightProductId,
    flyTo,
  };
}
