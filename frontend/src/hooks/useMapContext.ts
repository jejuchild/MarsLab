/**
 * useMapContext — Extracts current viewport bounds for the Workflow API.
 *
 * Provides the current map camera viewport so the Agentic panel
 * can scope searches to the user's visible region.
 */
import { useCallback, useRef, useState } from "react";

export type ViewportBounds = {
  west: number;
  south: number;
  east: number;
  north: number;
};

export type MapContext = {
  viewport: ViewportBounds | null;
  selected_features: string[];
};

/**
 * Hook that tracks the current map viewport and selected features.
 *
 * The AgenticPanel can call `getMapContext()` to pass viewport info
 * to the agent API.
 */
export default function useMapContext() {
  const [viewport, setViewport] = useState<ViewportBounds | null>(null);
  const selectedFeaturesRef = useRef<string[]>([]);

  /** Called by MapView on camera change (debounced). */
  const updateViewport = useCallback((bounds: ViewportBounds) => {
    setViewport(bounds);
  }, []);

  /** Called when products are selected on the map. */
  const updateSelectedFeatures = useCallback((productIds: string[]) => {
    selectedFeaturesRef.current = productIds;
  }, []);

  /** Returns the current map context for the workflow API. */
  const getMapContext = useCallback((): MapContext => {
    return {
      viewport,
      selected_features: selectedFeaturesRef.current,
    };
  }, [viewport]);

  return {
    viewport,
    updateViewport,
    updateSelectedFeatures,
    getMapContext,
  };
}
