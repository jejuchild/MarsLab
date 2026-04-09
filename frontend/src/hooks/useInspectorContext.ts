import { useCallback, useState } from "react";
import type { InspectorContext, RecentProduct } from "../components/inspector/types";
import type { TerrainPoint } from "../components/SlopeAnalysis";
import type { HiRiseDTMPoint } from "../components/HiRiseDTM3DViewer";

/* =========================================================
 * Hook
 * =======================================================*/

export interface UseInspectorContextReturn {
  // Currently selected product (clicking a footprint)
  selected: InspectorContext | null;
  setSelected: (ctx: InspectorContext | null) => void;

  // Recent products list (last 5)
  recentProducts: RecentProduct[];
  addRecentProduct: (ctx: InspectorContext) => void;
  removeRecentProduct: (productId: string) => void;

  // Terrain click point (slope analysis mode)
  terrainPoint: TerrainPoint | null;
  setTerrainPoint: (p: TerrainPoint | null) => void;

  // HiRISE DTM 3D analysis point
  hiRiseDTM3DPoint: HiRiseDTMPoint | null;
  setHiRiseDTM3DPoint: (p: HiRiseDTMPoint | null) => void;

  // Active HiRISE DTM product (for terrain clicks in dtm 3d mode)
  activeDTMProduct: string | null;
  setActiveDTMProduct: (id: string | null) => void;

  // Right panel collapse state
  rightPanelCollapsed: boolean;
  setRightPanelCollapsed: (v: boolean) => void;
}

/**
 * useInspectorContext — owns inspector state.
 *
 * Phase 2: lifts state out of MainPage. The legacy `selected` shape (a
 * single product context) is preserved for now. Phase 3 will replace this
 * with the 4-lane point/product context model from design.md.
 */
export default function useInspectorContext(): UseInspectorContextReturn {
  const [selected, setSelectedState] = useState<InspectorContext | null>(null);
  const [recentProducts, setRecentProducts] = useState<RecentProduct[]>([]);
  const [terrainPoint, setTerrainPoint] = useState<TerrainPoint | null>(null);
  const [hiRiseDTM3DPoint, setHiRiseDTM3DPoint] = useState<HiRiseDTMPoint | null>(null);
  const [activeDTMProduct, setActiveDTMProduct] = useState<string | null>(null);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);

  const setSelected = useCallback((ctx: InspectorContext | null) => {
    setSelectedState(ctx);
  }, []);

  const addRecentProduct = useCallback((ctx: InspectorContext) => {
    setRecentProducts((prev) => {
      const filtered = prev.filter((p) => p.productId !== ctx.productId);
      return [
        {
          productId: ctx.productId,
          instrument: ctx.instrument,
          lat: ctx.lat,
          lon: ctx.lon,
          title: ctx.title,
        },
        ...filtered,
      ].slice(0, 5);
    });
  }, []);

  const removeRecentProduct = useCallback((productId: string) => {
    setRecentProducts((prev) => {
      const next = prev.filter((p) => p.productId !== productId);
      // If we removed the currently-selected product, switch to the next one or close
      setSelectedState((cur) => {
        if (cur && cur.productId === productId) {
          const fallback = next[0];
          return fallback
            ? {
                instrument: fallback.instrument,
                productId: fallback.productId,
                lat: fallback.lat,
                lon: fallback.lon,
                title: fallback.title,
              }
            : null;
        }
        return cur;
      });
      return next;
    });
  }, []);

  return {
    selected,
    setSelected,
    recentProducts,
    addRecentProduct,
    removeRecentProduct,
    terrainPoint,
    setTerrainPoint,
    hiRiseDTM3DPoint,
    setHiRiseDTM3DPoint,
    activeDTMProduct,
    setActiveDTMProduct,
    rightPanelCollapsed,
    setRightPanelCollapsed,
  };
}
