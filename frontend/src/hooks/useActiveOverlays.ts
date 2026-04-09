import { useCallback, useRef, useState } from "react";
import type { ActiveOverlays, OverlayType, ProductOverlay } from "../pages/MainPage";

/* =========================================================
 * Hook
 * =======================================================*/

export interface UseActiveOverlaysReturn {
  /** The activeOverlays Map (productId → overlay state). */
  activeOverlays: ActiveOverlays;
  /** Stable ref to activeOverlays — read inside callbacks without depending on the Map identity. */
  activeOverlaysRef: React.MutableRefObject<ActiveOverlays>;

  /** Replace one product's overlay (or remove if `null`). */
  setOverlay: (productId: string, overlay: ProductOverlay | null) => void;
  /** Set just the opacity for an existing overlay. */
  setOpacity: (productId: string, opacity: number) => void;
  /** Convenience: change a product's overlay type while preserving opacity. */
  changeType: (productId: string, type: OverlayType, defaultOpacity?: number) => void;
  /** Remove all overlays. */
  clear: () => void;
  /** Remove a single product's overlay. */
  remove: (productId: string) => void;
  /** Remove all overlays whose productId is in the given set. */
  removeMany: (productIds: Iterable<string>) => void;
}

const DEFAULT_OPACITY = 100;

/**
 * useActiveOverlays — owns the productId → ProductOverlay map.
 *
 * Lifted from MainPage. The existing rendering hook (`useOverlays`) consumes
 * the resulting Map via props; this hook only manages state mutations.
 */
export default function useActiveOverlays(): UseActiveOverlaysReturn {
  const [activeOverlays, setActiveOverlaysState] = useState<ActiveOverlays>(() => new Map());

  // Stable ref pattern — keeps callbacks dependency-free.
  const activeOverlaysRef = useRef<ActiveOverlays>(activeOverlays);
  activeOverlaysRef.current = activeOverlays;

  const setOverlay = useCallback((productId: string, overlay: ProductOverlay | null) => {
    setActiveOverlaysState((prev) => {
      const next = new Map(prev);
      if (overlay === null) {
        next.delete(productId);
      } else {
        next.set(productId, overlay);
      }
      return next;
    });
  }, []);

  const setOpacity = useCallback((productId: string, opacity: number) => {
    setActiveOverlaysState((prev) => {
      const cur = prev.get(productId);
      if (!cur) return prev;
      const next = new Map(prev);
      next.set(productId, { ...cur, opacity });
      return next;
    });
  }, []);

  const changeType = useCallback(
    (productId: string, type: OverlayType, defaultOpacity = DEFAULT_OPACITY) => {
      setActiveOverlaysState((prev) => {
        const cur = prev.get(productId);
        const opacity = cur?.opacity ?? defaultOpacity;
        const next = new Map(prev);
        next.set(productId, { type, opacity });
        return next;
      });
    },
    []
  );

  const clear = useCallback(() => {
    setActiveOverlaysState(new Map());
  }, []);

  const remove = useCallback((productId: string) => {
    setActiveOverlaysState((prev) => {
      if (!prev.has(productId)) return prev;
      const next = new Map(prev);
      next.delete(productId);
      return next;
    });
  }, []);

  const removeMany = useCallback((productIds: Iterable<string>) => {
    setActiveOverlaysState((prev) => {
      const ids = new Set(productIds);
      let changed = false;
      const next = new Map(prev);
      for (const id of ids) {
        if (next.delete(id)) changed = true;
      }
      return changed ? next : prev;
    });
  }, []);

  return {
    activeOverlays,
    activeOverlaysRef,
    setOverlay,
    setOpacity,
    changeType,
    clear,
    remove,
    removeMany,
  };
}
