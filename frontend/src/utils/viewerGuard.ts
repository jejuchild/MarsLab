/**
 * Shared viewer safety utility.
 *
 * Multiple hooks access `viewer.scene` which can crash if the viewer is
 * not yet initialized (race condition on mount) or already destroyed
 * (cleanup after navigation away from the map page).
 *
 * This replaces per-hook `getViewer()` helpers with a single source.
 */
import type * as Cesium from "cesium";
import type React from "react";

/** Returns the viewer only if it exists and has not been destroyed. */
export function getViewer(
  ref: React.MutableRefObject<Cesium.Viewer | null>,
): Cesium.Viewer | null {
  const v = ref.current;
  return v && !v.isDestroyed() ? v : null;
}

/** Convenience: calls `viewer.scene.requestRender()` safely. */
export function safeRequestRender(
  ref: React.MutableRefObject<Cesium.Viewer | null>,
): void {
  const v = getViewer(ref);
  v?.scene.requestRender();
}
