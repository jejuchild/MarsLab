import { useEffect, useCallback } from "react";
import * as Cesium from "cesium";

interface UseMapKeyboardOptions {
  viewer: Cesium.Viewer | null;
  onToggleLayerPanel?: () => void;
  onToggleMeasurement?: () => void;
  onFlyToSelection?: () => void;
  onAddBookmark?: () => void;
  enabled?: boolean;
}

/**
 * Hook that binds keyboard shortcuts for map controls.
 * Ignores key events when focus is in input/textarea/select elements.
 */
export default function useMapKeyboard({
  viewer,
  onToggleLayerPanel,
  onToggleMeasurement,
  onFlyToSelection,
  onAddBookmark,
  enabled = true,
}: UseMapKeyboardOptions) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Skip if typing in form fields
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.target as HTMLElement)?.isContentEditable) return;

      // Skip if modifier keys are held (allow Ctrl/Cmd shortcuts to work)
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      if (!viewer || viewer.isDestroyed()) return;

      const PAN_AMOUNT = viewer.camera.positionCartographic.height * 0.1;

      switch (e.key) {
        case "+":
        case "=":
          e.preventDefault();
          viewer.camera.zoomIn(viewer.camera.positionCartographic.height * 0.3);
          break;
        case "-":
          e.preventDefault();
          viewer.camera.zoomOut(viewer.camera.positionCartographic.height * 0.3);
          break;
        case "ArrowUp":
          e.preventDefault();
          viewer.camera.moveUp(PAN_AMOUNT);
          break;
        case "ArrowDown":
          e.preventDefault();
          viewer.camera.moveDown(PAN_AMOUNT);
          break;
        case "ArrowLeft":
          e.preventDefault();
          viewer.camera.moveLeft(PAN_AMOUNT);
          break;
        case "ArrowRight":
          e.preventDefault();
          viewer.camera.moveRight(PAN_AMOUNT);
          break;
        case "l":
        case "L":
          e.preventDefault();
          onToggleLayerPanel?.();
          break;
        case "m":
          // Only lowercase 'm' — uppercase M is for measurement
          e.preventDefault();
          onToggleMeasurement?.();
          break;
        case "f":
        case "F":
          e.preventDefault();
          onFlyToSelection?.();
          break;
        case "b":
          // Only lowercase to avoid conflict with browser bookmarks (Ctrl+B)
          e.preventDefault();
          onAddBookmark?.();
          break;
        default:
          break;
      }
    },
    [viewer, onToggleLayerPanel, onToggleMeasurement, onFlyToSelection, onAddBookmark],
  );

  useEffect(() => {
    if (!enabled) return;
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [enabled, handleKeyDown]);
}
