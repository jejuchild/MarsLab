import { useEffect, useRef } from "react";
import type React from "react";
import * as Cesium from "cesium";

type BoundingBox = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;

type UseViewBoundSelectionParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  viewBoundSelectionMode: boolean;
  onViewBoundSelected?: (bounds: BoundingBox) => void;
};

export default function useViewBoundSelection({
  viewerRef,
  marsEllipsoid,
  viewBoundSelectionMode,
  onViewBoundSelected,
}: UseViewBoundSelectionParams): void {
  const onViewBoundSelectedRef = useRef(onViewBoundSelected);
  useEffect(() => {
    onViewBoundSelectedRef.current = onViewBoundSelected;
  }, [onViewBoundSelected]);

  // View Bound Selection Mode - drag to draw rectangle
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !viewBoundSelectionMode) return;

    // Store drag state
    let isDragging = false;
    let startCartographic: Cesium.Cartographic | null = null;
    const selectionRectId = "__VIEW_BOUND_SELECTION_RECT__";

    // Disable default camera controls during selection
    const scene = viewer.scene;
    scene.screenSpaceCameraController.enableRotate = false;
    scene.screenSpaceCameraController.enableTranslate = false;
    scene.screenSpaceCameraController.enableZoom = false;
    scene.screenSpaceCameraController.enableTilt = false;
    scene.screenSpaceCameraController.enableLook = false;

    // Change cursor to crosshair
    viewer.canvas.style.cursor = "crosshair";

    const handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);

    // Mouse down - start drag
    handler.setInputAction(
      (click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const cartesian = viewer.camera.pickEllipsoid(click.position, marsEllipsoid);
        if (!cartesian) return;

        startCartographic = Cesium.Cartographic.fromCartesian(cartesian, marsEllipsoid);
        isDragging = true;

        // Remove existing selection rect if any
        const existing = viewer.entities.getById(selectionRectId);
        if (existing) viewer.entities.remove(existing);
      },
      Cesium.ScreenSpaceEventType.LEFT_DOWN
    );

    // Mouse move - update rectangle
    handler.setInputAction(
      (movement: Cesium.ScreenSpaceEventHandler.MotionEvent) => {
        if (!isDragging || !startCartographic) return;

        const cartesian = viewer.camera.pickEllipsoid(movement.endPosition, marsEllipsoid);
        if (!cartesian) return;

        const endCartographic = Cesium.Cartographic.fromCartesian(cartesian, marsEllipsoid);

        // Compute rectangle bounds
        const west = Math.min(startCartographic.longitude, endCartographic.longitude);
        const east = Math.max(startCartographic.longitude, endCartographic.longitude);
        const south = Math.min(startCartographic.latitude, endCartographic.latitude);
        const north = Math.max(startCartographic.latitude, endCartographic.latitude);

        // Remove existing and add new rectangle
        const existing = viewer.entities.getById(selectionRectId);
        if (existing) viewer.entities.remove(existing);

        viewer.entities.add({
          id: selectionRectId,
          rectangle: {
            coordinates: new Cesium.Rectangle(west, south, east, north),
            material: Cesium.Color.YELLOW.withAlpha(0.3),
            outline: true,
            outlineColor: Cesium.Color.YELLOW,
            outlineWidth: 2,
            height: 0,
          },
        });

        scene.requestRender();
      },
      Cesium.ScreenSpaceEventType.MOUSE_MOVE
    );

    // Mouse up - finalize selection
    handler.setInputAction(
      (click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        if (!isDragging || !startCartographic) {
          isDragging = false;
          return;
        }

        const cartesian = viewer.camera.pickEllipsoid(click.position, marsEllipsoid);
        if (!cartesian) {
          isDragging = false;
          return;
        }

        const endCartographic = Cesium.Cartographic.fromCartesian(cartesian, marsEllipsoid);

        // Compute final bounds
        const westRad = Math.min(startCartographic.longitude, endCartographic.longitude);
        const eastRad = Math.max(startCartographic.longitude, endCartographic.longitude);
        const southRad = Math.min(startCartographic.latitude, endCartographic.latitude);
        const northRad = Math.max(startCartographic.latitude, endCartographic.latitude);

        // Convert to degrees
        const westLon = Cesium.Math.toDegrees(westRad);
        const eastLon = Cesium.Math.toDegrees(eastRad);
        const minLat = Cesium.Math.toDegrees(southRad);
        const maxLat = Cesium.Math.toDegrees(northRad);

        // Remove selection rectangle
        const existing = viewer.entities.getById(selectionRectId);
        if (existing) viewer.entities.remove(existing);

        // Only call callback if the selection is meaningful (not just a click)
        const lonSpan = eastLon - westLon;
        const latSpan = maxLat - minLat;
        if (lonSpan > 0.1 && latSpan > 0.1) {
          onViewBoundSelectedRef.current?.({ minLat, maxLat, westLon, eastLon });
        }

        isDragging = false;
        startCartographic = null;
        scene.requestRender();
      },
      Cesium.ScreenSpaceEventType.LEFT_UP
    );

    // Cleanup
    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      handler.destroy();

      // Re-enable camera controls
      scene.screenSpaceCameraController.enableRotate = true;
      scene.screenSpaceCameraController.enableTranslate = true;
      scene.screenSpaceCameraController.enableZoom = true;
      scene.screenSpaceCameraController.enableTilt = true;
      scene.screenSpaceCameraController.enableLook = true;

      // Reset cursor
      viewer.canvas.style.cursor = "default";

      // Remove selection rectangle if still exists
      const existing = viewer.entities.getById(selectionRectId);
      if (existing) viewer.entities.remove(existing);

      scene.requestRender();
    };
  }, [viewBoundSelectionMode, viewerRef, marsEllipsoid]);
}
