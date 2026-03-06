import { useEffect, useState, useRef } from "react";
import type React from "react";
import * as Cesium from "cesium";
import type { RouteResult, RouteGeoPoint } from "../api/pathfinder";

/* ==================================================
 * Entity ID constants
 * ==================================================*/
const PF_START = "PATHFINDER_START";
const PF_GOAL = "PATHFINDER_GOAL";
const PF_START_LABEL = "PATHFINDER_START_LABEL";
const PF_GOAL_LABEL = "PATHFINDER_GOAL_LABEL";
const PF_ROUTE = "PATHFINDER_ROUTE";
const PF_ROUTE_OUTLINE = "PATHFINDER_ROUTE_OUTLINE";

const ALL_IDS = [PF_START, PF_GOAL, PF_START_LABEL, PF_GOAL_LABEL, PF_ROUTE, PF_ROUTE_OUTLINE];

/* ==================================================
 * Types
 * ==================================================*/
type UsePathfinderOverlayParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  analysisMode: string | null;
  startPoint: { lat: number; lon: number } | null;
  goalPoint: { lat: number; lon: number } | null;
  routeResult: RouteResult | null;
};

/** Safe viewer access — returns viewer only if alive */
function getViewer(ref: React.MutableRefObject<Cesium.Viewer | null>): Cesium.Viewer | null {
  const v = ref.current;
  return v && !v.isDestroyed() ? v : null;
}

/* ==================================================
 * Hook
 * ==================================================*/
export default function usePathfinderOverlay({
  viewerRef,
  marsEllipsoid,
  analysisMode,
  startPoint,
  goalPoint,
  routeResult,
}: UsePathfinderOverlayParams): void {
  // Signal to re-trigger when viewer becomes available
  const [viewerReady, setViewerReady] = useState(false);
  const paramsRef = useRef({ viewerRef });
  paramsRef.current.viewerRef = viewerRef;

  // Poll for viewer readiness (same pattern as useRoverSimulation)
  useEffect(() => {
    if (viewerReady) return;
    const interval = setInterval(() => {
      const v = getViewer(paramsRef.current.viewerRef);
      if (v) {
        setViewerReady(true);
        clearInterval(interval);
      }
    }, 200);
    return () => clearInterval(interval);
  }, [viewerReady]);

  useEffect(() => {
    const viewer = getViewer(viewerRef);
    if (!viewer) return;

    // Clear all pathfinder entities
    for (const id of ALL_IDS) {
      const ent = viewer.entities.getById(id);
      if (ent) viewer.entities.remove(ent);
    }

    if (analysisMode !== "pathfinder") {
      viewer.scene.requestRender();
      return;
    }

    const fmtLabel = (lat: number, lon: number) =>
      `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;

    // ── Start Point ──────────────────────────────────
    if (startPoint) {
      viewer.entities.add({
        id: PF_START,
        position: Cesium.Cartesian3.fromDegrees(startPoint.lon, startPoint.lat, 0, marsEllipsoid),
        point: {
          pixelSize: 10,
          color: Cesium.Color.LIME,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      viewer.entities.add({
        id: PF_START_LABEL,
        position: Cesium.Cartesian3.fromDegrees(startPoint.lon, startPoint.lat, 0, marsEllipsoid),
        label: {
          text: `START\n${fmtLabel(startPoint.lat, startPoint.lon)}`,
          font: "11px monospace",
          fillColor: Cesium.Color.LIME,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -14),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }

    // ── Goal Point ───────────────────────────────────
    if (goalPoint) {
      viewer.entities.add({
        id: PF_GOAL,
        position: Cesium.Cartesian3.fromDegrees(goalPoint.lon, goalPoint.lat, 0, marsEllipsoid),
        point: {
          pixelSize: 10,
          color: Cesium.Color.RED,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      viewer.entities.add({
        id: PF_GOAL_LABEL,
        position: Cesium.Cartesian3.fromDegrees(goalPoint.lon, goalPoint.lat, 0, marsEllipsoid),
        label: {
          text: `GOAL\n${fmtLabel(goalPoint.lat, goalPoint.lon)}`,
          font: "11px monospace",
          fillColor: Cesium.Color.RED,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -14),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }

    // ── Route Polyline ───────────────────────────────
    if (routeResult?.route_geo && routeResult.route_geo.length >= 2) {
      const positions = routeResult.route_geo.map((p: RouteGeoPoint) =>
        Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 0, marsEllipsoid),
      );

      // Outline (wider, dark)
      viewer.entities.add({
        id: PF_ROUTE_OUTLINE,
        polyline: {
          positions,
          width: 5,
          material: Cesium.Color.BLACK.withAlpha(0.6),
          clampToGround: true,
        },
      });

      // Main route line (orange gradient)
      viewer.entities.add({
        id: PF_ROUTE,
        polyline: {
          positions,
          width: 3,
          material: Cesium.Color.ORANGE.withAlpha(0.9),
          clampToGround: true,
        },
      });
    }

    viewer.scene.requestRender();

    return () => {
      const v = getViewer(viewerRef);
      if (v) {
        for (const id of ALL_IDS) {
          const ent = v.entities.getById(id);
          if (ent) v.entities.remove(ent);
        }
        v.scene.requestRender();
      }
    };
    // viewerReady triggers re-run when viewer becomes available
  }, [viewerReady, analysisMode, startPoint, goalPoint, routeResult, viewerRef, marsEllipsoid]);
}
