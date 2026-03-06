import { useEffect, useRef, useCallback } from "react";
import type React from "react";
import * as Cesium from "cesium";
import type { RouteResult, Waypoint, VLMAnalysis, SolPlan } from "../api/pathfinder";

/* ==================================================
 * Entity ID Constants
 * ==================================================*/
const PF_ROVER = "PATHFINDER_ROVER";
const PF_ROVER_LABEL = "PATHFINDER_ROVER_LABEL";
const PF_TRAVERSED = "PATHFINDER_TRAVERSED";

/** Full route plays in 60s at 1x speed */
const BASE_DURATION_S = 60;
/** Telemetry callback throttle interval */
const TELEMETRY_THROTTLE_MS = 100;

/* ==================================================
 * Exported Types
 * ==================================================*/
export interface RoverTelemetry {
  distanceTraveled: number;
  totalDistance: number;
  currentSlope: number;
  currentElevation: number;
  currentHeading: number;
  currentSol: number;
  totalSols: number;
  currentTerrainType: string;
  speedMPerS: number;
  currentLat: number;
  currentLon: number;
  waypointIndex: number;
}

export type SpeedOption = 1 | 2 | 5 | 10;
export const SPEED_OPTIONS: readonly SpeedOption[] = [1, 2, 5, 10];

/* ==================================================
 * Hook Parameters
 * ==================================================*/
type UseRoverSimulationParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  routeResult: RouteResult | null;
  vlmAnalysis?: VLMAnalysis | null;
  analysisMode: string | null;
  /* Control props */
  isPlaying: boolean;
  speed: SpeedOption;
  cameraFollow: boolean;
  seekTo: number | null;
  /* Callbacks */
  onProgress?: (progress: number) => void;
  onTelemetry?: (telemetry: RoverTelemetry) => void;
  onComplete?: () => void;
};

/* ==================================================
 * Helpers
 * ==================================================*/

/** Binary-search waypoints by distance and linearly interpolate */
function interpolateWaypoint(
  waypoints: Waypoint[],
  progress: number,
): { wp: Waypoint; index: number } {
  const fallback: Waypoint = { lat: 0, lon: 0, elevation: 0, distance_from_start: 0, slope: 0, heading: 0 };
  if (waypoints.length === 0) return { wp: fallback, index: 0 };
  if (waypoints.length === 1 || progress <= 0) return { wp: waypoints[0]!, index: 0 };
  if (progress >= 1) return { wp: waypoints[waypoints.length - 1]!, index: waypoints.length - 1 };

  const totalDist = waypoints[waypoints.length - 1]!.distance_from_start;
  if (totalDist <= 0) return { wp: waypoints[0]!, index: 0 };

  const targetDist = progress * totalDist;
  let lo = 0;
  let hi = waypoints.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (waypoints[mid]!.distance_from_start <= targetDist) lo = mid;
    else hi = mid;
  }

  const wpA = waypoints[lo]!;
  const wpB = waypoints[hi]!;
  const segLen = wpB.distance_from_start - wpA.distance_from_start;
  if (segLen <= 0) return { wp: wpA, index: lo };

  const t = (targetDist - wpA.distance_from_start) / segLen;
  const lerp = (a: number, b: number) => a + (b - a) * t;

  return {
    wp: {
      lat: lerp(wpA.lat, wpB.lat),
      lon: lerp(wpA.lon, wpB.lon),
      elevation: lerp(wpA.elevation, wpB.elevation),
      distance_from_start: targetDist,
      slope: lerp(wpA.slope, wpB.slope),
      heading: lerp(wpA.heading, wpB.heading),
    },
    index: lo,
  };
}

function findCurrentSol(solPlan: SolPlan[] | undefined, wpIndex: number): number {
  if (!solPlan?.length) return 1;
  for (const sol of solPlan) {
    if (wpIndex >= sol.start_wp_id && wpIndex <= sol.end_wp_id) return sol.sol_number;
  }
  return solPlan[solPlan.length - 1]?.sol_number ?? 1;
}

function findTerrainType(vlm: VLMAnalysis | null | undefined, progress: number): string {
  if (!vlm?.zones?.length) return "N/A";
  const idx = Math.min(Math.floor(progress * vlm.zones.length), vlm.zones.length - 1);
  return vlm.zones[idx]?.terrain_type ?? "mixed";
}

/* ==================================================
 * Hook
 * ==================================================*/
export default function useRoverSimulation(params: UseRoverSimulationParams): void {
  // Keep latest params in ref (RAF reads these — avoids stale closures)
  const P = useRef(params);
  P.current = params;

  // Mutable animation state
  const progressRef = useRef(0);
  const completedRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number | null>(null);
  const lastTelemetryRef = useRef(0);

  // Mutable position refs — read by CallbackProperty
  const positionRef = useRef<Cesium.Cartesian3 | null>(null);
  const labelTextRef = useRef("ROVER | Ready");
  const traversedRef = useRef<Cesium.Cartesian3[]>([]);

  // Track entity lifecycle
  const entitiesAliveRef = useRef(false);

  /* ---------- entity update helper ---------- */
  const updatePosition = useCallback((progress: number) => {
    const { routeResult, vlmAnalysis, marsEllipsoid } = P.current;
    if (!routeResult?.waypoints?.length) return;

    const waypoints = routeResult.waypoints;
    const totalDist = waypoints[waypoints.length - 1]!.distance_from_start;
    const { wp, index } = interpolateWaypoint(waypoints, progress);

    // Update refs (Cesium reads via CallbackProperty)
    positionRef.current = Cesium.Cartesian3.fromDegrees(wp.lon, wp.lat, 0, marsEllipsoid);

    const sol = findCurrentSol(routeResult.sol_plan, index);
    const distKm = (wp.distance_from_start / 1000).toFixed(1);
    const totalKm = (totalDist / 1000).toFixed(1);
    labelTextRef.current = `Sol ${sol} | ${distKm}/${totalKm} km`;

    // Build traversed path: all waypoints up to current + interpolated pos
    const arr: Cesium.Cartesian3[] = [];
    const limit = Math.min(index + 1, waypoints.length);
    for (let i = 0; i < limit; i++) {
      arr.push(Cesium.Cartesian3.fromDegrees(waypoints[i]!.lon, waypoints[i]!.lat, 0, marsEllipsoid));
    }
    if (index < waypoints.length - 1 && positionRef.current) arr.push(positionRef.current);
    traversedRef.current = arr;

    // Request render
    const viewer = P.current.viewerRef.current;
    if (viewer && !viewer.isDestroyed()) viewer.scene.requestRender();

    // Throttled telemetry
    const now = performance.now();
    if (now - lastTelemetryRef.current > TELEMETRY_THROTTLE_MS) {
      lastTelemetryRef.current = now;
      const terrainType = findTerrainType(vlmAnalysis, progress);
      const totalTimeS = (routeResult.summary?.total_time_hours ?? 1) * 3600;
      P.current.onTelemetry?.({
        distanceTraveled: wp.distance_from_start,
        totalDistance: totalDist,
        currentSlope: wp.slope,
        currentElevation: wp.elevation,
        currentHeading: wp.heading,
        currentSol: sol,
        totalSols: routeResult.sol_plan?.length ?? 1,
        currentTerrainType: terrainType,
        speedMPerS: totalDist / totalTimeS,
        currentLat: wp.lat,
        currentLon: wp.lon,
        waypointIndex: index,
      });
      P.current.onProgress?.(progress);
    }
  }, []);

  /* ---------- Entity creation / cleanup ---------- */
  useEffect(() => {
    const viewer = P.current.viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const { marsEllipsoid, routeResult, analysisMode } = P.current;

    // Cleanup
    for (const id of [PF_ROVER, PF_ROVER_LABEL, PF_TRAVERSED]) {
      const ent = viewer.entities.getById(id);
      if (ent) viewer.entities.remove(ent);
    }
    entitiesAliveRef.current = false;

    if (analysisMode !== "pathfinder" || !routeResult?.waypoints?.length || routeResult.waypoints.length < 2) {
      viewer.scene.requestRender();
      return;
    }

    // Reset animation
    progressRef.current = 0;
    completedRef.current = false;

    const wp0 = routeResult.waypoints[0]!;
    positionRef.current = Cesium.Cartesian3.fromDegrees(wp0.lon, wp0.lat, 0, marsEllipsoid);
    labelTextRef.current = "ROVER | Ready";
    traversedRef.current = [positionRef.current];

    // Rover point
    viewer.entities.add({
      id: PF_ROVER,
      position: new Cesium.CallbackProperty(() => positionRef.current, false) as unknown as Cesium.PositionProperty,
      point: {
        pixelSize: 14,
        color: Cesium.Color.YELLOW,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // Rover label
    viewer.entities.add({
      id: PF_ROVER_LABEL,
      position: new Cesium.CallbackProperty(() => positionRef.current, false) as unknown as Cesium.PositionProperty,
      label: {
        text: new Cesium.CallbackProperty(() => labelTextRef.current, false) as unknown as Cesium.Property,
        font: "bold 10px monospace",
        fillColor: Cesium.Color.YELLOW,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -16),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // Traversed path
    viewer.entities.add({
      id: PF_TRAVERSED,
      polyline: {
        positions: new Cesium.CallbackProperty(() => traversedRef.current, false) as unknown as Cesium.Property,
        width: 4,
        material: Cesium.Color.LIME.withAlpha(0.8),
        clampToGround: true,
      },
    });

    entitiesAliveRef.current = true;
    viewer.scene.requestRender();

    return () => {
      if (!viewer.isDestroyed()) {
        for (const id of [PF_ROVER, PF_ROVER_LABEL, PF_TRAVERSED]) {
          const ent = viewer.entities.getById(id);
          if (ent) viewer.entities.remove(ent);
        }
        if (viewer.trackedEntity?.id === PF_ROVER) {
          viewer.trackedEntity = undefined;
        }
        entitiesAliveRef.current = false;
        viewer.scene.requestRender();
      }
    };
  }, [params.viewerRef, params.marsEllipsoid, params.routeResult, params.analysisMode]);

  /* ---------- Animation loop ---------- */
  useEffect(() => {
    const viewer = P.current.viewerRef.current;
    if (!viewer || viewer.isDestroyed() || !entitiesAliveRef.current) return;

    if (!params.isPlaying || completedRef.current) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      lastFrameRef.current = null;
      return;
    }

    const animate = (ts: number) => {
      if (!P.current.isPlaying || viewer.isDestroyed()) {
        rafRef.current = null;
        lastFrameRef.current = null;
        return;
      }
      if (lastFrameRef.current === null) {
        lastFrameRef.current = ts;
        rafRef.current = requestAnimationFrame(animate);
        return;
      }

      const deltaS = (ts - lastFrameRef.current) / 1000;
      lastFrameRef.current = ts;

      const rate = P.current.speed / BASE_DURATION_S;
      progressRef.current = Math.min(progressRef.current + rate * deltaS, 1);
      updatePosition(progressRef.current);

      if (progressRef.current >= 1 && !completedRef.current) {
        completedRef.current = true;
        P.current.onProgress?.(1);
        P.current.onComplete?.();
        return;
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    lastFrameRef.current = null;
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [params.isPlaying, updatePosition]);

  /* ---------- Camera follow ---------- */
  useEffect(() => {
    const viewer = P.current.viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    if (params.cameraFollow && entitiesAliveRef.current) {
      const rover = viewer.entities.getById(PF_ROVER);
      if (rover) viewer.trackedEntity = rover;
    } else {
      if (viewer.trackedEntity?.id === PF_ROVER) {
        viewer.trackedEntity = undefined;
      }
    }
  }, [params.cameraFollow]);

  /* ---------- Seek ---------- */
  useEffect(() => {
    if (params.seekTo === null) return;
    const p = Math.max(0, Math.min(1, params.seekTo));
    progressRef.current = p;
    completedRef.current = p >= 1;
    if (entitiesAliveRef.current) {
      updatePosition(p);
      // Immediate progress emit (bypass throttle)
      P.current.onProgress?.(p);
    }
  }, [params.seekTo, updatePosition]);

  /* ---------- Cleanup on unmount ---------- */
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);
}
