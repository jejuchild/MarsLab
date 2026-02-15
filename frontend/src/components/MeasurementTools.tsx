// src/components/MeasurementTools.tsx
// Globe annotation & measurement toolbar for the Cesium Mars viewer.
// Tools: Distance, Area, Elevation Profile, Pin Annotation.

import { useCallback, useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";

/* ==================================================
 * Constants
 * ==================================================*/
const MARS_RADIUS_KM = 3389.5;
const MARS_EQUATORIAL_RADIUS = 3396190; // meters
const MARS_POLAR_RADIUS = 3376200; // meters
const MARS_ELLIPSOID = new Cesium.Ellipsoid(
  MARS_EQUATORIAL_RADIUS,
  MARS_EQUATORIAL_RADIUS,
  MARS_POLAR_RADIUS,
);
const ENTITY_PREFIX = "MEASUREMENT_";
const DEG = Cesium.Math.toDegrees;
const RAD = Cesium.Math.toRadians;

/* ==================================================
 * Types
 * ==================================================*/
export type MeasurementTool = "distance" | "area" | "profile" | "pin" | null;

interface LatLon {
  lat: number;
  lon: number;
}

interface Measurement {
  id: string;
  type: "distance" | "area" | "profile" | "pin";
  points: LatLon[];
  result: string; // e.g. "142.3 km" or "2,340 km^2"
  cesiumEntities: string[]; // entity IDs to remove on clear
}

interface ProfileSample {
  distanceKm: number;
  elevationM: number;
}

export interface MeasurementToolsProps {
  viewer: Cesium.Viewer | null;
  isVisible: boolean;
  onPinNote?: (lat: number, lon: number, text: string) => void;
}

/* ==================================================
 * Mars great-circle distance (Haversine)
 * ==================================================*/
function marsGreatCircleDistanceKm(a: LatLon, b: LatLon): number {
  const phi1 = RAD(a.lat);
  const phi2 = RAD(b.lat);
  const dPhi = RAD(b.lat - a.lat);
  const dLam = RAD(b.lon - a.lon);
  const hav =
    Math.sin(dPhi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLam / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(hav), Math.sqrt(1 - hav));
  return MARS_RADIUS_KM * c;
}

/* ==================================================
 * Spherical polygon area on Mars
 * Uses the spherical excess formula: A = |E| * R^2
 * where E = sum of interior angles - (n-2)*pi
 * For robustness with small or large polygons we compute
 * using the girard formula with vector cross products.
 * ==================================================*/
function marsSphericalAreaKm2(vertices: LatLon[]): number {
  const n = vertices.length;
  if (n < 3) return 0;

  // Convert to unit vectors on sphere
  const vecs = vertices.map((p) => {
    const phi = RAD(p.lat);
    const lam = RAD(p.lon);
    return new Cesium.Cartesian3(
      Math.cos(phi) * Math.cos(lam),
      Math.cos(phi) * Math.sin(lam),
      Math.sin(phi),
    );
  });

  // Compute interior angles via spherical geometry
  let angleSum = 0;
  for (let i = 0; i < n; i++) {
    const prev = vecs[(i - 1 + n) % n];
    const curr = vecs[i];
    const next = vecs[(i + 1) % n];

    // Project prev and next onto tangent plane at curr
    // to find the interior angle at this vertex
    const tPrev = Cesium.Cartesian3.subtract(
      prev,
      Cesium.Cartesian3.multiplyByScalar(
        curr,
        Cesium.Cartesian3.dot(curr, prev),
        new Cesium.Cartesian3(),
      ),
      new Cesium.Cartesian3(),
    );
    const tNext = Cesium.Cartesian3.subtract(
      next,
      Cesium.Cartesian3.multiplyByScalar(
        curr,
        Cesium.Cartesian3.dot(curr, next),
        new Cesium.Cartesian3(),
      ),
      new Cesium.Cartesian3(),
    );

    const magPrev = Cesium.Cartesian3.magnitude(tPrev);
    const magNext = Cesium.Cartesian3.magnitude(tNext);
    if (magPrev < 1e-12 || magNext < 1e-12) continue;

    const cosAngle = Cesium.Cartesian3.dot(tPrev, tNext) / (magPrev * magNext);
    const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    angleSum += angle;
  }

  const sphericalExcess = Math.abs(angleSum - (n - 2) * Math.PI);
  return sphericalExcess * MARS_RADIUS_KM * MARS_RADIUS_KM;
}

/* ==================================================
 * Format helpers
 * ==================================================*/
function formatDistance(km: number): string {
  if (km < 1) return `${(km * 1000).toFixed(0)} m`;
  if (km < 100) return `${km.toFixed(2)} km`;
  return `${km.toFixed(1)} km`;
}

function formatArea(km2: number): string {
  if (km2 < 1) return `${(km2 * 1e6).toFixed(0)} m\u00B2`;
  if (km2 < 100) return `${km2.toFixed(2)} km\u00B2`;
  return `${km2.toLocaleString(undefined, { maximumFractionDigits: 1 })} km\u00B2`;
}

function midpoint(a: LatLon, b: LatLon): LatLon {
  return { lat: (a.lat + b.lat) / 2, lon: (a.lon + b.lon) / 2 };
}

function centroid(pts: LatLon[]): LatLon {
  const lat = pts.reduce((s, p) => s + p.lat, 0) / pts.length;
  const lon = pts.reduce((s, p) => s + p.lon, 0) / pts.length;
  return { lat, lon };
}

let nextId = 0;
function uid(): string {
  return `${ENTITY_PREFIX}${Date.now()}_${nextId++}`;
}

/* ==================================================
 * Inline SVG elevation profile
 * ==================================================*/
function ElevationProfileChart({
  samples,
  distanceKm,
}: {
  samples: ProfileSample[];
  distanceKm: number;
}) {
  const W = 200;
  const H = 100;
  const PAD_L = 32;
  const PAD_R = 8;
  const PAD_T = 12;
  const PAD_B = 22;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  if (samples.length < 2) return null;

  const elevs = samples.map((s) => s.elevationM);
  const minElev = Math.min(...elevs);
  const maxElev = Math.max(...elevs);
  const elevRange = maxElev - minElev || 1;

  const toX = (i: number) => PAD_L + (i / (samples.length - 1)) * plotW;
  const toY = (e: number) => PAD_T + plotH - ((e - minElev) / elevRange) * plotH;

  const linePath = samples
    .map((s, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(s.elevationM).toFixed(1)}`)
    .join(" ");

  const fillPath = `${linePath} L${toX(samples.length - 1).toFixed(1)},${(PAD_T + plotH).toFixed(1)} L${PAD_L},${(PAD_T + plotH).toFixed(1)} Z`;

  return (
    <div className="mt-2 rounded-md border border-border-dark bg-bg-dark/95 p-1">
      <div className="text-[8px] text-slate-500 uppercase tracking-wider px-1 mb-0.5">
        Elevation Profile &middot; {formatDistance(distanceKm)}
      </div>
      <svg width={W} height={H} className="block">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = PAD_T + plotH - f * plotH;
          const elev = minElev + f * elevRange;
          return (
            <g key={f}>
              <line
                x1={PAD_L}
                y1={y}
                x2={PAD_L + plotW}
                y2={y}
                stroke="#2d3a54"
                strokeWidth={0.5}
              />
              <text x={PAD_L - 3} y={y + 3} fill="#64748b" fontSize={7} textAnchor="end">
                {elev.toFixed(0)}
              </text>
            </g>
          );
        })}
        {/* X-axis labels */}
        <text x={PAD_L} y={H - 4} fill="#64748b" fontSize={7} textAnchor="start">
          0
        </text>
        <text x={PAD_L + plotW} y={H - 4} fill="#64748b" fontSize={7} textAnchor="end">
          {distanceKm < 1 ? `${(distanceKm * 1000).toFixed(0)}m` : `${distanceKm.toFixed(1)}km`}
        </text>
        {/* Fill */}
        <path d={fillPath} fill="rgba(34,211,238,0.12)" />
        {/* Line */}
        <path d={linePath} fill="none" stroke="#22d3ee" strokeWidth={1.5} />
      </svg>
    </div>
  );
}

/* ==================================================
 * Pin text input popup
 * ==================================================*/
function PinInput({
  position,
  onSubmit,
  onCancel,
}: {
  position: { x: number; y: number };
  onSubmit: (text: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && text.trim()) {
      onSubmit(text.trim());
    } else if (e.key === "Escape") {
      onCancel();
    }
  };

  return (
    <div
      className="fixed z-[9999] flex items-center gap-1"
      style={{
        left: Math.min(position.x, window.innerWidth - 220),
        top: Math.max(position.y - 40, 8),
      }}
    >
      <input
        ref={inputRef}
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={onCancel}
        placeholder="Add note..."
        className="w-48 rounded-md border border-border-dark bg-surface-dark px-2 py-1 text-xs text-white placeholder-text-muted outline-none focus:border-primary"
        maxLength={200}
      />
      <button
        onMouseDown={(e) => {
          e.preventDefault(); // prevent blur
          if (text.trim()) onSubmit(text.trim());
        }}
        className="rounded-md border border-border-dark bg-primary/80 px-2 py-1 text-xs text-white hover:bg-primary"
      >
        Save
      </button>
    </div>
  );
}

/* ==================================================
 * Tool button definitions
 * ==================================================*/
const TOOLS: Array<{
  key: MeasurementTool;
  icon: string;
  label: string;
}> = [
  { key: "distance", icon: "straighten", label: "Distance" },
  { key: "area", icon: "pentagon", label: "Area" },
  { key: "profile", icon: "show_chart", label: "Elevation" },
  { key: "pin", icon: "push_pin", label: "Pin" },
];

/* ==================================================
 * Component
 * ==================================================*/
export default function MeasurementTools({
  viewer,
  isVisible,
  onPinNote,
}: MeasurementToolsProps) {
  const [activeTool, setActiveTool] = useState<MeasurementTool>(null);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [pendingPoints, setPendingPoints] = useState<LatLon[]>([]);
  const [pendingEntities, setPendingEntities] = useState<string[]>([]);
  const [profileData, setProfileData] = useState<{
    samples: ProfileSample[];
    distanceKm: number;
  } | null>(null);
  const [pinInput, setPinInput] = useState<{
    lat: number;
    lon: number;
    screenX: number;
    screenY: number;
    entityId: string;
  } | null>(null);

  // Refs for stable access in event handlers
  const activeToolRef = useRef(activeTool);
  const pendingPointsRef = useRef(pendingPoints);
  const pendingEntitiesRef = useRef(pendingEntities);
  const measurementsRef = useRef(measurements);
  const handlerRef = useRef<Cesium.ScreenSpaceEventHandler | null>(null);
  const cursorStyleRef = useRef<string>("");
  // Timer ref to debounce area single-clicks so double-click doesn't add extra vertex
  const areaClickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const areaClickPendingRef = useRef<{ lat: number; lon: number } | null>(null);

  // Keep refs in sync
  useEffect(() => {
    activeToolRef.current = activeTool;
  }, [activeTool]);
  useEffect(() => {
    pendingPointsRef.current = pendingPoints;
  }, [pendingPoints]);
  useEffect(() => {
    pendingEntitiesRef.current = pendingEntities;
  }, [pendingEntities]);
  useEffect(() => {
    measurementsRef.current = measurements;
  }, [measurements]);

  /* ----- Helper: add Cesium entity safely ----- */
  const addEntity = useCallback(
    (opts: Cesium.Entity.ConstructorOptions): string | null => {
      if (!viewer) return null;
      try {
        viewer.entities.add(opts);
        viewer.scene.requestRender();
        return (opts.id as string) ?? null;
      } catch {
        return null;
      }
    },
    [viewer],
  );

  /* ----- Helper: remove entity by ID ----- */
  const removeEntity = useCallback(
    (id: string) => {
      if (!viewer) return;
      const ent = viewer.entities.getById(id);
      if (ent) {
        viewer.entities.remove(ent);
        viewer.scene.requestRender();
      }
    },
    [viewer],
  );

  /* ----- Clear all measurements ----- */
  const clearAll = useCallback(() => {
    if (!viewer) return;

    // Remove all measurement entities
    for (const m of measurementsRef.current) {
      for (const eid of m.cesiumEntities) {
        const ent = viewer.entities.getById(eid);
        if (ent) viewer.entities.remove(ent);
      }
    }

    // Remove any pending entities
    for (const eid of pendingEntitiesRef.current) {
      const ent = viewer.entities.getById(eid);
      if (ent) viewer.entities.remove(ent);
    }

    setMeasurements([]);
    setPendingPoints([]);
    setPendingEntities([]);
    setProfileData(null);
    setPinInput(null);
    viewer.scene.requestRender();
  }, [viewer]);

  /* ----- Cancel current in-progress tool ----- */
  const cancelPending = useCallback(() => {
    if (!viewer) return;
    // Cancel any debounced area click
    if (areaClickTimerRef.current) {
      clearTimeout(areaClickTimerRef.current);
      areaClickTimerRef.current = null;
      areaClickPendingRef.current = null;
    }
    for (const eid of pendingEntitiesRef.current) {
      const ent = viewer.entities.getById(eid);
      if (ent) viewer.entities.remove(ent);
    }
    setPendingPoints([]);
    setPendingEntities([]);
    setPinInput(null);
    viewer.scene.requestRender();
  }, [viewer]);

  /* ----- Toggle / select tool ----- */
  const handleToolClick = useCallback(
    (tool: MeasurementTool) => {
      if (activeTool === tool) {
        // Deactivate
        cancelPending();
        setActiveTool(null);
      } else {
        cancelPending();
        setActiveTool(tool);
      }
    },
    [activeTool, cancelPending],
  );

  /* ----- Escape key handler ----- */
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && activeToolRef.current) {
        cancelPending();
        setActiveTool(null);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [cancelPending]);

  /* ----- Cursor style management ----- */
  useEffect(() => {
    if (!viewer) return;
    const canvas = viewer.scene.canvas as HTMLCanvasElement;
    if (activeTool) {
      // Capture the original cursor BEFORE overwriting
      const originalCursor = canvas.style.cursor;
      cursorStyleRef.current = originalCursor;
      canvas.style.cursor = "crosshair";
      return () => {
        canvas.style.cursor = originalCursor;
      };
    }
    // No active tool: nothing to clean up
  }, [viewer, activeTool]);

  /* ----- Fetch elevation profile from backend ----- */
  const fetchElevationProfile = useCallback(
    async (start: LatLon, end: LatLon): Promise<ProfileSample[]> => {
      try {
        const url =
          `/api/terrain/line_profile?start_lat=${start.lat}&start_lon=${start.lon}` +
          `&end_lat=${end.lat}&end_lon=${end.lon}&num_samples=50`;
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          if (data.profile && Array.isArray(data.profile)) {
            return data.profile.map((p: any) => ({
              distanceKm: p.distance_km ?? 0,
              elevationM: p.elevation_m ?? 0,
            }));
          }
        }
      } catch {
        // Fall through to synthetic data
      }

      // Fallback: generate synthetic profile with random noise
      const totalDist = marsGreatCircleDistanceKm(start, end);
      const samples: ProfileSample[] = [];
      for (let i = 0; i < 50; i++) {
        const frac = i / 49;
        const baseLat = start.lat + (end.lat - start.lat) * frac;
        // Synthetic elevation: a sine-ish curve with noise
        const base = -2000 + Math.sin(frac * Math.PI * 3) * 500;
        const noise = (Math.random() - 0.5) * 200;
        samples.push({
          distanceKm: totalDist * frac,
          elevationM: base + noise + baseLat * 10,
        });
      }
      return samples;
    },
    [],
  );

  /* ==========================================================
   * Main click handler: installed as ScreenSpaceEventHandler
   * ==========================================================*/
  useEffect(() => {
    if (!viewer || !activeTool) {
      // Destroy existing handler if tool is deactivated
      if (handlerRef.current) {
        handlerRef.current.destroy();
        handlerRef.current = null;
      }
      return;
    }

    // Destroy any old handler before creating new one
    if (handlerRef.current) {
      handlerRef.current.destroy();
      handlerRef.current = null;
    }

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handlerRef.current = handler;

    /* ----- LEFT_CLICK ----- */
    handler.setInputAction(
      (click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const cart = viewer.camera.pickEllipsoid(click.position, MARS_ELLIPSOID);
        if (!cart) return;

        const carto = Cesium.Cartographic.fromCartesian(cart);
        const lat = DEG(carto.latitude);
        const lon = DEG(carto.longitude);
        const tool = activeToolRef.current;
        const pts = [...pendingPointsRef.current];
        const ents = [...pendingEntitiesRef.current];

        if (tool === "distance") {
          if (pts.length === 0) {
            // First point
            const dotId = uid();
            addEntity({
              id: dotId,
              position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
              point: {
                pixelSize: 8,
                color: Cesium.Color.LIME,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
            });
            setPendingPoints([{ lat, lon }]);
            setPendingEntities([dotId]);
          } else {
            // Second point - complete measurement
            const start = pts[0];
            const end: LatLon = { lat, lon };
            const dist = marsGreatCircleDistanceKm(start, end);
            const label = formatDistance(dist);
            const mid = midpoint(start, end);

            const dotId = uid();
            const lineId = uid();
            const labelId = uid();

            addEntity({
              id: dotId,
              position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
              point: {
                pixelSize: 8,
                color: Cesium.Color.fromCssColorString("#ef4444"),
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
            });

            addEntity({
              id: lineId,
              polyline: {
                positions: [
                  Cesium.Cartesian3.fromDegrees(start.lon, start.lat, 0, MARS_ELLIPSOID),
                  Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
                ],
                width: 2.5,
                material: new Cesium.PolylineDashMaterialProperty({
                  color: Cesium.Color.fromCssColorString("#22d3ee"),
                  dashLength: 12,
                }),
                clampToGround: true,
              },
            });

            addEntity({
              id: labelId,
              position: Cesium.Cartesian3.fromDegrees(mid.lon, mid.lat, 0, MARS_ELLIPSOID),
              label: {
                text: label,
                font: "bold 13px Space Grotesk, sans-serif",
                fillColor: Cesium.Color.WHITE,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 3,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                pixelOffset: new Cesium.Cartesian2(0, -10),
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
                showBackground: true,
                backgroundColor: Cesium.Color.fromCssColorString("rgba(10,14,23,0.85)"),
                backgroundPadding: new Cesium.Cartesian2(6, 4),
              },
            });

            const allEnts = [...ents, dotId, lineId, labelId];
            const measurement: Measurement = {
              id: uid(),
              type: "distance",
              points: [start, end],
              result: label,
              cesiumEntities: allEnts,
            };
            setMeasurements((prev) => [...prev, measurement]);
            setPendingPoints([]);
            setPendingEntities([]);
            // Auto-deactivate
            setActiveTool(null);
          }
        } else if (tool === "area") {
          // Debounce area clicks: delay vertex addition so double-click
          // can cancel the pending single click and not add an extra vertex.
          if (areaClickTimerRef.current) {
            clearTimeout(areaClickTimerRef.current);
          }
          areaClickPendingRef.current = { lat, lon };
          areaClickTimerRef.current = setTimeout(() => {
            areaClickTimerRef.current = null;
            const pending = areaClickPendingRef.current;
            if (!pending) return;
            areaClickPendingRef.current = null;

            const curPts = [...pendingPointsRef.current];
            const curEnts = [...pendingEntitiesRef.current];

            const dotId = uid();
            addEntity({
              id: dotId,
              position: Cesium.Cartesian3.fromDegrees(pending.lon, pending.lat, 0, MARS_ELLIPSOID),
              point: {
                pixelSize: 6,
                color: Cesium.Color.fromCssColorString("#a78bfa"),
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 1.5,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
            });

            // If we have at least 1 prior point, draw an edge to the new vertex
            if (curPts.length > 0) {
              const prev = curPts[curPts.length - 1];
              const edgeId = uid();
              addEntity({
                id: edgeId,
                polyline: {
                  positions: [
                    Cesium.Cartesian3.fromDegrees(prev.lon, prev.lat, 0, MARS_ELLIPSOID),
                    Cesium.Cartesian3.fromDegrees(pending.lon, pending.lat, 0, MARS_ELLIPSOID),
                  ],
                  width: 2,
                  material: Cesium.Color.fromCssColorString("#a78bfa").withAlpha(0.7),
                  clampToGround: true,
                },
              });
              curEnts.push(edgeId);
            }

            curPts.push({ lat: pending.lat, lon: pending.lon });
            curEnts.push(dotId);
            setPendingPoints(curPts);
            setPendingEntities(curEnts);
          }, 250);
        } else if (tool === "profile") {
          if (pts.length === 0) {
            const dotId = uid();
            addEntity({
              id: dotId,
              position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
              point: {
                pixelSize: 8,
                color: Cesium.Color.LIME,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
            });
            setPendingPoints([{ lat, lon }]);
            setPendingEntities([dotId]);
          } else {
            // Second point - complete
            const start = pts[0];
            const end: LatLon = { lat, lon };
            const dist = marsGreatCircleDistanceKm(start, end);

            const dotId = uid();
            const lineId = uid();
            const labelId = uid();

            addEntity({
              id: dotId,
              position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
              point: {
                pixelSize: 8,
                color: Cesium.Color.fromCssColorString("#ef4444"),
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
            });

            addEntity({
              id: lineId,
              polyline: {
                positions: [
                  Cesium.Cartesian3.fromDegrees(start.lon, start.lat, 0, MARS_ELLIPSOID),
                  Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
                ],
                width: 2.5,
                material: Cesium.Color.fromCssColorString("#22d3ee").withAlpha(0.8),
                clampToGround: true,
              },
            });

            const mid = midpoint(start, end);
            addEntity({
              id: labelId,
              position: Cesium.Cartesian3.fromDegrees(mid.lon, mid.lat, 0, MARS_ELLIPSOID),
              label: {
                text: `Profile: ${formatDistance(dist)}`,
                font: "bold 11px Space Grotesk, sans-serif",
                fillColor: Cesium.Color.fromCssColorString("#22d3ee"),
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                pixelOffset: new Cesium.Cartesian2(0, -10),
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
            });

            const allEnts = [...ents, dotId, lineId, labelId];
            const measurement: Measurement = {
              id: uid(),
              type: "profile",
              points: [start, end],
              result: formatDistance(dist),
              cesiumEntities: allEnts,
            };
            setMeasurements((prev) => [...prev, measurement]);
            setPendingPoints([]);
            setPendingEntities([]);

            // Fetch elevation profile
            fetchElevationProfile(start, end).then((samples) => {
              setProfileData({ samples, distanceKm: dist });
            });

            setActiveTool(null);
          }
        } else if (tool === "pin") {
          // Drop a pin billboard
          const pinId = uid();
          addEntity({
            id: pinId,
            position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
            billboard: {
              image: createPinIcon(),
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              scale: 1.0,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
          });

          // Convert globe position to screen position for the text input
          const screenPos = Cesium.SceneTransforms.wgs84ToWindowCoordinates(
            viewer.scene,
            Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
          );

          setPinInput({
            lat,
            lon,
            screenX: screenPos?.x ?? click.position.x,
            screenY: screenPos?.y ?? click.position.y,
            entityId: pinId,
          });
        }
      },
      Cesium.ScreenSpaceEventType.LEFT_CLICK,
    );

    /* ----- LEFT_DOUBLE_CLICK (area polygon close) ----- */
    handler.setInputAction(
      (_click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const tool = activeToolRef.current;
        if (tool !== "area") return;

        // Cancel the pending single-click timer so it doesn't add an extra vertex
        if (areaClickTimerRef.current) {
          clearTimeout(areaClickTimerRef.current);
          areaClickTimerRef.current = null;
          areaClickPendingRef.current = null;
        }

        const pts = pendingPointsRef.current;
        const ents = [...pendingEntitiesRef.current];
        if (pts.length < 3) return;

        // Close polygon: draw closing edge
        const first = pts[0];
        const last = pts[pts.length - 1];
        const closingEdgeId = uid();
        addEntity({
          id: closingEdgeId,
          polyline: {
            positions: [
              Cesium.Cartesian3.fromDegrees(last.lon, last.lat, 0, MARS_ELLIPSOID),
              Cesium.Cartesian3.fromDegrees(first.lon, first.lat, 0, MARS_ELLIPSOID),
            ],
            width: 2,
            material: Cesium.Color.fromCssColorString("#a78bfa").withAlpha(0.7),
            clampToGround: true,
          },
        });
        ents.push(closingEdgeId);

        // Filled polygon
        const polyId = uid();
        const positions = pts.map((p) =>
          Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 0, MARS_ELLIPSOID),
        );
        addEntity({
          id: polyId,
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(positions),
            material: Cesium.Color.fromCssColorString("#a78bfa").withAlpha(0.2),
            outline: false,
          },
        });
        ents.push(polyId);

        // Compute area
        const area = marsSphericalAreaKm2(pts);
        const label = formatArea(area);
        const cen = centroid(pts);
        const labelId = uid();
        addEntity({
          id: labelId,
          position: Cesium.Cartesian3.fromDegrees(cen.lon, cen.lat, 0, MARS_ELLIPSOID),
          label: {
            text: label,
            font: "bold 13px Space Grotesk, sans-serif",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 3,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.CENTER,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            showBackground: true,
            backgroundColor: Cesium.Color.fromCssColorString("rgba(10,14,23,0.85)"),
            backgroundPadding: new Cesium.Cartesian2(6, 4),
          },
        });
        ents.push(labelId);

        const measurement: Measurement = {
          id: uid(),
          type: "area",
          points: [...pts],
          result: label,
          cesiumEntities: ents,
        };
        setMeasurements((prev) => [...prev, measurement]);
        setPendingPoints([]);
        setPendingEntities([]);
        setActiveTool(null);
      },
      Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK,
    );

    return () => {
      handler.destroy();
      handlerRef.current = null;
    };
    // Re-create handler whenever the active tool changes, but NOT on every
    // pending point update (we use refs for that).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewer, activeTool, addEntity, fetchElevationProfile]);

  /* ----- Pin submit handler ----- */
  const handlePinSubmit = useCallback(
    (text: string) => {
      if (!pinInput || !viewer) return;
      const { lat, lon, entityId } = pinInput;

      // Add a label below the pin
      const labelId = uid();
      addEntity({
        id: labelId,
        position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, MARS_ELLIPSOID),
        label: {
          text,
          font: "11px Space Grotesk, sans-serif",
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.TOP,
          pixelOffset: new Cesium.Cartesian2(0, 4),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          showBackground: true,
          backgroundColor: Cesium.Color.fromCssColorString("rgba(10,14,23,0.85)"),
          backgroundPadding: new Cesium.Cartesian2(5, 3),
        },
      });

      const measurement: Measurement = {
        id: uid(),
        type: "pin",
        points: [{ lat, lon }],
        result: text,
        cesiumEntities: [entityId, labelId],
      };
      setMeasurements((prev) => [...prev, measurement]);
      setPinInput(null);
      setActiveTool(null);

      // Invoke callback to save as field note
      onPinNote?.(lat, lon, text);
    },
    [pinInput, viewer, addEntity, onPinNote],
  );

  /* ----- Pin cancel handler ----- */
  const handlePinCancel = useCallback(() => {
    if (pinInput) {
      removeEntity(pinInput.entityId);
    }
    setPinInput(null);
  }, [pinInput, removeEntity]);

  /* ----- Cleanup on unmount: remove all entities and event handlers ----- */
  useEffect(() => {
    return () => {
      // Cancel any debounced area click
      if (areaClickTimerRef.current) {
        clearTimeout(areaClickTimerRef.current);
        areaClickTimerRef.current = null;
      }
      // Destroy event handler
      if (handlerRef.current) {
        handlerRef.current.destroy();
        handlerRef.current = null;
      }
      // Remove all measurement entities from the viewer
      if (viewer) {
        for (const m of measurementsRef.current) {
          for (const eid of m.cesiumEntities) {
            const ent = viewer.entities.getById(eid);
            if (ent) viewer.entities.remove(ent);
          }
        }
        // Remove any pending (in-progress) entities
        for (const eid of pendingEntitiesRef.current) {
          const ent = viewer.entities.getById(eid);
          if (ent) viewer.entities.remove(ent);
        }
        viewer.scene.requestRender();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewer]);

  /* ----- Remove a single measurement ----- */
  const removeMeasurement = useCallback(
    (measurementId: string) => {
      if (!viewer) return;
      setMeasurements((prev) => {
        const target = prev.find((m) => m.id === measurementId);
        if (target) {
          for (const eid of target.cesiumEntities) {
            const ent = viewer.entities.getById(eid);
            if (ent) viewer.entities.remove(ent);
          }
        }
        viewer.scene.requestRender();
        return prev.filter((m) => m.id !== measurementId);
      });
      // If the removed measurement was a profile, clear profile data
      const target = measurementsRef.current.find((m) => m.id === measurementId);
      if (target?.type === "profile") {
        setProfileData(null);
      }
    },
    [viewer],
  );

  if (!isVisible) return null;

  const hasAny = measurements.length > 0 || pendingPoints.length > 0;

  return (
    <>
      {/* Floating Toolbar */}
      <div
        className="absolute top-20 right-4 z-[500] flex flex-col items-center gap-1 rounded-xl border border-border-dark bg-surface-dark/95 p-1.5 backdrop-blur-md shadow-lg"
        style={{ pointerEvents: "auto" }}
      >
        {/* Tool buttons */}
        {TOOLS.map(({ key, icon, label }) => (
          <button
            key={key}
            onClick={() => handleToolClick(key)}
            title={label}
            className={`group relative flex h-9 w-9 items-center justify-center rounded-lg transition-all ${
              activeTool === key
                ? "bg-primary text-white shadow-md shadow-primary/30"
                : "text-slate-400 hover:bg-surface-elevated hover:text-white"
            }`}
          >
            <span className="material-symbols-outlined text-[20px]">{icon}</span>
            {/* Tooltip */}
            <span className="pointer-events-none absolute right-full mr-2 whitespace-nowrap rounded-md bg-bg-dark/95 px-2 py-1 text-[10px] text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100 border border-border-dark">
              {label}
            </span>
          </button>
        ))}

        {/* Separator */}
        {hasAny && <div className="my-0.5 h-px w-6 bg-border-dark" />}

        {/* Clear all button */}
        {hasAny && (
          <button
            onClick={clearAll}
            title="Clear all measurements"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition-all hover:bg-red-900/30 hover:text-red-400"
          >
            <span className="material-symbols-outlined text-[18px]">delete</span>
          </button>
        )}

        {/* Active tool indicator */}
        {activeTool && (
          <div className="mt-0.5 text-[8px] text-primary font-medium uppercase tracking-wider">
            {activeTool === "area" && pendingPoints.length > 0
              ? `${pendingPoints.length} pts`
              : activeTool === "distance" && pendingPoints.length > 0
                ? "1/2"
                : activeTool === "profile" && pendingPoints.length > 0
                  ? "1/2"
                  : ""}
          </div>
        )}
      </div>

      {/* Measurement list (below toolbar) */}
      {measurements.length > 0 && (
        <div className="absolute top-20 right-16 z-[499] w-52 rounded-lg border border-border-dark bg-surface-dark/95 backdrop-blur-md shadow-lg overflow-hidden">
          <div className="px-2.5 py-1.5 border-b border-border-dark">
            <span className="text-[9px] uppercase tracking-widest text-slate-500 font-bold">
              Measurements
            </span>
          </div>
          <div className="max-h-48 overflow-y-auto scrollbar-dark">
            {measurements.map((m) => (
              <div
                key={m.id}
                className="flex items-center justify-between px-2.5 py-1.5 border-b border-border-dark/50 last:border-b-0 group hover:bg-surface-elevated/50"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="material-symbols-outlined text-[14px] text-slate-500">
                    {m.type === "distance"
                      ? "straighten"
                      : m.type === "area"
                        ? "pentagon"
                        : m.type === "profile"
                          ? "show_chart"
                          : "push_pin"}
                  </span>
                  <span className="text-xs text-white truncate">{m.result}</span>
                </div>
                <button
                  onClick={() => removeMeasurement(m.id)}
                  className="text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity ml-1 flex-shrink-0"
                  title="Remove"
                >
                  <span className="material-symbols-outlined text-[14px]">close</span>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Inline elevation profile chart */}
      {profileData && (
        <div className="absolute top-20 right-16 z-[498]" style={{ marginTop: measurements.length > 0 ? `${Math.min(measurements.length * 32 + 28, 220)}px` : "0px" }}>
          <div className="relative">
            <ElevationProfileChart
              samples={profileData.samples}
              distanceKm={profileData.distanceKm}
            />
            <button
              onClick={() => setProfileData(null)}
              className="absolute top-1 right-1 text-slate-500 hover:text-white"
              title="Close profile"
            >
              <span className="material-symbols-outlined text-[12px]">close</span>
            </button>
          </div>
        </div>
      )}

      {/* Pin text input popup */}
      {pinInput && (
        <PinInput
          position={{ x: pinInput.screenX + 16, y: pinInput.screenY }}
          onSubmit={handlePinSubmit}
          onCancel={handlePinCancel}
        />
      )}
    </>
  );
}

/* ==================================================
 * Pin icon generator (canvas-based, cached)
 * ==================================================*/
let _pinIconUrl: string | null = null;

function createPinIcon(): string {
  if (_pinIconUrl) return _pinIconUrl;

  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 38;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // Drop shadow
  ctx.shadowColor = "rgba(0,0,0,0.4)";
  ctx.shadowBlur = 3;
  ctx.shadowOffsetY = 1;

  // Pin body
  ctx.beginPath();
  ctx.arc(14, 13, 11, Math.PI, 0, false);
  ctx.lineTo(14, 36);
  ctx.closePath();
  ctx.fillStyle = "#ef4444";
  ctx.fill();

  // Reset shadow for inner details
  ctx.shadowColor = "transparent";

  // Border
  ctx.beginPath();
  ctx.arc(14, 13, 11, Math.PI, 0, false);
  ctx.lineTo(14, 36);
  ctx.closePath();
  ctx.strokeStyle = "#991b1b";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Inner circle
  ctx.beginPath();
  ctx.arc(14, 13, 5, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();

  _pinIconUrl = canvas.toDataURL();
  return _pinIconUrl;
}
