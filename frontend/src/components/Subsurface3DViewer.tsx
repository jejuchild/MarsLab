/**
 * Subsurface3DViewer - 3D visualization of SHARAD subsurface interface
 *
 * Displays:
 * - MOLA terrain mesh (surface)
 * - Subsurface interface ribbon (from boundary line)
 * - Vertical exaggeration controls
 */

import { useState, useEffect, useMemo } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";

// =============================================================================
// Types
// =============================================================================

interface SubsurfacePoint {
  lat: number;
  lon: number;
  surfaceElevation: number;
  subsurfaceElevation: number;
  depthM: number;
}

interface Subsurface3DViewerProps {
  // Trace range
  startTrace: number;
  endTrace: number;
  // Geometry data for each trace in range
  lats: number[];
  lons: number[];
  // Boundary offset in bins
  boundaryBinOffset: number;
  // Dielectric constant
  epsilonR: number;
  // MOLA elevations for each trace
  molaElevations: (number | null)[];
  // Callbacks
  onClose: () => void;
}

interface DEMPatchData {
  elevations: number[];
  rows: number;
  cols: number;
  spacing_m: number;
  bounds: {
    west: number;
    east: number;
    south: number;
    north: number;
  };
  center: { lat: number; lon: number };
  center_elevation_m: number;
  min_elevation_m: number;
  max_elevation_m: number;
  elevation_range_m: number;
  radius_m: number;
}

// =============================================================================
// Constants
// =============================================================================

const SPEED_OF_LIGHT = 299792458; // m/s
const BIN_DT_SEC = 0.0375e-6; // seconds per range bin
const MARS_RADIUS = 3389500; // meters (mean radius)
const RIBBON_WIDTH_M = 500; // width of subsurface ribbon in meters

// =============================================================================
// Coordinate Conversion
// =============================================================================

/**
 * Convert lat/lon to local ENU (East-North-Up) coordinates in meters
 * relative to an origin point
 */
function latLonToENU(
  lat: number,
  lon: number,
  elevation: number,
  originLat: number,
  originLon: number
): [number, number, number] {
  const originLatRad = (originLat * Math.PI) / 180;

  // Approximate conversion to meters
  const dLat = lat - originLat;
  const dLon = lon - originLon;

  const metersPerDegLat = (Math.PI / 180) * MARS_RADIUS;
  const metersPerDegLon = metersPerDegLat * Math.cos(originLatRad);

  const east = dLon * metersPerDegLon;
  const north = dLat * metersPerDegLat;
  const up = elevation;

  return [east, up, -north]; // Three.js: X=east, Y=up, Z=-north
}

/**
 * Normalize longitude to 0-360 or -180 to 180 range
 */
function normalizeLon(lon: number, range180: boolean = false): number {
  let normalized = ((lon % 360) + 360) % 360;
  if (range180 && normalized > 180) {
    normalized -= 360;
  }
  return normalized;
}

// =============================================================================
// API Functions
// =============================================================================

async function fetchDEMPatch(
  lat: number,
  lon: number,
  radiusM: number,
  gridSize: number = 128
): Promise<DEMPatchData> {
  const params = new URLSearchParams({
    lat: lat.toString(),
    lon: lon.toString(),
    radius_m: radiusM.toString(),
    grid_size: gridSize.toString(),
  });

  const url = `/terrain/dem_patch?${params}`;
  console.log("[Subsurface3D] Fetching DEM:", url);

  const res = await fetch(url);
  if (!res.ok) {
    const errorText = await res.text().catch(() => "Unknown error");
    console.error("[Subsurface3D] DEM fetch failed:", res.status, errorText);
    throw new Error(`DEM fetch failed (${res.status}): ${errorText.slice(0, 200)}`);
  }

  return res.json();
}

// =============================================================================
// Color Utilities
// =============================================================================

function elevationToTerrainColor(
  elevation: number,
  minElev: number,
  maxElev: number
): [number, number, number] {
  const range = maxElev - minElev;
  const t = range > 0 ? (elevation - minElev) / range : 0.5;

  // Mars terrain colormap: brown/tan
  const color = new THREE.Color();
  if (t < 0.25) {
    color.setHSL(0.05, 0.6, 0.2 + t * 0.8);
  } else if (t < 0.5) {
    color.setHSL(0.08, 0.5, 0.4 + (t - 0.25) * 0.6);
  } else if (t < 0.75) {
    color.setHSL(0.1, 0.4, 0.55 + (t - 0.5) * 0.4);
  } else {
    color.setHSL(0.1, 0.2 * (1 - t), 0.7 + (t - 0.75) * 0.8);
  }
  return [color.r, color.g, color.b];
}

function depthToSubsurfaceColor(
  depth: number,
  maxDepth: number
): [number, number, number] {
  const t = maxDepth > 0 ? Math.min(depth / maxDepth, 1) : 0;
  // Cyan to blue gradient for subsurface
  const color = new THREE.Color();
  color.setHSL(0.55 - t * 0.1, 0.7, 0.5 - t * 0.2);
  return [color.r, color.g, color.b];
}

// =============================================================================
// 3D Terrain Mesh Component
// =============================================================================

function TerrainMesh({
  data,
  verticalExaggeration,
  originLat,
  originLon,
  opacity,
}: {
  data: DEMPatchData;
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  opacity: number;
}) {
  const geometry = useMemo(() => {
    const { elevations, rows, cols, bounds, min_elevation_m, max_elevation_m } = data;

    const geo = new THREE.BufferGeometry();
    const numVertices = rows * cols;
    const numTriangles = (rows - 1) * (cols - 1) * 2;

    const positions = new Float32Array(numVertices * 3);
    const colors = new Float32Array(numVertices * 3);
    const indices = new Uint32Array(numTriangles * 3);

    const latStep = (bounds.north - bounds.south) / (rows - 1);
    const lonStep = (bounds.east - bounds.west) / (cols - 1);

    // Center elevation for normalization
    const centerElev = (min_elevation_m + max_elevation_m) / 2;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const vertexIndex = row * cols + col;
        const elev = elevations[vertexIndex];

        // Compute lat/lon for this vertex
        const lat = bounds.north - row * latStep;
        const lon = bounds.west + col * lonStep;

        // Convert to local ENU coordinates
        const normalizedElev = (elev - centerElev) * verticalExaggeration;
        const [x, y, z] = latLonToENU(lat, lon, normalizedElev, originLat, originLon);

        positions[vertexIndex * 3] = x;
        positions[vertexIndex * 3 + 1] = y;
        positions[vertexIndex * 3 + 2] = z;

        const [r, g, b] = elevationToTerrainColor(elev, min_elevation_m, max_elevation_m);
        colors[vertexIndex * 3] = r;
        colors[vertexIndex * 3 + 1] = g;
        colors[vertexIndex * 3 + 2] = b;
      }
    }

    // Generate triangle indices
    let indexOffset = 0;
    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const topLeft = row * cols + col;
        const topRight = row * cols + col + 1;
        const bottomLeft = (row + 1) * cols + col;
        const bottomRight = (row + 1) * cols + col + 1;

        indices[indexOffset++] = topLeft;
        indices[indexOffset++] = bottomLeft;
        indices[indexOffset++] = topRight;

        indices[indexOffset++] = topRight;
        indices[indexOffset++] = bottomLeft;
        indices[indexOffset++] = bottomRight;
      }
    }

    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();

    return geo;
  }, [data, verticalExaggeration, originLat, originLon]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent={opacity < 1}
        opacity={opacity}
      />
    </mesh>
  );
}

// =============================================================================
// Subsurface Ribbon Mesh Component
// =============================================================================

function SubsurfaceRibbonMesh({
  points,
  verticalExaggeration,
  originLat,
  originLon,
  centerElev,
  opacity,
}: {
  points: SubsurfacePoint[];
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  centerElev: number;
  opacity: number;
}) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;

    const geo = new THREE.BufferGeometry();
    const n = points.length;

    // Create a ribbon: 2 vertices per point (left and right of track)
    const numVertices = n * 2;
    const numTriangles = (n - 1) * 2;

    const positions = new Float32Array(numVertices * 3);
    const colors = new Float32Array(numVertices * 3);
    const indices = new Uint32Array(numTriangles * 3);

    // Find max depth for coloring
    const maxDepth = Math.max(...points.map((p) => p.depthM));

    // Compute track direction for each point to determine ribbon width direction
    for (let i = 0; i < n; i++) {
      const p = points[i];

      // Compute perpendicular direction to track
      let perpLat = 0,
        perpLon = 0;
      if (i < n - 1) {
        const next = points[i + 1];
        const dLat = next.lat - p.lat;
        const dLon = next.lon - p.lon;
        // Perpendicular: rotate 90 degrees
        const len = Math.sqrt(dLat * dLat + dLon * dLon);
        if (len > 0) {
          perpLat = -dLon / len;
          perpLon = dLat / len;
        }
      } else if (i > 0) {
        const prev = points[i - 1];
        const dLat = p.lat - prev.lat;
        const dLon = p.lon - prev.lon;
        const len = Math.sqrt(dLat * dLat + dLon * dLon);
        if (len > 0) {
          perpLat = -dLon / len;
          perpLon = dLat / len;
        }
      }

      // Convert ribbon width from meters to degrees (approximately)
      const metersPerDegLat = (Math.PI / 180) * MARS_RADIUS;
      const metersPerDegLon = metersPerDegLat * Math.cos((p.lat * Math.PI) / 180);
      const widthDegLat = (RIBBON_WIDTH_M / 2 / metersPerDegLat) * perpLat;
      const widthDegLon = (RIBBON_WIDTH_M / 2 / metersPerDegLon) * perpLon;

      // Left vertex
      const leftLat = p.lat + widthDegLat;
      const leftLon = p.lon + widthDegLon;
      const leftElev = (p.subsurfaceElevation - centerElev) * verticalExaggeration;
      const [lx, ly, lz] = latLonToENU(leftLat, leftLon, leftElev, originLat, originLon);

      // Right vertex
      const rightLat = p.lat - widthDegLat;
      const rightLon = p.lon - widthDegLon;
      const rightElev = (p.subsurfaceElevation - centerElev) * verticalExaggeration;
      const [rx, ry, rz] = latLonToENU(rightLat, rightLon, rightElev, originLat, originLon);

      const leftIdx = i * 2;
      const rightIdx = i * 2 + 1;

      positions[leftIdx * 3] = lx;
      positions[leftIdx * 3 + 1] = ly;
      positions[leftIdx * 3 + 2] = lz;

      positions[rightIdx * 3] = rx;
      positions[rightIdx * 3 + 1] = ry;
      positions[rightIdx * 3 + 2] = rz;

      // Color based on depth
      const [r, g, b] = depthToSubsurfaceColor(p.depthM, maxDepth);
      colors[leftIdx * 3] = r;
      colors[leftIdx * 3 + 1] = g;
      colors[leftIdx * 3 + 2] = b;
      colors[rightIdx * 3] = r;
      colors[rightIdx * 3 + 1] = g;
      colors[rightIdx * 3 + 2] = b;
    }

    // Generate triangle indices (strip pattern)
    let indexOffset = 0;
    for (let i = 0; i < n - 1; i++) {
      const l0 = i * 2;
      const r0 = i * 2 + 1;
      const l1 = (i + 1) * 2;
      const r1 = (i + 1) * 2 + 1;

      // Triangle 1: l0, l1, r0
      indices[indexOffset++] = l0;
      indices[indexOffset++] = l1;
      indices[indexOffset++] = r0;

      // Triangle 2: r0, l1, r1
      indices[indexOffset++] = r0;
      indices[indexOffset++] = l1;
      indices[indexOffset++] = r1;
    }

    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();

    return geo;
  }, [points, verticalExaggeration, originLat, originLon, centerElev]);

  if (!geometry) return null;

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={opacity}
      />
    </mesh>
  );
}

// =============================================================================
// SHARAD Track Line Component
// =============================================================================

function TrackLine({
  points,
  verticalExaggeration,
  originLat,
  originLon,
  centerElev,
  showSurface,
  showSubsurface,
}: {
  points: SubsurfacePoint[];
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  centerElev: number;
  showSurface: boolean;
  showSubsurface: boolean;
}) {
  const surfacePoints = useMemo(() => {
    if (!showSurface || points.length < 2) return null;

    const positions: [number, number, number][] = [];
    for (const p of points) {
      const elev = (p.surfaceElevation - centerElev) * verticalExaggeration;
      const [x, y, z] = latLonToENU(p.lat, p.lon, elev, originLat, originLon);
      positions.push([x, y, z]);
    }
    return positions;
  }, [points, verticalExaggeration, originLat, originLon, centerElev, showSurface]);

  const subsurfacePoints = useMemo(() => {
    if (!showSubsurface || points.length < 2) return null;

    const positions: [number, number, number][] = [];
    for (const p of points) {
      const elev = (p.subsurfaceElevation - centerElev) * verticalExaggeration;
      const [x, y, z] = latLonToENU(p.lat, p.lon, elev, originLat, originLon);
      positions.push([x, y, z]);
    }
    return positions;
  }, [points, verticalExaggeration, originLat, originLon, centerElev, showSubsurface]);

  return (
    <group>
      {surfacePoints && (
        <Line points={surfacePoints} color="#22c55e" lineWidth={2} />
      )}
      {subsurfacePoints && (
        <Line points={subsurfacePoints} color="#22d3ee" lineWidth={2} />
      )}
    </group>
  );
}

// =============================================================================
// Vertical Connector Lines Component
// =============================================================================

function VerticalConnectors({
  points,
  verticalExaggeration,
  originLat,
  originLon,
  centerElev,
  interval,
}: {
  points: SubsurfacePoint[];
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  centerElev: number;
  interval: number;
}) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;

    const positions: number[] = [];
    for (let i = 0; i < points.length; i += interval) {
      const p = points[i];
      const surfElev = (p.surfaceElevation - centerElev) * verticalExaggeration;
      const subElev = (p.subsurfaceElevation - centerElev) * verticalExaggeration;

      const [sx, sy, sz] = latLonToENU(p.lat, p.lon, surfElev, originLat, originLon);
      const [bx, by, bz] = latLonToENU(p.lat, p.lon, subElev, originLat, originLon);

      positions.push(sx, sy, sz);
      positions.push(bx, by, bz);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, [points, verticalExaggeration, originLat, originLon, centerElev, interval]);

  if (!geometry) return null;

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#94a3b8" linewidth={1} transparent opacity={0.5} />
    </lineSegments>
  );
}

// =============================================================================
// Camera Setup Component
// =============================================================================

function CameraSetup({
  sceneSize,
}: {
  sceneSize: number;
}) {
  const { camera } = useThree();

  useEffect(() => {
    const distance = sceneSize * 1.5;
    camera.position.set(distance * 0.7, distance * 0.5, distance * 0.7);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, sceneSize]);

  return null;
}

// =============================================================================
// Main Component
// =============================================================================

export default function Subsurface3DViewer({
  startTrace,
  endTrace,
  lats,
  lons,
  boundaryBinOffset,
  epsilonR,
  molaElevations,
  onClose,
}: Subsurface3DViewerProps) {
  const [terrainData, setTerrainData] = useState<DEMPatchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Controls
  const [verticalExaggeration, setVerticalExaggeration] = useState(5);
  const [showTerrain, setShowTerrain] = useState(true);
  const [showSubsurface, setShowSubsurface] = useState(true);
  const [showTrackLines, setShowTrackLines] = useState(true);
  const [showConnectors, setShowConnectors] = useState(false);
  const [terrainOpacity, setTerrainOpacity] = useState(0.8);
  const [subsurfaceOpacity, setSubsurfaceOpacity] = useState(0.7);

  // Compute subsurface points from boundary data
  const subsurfacePoints = useMemo((): SubsurfacePoint[] => {
    const points: SubsurfacePoint[] = [];

    for (let trace = startTrace; trace <= endTrace; trace++) {
      if (trace < 0 || trace >= lats.length) continue;

      const lat = lats[trace];
      const lon = lons[trace];
      if (lat === undefined || lon === undefined) continue;

      // Get MOLA elevation at this trace
      const molaElev = molaElevations[trace];
      if (molaElev === null || molaElev === undefined) continue;

      // Compute depth from boundary offset
      // depth = (c * t) / (2 * sqrt(eps_r))
      // t = boundaryBinOffset * BIN_DT_SEC
      const timeDelaySec = boundaryBinOffset * BIN_DT_SEC;
      const depthM = (SPEED_OF_LIGHT * timeDelaySec) / (2 * Math.sqrt(epsilonR));

      const subsurfaceElevation = molaElev - depthM;

      points.push({
        lat,
        lon: normalizeLon(lon),
        surfaceElevation: molaElev,
        subsurfaceElevation,
        depthM,
      });
    }

    return points;
  }, [startTrace, endTrace, lats, lons, molaElevations, boundaryBinOffset, epsilonR]);

  // Compute scene center and bounds
  const { originLat, originLon, centerElev, boundsRadiusM } = useMemo(() => {
    if (subsurfacePoints.length === 0) {
      return { originLat: 0, originLon: 0, centerElev: 0, boundsRadiusM: 5000 };
    }

    const lats = subsurfacePoints.map((p) => p.lat);
    const lons = subsurfacePoints.map((p) => p.lon);
    const elevs = subsurfacePoints.map((p) => p.surfaceElevation);

    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const minElev = Math.min(...elevs);
    const maxElev = Math.max(...elevs);

    const oLat = (minLat + maxLat) / 2;
    const oLon = (minLon + maxLon) / 2;
    const cElev = (minElev + maxElev) / 2;

    // Compute radius in meters
    const dLat = maxLat - minLat;
    const dLon = maxLon - minLon;
    const metersPerDegLat = (Math.PI / 180) * MARS_RADIUS;
    const metersPerDegLon = metersPerDegLat * Math.cos((oLat * Math.PI) / 180);

    const latExtentM = dLat * metersPerDegLat;
    const lonExtentM = dLon * metersPerDegLon;
    const radius = Math.max(latExtentM, lonExtentM) / 2 + 2000; // Add margin

    // Cap radius to backend limit (50km max)
    const clampedRadius = Math.min(Math.max(radius, 5000), 50000);

    return {
      originLat: oLat,
      originLon: oLon,
      centerElev: cElev,
      boundsRadiusM: clampedRadius,
    };
  }, [subsurfacePoints]);

  // Fetch terrain data
  useEffect(() => {
    let cancelled = false;

    async function loadTerrain() {
      console.log("[Subsurface3D] Loading terrain:", {
        subsurfacePointsCount: subsurfacePoints.length,
        originLat,
        originLon,
        boundsRadiusM,
        startTrace,
        endTrace,
      });

      if (subsurfacePoints.length === 0) {
        setLoading(false);
        setError("No valid subsurface points in selected range. Check that MOLA data exists for the selected traces.");
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const data = await fetchDEMPatch(originLat, originLon, boundsRadiusM, 128);

        if (!cancelled) {
          setTerrainData(data);
        }
      } catch (e) {
        if (!cancelled) {
          console.error("[Subsurface3D] Error loading terrain:", e);
          setError(e instanceof Error ? e.message : "Failed to load terrain");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadTerrain();

    return () => {
      cancelled = true;
    };
  }, [originLat, originLon, boundsRadiusM, subsurfacePoints.length]);

  // Compute scene size for camera
  const sceneSize = boundsRadiusM;

  // Stats
  const stats = useMemo(() => {
    if (subsurfacePoints.length === 0) return null;
    const depths = subsurfacePoints.map((p) => p.depthM);
    return {
      numPoints: subsurfacePoints.length,
      minDepth: Math.min(...depths).toFixed(0),
      maxDepth: Math.max(...depths).toFixed(0),
      avgDepth: (depths.reduce((a, b) => a + b, 0) / depths.length).toFixed(0),
    };
  }, [subsurfacePoints]);

  return (
    <div className="flex flex-col h-full bg-[#0a0f18]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-cyan-400">layers</span>
          <div>
            <h2 className="text-white font-bold text-sm">3D Subsurface View</h2>
            <p className="text-[#6b7c9c] text-[10px]">
              Traces {startTrace} - {endTrace} ({endTrace - startTrace + 1} traces)
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded hover:bg-[#232f48] transition-colors text-[#92a4c9] hover:text-white"
        >
          <span className="material-symbols-outlined">close</span>
        </button>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 border-b border-[#232f48] space-y-3">
        {/* Vertical Exaggeration */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase text-[#6b7c9c] w-24">V. Exag.</span>
          <input
            type="range"
            min="1"
            max="30"
            value={verticalExaggeration}
            onChange={(e) => setVerticalExaggeration(Number(e.target.value))}
            className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-cyan-500
              [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <span className="text-[11px] text-white font-mono w-8 text-right">
            {verticalExaggeration}×
          </span>
        </div>

        {/* Layer toggles */}
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showTerrain}
              onChange={(e) => setShowTerrain(e.target.checked)}
              className="rounded border-slate-600 bg-transparent text-amber-500 focus:ring-amber-500/30"
            />
            <span className="text-[10px] text-slate-300">Terrain</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showSubsurface}
              onChange={(e) => setShowSubsurface(e.target.checked)}
              className="rounded border-slate-600 bg-transparent text-cyan-500 focus:ring-cyan-500/30"
            />
            <span className="text-[10px] text-slate-300">Interface</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showTrackLines}
              onChange={(e) => setShowTrackLines(e.target.checked)}
              className="rounded border-slate-600 bg-transparent text-green-500 focus:ring-green-500/30"
            />
            <span className="text-[10px] text-slate-300">Track</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showConnectors}
              onChange={(e) => setShowConnectors(e.target.checked)}
              className="rounded border-slate-600 bg-transparent text-slate-500 focus:ring-slate-500/30"
            />
            <span className="text-[10px] text-slate-300">Connectors</span>
          </label>
        </div>

        {/* Opacity controls */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 flex-1">
            <span className="text-[9px] text-[#6b7c9c]">Terrain</span>
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.1"
              value={terrainOpacity}
              onChange={(e) => setTerrainOpacity(Number(e.target.value))}
              className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-2
                [&::-webkit-slider-thumb]:w-2
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-amber-400
                [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
          <div className="flex items-center gap-2 flex-1">
            <span className="text-[9px] text-[#6b7c9c]">Interface</span>
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.1"
              value={subsurfaceOpacity}
              onChange={(e) => setSubsurfaceOpacity(Number(e.target.value))}
              className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-2
                [&::-webkit-slider-thumb]:w-2
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-cyan-400
                [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <div className="text-center">
              <span className="material-symbols-outlined text-4xl text-cyan-400 animate-spin">
                progress_activity
              </span>
              <p className="text-[#92a4c9] text-sm mt-2">Loading terrain...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <div className="text-center">
              <span className="material-symbols-outlined text-4xl text-red-400">error</span>
              <p className="text-red-400 text-sm mt-2">{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && terrainData && (
          <Canvas
            camera={{
              fov: 50,
              near: 1,
              far: 1000000,
              position: [sceneSize * 2, sceneSize, sceneSize * 2],
            }}
            gl={{ antialias: true }}
          >
            <CameraSetup sceneSize={sceneSize} />

            <OrbitControls
              enableDamping
              dampingFactor={0.05}
              minDistance={sceneSize * 0.1}
              maxDistance={sceneSize * 10}
              target={[0, 0, 0]}
            />

            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[1, 2, 1]} intensity={0.8} />
            <directionalLight position={[-1, 1, -1]} intensity={0.3} />

            {/* Terrain mesh */}
            {showTerrain && (
              <TerrainMesh
                data={terrainData}
                verticalExaggeration={verticalExaggeration}
                originLat={originLat}
                originLon={originLon}
                opacity={terrainOpacity}
              />
            )}

            {/* Subsurface ribbon */}
            {showSubsurface && subsurfacePoints.length > 1 && (
              <SubsurfaceRibbonMesh
                points={subsurfacePoints}
                verticalExaggeration={verticalExaggeration}
                originLat={originLat}
                originLon={originLon}
                centerElev={centerElev}
                opacity={subsurfaceOpacity}
              />
            )}

            {/* Track lines */}
            {showTrackLines && (
              <TrackLine
                points={subsurfacePoints}
                verticalExaggeration={verticalExaggeration}
                originLat={originLat}
                originLon={originLon}
                centerElev={centerElev}
                showSurface={true}
                showSubsurface={showSubsurface}
              />
            )}

            {/* Vertical connectors */}
            {showConnectors && (
              <VerticalConnectors
                points={subsurfacePoints}
                verticalExaggeration={verticalExaggeration}
                originLat={originLat}
                originLon={originLon}
                centerElev={centerElev}
                interval={Math.max(1, Math.floor(subsurfacePoints.length / 20))}
              />
            )}

            {/* Grid helper */}
            <gridHelper
              args={[sceneSize * 4, 20, "#333333", "#222222"]}
              position={[0, -sceneSize * 0.3, 0]}
            />
          </Canvas>
        )}
      </div>

      {/* Stats Footer */}
      {stats && (
        <div className="px-4 py-3 border-t border-[#232f48] bg-[#101622]">
          <div className="grid grid-cols-5 gap-4 text-center">
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Points</p>
              <p className="text-white text-sm font-mono">{stats.numPoints}</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Min Depth</p>
              <p className="text-white text-sm font-mono">{stats.minDepth} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Max Depth</p>
              <p className="text-white text-sm font-mono">{stats.maxDepth} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Avg Depth</p>
              <p className="text-white text-sm font-mono">{stats.avgDepth} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">εr</p>
              <p className="text-amber-400 text-sm font-mono">{epsilonR.toFixed(1)}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
