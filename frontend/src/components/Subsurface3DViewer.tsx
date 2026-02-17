/**
 * Subsurface3DViewer - 3D visualization of SHARAD subsurface interface
 *
 * Displays MOLA terrain as a 3D mesh surface (same as SlopeAnalysis3DTab)
 * with the subsurface boundary as a 1D line following the track path at depth.
 */

import { useState, useEffect, useMemo } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";

// =============================================================================
// Types
// =============================================================================

interface Subsurface3DViewerProps {
  startTrace: number;
  endTrace: number;
  lats: number[];
  lons: number[];
  boundaryBinOffset: number;
  epsilonR: number;
  molaElevations: (number | null)[];
  onClose: () => void;
}

interface DEMPatchData {
  elevations: number[];
  rows: number;
  cols: number;
  spacing_m: number;
  bounds: { west: number; east: number; south: number; north: number };
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

const SPEED_OF_LIGHT = 299792458;
const BIN_DT_SEC = 0.0375e-6; // 1/26.67 MHz ADC — seconds per range bin (two-way)
const MARS_RADIUS = 3389500;

// =============================================================================
// API
// =============================================================================

async function fetchDEMPatch(
  lat: number,
  lon: number,
  radiusM: number,
  gridSize: number = 128
): Promise<DEMPatchData> {
  const params = new URLSearchParams({
    lat: lat.toString(),
    lon: (((lon % 360) + 360) % 360).toString(),
    radius_m: radiusM.toString(),
    grid_size: gridSize.toString(),
  });

  const res = await fetch(`/terrain/dem_patch?${params}`);
  if (!res.ok) {
    const errorText = await res.text().catch(() => "Unknown error");
    throw new Error(`DEM fetch failed (${res.status}): ${errorText.slice(0, 200)}`);
  }
  return res.json();
}

// =============================================================================
// Color Utilities (same as SlopeAnalysis3DTab)
// =============================================================================

function elevationToColor(
  elevation: number,
  minElev: number,
  maxElev: number
): [number, number, number] {
  const range = maxElev - minElev;
  const t = range > 0 ? (elevation - minElev) / range : 0.5;

  // Mars terrain colormap: brown/tan to white (high elevation)
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

// =============================================================================
// Terrain Mesh (same approach as SlopeAnalysis3DTab)
// =============================================================================

function TerrainMesh({
  data,
  verticalExaggeration,
  wireframe,
}: {
  data: DEMPatchData;
  verticalExaggeration: number;
  wireframe: boolean;
}) {
  const geometry = useMemo(() => {
    const { elevations, rows, cols, min_elevation_m, max_elevation_m, radius_m } = data;

    if (rows < 2 || cols < 2) return new THREE.BufferGeometry();

    const meshWidth = radius_m * 2;
    const meshHeight = radius_m * 2;
    const centerElev = (min_elevation_m + max_elevation_m) / 2;

    const geo = new THREE.BufferGeometry();
    const numVertices = rows * cols;
    const numTriangles = (rows - 1) * (cols - 1) * 2;

    const positions = new Float32Array(numVertices * 3);
    const colors = new Float32Array(numVertices * 3);
    const indices = new Uint32Array(numTriangles * 3);

    const stepX = meshWidth / (cols - 1);
    const stepZ = meshHeight / (rows - 1);

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const vertexIndex = row * cols + col;
        const x = -meshWidth / 2 + col * stepX;
        const z = meshHeight / 2 - row * stepZ;
        const elev = elevations[vertexIndex];
        const y = (elev - centerElev) * verticalExaggeration;

        positions[vertexIndex * 3] = x;
        positions[vertexIndex * 3 + 1] = y;
        positions[vertexIndex * 3 + 2] = z;

        const [r, g, b] = elevationToColor(elev, min_elevation_m, max_elevation_m);
        colors[vertexIndex * 3] = r;
        colors[vertexIndex * 3 + 1] = g;
        colors[vertexIndex * 3 + 2] = b;
      }
    }

    let indexOffset = 0;
    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const tl = row * cols + col;
        const tr = row * cols + col + 1;
        const bl = (row + 1) * cols + col;
        const br = (row + 1) * cols + col + 1;
        indices[indexOffset++] = tl;
        indices[indexOffset++] = bl;
        indices[indexOffset++] = tr;
        indices[indexOffset++] = tr;
        indices[indexOffset++] = bl;
        indices[indexOffset++] = br;
      }
    }

    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    return geo;
  }, [data, verticalExaggeration]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} wireframe={wireframe} flatShading={false} />
    </mesh>
  );
}

// =============================================================================
// Track Line on Surface (uses track geometry for positioning)
// =============================================================================

function TrackLine({
  points,
  trackGeometry,
  terrainData,
  verticalExaggeration,
}: {
  points: { lat: number; lon: number; elev: number }[];
  trackGeometry: {
    centerLat: number;
    centerLon: number;
    metersPerDegLat: number;
    metersPerDegLon: number;
  };
  terrainData: DEMPatchData;
  verticalExaggeration: number;
}) {
  const linePoints = useMemo(() => {
    const { centerLat, centerLon, metersPerDegLat, metersPerDegLon } = trackGeometry;
    const centerElev = (terrainData.min_elevation_m + terrainData.max_elevation_m) / 2;

    return points.map((p) => {
      // Convert lat/lon to local coordinates relative to track center
      const x = (p.lon - centerLon) * metersPerDegLon;
      const z = -(p.lat - centerLat) * metersPerDegLat; // Negative because north is -Z in Three.js
      const y = (p.elev - centerElev) * verticalExaggeration + 5; // Slight offset above terrain
      return [x, y, z] as [number, number, number];
    });
  }, [points, trackGeometry, terrainData, verticalExaggeration]);

  if (linePoints.length < 2) return null;
  return <Line points={linePoints} color="#22c55e" lineWidth={4} />;
}

// =============================================================================
// Subsurface Line (1D line at depth below surface)
// =============================================================================

function SubsurfaceLine({
  points,
  trackGeometry,
  terrainData,
  meanDepth,
  verticalExaggeration,
}: {
  points: { lat: number; lon: number; elev: number }[];
  trackGeometry: {
    centerLat: number;
    centerLon: number;
    metersPerDegLat: number;
    metersPerDegLon: number;
  };
  terrainData: DEMPatchData;
  meanDepth: number;
  verticalExaggeration: number;
}) {
  const linePoints = useMemo(() => {
    const { centerLat, centerLon, metersPerDegLat, metersPerDegLon } = trackGeometry;
    const centerElev = (terrainData.min_elevation_m + terrainData.max_elevation_m) / 2;

    return points.map((p) => {
      // Convert lat/lon to local coordinates relative to track center
      const x = (p.lon - centerLon) * metersPerDegLon;
      const z = -(p.lat - centerLat) * metersPerDegLat;
      // Subsurface: surface elevation minus depth
      const subsurfaceElev = p.elev - meanDepth;
      const y = (subsurfaceElev - centerElev) * verticalExaggeration;
      return [x, y, z] as [number, number, number];
    });
  }, [points, trackGeometry, terrainData, meanDepth, verticalExaggeration]);

  if (linePoints.length < 2) return null;
  return <Line points={linePoints} color="#22d3ee" lineWidth={4} />;
}

// =============================================================================
// End Connectors (vertical lines at track endpoints)
// =============================================================================

function EndConnectors({
  points,
  trackGeometry,
  terrainData,
  meanDepth,
  verticalExaggeration,
}: {
  points: { lat: number; lon: number; elev: number }[];
  trackGeometry: {
    centerLat: number;
    centerLon: number;
    metersPerDegLat: number;
    metersPerDegLon: number;
  };
  terrainData: DEMPatchData;
  meanDepth: number;
  verticalExaggeration: number;
}) {
  const lines = useMemo(() => {
    if (points.length < 2) return [];

    const { centerLat, centerLon, metersPerDegLat, metersPerDegLon } = trackGeometry;
    const centerElev = (terrainData.min_elevation_m + terrainData.max_elevation_m) / 2;

    const result: [number, number, number][][] = [];

    for (const p of [points[0], points[points.length - 1]]) {
      const x = (p.lon - centerLon) * metersPerDegLon;
      const z = -(p.lat - centerLat) * metersPerDegLat;
      const surfaceY = (p.elev - centerElev) * verticalExaggeration + 5;
      const subsurfaceY = (p.elev - meanDepth - centerElev) * verticalExaggeration;
      result.push([[x, surfaceY, z], [x, subsurfaceY, z]]);
    }
    return result;
  }, [points, trackGeometry, terrainData, meanDepth, verticalExaggeration]);

  return (
    <>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#f59e0b" lineWidth={3} />
      ))}
    </>
  );
}

// =============================================================================
// Camera Setup
// =============================================================================

function CameraSetup({
  data,
  verticalExaggeration,
  subsurfaceDepthY,
}: {
  data: DEMPatchData;
  verticalExaggeration: number;
  subsurfaceDepthY: number;
}) {
  const { camera } = useThree();
  const { radius_m } = data;

  useEffect(() => {
    const meshSize = radius_m * 2;
    const distance = meshSize * 1.5;
    // Look at midpoint between surface and subsurface
    const lookAtY = subsurfaceDepthY / 2;
    camera.position.set(distance * 0.7, distance * 0.5 + lookAtY, distance * 0.7);
    camera.lookAt(0, lookAtY, 0);
    camera.updateProjectionMatrix();
  }, [camera, radius_m, verticalExaggeration, subsurfaceDepthY]);

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

  const [verticalExaggeration, setVerticalExaggeration] = useState(5);
  const [showSubsurface, setShowSubsurface] = useState(true);
  const [showWireframe, setShowWireframe] = useState(false);

  // Ensure start <= end
  const rangeStart = Math.min(startTrace, endTrace);
  const rangeEnd = Math.max(startTrace, endTrace);

  // Compute mean depth from boundary offset
  const meanDepth = useMemo(() => {
    const timeDelaySec = boundaryBinOffset * BIN_DT_SEC;
    return (SPEED_OF_LIGHT * timeDelaySec) / (2 * Math.sqrt(epsilonR));
  }, [boundaryBinOffset, epsilonR]);

  // Collect valid track points
  const trackPoints = useMemo(() => {
    const points: { lat: number; lon: number; elev: number }[] = [];
    for (let trace = rangeStart; trace <= rangeEnd; trace++) {
      if (trace < 0 || trace >= lats.length) continue;
      const lat = lats[trace];
      const lon = lons[trace];
      const elev = molaElevations[trace];
      if (lat == null || lon == null || elev == null) continue;
      points.push({ lat, lon: ((lon % 360) + 360) % 360, elev });
    }
    return points;
  }, [rangeStart, rangeEnd, lats, lons, molaElevations]);

  // Compute track geometry (this defines the subsurface extent)
  const trackGeometry = useMemo(() => {
    if (trackPoints.length === 0) {
      return {
        centerLat: 0,
        centerLon: 0,
        minLat: 0,
        maxLat: 0,
        minLon: 0,
        maxLon: 0,
        trackLengthM: 5000,
        metersPerDegLat: 59274,
        metersPerDegLon: 59274,
      };
    }

    const latArr = trackPoints.map((p) => p.lat);
    const lonArr = trackPoints.map((p) => p.lon);

    const minLat = Math.min(...latArr);
    const maxLat = Math.max(...latArr);
    const minLon = Math.min(...lonArr);
    const maxLon = Math.max(...lonArr);

    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;

    const metersPerDegLat = (Math.PI / 180) * MARS_RADIUS;
    const metersPerDegLon = metersPerDegLat * Math.cos((centerLat * Math.PI) / 180);

    const latExtentM = (maxLat - minLat) * metersPerDegLat;
    const lonExtentM = (maxLon - minLon) * metersPerDegLon;
    const trackLength = Math.sqrt(latExtentM ** 2 + lonExtentM ** 2);

    return {
      centerLat,
      centerLon,
      minLat,
      maxLat,
      minLon,
      maxLon,
      trackLengthM: Math.max(trackLength, 5000),
      metersPerDegLat,
      metersPerDegLon,
    };
  }, [trackPoints]);

  // Terrain radius = track length (so diameter = 2× track length)
  const terrainRadiusM = useMemo(() => {
    return Math.min(trackGeometry.trackLengthM, 50000);
  }, [trackGeometry.trackLengthM]);

  // Fetch terrain centered on track
  useEffect(() => {
    if (trackPoints.length === 0) {
      setLoading(false);
      setError("No valid points in selected range");
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchDEMPatch(trackGeometry.centerLat, trackGeometry.centerLon, terrainRadiusM, 128)
      .then((data) => {
        if (!cancelled) {
          console.log("[Subsurface3D] Loaded terrain:", {
            rows: data.rows,
            cols: data.cols,
            radius_m: data.radius_m,
            trackLengthM: trackGeometry.trackLengthM,
            terrainDiameter: data.radius_m * 2,
          });
          setTerrainData(data);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [trackGeometry.centerLat, trackGeometry.centerLon, terrainRadiusM, trackPoints.length, trackGeometry.trackLengthM]);

  return (
    <div className="flex flex-col h-full bg-[#0a0f18]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-cyan-400 text-lg">layers</span>
          <div>
            <h2 className="text-white font-bold text-sm">3D Subsurface View</h2>
            <p className="text-[#6b7c9c] text-[10px]">
              {trackPoints.length} traces · Depth: {meanDepth.toFixed(0)}m · Track: {(trackGeometry.trackLengthM / 1000).toFixed(1)}km · Terrain: {(terrainRadiusM * 2 / 1000).toFixed(1)}km ⌀
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-[#232f48] text-[#92a4c9] hover:text-white"
        >
          <span className="material-symbols-outlined">close</span>
        </button>
      </div>

      {/* Controls */}
      <div className="px-4 py-2 border-b border-[#232f48] flex items-center gap-4">
        <div className="flex items-center gap-2 flex-1">
          <span className="text-[9px] text-[#6b7c9c] uppercase">V.Exag</span>
          <input
            type="range"
            min="1"
            max="30"
            value={verticalExaggeration}
            onChange={(e) => setVerticalExaggeration(Number(e.target.value))}
            className="flex-1 h-1 bg-[#232f48] rounded appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-cyan-500 [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <span className="text-[10px] text-white font-mono w-6">{verticalExaggeration}×</span>
        </div>

        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showSubsurface}
            onChange={(e) => setShowSubsurface(e.target.checked)}
            className="rounded border-slate-600 bg-transparent text-cyan-500"
          />
          <span className="text-[10px] text-slate-300">Interface</span>
        </label>

        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showWireframe}
            onChange={(e) => setShowWireframe(e.target.checked)}
            className="rounded border-slate-600 bg-transparent text-cyan-500"
          />
          <span className="text-[10px] text-slate-300">Wireframe</span>
        </label>
      </div>

      {/* 3D Canvas */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <span className="material-symbols-outlined text-4xl text-cyan-400 animate-spin">progress_activity</span>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <div className="text-center">
              <span className="material-symbols-outlined text-3xl text-red-400">error</span>
              <p className="text-red-400 text-sm mt-2">{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && terrainData && (() => {
          // Calculate subsurface depth in scene units
          const subsurfaceDepthY = -meanDepth * verticalExaggeration;
          return (
          <Canvas
            camera={{ fov: 50, near: 1, far: 1000000 }}
            gl={{ antialias: true }}
          >
            <CameraSetup data={terrainData} verticalExaggeration={verticalExaggeration} subsurfaceDepthY={subsurfaceDepthY} />
            <OrbitControls
              enableDamping
              dampingFactor={0.05}
              minDistance={terrainData.radius_m * 0.1}
              maxDistance={terrainData.radius_m * 10}
              target={[0, subsurfaceDepthY / 2, 0]}
            />

            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[1, 2, 1]} intensity={0.8} />
            <directionalLight position={[-1, 1, -1]} intensity={0.3} />

            {/* 3D MOLA Terrain Mesh */}
            <TerrainMesh
              data={terrainData}
              verticalExaggeration={verticalExaggeration}
              wireframe={showWireframe}
            />

            {/* Track line on surface (green) */}
            <TrackLine
              points={trackPoints}
              trackGeometry={trackGeometry}
              terrainData={terrainData}
              verticalExaggeration={verticalExaggeration}
            />

            {/* Subsurface line (cyan) */}
            {showSubsurface && (
              <SubsurfaceLine
                points={trackPoints}
                trackGeometry={trackGeometry}
                terrainData={terrainData}
                meanDepth={meanDepth}
                verticalExaggeration={verticalExaggeration}
              />
            )}

            {/* Vertical connectors at ends */}
            {showSubsurface && (
              <EndConnectors
                points={trackPoints}
                trackGeometry={trackGeometry}
                terrainData={terrainData}
                meanDepth={meanDepth}
                verticalExaggeration={verticalExaggeration}
              />
            )}

            {/* Ground reference grid - position at subsurface level */}
            <gridHelper
              args={[terrainData.radius_m * 4, 20, "#333333", "#222222"]}
              position={[0, subsurfaceDepthY - 50, 0]}
            />
          </Canvas>
          );
        })()}
      </div>

      {/* Stats Footer */}
      {terrainData && (
        <div className="px-4 py-2 border-t border-[#232f48] bg-[#101622]">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Track Length</p>
              <p className="text-white text-sm font-mono">{(trackGeometry.trackLengthM / 1000).toFixed(1)} km</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Terrain ⌀</p>
              <p className="text-white text-sm font-mono">{(terrainData.radius_m * 2 / 1000).toFixed(1)} km</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Elev Range</p>
              <p className="text-white text-sm font-mono">{terrainData.elevation_range_m.toFixed(0)} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Sub. Depth</p>
              <p className="text-cyan-400 text-sm font-mono">{meanDepth.toFixed(0)} m</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
