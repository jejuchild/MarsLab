/**
 * Subsurface3DViewer - 3D visualization of SHARAD subsurface interface
 *
 * Prototype: Displays MOLA terrain with a FLAT subsurface boundary layer
 * at mean depth (no undulations).
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
  bounds: { west: number; east: number; south: number; north: number };
  min_elevation_m: number;
  max_elevation_m: number;
  radius_m: number;
}

// =============================================================================
// Constants
// =============================================================================

const SPEED_OF_LIGHT = 299792458;
const BIN_DT_SEC = 0.0375e-6;
const MARS_RADIUS = 3389500;

// =============================================================================
// Coordinate Conversion
// =============================================================================

function latLonToENU(
  lat: number,
  lon: number,
  elevation: number,
  originLat: number,
  originLon: number
): [number, number, number] {
  const originLatRad = (originLat * Math.PI) / 180;
  const dLat = lat - originLat;
  const dLon = lon - originLon;
  const metersPerDegLat = (Math.PI / 180) * MARS_RADIUS;
  const metersPerDegLon = metersPerDegLat * Math.cos(originLatRad);
  const east = dLon * metersPerDegLon;
  const north = dLat * metersPerDegLat;
  return [east, elevation, -north];
}

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
// Terrain Mesh
// =============================================================================

function TerrainMesh({
  data,
  verticalExaggeration,
  originLat,
  originLon,
  centerElev,
  opacity,
}: {
  data: DEMPatchData;
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  centerElev: number;
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

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const vertexIndex = row * cols + col;
        const elev = elevations[vertexIndex];
        const lat = bounds.north - row * latStep;
        const lon = bounds.west + col * lonStep;
        const normalizedElev = (elev - centerElev) * verticalExaggeration;
        const [x, y, z] = latLonToENU(lat, lon, normalizedElev, originLat, originLon);

        positions[vertexIndex * 3] = x;
        positions[vertexIndex * 3 + 1] = y;
        positions[vertexIndex * 3 + 2] = z;

        // Brown/tan Mars terrain colors
        const t = (elev - min_elevation_m) / (max_elevation_m - min_elevation_m || 1);
        const color = new THREE.Color();
        color.setHSL(0.07, 0.5 - t * 0.2, 0.35 + t * 0.35);
        colors[vertexIndex * 3] = color.r;
        colors[vertexIndex * 3 + 1] = color.g;
        colors[vertexIndex * 3 + 2] = color.b;
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
  }, [data, verticalExaggeration, originLat, originLon, centerElev]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} transparent opacity={opacity} />
    </mesh>
  );
}

// =============================================================================
// Flat Subsurface Plane (at mean depth)
// =============================================================================

function SubsurfacePlane({
  centerLat,
  centerLon,
  originLat,
  originLon,
  centerElev,
  meanDepth,
  extentM,
  verticalExaggeration,
  opacity,
}: {
  centerLat: number;
  centerLon: number;
  originLat: number;
  originLon: number;
  centerElev: number;
  meanDepth: number;
  extentM: number;
  verticalExaggeration: number;
  opacity: number;
}) {
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(extentM * 2, extentM * 2, 1, 1);
    // Rotate to be horizontal (XZ plane)
    geo.rotateX(-Math.PI / 2);
    return geo;
  }, [extentM]);

  // Position at center, at mean subsurface depth
  const subsurfaceElev = (0 - meanDepth - centerElev) * verticalExaggeration;
  const [cx, , cz] = latLonToENU(centerLat, centerLon, 0, originLat, originLon);

  return (
    <mesh geometry={geometry} position={[cx, subsurfaceElev, cz]}>
      <meshStandardMaterial
        color="#22d3ee"
        side={THREE.DoubleSide}
        transparent
        opacity={opacity}
        emissive="#0891b2"
        emissiveIntensity={0.2}
      />
    </mesh>
  );
}

// =============================================================================
// Track Line (surface)
// =============================================================================

function TrackLine({
  points,
  verticalExaggeration,
  originLat,
  originLon,
  centerElev,
}: {
  points: { lat: number; lon: number; elev: number }[];
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  centerElev: number;
}) {
  const linePoints = useMemo(() => {
    return points.map((p) => {
      const elev = (p.elev - centerElev) * verticalExaggeration;
      return latLonToENU(p.lat, p.lon, elev, originLat, originLon);
    });
  }, [points, verticalExaggeration, originLat, originLon, centerElev]);

  if (linePoints.length < 2) return null;
  return <Line points={linePoints} color="#22c55e" lineWidth={3} />;
}

// =============================================================================
// Vertical Connectors at ends
// =============================================================================

function EndConnectors({
  startPoint,
  endPoint,
  meanDepth,
  verticalExaggeration,
  originLat,
  originLon,
  centerElev,
}: {
  startPoint: { lat: number; lon: number; elev: number } | null;
  endPoint: { lat: number; lon: number; elev: number } | null;
  meanDepth: number;
  verticalExaggeration: number;
  originLat: number;
  originLon: number;
  centerElev: number;
}) {
  const lines = useMemo(() => {
    const result: [number, number, number][][] = [];

    for (const p of [startPoint, endPoint]) {
      if (!p) continue;
      const surfElev = (p.elev - centerElev) * verticalExaggeration;
      const subElev = (p.elev - meanDepth - centerElev) * verticalExaggeration;
      const [x, , z] = latLonToENU(p.lat, p.lon, 0, originLat, originLon);
      result.push([
        [x, surfElev, z],
        [x, subElev, z],
      ]);
    }
    return result;
  }, [startPoint, endPoint, meanDepth, verticalExaggeration, originLat, originLon, centerElev]);

  return (
    <>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#94a3b8" lineWidth={2} dashed dashSize={100} gapSize={50} />
      ))}
    </>
  );
}

// =============================================================================
// Camera Setup
// =============================================================================

function CameraSetup({ sceneSize }: { sceneSize: number }) {
  const { camera } = useThree();
  useEffect(() => {
    const distance = sceneSize * 1.8;
    camera.position.set(distance * 0.6, distance * 0.5, distance * 0.6);
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

  const [verticalExaggeration, setVerticalExaggeration] = useState(5);
  const [showSubsurface, setShowSubsurface] = useState(true);
  const [terrainOpacity, setTerrainOpacity] = useState(0.85);
  const [subsurfaceOpacity] = useState(0.6);

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

  // Compute scene geometry
  const { originLat, originLon, centerElev, boundsRadiusM } = useMemo(() => {
    if (trackPoints.length === 0) {
      return { originLat: 0, originLon: 0, centerElev: 0, boundsRadiusM: 5000 };
    }

    const latArr = trackPoints.map((p) => p.lat);
    const lonArr = trackPoints.map((p) => p.lon);
    const elevArr = trackPoints.map((p) => p.elev);

    const minLat = Math.min(...latArr);
    const maxLat = Math.max(...latArr);
    const minLon = Math.min(...lonArr);
    const maxLon = Math.max(...lonArr);
    const minElev = Math.min(...elevArr);
    const maxElev = Math.max(...elevArr);

    const oLat = (minLat + maxLat) / 2;
    const oLon = (minLon + maxLon) / 2;
    const cElev = (minElev + maxElev) / 2;

    const metersPerDegLat = (Math.PI / 180) * MARS_RADIUS;
    const metersPerDegLon = metersPerDegLat * Math.cos((oLat * Math.PI) / 180);

    const latExtentM = (maxLat - minLat) * metersPerDegLat;
    const lonExtentM = (maxLon - minLon) * metersPerDegLon;
    const trackLength = Math.sqrt(latExtentM ** 2 + lonExtentM ** 2);

    // Terrain extent = 2× selected length (per spec)
    const radius = Math.max(trackLength, 5000);
    const clampedRadius = Math.min(radius, 50000);

    return {
      originLat: oLat,
      originLon: oLon,
      centerElev: cElev,
      boundsRadiusM: clampedRadius,
    };
  }, [trackPoints]);

  // Fetch terrain
  useEffect(() => {
    if (trackPoints.length === 0) {
      setLoading(false);
      setError("No valid points in selected range");
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchDEMPatch(originLat, originLon, boundsRadiusM, 128)
      .then((data) => {
        if (!cancelled) setTerrainData(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [originLat, originLon, boundsRadiusM, trackPoints.length]);

  const startPoint = trackPoints[0] || null;
  const endPoint = trackPoints[trackPoints.length - 1] || null;

  return (
    <div className="flex flex-col h-full bg-[#0a0f18]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-cyan-400 text-lg">layers</span>
          <div>
            <h2 className="text-white font-bold text-sm">3D Subsurface View</h2>
            <p className="text-[#6b7c9c] text-[10px]">
              {trackPoints.length} traces · Mean depth: {meanDepth.toFixed(0)}m · εr={epsilonR.toFixed(1)}
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

        <div className="flex items-center gap-1">
          <span className="text-[9px] text-[#6b7c9c]">Opacity</span>
          <input
            type="range"
            min="0.3"
            max="1"
            step="0.1"
            value={terrainOpacity}
            onChange={(e) => setTerrainOpacity(Number(e.target.value))}
            className="w-12 h-1 bg-[#232f48] rounded appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-2
              [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-amber-400"
          />
        </div>
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

        {!loading && !error && terrainData && (
          <Canvas
            camera={{ fov: 50, near: 1, far: 1000000 }}
            gl={{ antialias: true }}
          >
            <CameraSetup sceneSize={boundsRadiusM} />
            <OrbitControls enableDamping dampingFactor={0.05} minDistance={boundsRadiusM * 0.1} maxDistance={boundsRadiusM * 10} />

            <ambientLight intensity={0.5} />
            <directionalLight position={[1, 2, 1]} intensity={0.8} />
            <directionalLight position={[-1, 1, -1]} intensity={0.3} />

            {/* Terrain */}
            <TerrainMesh
              data={terrainData}
              verticalExaggeration={verticalExaggeration}
              originLat={originLat}
              originLon={originLon}
              centerElev={centerElev}
              opacity={terrainOpacity}
            />

            {/* Flat subsurface plane */}
            {showSubsurface && (
              <SubsurfacePlane
                centerLat={originLat}
                centerLon={originLon}
                originLat={originLat}
                originLon={originLon}
                centerElev={centerElev}
                meanDepth={meanDepth}
                extentM={boundsRadiusM}
                verticalExaggeration={verticalExaggeration}
                opacity={subsurfaceOpacity}
              />
            )}

            {/* Track line on surface */}
            <TrackLine
              points={trackPoints}
              verticalExaggeration={verticalExaggeration}
              originLat={originLat}
              originLon={originLon}
              centerElev={centerElev}
            />

            {/* Vertical connectors at ends */}
            {showSubsurface && (
              <EndConnectors
                startPoint={startPoint}
                endPoint={endPoint}
                meanDepth={meanDepth}
                verticalExaggeration={verticalExaggeration}
                originLat={originLat}
                originLon={originLon}
                centerElev={centerElev}
              />
            )}

            <gridHelper args={[boundsRadiusM * 4, 20, "#333", "#222"]} position={[0, -boundsRadiusM * 0.3, 0]} />
          </Canvas>
        )}
      </div>
    </div>
  );
}
