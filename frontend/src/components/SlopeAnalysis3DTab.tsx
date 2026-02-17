/**
 * SlopeAnalysis3DTab — 3D terrain visualization tab
 * Lazy-loaded from SlopeAnalysis to keep Three.js out of the main bundle.
 */

import { useState, useEffect, useMemo } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { TerrainPoint } from "./SlopeAnalysis";

// =============================================================================
// Types
// =============================================================================

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
  slope_mean: number;
  slope_max: number;
  radius_m: number;
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
    lon: lon.toString(),
    radius_m: radiusM.toString(),
    grid_size: gridSize.toString(),
  });

  const res = await fetch(`/terrain/dem_patch?${params}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "Failed to fetch DEM patch" }));
    throw new Error(error.error || "Failed to fetch DEM patch");
  }

  return res.json();
}

// =============================================================================
// Color Utilities
// =============================================================================

function elevationToColor(
  elevation: number,
  minElev: number,
  maxElev: number
): [number, number, number] {
  const range = maxElev - minElev;
  const t = range > 0 ? (elevation - minElev) / range : 0.5;

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
// 3D Terrain Mesh
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
        const vi = row * cols + col;
        const x = -meshWidth / 2 + col * stepX;
        const z = meshHeight / 2 - row * stepZ;
        const elev = elevations[vi];
        const y = (elev - centerElev) * verticalExaggeration;

        positions[vi * 3] = x;
        positions[vi * 3 + 1] = y;
        positions[vi * 3 + 2] = z;

        const [r, g, b] = elevationToColor(elev, min_elevation_m, max_elevation_m);
        colors[vi * 3] = r;
        colors[vi * 3 + 1] = g;
        colors[vi * 3 + 2] = b;
      }
    }

    let idx = 0;
    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const tl = row * cols + col;
        const tr = tl + 1;
        const bl = (row + 1) * cols + col;
        const br = bl + 1;
        indices[idx++] = tl;
        indices[idx++] = bl;
        indices[idx++] = tr;
        indices[idx++] = tr;
        indices[idx++] = bl;
        indices[idx++] = br;
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
// Bounding Box Helper
// =============================================================================

function BoundingBoxHelper({
  data,
  verticalExaggeration,
}: {
  data: DEMPatchData;
  verticalExaggeration: number;
}) {
  const { radius_m, min_elevation_m, max_elevation_m } = data;
  const meshWidth = radius_m * 2;
  const meshHeight = radius_m * 2;
  const elevRange = (max_elevation_m - min_elevation_m) * verticalExaggeration;

  return (
    <lineSegments>
      <edgesGeometry args={[new THREE.BoxGeometry(meshWidth, elevRange || 100, meshHeight)]} />
      <lineBasicMaterial color="#00ff00" linewidth={2} />
    </lineSegments>
  );
}

// =============================================================================
// Center Marker
// =============================================================================

function CenterMarker({
  data,
  verticalExaggeration,
}: {
  data: DEMPatchData;
  verticalExaggeration: number;
}) {
  const { center_elevation_m, min_elevation_m, max_elevation_m, radius_m } = data;
  const centerElev = (min_elevation_m + max_elevation_m) / 2;
  const y = (center_elevation_m - centerElev) * verticalExaggeration;
  const s = radius_m / 50;

  return (
    <group position={[0, y, 0]}>
      <mesh position={[0, s * 2, 0]}>
        <cylinderGeometry args={[s * 0.1, s * 0.1, s * 4, 8]} />
        <meshStandardMaterial color="#ff4444" />
      </mesh>
      <mesh position={[0, s * 4.5, 0]}>
        <sphereGeometry args={[s * 0.3, 16, 16]} />
        <meshStandardMaterial color="#ff4444" emissive="#ff0000" emissiveIntensity={0.3} />
      </mesh>
    </group>
  );
}

// =============================================================================
// Camera Setup
// =============================================================================

function CameraSetup({ data, verticalExaggeration }: { data: DEMPatchData; verticalExaggeration: number }) {
  const { camera } = useThree();
  const { radius_m } = data;

  useEffect(() => {
    const d = radius_m * 2 * 1.5;
    camera.position.set(d * 0.7, d * 0.5, d * 0.7);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, radius_m, verticalExaggeration]);

  return null;
}

// =============================================================================
// Main Exported Component
// =============================================================================

export default function SlopeAnalysis3DTab({ point }: { point: TerrainPoint }) {
  const [data, setData] = useState<DEMPatchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [patchSizeKm, setPatchSizeKm] = useState(5);
  const [verticalExaggeration, setVerticalExaggeration] = useState(3);
  const [showWireframe, setShowWireframe] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        const radiusM = (patchSizeKm * 1000) / 2;
        const patchData = await fetchDEMPatch(point.lat, point.lon, radiusM, 128);

        if (!cancelled) {
          if (!patchData.elevations || patchData.elevations.length === 0) {
            throw new Error("No elevation data received");
          }
          if (patchData.rows < 2 || patchData.cols < 2) {
            throw new Error(`Invalid grid size: ${patchData.rows}x${patchData.cols}`);
          }
          if (patchData.elevations.length !== patchData.rows * patchData.cols) {
            throw new Error(`Data mismatch: ${patchData.elevations.length} values for ${patchData.rows}x${patchData.cols} grid`);
          }
          setData(patchData);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load terrain data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadData();
    return () => { cancelled = true; };
  }, [point.lat, point.lon, patchSizeKm]);

  return (
    <>
      {/* Controls */}
      <div className="px-4 py-3 border-b border-[#232f48] space-y-3 bg-[#0a0f18]">
        {/* Patch Size */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase text-[#6b7c9c] w-20">Patch Size</span>
          <div className="flex gap-1">
            {[2, 5, 10, 50].map((size) => (
              <button
                key={size}
                onClick={() => setPatchSizeKm(size)}
                className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${
                  patchSizeKm === size
                    ? "bg-purple-500/20 text-purple-400 border border-purple-500/50"
                    : "bg-[#1a2333] text-[#92a4c9] border border-[#232f48] hover:border-purple-500/30"
                }`}
              >
                {size} km
              </button>
            ))}
          </div>
        </div>

        {/* Vertical Exaggeration */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase text-[#6b7c9c] w-20">V. Exag.</span>
          <input
            type="range"
            min="1"
            max="20"
            value={verticalExaggeration}
            onChange={(e) => setVerticalExaggeration(Number(e.target.value))}
            className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-purple-500
              [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <span className="text-[11px] text-white font-mono w-8 text-right">
            {verticalExaggeration}×
          </span>
        </div>

        {/* Toggle buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowWireframe(!showWireframe)}
            className={`px-2 py-1 text-[9px] font-bold rounded transition-colors ${
              showWireframe
                ? "bg-green-500/20 text-green-400 border border-green-500/50"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48]"
            }`}
          >
            Wireframe
          </button>
          <button
            onClick={() => setShowBoundingBox(!showBoundingBox)}
            className={`px-2 py-1 text-[9px] font-bold rounded transition-colors ${
              showBoundingBox
                ? "bg-green-500/20 text-green-400 border border-green-500/50"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48]"
            }`}
          >
            Bounds
          </button>
          <button
            onClick={() => setShowDebug(!showDebug)}
            className={`px-2 py-1 text-[9px] font-bold rounded transition-colors ${
              showDebug
                ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/50"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48]"
            }`}
          >
            Debug
          </button>
        </div>
      </div>

      {/* Debug Overlay */}
      {showDebug && data && (
        <div className="absolute top-2 left-6 z-20 bg-black/80 text-[10px] font-mono text-green-400 p-2 rounded border border-green-500/30">
          <div>Grid: {data.rows} × {data.cols}</div>
          <div>Radius: {data.radius_m} m</div>
          <div>Spacing: {data.spacing_m.toFixed(1)} m</div>
          <div>Elev: {data.min_elevation_m.toFixed(0)} → {data.max_elevation_m.toFixed(0)} m</div>
          <div>Range: {data.elevation_range_m.toFixed(0)} m</div>
          <div>Vertices: {data.rows * data.cols}</div>
          <div>Triangles: {(data.rows - 1) * (data.cols - 1) * 2}</div>
        </div>
      )}

      {/* 3D Canvas */}
      <div className="flex-1 relative bg-[#0a0f18]">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <div className="text-center">
              <span className="material-symbols-outlined text-4xl text-purple-400 animate-spin">
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

        {data && !loading && (
          <Canvas
            camera={{
              fov: 50,
              near: 1,
              far: 1000000,
              position: [data.radius_m * 2, data.radius_m, data.radius_m * 2],
            }}
            gl={{ antialias: true }}
          >
            <CameraSetup data={data} verticalExaggeration={verticalExaggeration} />
            <OrbitControls
              enableDamping
              dampingFactor={0.05}
              minDistance={data.radius_m * 0.1}
              maxDistance={data.radius_m * 10}
              target={[0, 0, 0]}
            />
            <ambientLight intensity={0.5} />
            <directionalLight position={[1, 2, 1]} intensity={0.8} />
            <directionalLight position={[-1, 1, -1]} intensity={0.3} />
            <TerrainMesh data={data} verticalExaggeration={verticalExaggeration} wireframe={showWireframe} />
            {showBoundingBox && <BoundingBoxHelper data={data} verticalExaggeration={verticalExaggeration} />}
            <CenterMarker data={data} verticalExaggeration={verticalExaggeration} />
            <gridHelper
              args={[data.radius_m * 4, 20, "#333333", "#222222"]}
              position={[0, (data.min_elevation_m - (data.min_elevation_m + data.max_elevation_m) / 2) * verticalExaggeration - 10, 0]}
            />
          </Canvas>
        )}
      </div>

      {/* Stats Footer */}
      {data && (
        <div className="px-4 py-3 border-t border-[#232f48] bg-[#101622]">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Elev Range</p>
              <p className="text-white text-sm font-mono">{data.elevation_range_m.toFixed(0)} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Center Elev</p>
              <p className="text-white text-sm font-mono">{data.center_elevation_m.toFixed(0)} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Mean Slope</p>
              <p className="text-white text-sm font-mono">{data.slope_mean.toFixed(1)}{"\u00b0"}</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Max Slope</p>
              <p className="text-white text-sm font-mono">{data.slope_max.toFixed(1)}{"\u00b0"}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
