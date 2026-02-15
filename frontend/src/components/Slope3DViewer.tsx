/**
 * Slope3DViewer - 3D terrain visualization component
 *
 * Displays a local DEM patch in an interactive 3D view with:
 * - Orbit controls (rotate, zoom, pan)
 * - Vertical exaggeration control
 * - Patch size selection
 * - Elevation-based coloring
 */

import { useState, useEffect, useMemo, useCallback } from "react";
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
  slope_mean: number;
  slope_max: number;
  radius_m: number;
}

interface Slope3DViewerProps {
  point: TerrainPoint;
  onClose: () => void;
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
// 3D Terrain Mesh Component - Fixed Implementation
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

    // Calculate actual mesh dimensions in world units
    // Use radius_m * 2 as the total width/height
    const meshWidth = radius_m * 2;
    const meshHeight = radius_m * 2;

    // Center elevation for normalization
    const centerElev = (min_elevation_m + max_elevation_m) / 2;

    // Create BufferGeometry manually for precise control
    const geo = new THREE.BufferGeometry();

    // Calculate number of vertices and triangles
    const numVertices = rows * cols;
    const numTriangles = (rows - 1) * (cols - 1) * 2;

    // Create typed arrays
    const positions = new Float32Array(numVertices * 3);
    const colors = new Float32Array(numVertices * 3);
    const indices = new Uint32Array(numTriangles * 3);

    // Step sizes
    const stepX = meshWidth / (cols - 1);
    const stepZ = meshHeight / (rows - 1);

    // Fill vertex positions and colors
    // Elevation data is row-major: row 0 is northernmost
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const vertexIndex = row * cols + col;
        const elevIndex = row * cols + col;

        // X: from -meshWidth/2 to +meshWidth/2 (west to east)
        const x = -meshWidth / 2 + col * stepX;

        // Z: from +meshHeight/2 to -meshHeight/2 (north to south)
        // Row 0 is north (positive Z), row (rows-1) is south (negative Z)
        const z = meshHeight / 2 - row * stepZ;

        // Y: elevation (normalized and exaggerated)
        const elev = elevations[elevIndex];
        const y = (elev - centerElev) * verticalExaggeration;

        // Set position
        positions[vertexIndex * 3] = x;
        positions[vertexIndex * 3 + 1] = y;
        positions[vertexIndex * 3 + 2] = z;

        // Set color
        const [r, g, b] = elevationToColor(elev, min_elevation_m, max_elevation_m);
        colors[vertexIndex * 3] = r;
        colors[vertexIndex * 3 + 1] = g;
        colors[vertexIndex * 3 + 2] = b;
      }
    }

    // Fill indices for triangles
    let indexOffset = 0;
    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        // Vertex indices for this cell
        const topLeft = row * cols + col;
        const topRight = row * cols + col + 1;
        const bottomLeft = (row + 1) * cols + col;
        const bottomRight = (row + 1) * cols + col + 1;

        // Triangle 1: top-left, bottom-left, top-right
        indices[indexOffset++] = topLeft;
        indices[indexOffset++] = bottomLeft;
        indices[indexOffset++] = topRight;

        // Triangle 2: top-right, bottom-left, bottom-right
        indices[indexOffset++] = topRight;
        indices[indexOffset++] = bottomLeft;
        indices[indexOffset++] = bottomRight;
      }
    }

    // Set attributes
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    // Compute normals for lighting
    geo.computeVertexNormals();

    return geo;
  }, [data, verticalExaggeration]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        wireframe={wireframe}
        flatShading={false}
      />
    </mesh>
  );
}

// =============================================================================
// Bounding Box Helper Component
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
      <edgesGeometry
        args={[new THREE.BoxGeometry(meshWidth, elevRange || 100, meshHeight)]}
      />
      <lineBasicMaterial color="#00ff00" linewidth={2} />
    </lineSegments>
  );
}

// =============================================================================
// Center Marker Component
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

  // Scale marker size relative to patch size
  const markerScale = radius_m / 50;

  return (
    <group position={[0, y, 0]}>
      {/* Vertical line */}
      <mesh position={[0, markerScale * 2, 0]}>
        <cylinderGeometry args={[markerScale * 0.1, markerScale * 0.1, markerScale * 4, 8]} />
        <meshStandardMaterial color="#ff4444" />
      </mesh>
      {/* Sphere at top */}
      <mesh position={[0, markerScale * 4.5, 0]}>
        <sphereGeometry args={[markerScale * 0.3, 16, 16]} />
        <meshStandardMaterial color="#ff4444" emissive="#ff0000" emissiveIntensity={0.3} />
      </mesh>
    </group>
  );
}

// =============================================================================
// Camera Setup Component
// =============================================================================

function CameraSetup({ data, verticalExaggeration }: { data: DEMPatchData; verticalExaggeration: number }) {
  const { camera } = useThree();
  const { radius_m } = data;

  useEffect(() => {
    // Position camera to view the entire terrain
    const meshSize = radius_m * 2;

    // Camera distance based on patch size
    const distance = meshSize * 1.5;

    // Position camera at 45-degree angle
    camera.position.set(distance * 0.7, distance * 0.5, distance * 0.7);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, radius_m, verticalExaggeration]);

  return null;
}

// =============================================================================
// Main Component
// =============================================================================

export default function Slope3DViewer({ point, onClose }: Slope3DViewerProps) {
  const [data, setData] = useState<DEMPatchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Controls
  const [patchSizeKm, setPatchSizeKm] = useState(5); // 2, 5, or 10 km
  const [verticalExaggeration, setVerticalExaggeration] = useState(3);
  const [showWireframe, setShowWireframe] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  // Panel width (resizable)
  const [panelWidth, setPanelWidth] = useState(384);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.6);
      setPanelWidth(Math.max(280, Math.min(maxW, startW + delta)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [panelWidth]);

  // Fetch DEM data when point or patch size changes
  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        const radiusM = (patchSizeKm * 1000) / 2; // Convert diameter to radius
        const patchData = await fetchDEMPatch(point.lat, point.lon, radiusM, 128);

        if (!cancelled) {
          // Validate data
          if (!patchData.elevations || patchData.elevations.length === 0) {
            throw new Error("No elevation data received");
          }
          if (patchData.rows < 2 || patchData.cols < 2) {
            throw new Error(`Invalid grid size: ${patchData.rows}x${patchData.cols}`);
          }
          if (patchData.elevations.length !== patchData.rows * patchData.cols) {
            throw new Error(
              `Data mismatch: ${patchData.elevations.length} values for ${patchData.rows}x${patchData.cols} grid`
            );
          }

          console.log("[Slope3D] Loaded patch:", {
            patchSizeKm,
            rows: patchData.rows,
            cols: patchData.cols,
            radiusM: patchData.radius_m,
            spacing_m: patchData.spacing_m,
            elevMin: patchData.min_elevation_m,
            elevMax: patchData.max_elevation_m,
            elevRange: patchData.elevation_range_m,
          });

          setData(patchData);
        }
      } catch (e) {
        if (!cancelled) {
          console.error("[Slope3D] Error:", e);
          setError(e instanceof Error ? e.message : "Failed to load terrain data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadData();

    return () => {
      cancelled = true;
    };
  }, [point.lat, point.lon, patchSizeKm]);

  return (
    <div className="relative flex flex-col h-full bg-[#0a0f18]" style={{ width: panelWidth }}>
      {/* Resize handle (left edge) */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
        onMouseDown={handleResizeStart}
      />
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-purple-400">landscape</span>
          <div>
            <h2 className="text-white font-bold text-sm">3D Terrain Viewer</h2>
            <p className="text-[#6b7c9c] text-[10px]">
              {point.lat.toFixed(4)}°, {point.lon.toFixed(4)}°
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

        {/* Debug Controls */}
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

      {/* Debug Info Overlay */}
      {showDebug && data && (
        <div className="absolute top-32 left-4 z-20 bg-black/80 text-[10px] font-mono text-green-400 p-2 rounded border border-green-500/30">
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
      <div className="flex-1 relative">
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

            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[1, 2, 1]} intensity={0.8} />
            <directionalLight position={[-1, 1, -1]} intensity={0.3} />

            {/* Terrain */}
            <TerrainMesh
              data={data}
              verticalExaggeration={verticalExaggeration}
              wireframe={showWireframe}
            />

            {/* Bounding box */}
            {showBoundingBox && (
              <BoundingBoxHelper data={data} verticalExaggeration={verticalExaggeration} />
            )}

            {/* Center marker */}
            <CenterMarker data={data} verticalExaggeration={verticalExaggeration} />

            {/* Ground reference plane (optional) */}
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
              <p className="text-[9px] uppercase text-[#6b7c9c]">Elevation Range</p>
              <p className="text-white text-sm font-mono">{data.elevation_range_m.toFixed(0)} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Center Elev.</p>
              <p className="text-white text-sm font-mono">{data.center_elevation_m.toFixed(0)} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Mean Slope</p>
              <p className="text-white text-sm font-mono">{data.slope_mean.toFixed(1)}°</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Max Slope</p>
              <p className="text-white text-sm font-mono">{data.slope_max.toFixed(1)}°</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
