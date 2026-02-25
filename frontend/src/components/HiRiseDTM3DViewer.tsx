/**
 * HiRiseDTM3DViewer - 3D terrain visualization for HiRISE DTM data
 *
 * Displays a high-resolution (1m) DEM patch from HiRISE DTM in an interactive 3D view:
 * - Orbit controls (rotate, zoom, pan)
 * - Vertical exaggeration control
 * - Patch size selection (smaller sizes for high-res data)
 * - Elevation-based coloring
 */

import { useState, useEffect, useMemo, useCallback, useRef, forwardRef } from "react";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { throttle } from "../utils/dtmHover";

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
  product_id?: string;
}

export interface HiRiseDTMPoint {
  lat: number;
  lon: number;
  productId: string;
}

interface HiRiseDTM3DViewerProps {
  point: HiRiseDTMPoint;
  onClose: () => void;
}

interface FootprintOverlay {
  instrument: string;
  productId: string;
  coords: number[][][]; // [lon, lat][] per ring/segment
  type: string;
  color: string;
}

const FOOTPRINT_COLORS: Record<string, string> = {
  CRISM: "#00FFFF",
  HIRISE: "#FFFF00",
  SHARAD: "#FFA500",
  SHARAD_HIGHRES: "#FFD700",
  CTX: "#FF69B4",
};

const INSTRUMENTS_TO_QUERY = ["CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES", "CTX"];

// =============================================================================
// API Functions
// =============================================================================

async function fetchHiRiseDTMPatch(
  productId: string,
  lat: number,
  lon: number,
  radiusM: number,
  gridSize: number = 128
): Promise<DEMPatchData> {
  const params = new URLSearchParams({
    product_id: productId,
    lat: lat.toString(),
    lon: lon.toString(),
    radius_m: radiusM.toString(),
    grid_size: gridSize.toString(),
  });

  const res = await fetch(`/api/terrain/hirise_dtm_patch?${params}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "Failed to fetch HiRISE DTM patch" }));
    throw new Error(error.error || error.detail || "Failed to fetch HiRISE DTM patch");
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
// 3D Terrain Mesh Component (with forwardRef for raycasting)
// =============================================================================

const TerrainMesh = forwardRef<
  THREE.Mesh,
  {
    data: DEMPatchData;
    verticalExaggeration: number;
    wireframe: boolean;
  }
>(function TerrainMesh({ data, verticalExaggeration, wireframe }, ref) {
  const geometry = useMemo(() => {
    const { elevations, rows, cols, min_elevation_m, max_elevation_m, radius_m } = data;

    // Calculate actual mesh dimensions in world units
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
        const z = meshHeight / 2 - row * stepZ;

        // Y: elevation (normalized and exaggerated)
        const elev = elevations[elevIndex]!;
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

        // Triangle 1
        indices[indexOffset++] = topLeft;
        indices[indexOffset++] = bottomLeft;
        indices[indexOffset++] = topRight;

        // Triangle 2
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
    <mesh ref={ref} geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        wireframe={wireframe}
        flatShading={false}
      />
    </mesh>
  );
});

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
        args={[new THREE.BoxGeometry(meshWidth, elevRange || 10, meshHeight)]}
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

function CameraSetup({ data }: { data: DEMPatchData }) {
  const { camera } = useThree();
  const { radius_m } = data;

  useEffect(() => {
    const meshSize = radius_m * 2;
    const distance = meshSize * 1.5;

    camera.position.set(distance * 0.7, distance * 0.5, distance * 0.7);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, radius_m]);

  return null;
}

// =============================================================================
// Hover Raycast Component - Throttled terrain probing
// =============================================================================

interface HoverInfo {
  lat: number;
  lon: number;
  elevation: number;
  localX: number;
  localZ: number;
}

function HoverRaycast({
  data,
  meshRef,
  verticalExaggeration,
  onHover,
  mode,
}: {
  data: DEMPatchData;
  meshRef: React.RefObject<THREE.Mesh | null>;
  verticalExaggeration: number;
  onHover: (info: HoverInfo | null) => void;
  mode: "hover" | "click";
}) {
  const { camera, raycaster, pointer } = useThree();
  const lastHoverRef = useRef<HoverInfo | null>(null);

  // Throttled raycast (20 Hz for 3D - more aggressive than 2D)
  const throttledRaycast = useMemo(
    () =>
      throttle(() => {
        if (mode !== "hover" || !meshRef.current) return;

        raycaster.setFromCamera(pointer, camera);
        const intersects = raycaster.intersectObject(meshRef.current);

        if (intersects.length > 0) {
          const hit = intersects[0]!;
          const { x, y, z } = hit.point;

          // Convert local coords back to lat/lon
          const { bounds, radius_m, min_elevation_m, max_elevation_m } = data;
          const centerElev = (min_elevation_m + max_elevation_m) / 2;

          // X maps to longitude, Z maps to latitude
          const lonRange = bounds.east - bounds.west;
          const latRange = bounds.north - bounds.south;

          const lonFrac = (x + radius_m) / (radius_m * 2);
          const latFrac = (radius_m - z) / (radius_m * 2);

          const lon = bounds.west + lonFrac * lonRange;
          const lat = bounds.south + latFrac * latRange;

          // Y is elevation (de-exaggerate)
          const elevation = y / verticalExaggeration + centerElev;

          const info: HoverInfo = { lat, lon, elevation, localX: x, localZ: z };
          lastHoverRef.current = info;
          onHover(info);
        } else {
          if (lastHoverRef.current) {
            lastHoverRef.current = null;
            onHover(null);
          }
        }
      }, 50),
    [camera, data, meshRef, mode, onHover, pointer, raycaster, verticalExaggeration]
  );

  useFrame(() => {
    throttledRaycast();
  });

  return null;
}

// =============================================================================
// Hover Marker Component - Small sphere at hover point
// =============================================================================

function HoverMarker({
  position,
  visible,
  radius_m,
}: {
  position: { x: number; y: number; z: number } | null;
  visible: boolean;
  radius_m: number;
}) {
  if (!visible || !position) return null;

  const markerScale = radius_m / 80;

  return (
    <mesh position={[position.x, position.y + markerScale, position.z]}>
      <sphereGeometry args={[markerScale * 0.5, 12, 12]} />
      <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={0.5} />
    </mesh>
  );
}

// =============================================================================
// Footprint Lines Component - Renders overlapping footprints on 3D terrain
// =============================================================================

function FootprintLines({
  footprints,
  data,
  verticalExaggeration,
}: {
  footprints: FootprintOverlay[];
  data: DEMPatchData;
  verticalExaggeration: number;
}) {
  const lines = useMemo(() => {
    const { bounds, radius_m, min_elevation_m, max_elevation_m } = data;
    const centerElev = (min_elevation_m + max_elevation_m) / 2;
    const lonRange = bounds.east - bounds.west;
    const latRange = bounds.north - bounds.south;
    // Slight Y offset above terrain surface
    const yOffset = (max_elevation_m - centerElev + 5) * verticalExaggeration;

    const result: { points: Float32Array; color: string; key: string }[] = [];

    for (const fp of footprints) {
      for (let ri = 0; ri < fp.coords.length; ri++) {
        const ring = fp.coords[ri];
        if (!ring || ring.length < 2) continue;

        const pts: number[] = [];
        for (const coord of ring) {
          if (coord.length < 2) continue;
          const lon = coord[0]!;
          const lat = coord[1]!;
          // Convert lat/lon to mesh coords
          const x = ((lon - bounds.west) / lonRange) * radius_m * 2 - radius_m;
          const z = radius_m - ((lat - bounds.south) / latRange) * radius_m * 2;
          pts.push(x, yOffset, z);
        }

        if (pts.length >= 6) {
          result.push({
            points: new Float32Array(pts),
            color: fp.color,
            key: `${fp.productId}_${ri}`,
          });
        }
      }
    }
    return result;
  }, [footprints, data, verticalExaggeration]);

  return (
    <>
      {lines.map(({ points, color, key }) => (
        <line key={key}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[points, 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial color={color} linewidth={2} transparent opacity={0.9} />
        </line>
      ))}
    </>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export default function HiRiseDTM3DViewer({ point, onClose }: HiRiseDTM3DViewerProps) {
  const [data, setData] = useState<DEMPatchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Controls - patch sizes for HiRISE DTM
  const [patchSizeM, setPatchSizeM] = useState(1000); // 1km, 5km, 10km, or full
  const [verticalExaggeration, setVerticalExaggeration] = useState(2);
  const [showWireframe, setShowWireframe] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  // Footprint overlay state
  const [showFootprints, setShowFootprints] = useState(false);
  const [footprintData, setFootprintData] = useState<FootprintOverlay[]>([]);
  const [footprintsLoading, setFootprintsLoading] = useState(false);

  // Hover state (for 3D terrain probing)
  const [hoverMode, setHoverMode] = useState<"hover" | "click">("hover");
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const meshRef = useRef<THREE.Mesh>(null);

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
        const radiusM = patchSizeM / 2;
        // Use higher grid resolution for larger patches
        const gridSize = patchSizeM >= 10000 ? 256 : patchSizeM >= 5000 ? 192 : 128;
        const patchData = await fetchHiRiseDTMPatch(
          point.productId,
          point.lat,
          point.lon,
          radiusM,
          gridSize
        );

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



          setData(patchData);
        }
      } catch (e) {
        if (!cancelled) {
          console.error("[HiRiseDTM3D] Error:", e);
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
  }, [point.lat, point.lon, point.productId, patchSizeM]);

  // Fetch overlapping footprints when toggle is on
  useEffect(() => {
    if (!showFootprints || !data) {
      setFootprintData([]);
      return;
    }

    let cancelled = false;
    async function fetchFootprints() {
      setFootprintsLoading(true);
      try {
        const { west, south, east, north } = data!.bounds;
        const bbox = `${west},${south},${east},${north}`;
        const allOverlays: FootprintOverlay[] = [];

        const fetches = INSTRUMENTS_TO_QUERY.map(async (inst) => {
          try {
            const res = await fetch(
              `/api/footprints?instrument=${inst}&bbox=${bbox}&lod=poly`
            );
            if (!res.ok) return [];
            const geojson = await res.json();
            if (!geojson.features) return [];

            const overlays: FootprintOverlay[] = [];
            for (const feature of geojson.features) {
              const geomType = feature.geometry?.type;
              const productId = feature.properties?.product_id || feature.properties?.id || "unknown";
              let coords: number[][][] = [];

              if (geomType === "Polygon") {
                coords = feature.geometry.coordinates; // [[lon,lat], ...][]
              } else if (geomType === "MultiPolygon") {
                // Flatten multi-polygon rings
                for (const poly of feature.geometry.coordinates) {
                  coords.push(...poly);
                }
              } else if (geomType === "LineString") {
                coords = [feature.geometry.coordinates]; // wrap single line
              } else if (geomType === "MultiLineString") {
                coords = feature.geometry.coordinates;
              } else {
                continue; // skip Points etc.
              }

              overlays.push({
                instrument: inst,
                productId,
                coords,
                type: geomType,
                color: FOOTPRINT_COLORS[inst] || "#FFFFFF",
              });
            }
            return overlays;
          } catch {
            return [];
          }
        });

        const results = await Promise.all(fetches);
        for (const r of results) allOverlays.push(...r);

        if (!cancelled) {
          setFootprintData(allOverlays);

        }
      } catch (e) {
        console.error("[HiRiseDTM3D] Footprint fetch error:", e);
      } finally {
        if (!cancelled) setFootprintsLoading(false);
      }
    }

    fetchFootprints();
    return () => { cancelled = true; };
  }, [showFootprints, data]);

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
          <span className="material-symbols-outlined text-amber-600">terrain</span>
          <div>
            <h2 className="text-white font-bold text-sm">HiRISE DTM 3D View</h2>
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

      {/* Product ID info */}
      <div className="px-4 py-2 border-b border-[#232f48] bg-[#101622]">
        <span className="text-[9px] uppercase text-[#6b7c9c]">Product:</span>
        <span className="text-[10px] font-mono text-amber-600 ml-2">{point.productId}</span>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 border-b border-[#232f48] space-y-3">
        {/* Patch Size */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase text-[#6b7c9c] w-20">Patch Size</span>
          <div className="flex gap-1">
            {([
              { value: 1000, label: "1 km" },
              { value: 5000, label: "5 km" },
              { value: 10000, label: "10 km" },
              { value: 100000, label: "Full" },
            ] as const).map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setPatchSizeM(value)}
                className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${
                  patchSizeM === value
                    ? "bg-amber-600/20 text-amber-600 border border-amber-600/50"
                    : "bg-[#1a2333] text-[#92a4c9] border border-[#232f48] hover:border-amber-600/30"
                }`}
              >
                {label}
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
              [&::-webkit-slider-thumb]:bg-amber-600
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
          <button
            onClick={() => setShowFootprints(!showFootprints)}
            className={`px-2 py-1 text-[9px] font-bold rounded transition-colors ${
              showFootprints
                ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/50"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48]"
            }`}
          >
            Footprints
            {footprintsLoading && (
              <span className="ml-1 inline-block animate-spin text-[8px]">&#9696;</span>
            )}
            {!footprintsLoading && showFootprints && footprintData.length > 0 && (
              <span className="ml-1 text-[8px] opacity-70">({footprintData.length})</span>
            )}
          </button>
        </div>
      </div>

      {/* Debug Info Overlay */}
      {showDebug && data && (
        <div className="absolute top-40 left-4 z-20 bg-black/80 text-[10px] font-mono text-green-400 p-2 rounded border border-green-500/30">
          <div>Grid: {data.rows} × {data.cols}</div>
          <div>Radius: {data.radius_m} m</div>
          <div>Spacing: {data.spacing_m.toFixed(2)} m</div>
          <div>Elev: {data.min_elevation_m.toFixed(1)} → {data.max_elevation_m.toFixed(1)} m</div>
          <div>Range: {data.elevation_range_m.toFixed(1)} m</div>
          <div>Vertices: {data.rows * data.cols}</div>
          <div>Triangles: {(data.rows - 1) * (data.cols - 1) * 2}</div>
        </div>
      )}

      {/* 3D Canvas */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <div className="text-center">
              <span className="material-symbols-outlined text-4xl text-amber-600 animate-spin">
                progress_activity
              </span>
              <p className="text-[#92a4c9] text-sm mt-2">Loading HiRISE DTM...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0a0f18]/80 z-10">
            <div className="text-center max-w-xs">
              <span className="material-symbols-outlined text-4xl text-red-400">error</span>
              <p className="text-red-400 text-sm mt-2">{error}</p>
            </div>
          </div>
        )}

        {data && !loading && (
          <Canvas
            camera={{
              fov: 50,
              near: 0.1,
              far: 100000,
              position: [data.radius_m * 2, data.radius_m, data.radius_m * 2],
            }}
            gl={{ antialias: true }}
          >
            <CameraSetup data={data} />

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
              ref={meshRef}
              data={data}
              verticalExaggeration={verticalExaggeration}
              wireframe={showWireframe}
            />

            {/* Hover Raycast */}
            <HoverRaycast
              data={data}
              meshRef={meshRef}
              verticalExaggeration={verticalExaggeration}
              onHover={setHoverInfo}
              mode={hoverMode}
            />

            {/* Hover Marker */}
            {hoverInfo && (
              <HoverMarker
                position={{
                  x: hoverInfo.localX,
                  y: (hoverInfo.elevation - (data.min_elevation_m + data.max_elevation_m) / 2) * verticalExaggeration,
                  z: hoverInfo.localZ,
                }}
                visible={true}
                radius_m={data.radius_m}
              />
            )}

            {/* Bounding box */}
            {showBoundingBox && (
              <BoundingBoxHelper data={data} verticalExaggeration={verticalExaggeration} />
            )}

            {/* Center marker */}
            <CenterMarker data={data} verticalExaggeration={verticalExaggeration} />

            {/* Footprint overlays */}
            {showFootprints && footprintData.length > 0 && (
              <FootprintLines
                footprints={footprintData}
                data={data}
                verticalExaggeration={verticalExaggeration}
              />
            )}

            {/* Ground reference plane */}
            <gridHelper
              args={[data.radius_m * 4, 20, "#333333", "#222222"]}
              position={[0, (data.min_elevation_m - (data.min_elevation_m + data.max_elevation_m) / 2) * verticalExaggeration - 5, 0]}
            />
          </Canvas>
        )}

        {/* Footprint Legend Overlay */}
        {showFootprints && footprintData.length > 0 && (
          <div className="absolute top-3 right-3 z-20 bg-[#0a0f18]/85 backdrop-blur-sm rounded border border-[#232f48] px-2 py-1.5">
            <p className="text-[8px] uppercase text-[#6b7c9c] mb-1">Footprints</p>
            {(() => {
              const instruments = [...new Set(footprintData.map((f) => f.instrument))];
              return instruments.map((inst) => {
                const count = footprintData.filter((f) => f.instrument === inst).length;
                return (
                  <div key={inst} className="flex items-center gap-1.5 py-0.5">
                    <span
                      className="inline-block w-3 h-[2px] rounded"
                      style={{ backgroundColor: FOOTPRINT_COLORS[inst] || "#fff" }}
                    />
                    <span className="text-[9px] text-white">{inst}</span>
                    <span className="text-[8px] text-[#6b7c9c]">({count})</span>
                  </div>
                );
              });
            })()}
          </div>
        )}

        {/* Hover Info Overlay */}
        {data && !loading && (
          <div className="absolute bottom-3 left-3 right-3 z-20 flex items-center justify-between">
            {/* Hover readout */}
            <div className="flex items-center gap-3 px-3 py-1.5 rounded bg-[#0a0f18]/80 border border-amber-600/20 backdrop-blur-sm">
              <span className="material-symbols-outlined text-amber-600 text-sm">my_location</span>
              {hoverInfo ? (
                <>
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] text-[#6b7c9c] uppercase">Lat</span>
                    <span className="text-[10px] font-mono text-white">{hoverInfo.lat.toFixed(5)}°</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] text-[#6b7c9c] uppercase">Lon</span>
                    <span className="text-[10px] font-mono text-white">{hoverInfo.lon.toFixed(5)}°</span>
                  </div>
                  <div className="w-px h-3 bg-[#232f48]" />
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] text-[#6b7c9c] uppercase">Elev</span>
                    <span className="text-[10px] font-mono text-amber-400 font-bold">
                      {hoverInfo.elevation.toFixed(1)} m
                    </span>
                  </div>
                </>
              ) : (
                <span className="text-[10px] text-[#6b7c9c]">Hover over terrain</span>
              )}
            </div>

            {/* Mode toggle */}
            <button
              onClick={() => setHoverMode(hoverMode === "hover" ? "click" : "hover")}
              className={`px-2 py-1 text-[8px] font-bold uppercase rounded transition-colors ${
                hoverMode === "hover"
                  ? "bg-amber-600/20 text-amber-400 border border-amber-600/30"
                  : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
              }`}
            >
              {hoverMode === "hover" ? "Hover" : "Click"}
            </button>
          </div>
        )}
      </div>

      {/* Stats Footer */}
      {data && (
        <div className="px-4 py-3 border-t border-[#232f48] bg-[#101622]">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Resolution</p>
              <p className="text-white text-sm font-mono">{data.spacing_m.toFixed(1)} m</p>
            </div>
            <div>
              <p className="text-[9px] uppercase text-[#6b7c9c]">Elev Range</p>
              <p className="text-white text-sm font-mono">{data.elevation_range_m.toFixed(1)} m</p>
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
