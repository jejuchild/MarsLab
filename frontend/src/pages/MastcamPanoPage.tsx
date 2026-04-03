// src/pages/MastcamPanoPage.tsx
import { useState, useEffect, useRef, useCallback, memo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

// ── Types ──────────────────────────────────────────────
type Scene = {
  id: string;
  title: string;
  has_thumb: boolean;
  has_preview: boolean;
  has_equirectangular: boolean;
  has_webview: boolean;
  equirect_size_mb: number | null;
  lon: number | null;
  lat: number | null;
  pano_id: number | null;
};

// ── Panorama Sphere (Three.js) ─────────────────────────
function PanoSphere({ url }: { url: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const loader = new THREE.TextureLoader();
    const texture = loader.load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        if (meshRef.current) {
          const mat = meshRef.current.material as THREE.MeshBasicMaterial;
          if (mat.map) mat.map.dispose();
          mat.map = tex;
          mat.needsUpdate = true;
        }
        setLoading(false);
      },
      undefined,
      () => setLoading(false)
    );

    return () => {
      texture.dispose();
      if (meshRef.current) {
        const mat = meshRef.current.material as THREE.MeshBasicMaterial;
        if (mat.map) {
          mat.map.dispose();
          mat.map = null;
          mat.needsUpdate = true;
        }
      }
    };
  }, [url]);

  return (
    <>
      <mesh ref={meshRef} scale={[-1, 1, 1]}>
        <sphereGeometry args={[500, 64, 32]} />
        <meshBasicMaterial side={THREE.BackSide} />
      </mesh>
      {loading && <LoadingSpinner />}
    </>
  );
}

function LoadingSpinner() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_state, delta) => {
    if (meshRef.current) meshRef.current.rotation.z -= delta * 2;
  });
  return (
    <mesh ref={meshRef} position={[0, 0, -10]}>
      <torusGeometry args={[1, 0.15, 8, 32]} />
      <meshBasicMaterial color="#f59e0b" wireframe />
    </mesh>
  );
}

// ── Camera Drag Controls ───────────────────────────────
function DragControls() {
  const { camera, gl } = useThree();
  const isDragging = useRef(false);
  const prevMouse = useRef({ x: 0, y: 0 });
  const lon = useRef(0);
  const lat = useRef(0);
  const fov = useRef(75);

  useEffect(() => {
    const el = gl.domElement;
    const onDown = (e: PointerEvent) => {
      isDragging.current = true;
      prevMouse.current = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!isDragging.current) return;
      lon.current -= (e.clientX - prevMouse.current.x) * 0.2;
      lat.current += (e.clientY - prevMouse.current.y) * 0.2;
      lat.current = Math.max(-85, Math.min(85, lat.current));
      prevMouse.current = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e: PointerEvent) => {
      isDragging.current = false;
      el.releasePointerCapture(e.pointerId);
    };
    const onWheel = (e: WheelEvent) => {
      fov.current = Math.max(20, Math.min(120, fov.current + e.deltaY * 0.03));
      (camera as THREE.PerspectiveCamera).fov = fov.current;
      camera.updateProjectionMatrix();
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("wheel", onWheel, { passive: true });
    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("wheel", onWheel);
    };
  }, [camera, gl]);

  useFrame(() => {
    const phi = THREE.MathUtils.degToRad(90 - lat.current);
    const theta = THREE.MathUtils.degToRad(lon.current);
    camera.lookAt(
      500 * Math.sin(phi) * Math.cos(theta),
      500 * Math.cos(phi),
      500 * Math.sin(phi) * Math.sin(theta)
    );
  });

  return null;
}

// ── Scene Card ─────────────────────────────────────────
const SceneCard = memo(function SceneCard({
  scene,
  onClick,
  isSelected,
}: {
  scene: Scene;
  onClick: () => void;
  isSelected: boolean;
}) {
  const has360 = scene.has_equirectangular || scene.has_webview;
  return (
    <button
      onClick={onClick}
      className={`group relative w-full rounded-lg overflow-hidden border transition-all text-left ${
        isSelected
          ? "border-amber-500 ring-1 ring-amber-500/40"
          : "border-[#2d3a54] hover:border-[#4a5a7a]"
      }`}
    >
      <div className="aspect-[3/1] bg-[#0d1420] overflow-hidden">
        <img
          src={`/api/mastcam/preview/${scene.id}`}
          alt={scene.title}
          loading="lazy"
          className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
        />
      </div>
      <div className="p-2">
        <p className="text-xs text-[#c8d6e5] truncate font-medium">
          {scene.title}
        </p>
        <div className="flex items-center gap-1.5 mt-1 flex-wrap">
          {has360 ? (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-900/50 text-emerald-400">
              360&deg;
            </span>
          ) : (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a2233] text-[#6b7c9c]">
              Preview
            </span>
          )}
          {scene.lon != null && (
            <span className="text-[10px] text-[#4a5a7a]">
              {scene.lon.toFixed(3)}&deg;E {scene.lat!.toFixed(3)}&deg;N
            </span>
          )}
        </div>
      </div>
    </button>
  );
});

// ── Main Page ──────────────────────────────────────────
export default function MastcamPanoPage() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Scene | null>(null);
  const [viewerMode, setViewerMode] = useState<"preview" | "panorama">("preview");
  const [filter, setFilter] = useState("");

  useEffect(() => {
    fetch("/api/mastcam/scenes")
      .then((r) => r.json())
      .then((data: Scene[]) => {
        setScenes(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filtered = scenes.filter(
    (s) =>
      (s.has_equirectangular || s.has_webview) &&
      (s.title.toLowerCase().includes(filter.toLowerCase()) ||
       s.id.toLowerCase().includes(filter.toLowerCase()))
  );

  const handleSelect = useCallback((scene: Scene) => {
    setSelected(scene);
    setViewerMode("panorama");
  }, []);

  // Use webview (4096px, ~2MB) for 3D viewer instead of full-res (100MB+)
  const panoUrl = selected
    ? `/api/mastcam/webview/${selected.id}`
    : null;

  const previewUrl = selected
    ? `/api/mastcam/preview/${selected.id}`
    : null;

  const has360 = selected
    ? selected.has_equirectangular || selected.has_webview
    : false;

  return (
    <div className="h-screen flex flex-col bg-[#0a0f18] text-[#e2e8f0]">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-[#1e293b] bg-[#0d1420]/80 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <a
            href="/"
            className="text-[#6b7c9c] hover:text-[#c8d6e5] transition-colors text-sm"
          >
            &larr; MarsLab
          </a>
          <h1 className="text-base font-semibold tracking-tight">
            Mastcam-Z 360&deg; Panoramas
          </h1>
          <span className="text-xs text-[#4a5a7a]">
            Perseverance &middot; Jezero Crater
          </span>
        </div>
        <span className="text-xs text-[#4a5a7a]">
          {scenes.length} scenes
        </span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 shrink-0 border-r border-[#1e293b] flex flex-col bg-[#0d1420]">
          <div className="p-2 border-b border-[#1e293b]">
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Search panoramas..."
              className="w-full px-3 py-1.5 rounded bg-[#1a2233] border border-[#2d3a54] text-sm text-[#c8d6e5] placeholder-[#4a5a7a] outline-none focus:border-amber-600"
            />
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {loading ? (
              <p className="text-xs text-[#4a5a7a] text-center py-8">
                Loading scenes...
              </p>
            ) : filtered.length === 0 ? (
              <p className="text-xs text-[#4a5a7a] text-center py-8">
                No matches
              </p>
            ) : (
              filtered.map((s) => (
                <SceneCard
                  key={s.id}
                  scene={s}
                  onClick={() => handleSelect(s)}
                  isSelected={selected?.id === s.id}
                />
              ))
            )}
          </div>
        </aside>

        {/* Viewer */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {!selected ? (
            <div className="flex-1 flex items-center justify-center text-[#4a5a7a]">
              <div className="text-center">
                <p className="text-lg">Select a panorama</p>
                <p className="text-xs mt-1">Click a scene for 360&deg; view</p>
              </div>
            </div>
          ) : (
            <>
              {/* Toolbar */}
              <div className="flex items-center justify-between px-4 py-1.5 border-b border-[#1e293b] bg-[#0d1420]/60 shrink-0">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="text-sm font-medium truncate">
                    {selected.title}
                  </span>
                  {selected.lon != null && (
                    <span className="text-[11px] text-[#6b7c9c] shrink-0">
                      {selected.lon.toFixed(4)}&deg;E, {selected.lat!.toFixed(4)}&deg;N
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {has360 && (
                    <>
                      <button
                        onClick={() => setViewerMode("panorama")}
                        className={`px-2.5 py-1 rounded text-xs transition-colors ${
                          viewerMode === "panorama"
                            ? "bg-amber-600 text-white"
                            : "bg-[#1a2233] text-[#6b7c9c] hover:text-[#c8d6e5]"
                        }`}
                      >
                        360&deg;
                      </button>
                      <button
                        onClick={() => setViewerMode("preview")}
                        className={`px-2.5 py-1 rounded text-xs transition-colors ${
                          viewerMode === "preview"
                            ? "bg-amber-600 text-white"
                            : "bg-[#1a2233] text-[#6b7c9c] hover:text-[#c8d6e5]"
                        }`}
                      >
                        Flat
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 relative">
                {viewerMode === "panorama" && has360 ? (
                  <Canvas
                    key={selected.id}
                    camera={{
                      fov: 75,
                      near: 0.1,
                      far: 1000,
                      position: [0, 0, 0.1],
                    }}
                    style={{ width: "100%", height: "100%" }}
                    gl={{ antialias: false, powerPreference: "high-performance" }}
                  >
                    <PanoSphere url={panoUrl!} />
                    <DragControls />
                  </Canvas>
                ) : (
                  <div className="w-full h-full overflow-auto flex items-center justify-center bg-black">
                    <img
                      src={previewUrl!}
                      alt={selected.title}
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>
                )}

                {viewerMode === "panorama" && (
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded bg-black/60 text-[10px] text-[#8899aa] pointer-events-none">
                    Drag to look around &middot; Scroll to zoom
                  </div>
                )}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
