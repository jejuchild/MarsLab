import { useState, useRef, useEffect, useCallback } from "react";

type HiResImageViewerProps = {
  productId: string;
  onClose: () => void;
};

const ZOOM_PRESETS = [0.25, 0.5, 1, 2, 4, 8];
const ZOOM_MIN = 0.1;
const ZOOM_MAX = 12;
const SCROLL_FACTOR = 1.06; // gentle scroll zoom

export default function HiResImageViewer({ productId, onClose }: HiResImageViewerProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [isJP2, setIsJP2] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch image — try JP2-backed high-res first, fall back to quickview
  useEffect(() => {
    let revoke: string | null = null;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setIsJP2(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });

    (async () => {
      try {
        // Check if JP2 core data exists
        const existsRes = await fetch(`/api/exists/hirise/${encodeURIComponent(productId)}`);
        const hasJP2 = existsRes.ok && (await existsRes.json()).has_core;

        if (cancelled) return;

        // Use higher max_size for JP2 data, lower for quickview
        const maxSize = hasJP2 ? 8192 : 2048;
        const res = await fetch(`/hirise/overlay/${productId}.png?max_size=${maxSize}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();

        if (cancelled) return;
        revoke = URL.createObjectURL(blob);
        setImgUrl(revoke);
        setIsJP2(hasJP2);
        setLoading(false);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      if (revoke) URL.revokeObjectURL(revoke);
    };
  }, [productId]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "=" || e.key === "+") zoomBy(1.2);
      else if (e.key === "-") zoomBy(1 / 1.2);
      else if (e.key === "0") { setZoom(1); setPan({ x: 0, y: 0 }); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const clampZoom = (z: number) => Math.min(Math.max(z, ZOOM_MIN), ZOOM_MAX);
  const zoomBy = useCallback((factor: number) => {
    setZoom((z) => clampZoom(z * factor));
  }, []);

  // Mouse wheel zoom — cursor-centered
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const steps = -e.deltaY / 100; // normalize across browsers
    const factor = Math.pow(SCROLL_FACTOR, steps);
    setZoom((prevZoom) => {
      const newZoom = clampZoom(prevZoom * factor);
      const scale = newZoom / prevZoom;
      // Adjust pan so zoom centers on cursor position
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) {
        const cx = e.clientX - rect.left - rect.width / 2;
        const cy = e.clientY - rect.top - rect.height / 2;
        setPan((p) => ({
          x: cx - scale * (cx - p.x),
          y: cy - scale * (cy - p.y),
        }));
      }
      return newZoom;
    });
  }, []);

  // Drag to pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragging) return;
    setPan({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y,
    });
  };
  const handleMouseUp = () => setDragging(false);

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90 backdrop-blur-sm">
      {/* Close overlay — click outside header/image to close */}

      {/* Header bar */}
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-black/90 border-b border-white/10 z-10">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-purple-400">hd</span>
          <span className="text-sm font-bold text-white">{productId}</span>
          <span className={`text-xs px-2 py-0.5 rounded ${isJP2 ? "bg-green-500/20 text-green-400" : "bg-yellow-500/20 text-yellow-400"}`}>
            {isJP2 ? "JP2 High-Res" : "Browse Preview"}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {/* Zoom out */}
          <button
            onClick={() => zoomBy(1 / 1.3)}
            className="p-1.5 rounded bg-white/10 hover:bg-white/20 text-white/70 transition-colors"
            title="Zoom Out (-)"
          >
            <span className="material-symbols-outlined text-sm">remove</span>
          </button>

          {/* Zoom presets */}
          {ZOOM_PRESETS.map((preset) => (
            <button
              key={preset}
              onClick={() => { setZoom(preset); setPan({ x: 0, y: 0 }); }}
              className={`px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                Math.abs(zoom - preset) < 0.05
                  ? "bg-purple-500/30 text-purple-300 border border-purple-500/40"
                  : "bg-white/10 hover:bg-white/20 text-white/60"
              }`}
              title={`${preset * 100}%`}
            >
              {preset < 1 ? `${preset * 100}%` : `${preset}x`}
            </button>
          ))}

          {/* Zoom in */}
          <button
            onClick={() => zoomBy(1.3)}
            className="p-1.5 rounded bg-white/10 hover:bg-white/20 text-white/70 transition-colors"
            title="Zoom In (+)"
          >
            <span className="material-symbols-outlined text-sm">add</span>
          </button>

          {/* Current zoom display */}
          <span className="text-xs text-white/50 font-mono w-14 text-center">
            {Math.round(zoom * 100)}%
          </span>

          {/* Fit */}
          <button
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
            className="p-1.5 rounded bg-white/10 hover:bg-white/20 text-white/70 transition-colors"
            title="Fit to screen (0)"
          >
            <span className="material-symbols-outlined text-sm">fit_screen</span>
          </button>

          <div className="w-px h-5 bg-white/20 mx-1" />
          <button
            onClick={onClose}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-red-500/30 hover:bg-red-500/60 text-white border border-red-500/40 transition-colors"
            title="Close (Esc)"
          >
            <span className="material-symbols-outlined text-sm">close</span>
            <span className="text-xs font-medium">Close</span>
          </button>
        </div>
      </div>

      {/* Image area */}
      <div
        ref={containerRef}
        className="w-full h-full cursor-grab active:cursor-grabbing overflow-hidden"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {loading && (
          <div className="flex items-center justify-center h-full gap-3">
            <span className="material-symbols-outlined text-2xl text-purple-400 animate-spin">progress_activity</span>
            <span className="text-white/60 text-sm">Loading high-resolution image...</span>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-full gap-3">
            <span className="material-symbols-outlined text-2xl text-red-400">error</span>
            <span className="text-red-400 text-sm">{error}</span>
          </div>
        )}
        {imgUrl && !loading && (
          <div
            className="w-full h-full flex items-center justify-center"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transition: dragging ? "none" : "transform 0.1s ease-out",
            }}
          >
            <img
              src={imgUrl}
              alt={`HiRISE ${productId}`}
              className="max-w-none select-none"
              draggable={false}
              style={{ imageRendering: zoom > 2 ? "pixelated" : "auto" }}
            />
          </div>
        )}
      </div>

      {/* Bottom info bar */}
      <div className="absolute bottom-0 left-0 right-0 px-4 py-2 bg-gradient-to-t from-black/80 to-transparent z-10">
        <span className="text-[10px] text-white/40">
          Scroll to zoom · Drag to pan · +/- keys · Press 0 to reset · Esc to close
        </span>
      </div>
    </div>
  );
}
