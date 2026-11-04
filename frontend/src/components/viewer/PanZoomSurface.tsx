import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

export interface PanZoomSurfaceProps {
  src: string;
  alt: string;
  /** Point to mark, as fractions of the image (0..1 from the top-left). */
  marker?: { fx: number; fy: number; label?: string } | null;
  /** Called with the cursor position in image fractions (null when outside). */
  onHover?: (pos: { fx: number; fy: number } | null) => void;
  /** Zoom range relative to native pixels. */
  maxZoom?: number;
  /** Polylines drawn on the image (image fractions), e.g. radar ground tracks. */
  lines?: { id: string; label: string; points: { fx: number; fy: number }[] }[];
  /** Click on a line, with the clicked position in image fractions. */
  onLineClick?: (id: string, pos: { fx: number; fy: number }) => void;
}

type View = { zoom: number; x: number; y: number };

const WHEEL_FACTOR = 1.0015;

/**
 * Image surface shared by the viewers: opens fitted and centred, wheel zoom
 * around the cursor, drag (mouse/touch/pen) to pan, keyboard + − 0 1 and
 * arrows, and a marker that keeps its size at any zoom.
 */
export default function PanZoomSurface({
  src,
  alt,
  marker,
  onHover,
  maxZoom = 16,
  lines,
  onLineClick,
}: PanZoomSurfaceProps) {
  const [hoverLine, setHoverLine] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [view, setView] = useState<View>({ zoom: 1, x: 0, y: 0 });
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const drag = useRef<{ id: number; x: number; y: number } | null>(null);
  const viewRef = useRef(view);
  viewRef.current = view;

  const fitView = useCallback((): View | null => {
    const box = boxRef.current;
    if (!box || !natural) return null;
    const { clientWidth: cw, clientHeight: ch } = box;
    const zoom = Math.min(cw / natural.w, ch / natural.h) * 0.96;
    return { zoom, x: (cw - natural.w * zoom) / 2, y: (ch - natural.h * zoom) / 2 };
  }, [natural]);

  const minZoom = useCallback(() => Math.min((fitView()?.zoom ?? 1) * 0.5, 1), [fitView]);

  /** Zoom by `factor` keeping the screen point (px, py) fixed (default: centre). */
  const zoomAt = useCallback(
    (factor: number, px?: number, py?: number) => {
      const box = boxRef.current;
      if (!box) return;
      const cx = px ?? box.clientWidth / 2;
      const cy = py ?? box.clientHeight / 2;
      setView((v) => {
        const zoom = Math.min(Math.max(v.zoom * factor, minZoom()), maxZoom);
        const k = zoom / v.zoom;
        return { zoom, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k };
      });
    },
    [maxZoom, minZoom],
  );

  const fit = useCallback(() => {
    const f = fitView();
    if (f) setView(f);
  }, [fitView]);

  const actualSize = useCallback(() => zoomAt(1 / viewRef.current.zoom), [zoomAt]);

  // New image → loading state; fitted once its size is known.
  useEffect(() => {
    setStatus("loading");
    setNatural(null);
  }, [src]);
  useLayoutEffect(() => {
    if (natural) fit();
  }, [natural, fit]);

  // Wheel must be non-passive to stop the page from scrolling.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = box.getBoundingClientRect();
      zoomAt(Math.pow(WHEEL_FACTOR, -e.deltaY), e.clientX - r.left, e.clientY - r.top);
    };
    box.addEventListener("wheel", onWheel, { passive: false });
    return () => box.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  // Keyboard (ignored while typing in a field). Captured on window so the
  // map's own shortcuts underneath don't also react.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const step = 60;
      const keys: Record<string, () => void> = {
        "+": () => zoomAt(1.25),
        "=": () => zoomAt(1.25),
        "-": () => zoomAt(0.8),
        "0": fit,
        "1": actualSize,
        ArrowLeft: () => setView((v) => ({ ...v, x: v.x + step })),
        ArrowRight: () => setView((v) => ({ ...v, x: v.x - step })),
        ArrowUp: () => setView((v) => ({ ...v, y: v.y + step })),
        ArrowDown: () => setView((v) => ({ ...v, y: v.y - step })),
      };
      const fn = keys[e.key];
      if (fn) {
        // The viewer is in front: keep these keys from also driving the map.
        e.preventDefault();
        e.stopImmediatePropagation();
        fn();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [zoomAt, fit, actualSize]);

  // Refit when the viewer is resized (window resize, panel toggle).
  useEffect(() => {
    const box = boxRef.current;
    if (!box || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => fit());
    ro.observe(box);
    return () => ro.disconnect();
  }, [fit]);

  const toImageFraction = (clientX: number, clientY: number) => {
    const box = boxRef.current;
    if (!box || !natural) return null;
    const r = box.getBoundingClientRect();
    const v = viewRef.current;
    const fx = (clientX - r.left - v.x) / (natural.w * v.zoom);
    const fy = (clientY - r.top - v.y) / (natural.h * v.zoom);
    return fx >= 0 && fx <= 1 && fy >= 0 && fy <= 1 ? { fx, fy } : null;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { id: e.pointerId, x: e.clientX - view.x, y: e.clientY - view.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (d && d.id === e.pointerId) setView((v) => ({ ...v, x: e.clientX - d.x, y: e.clientY - d.y }));
    onHover?.(toImageFraction(e.clientX, e.clientY));
  };
  const endDrag = (e: React.PointerEvent) => {
    if (drag.current?.id === e.pointerId) drag.current = null;
  };

  const pct = Math.round(view.zoom * 100);
  const btn =
    "flex h-8 min-w-8 items-center justify-center rounded-md px-1.5 text-slate-200 hover:bg-white/10 disabled:opacity-40";

  return (
    <div className="absolute inset-0 overflow-hidden bg-black">
      <div
        ref={boxRef}
        className="absolute inset-0 cursor-grab touch-none select-none active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => onHover?.(null)}
        onDoubleClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          zoomAt(2, e.clientX - r.left, e.clientY - r.top);
        }}
      >
        <div
          className="absolute left-0 top-0 origin-top-left"
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})` }}
        >
          <img
            src={src}
            alt={alt}
            draggable={false}
            onLoad={(e) => {
              const img = e.currentTarget;
              setNatural({ w: img.naturalWidth, h: img.naturalHeight });
              setStatus("ok");
            }}
            onError={() => setStatus("error")}
            className="block max-w-none"
            style={{ imageRendering: view.zoom > 2 ? "pixelated" : "auto", visibility: status === "ok" ? "visible" : "hidden" }}
          />
          {lines && lines.length > 0 && natural && status === "ok" && (
            <svg
              className="absolute left-0 top-0 overflow-visible"
              width={natural.w}
              height={natural.h}
              viewBox={`0 0 ${natural.w} ${natural.h}`}
              aria-label="Tracks on this image"
            >
              {/* The zoom is a CSS transform outside the SVG, so stroke widths
                  are divided by it to stay constant on screen. */}
              {lines.map((l) => {
                const pts = l.points.map((p) => `${p.fx * natural.w},${p.fy * natural.h}`).join(" ");
                const hot = hoverLine === l.id;
                return (
                  <g key={l.id}>
                    <polyline points={pts} fill="none" stroke="black" strokeOpacity={0.7} strokeWidth={(hot ? 7 : 5) / view.zoom} />
                    <polyline points={pts} fill="none" stroke={hot ? "#fde68a" : "#fbbf24"} strokeWidth={(hot ? 4 : 2.5) / view.zoom} />
                    {onLineClick && (
                      <polyline
                        points={pts}
                        fill="none"
                        stroke="transparent"
                        strokeWidth={16 / view.zoom}
                        style={{ pointerEvents: "stroke", cursor: "pointer" }}
                        role="button"
                        aria-label={l.label}
                        onPointerDown={(e) => e.stopPropagation()}
                        onPointerEnter={() => setHoverLine(l.id)}
                        onPointerLeave={() => setHoverLine(null)}
                        onClick={(e) => {
                          const pos = toImageFraction(e.clientX, e.clientY);
                          if (pos) onLineClick(l.id, pos);
                        }}
                      >
                        <title>{l.label}</title>
                      </polyline>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
          {marker && natural && status === "ok" && (
            <div
              className="pointer-events-none absolute"
              style={{
                left: marker.fx * natural.w,
                top: marker.fy * natural.h,
                transform: `translate(-50%, -50%) scale(${1 / view.zoom})`,
              }}
              aria-label={marker.label ?? "Inspected point"}
            >
              <div className="h-6 w-6 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(0,0,0,0.7)]" />
              <div className="absolute left-1/2 top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-red-500" />
              {marker.label && (
                <div className="absolute left-7 top-0 whitespace-nowrap rounded bg-black/80 px-1.5 py-0.5 font-mono text-[11px] text-white">
                  {marker.label}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {status === "loading" && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center gap-3 text-sm text-slate-300">
          <span className="material-symbols-outlined animate-spin text-2xl text-primary">progress_activity</span>
          Loading image…
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-sm text-slate-400">
          <span className="material-symbols-outlined text-3xl text-red-400">broken_image</span>
          Image not available
        </div>
      )}

      {status === "ok" && (
        <div
          className="absolute bottom-3 right-3 flex items-center gap-0.5 rounded-lg border border-border-dark bg-surface-dark/90 p-1 shadow-lg backdrop-blur-md"
          role="toolbar"
          aria-label="Zoom"
        >
          <button type="button" className={btn} onClick={() => zoomAt(0.8)} aria-label="Zoom out" title="Zoom out (−)">
            <span className="material-symbols-outlined text-[18px]">remove</span>
          </button>
          <span className="w-12 text-center font-mono text-[11px] text-slate-300" aria-live="polite">
            {pct}%
          </span>
          <button type="button" className={btn} onClick={() => zoomAt(1.25)} aria-label="Zoom in" title="Zoom in (+)">
            <span className="material-symbols-outlined text-[18px]">add</span>
          </button>
          <div className="mx-0.5 h-5 w-px bg-border-dark" />
          <button type="button" className={btn} onClick={fit} aria-label="Fit to screen" title="Fit to screen (0)">
            <span className="material-symbols-outlined text-[18px]">fit_screen</span>
          </button>
          <button type="button" className={`${btn} text-[11px] font-mono`} onClick={actualSize} aria-label="Actual pixels" title="Actual pixels (1)">
            1:1
          </button>
        </div>
      )}
    </div>
  );
}
