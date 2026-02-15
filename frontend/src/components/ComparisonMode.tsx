import { useCallback, useEffect, useRef, useState, memo } from "react";

/* ==================================================
 * Types
 * ==================================================*/
export interface ComparisonProduct {
  productId: string;
  instrument: string;
  overlayUrl: string;
  label: string;
  bounds: { west: number; south: number; east: number; north: number };
}

interface ComparisonModeProps {
  isActive: boolean;
  onClose: () => void;
  products: ComparisonProduct[];
  onRemoveProduct: (productId: string) => void;
}

type LayoutMode = "2-panel" | "4-panel";

/** Internal transform state (shared across all panes). */
type TransformState = {
  scale: number;
  translateX: number;
  translateY: number;
};

/* ==================================================
 * Constants
 * ==================================================*/
const MIN_SCALE = 0.25;
const MAX_SCALE = 12;
const ZOOM_SENSITIVITY = 0.002;

const INSTRUMENT_COLORS: Record<string, string> = {
  CRISM: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  HIRISE: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  SHARAD: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  SHARAD_HIGHRES: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  CTX: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  HIRISE_DTM: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  CRISM_TRR3: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  CUSTOM: "bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/30",
};

/* ==================================================
 * Pane Component (single viewer pane)
 * ==================================================*/
interface PaneProps {
  product: ComparisonProduct | null;
  paneIndex: number;
  transformRef: React.RefObject<TransformState>;
  onStartDrag: (e: React.MouseEvent | React.TouchEvent) => void;
  onRemove: (productId: string) => void;
  isDragging: boolean;
}

/** Single comparison pane -- renders image with synchronized transform. */
const Pane = memo(function Pane({
  product,
  paneIndex,
  transformRef,
  onStartDrag,
  onRemove,
  isDragging,
}: PaneProps) {
  const paneRef = useRef<HTMLDivElement>(null);
  const imgContainerRef = useRef<HTMLDivElement>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgError, setImgError] = useState(false);

  // Reset image loading states when the product (overlay URL) changes
  const prevUrlRef = useRef(product?.overlayUrl);
  useEffect(() => {
    if (product?.overlayUrl !== prevUrlRef.current) {
      prevUrlRef.current = product?.overlayUrl;
      setImgLoaded(false);
      setImgError(false);
    }
  }, [product?.overlayUrl]);

  // Apply the shared transform to this pane's image container.
  // Listens for a custom event dispatched by the parent whenever the transform changes.
  useEffect(() => {
    const container = imgContainerRef.current;
    if (!container) return;

    const applyTransform = () => {
      const t = transformRef.current;
      if (t) {
        container.style.transform = `translate(${t.translateX}px, ${t.translateY}px) scale(${t.scale})`;
      }
    };

    // Initial apply
    applyTransform();

    const handler = () => applyTransform();
    window.addEventListener("comparison-transform-update", handler);
    return () => window.removeEventListener("comparison-transform-update", handler);
  }, [transformRef]);

  // Attach a non-passive wheel listener to prevent default page scroll.
  // React's synthetic onWheel is passive, so e.preventDefault() would log a warning.
  useEffect(() => {
    const pane = paneRef.current;
    if (!pane) return;

    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const delta = -e.deltaY * ZOOM_SENSITIVITY;
      const t = transformRef.current;
      if (!t) return;

      const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, t.scale * (1 + delta)));

      // Zoom toward cursor position relative to pane center
      const rect = pane.getBoundingClientRect();
      const cursorX = e.clientX - rect.left - rect.width / 2;
      const cursorY = e.clientY - rect.top - rect.height / 2;

      const scaleFactor = newScale / t.scale;
      transformRef.current = {
        scale: newScale,
        translateX: cursorX - scaleFactor * (cursorX - t.translateX),
        translateY: cursorY - scaleFactor * (cursorY - t.translateY),
      };
      window.dispatchEvent(new CustomEvent("comparison-transform-update"));
    };

    pane.addEventListener("wheel", handler, { passive: false });
    return () => pane.removeEventListener("wheel", handler);
  }, [transformRef]);

  const handleImgLoad = useCallback(() => {
    setImgLoaded(true);
    setImgError(false);
  }, []);

  const handleImgError = useCallback(() => {
    setImgError(true);
    setImgLoaded(false);
  }, []);

  // Empty pane
  if (!product) {
    return (
      <div className="relative flex items-center justify-center border border-dashed border-[#1e293b] bg-[#080c14] rounded-lg overflow-hidden">
        <div className="flex flex-col items-center gap-2 text-slate-600">
          <span className="material-symbols-outlined text-3xl">add_photo_alternate</span>
          <span className="text-xs font-medium">Slot {paneIndex + 1} -- Empty</span>
          <span className="text-[10px] text-slate-700">Add a product to compare</span>
        </div>
      </div>
    );
  }

  const instrumentColor =
    INSTRUMENT_COLORS[product.instrument] || "bg-slate-500/20 text-slate-400 border-slate-500/30";

  return (
    <div
      ref={paneRef}
      className="relative border border-[#1e293b] bg-[#080c14] rounded-lg overflow-hidden select-none"
      onMouseDown={onStartDrag}
      onTouchStart={onStartDrag}
      style={{ cursor: isDragging ? "grabbing" : "grab" }}
    >
      {/* Image container with shared transform */}
      <div
        ref={imgContainerRef}
        className="absolute inset-0 flex items-center justify-center"
        style={{
          transformOrigin: "center center",
          willChange: "transform",
        }}
      >
        {!imgError ? (
          <img
            src={product.overlayUrl}
            alt={product.label}
            className="max-w-none"
            style={{
              imageRendering: "auto",
              pointerEvents: "none",
              userSelect: "none",
            }}
            onLoad={handleImgLoad}
            onError={handleImgError}
            draggable={false}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-red-400/60">
            <span className="material-symbols-outlined text-3xl">broken_image</span>
            <span className="text-xs">Failed to load image</span>
          </div>
        )}
      </div>

      {/* Loading indicator */}
      {!imgLoaded && !imgError && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#080c14]">
          <div className="flex flex-col items-center gap-2">
            <div className="w-6 h-6 border-2 border-sky-500/30 border-t-sky-500 rounded-full animate-spin" />
            <span className="text-[10px] text-slate-500">Loading overlay...</span>
          </div>
        </div>
      )}

      {/* Product label (top-left) */}
      <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#0d1520]/90 backdrop-blur-sm border border-[#1e293b]">
        <span
          className={`px-1.5 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider border ${instrumentColor}`}
        >
          {product.instrument}
        </span>
        <span className="text-[10px] font-mono text-slate-300 max-w-[150px] truncate">
          {product.label}
        </span>
      </div>

      {/* Remove button (top-right) */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRemove(product.productId);
        }}
        className="absolute top-2 right-2 z-10 w-6 h-6 flex items-center justify-center rounded-md bg-[#0d1520]/90 backdrop-blur-sm border border-[#1e293b] text-slate-500 hover:text-red-400 hover:border-red-500/30 transition-colors"
        title="Remove from comparison"
      >
        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
          close
        </span>
      </button>

      {/* Bounds info (bottom-left, subtle) */}
      <div className="absolute bottom-2 left-2 z-10 px-1.5 py-0.5 rounded bg-[#0d1520]/70 text-[8px] font-mono text-slate-600">
        {product.bounds.south.toFixed(2)}N {product.bounds.west.toFixed(2)}E
      </div>
    </div>
  );
});

/* ==================================================
 * ComparisonMode Component (main)
 * ==================================================*/
function ComparisonModeInner({
  isActive,
  onClose,
  products,
  onRemoveProduct,
}: ComparisonModeProps) {
  const [layout, setLayout] = useState<LayoutMode>("2-panel");

  // Zoom display state -- kept in sync with transformRef for UI rendering.
  // The transform ref is the source of truth (mutated during drag/zoom for performance),
  // and this state is updated in a batched manner for display only.
  const [zoomPercent, setZoomPercent] = useState(100);

  // Shared transform state (mutable ref to avoid re-renders during interaction)
  const transformRef = useRef<TransformState>({
    scale: 1,
    translateX: 0,
    translateY: 0,
  });

  // Track dragging
  const isDraggingRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  // Broadcast transform update to all panes and update zoom display
  const broadcastTransform = useCallback(() => {
    window.dispatchEvent(new CustomEvent("comparison-transform-update"));
    setZoomPercent(Math.round(transformRef.current.scale * 100));
  }, []);

  // Reset transform when entering comparison mode
  useEffect(() => {
    if (isActive) {
      transformRef.current = { scale: 1, translateX: 0, translateY: 0 };
      setZoomPercent(100);
      broadcastTransform();
    }
  }, [isActive, broadcastTransform]);

  // Escape key handler
  useEffect(() => {
    if (!isActive) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isActive, onClose]);

  // Drag start
  const handleStartDrag = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if ("touches" in e) {
        const touch = e.touches[0];
        dragStartRef.current = {
          x: touch.clientX,
          y: touch.clientY,
          tx: transformRef.current.translateX,
          ty: transformRef.current.translateY,
        };
      } else {
        // Only handle left mouse button
        if ((e as React.MouseEvent).button !== 0) return;
        dragStartRef.current = {
          x: (e as React.MouseEvent).clientX,
          y: (e as React.MouseEvent).clientY,
          tx: transformRef.current.translateX,
          ty: transformRef.current.translateY,
        };
      }
      isDraggingRef.current = true;
      setIsDragging(true);
    },
    []
  );

  // Global mouse/touch move and up listeners for drag
  useEffect(() => {
    if (!isActive) return;

    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (!isDraggingRef.current || !dragStartRef.current) return;

      let clientX: number, clientY: number;
      if ("touches" in e) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      } else {
        clientX = e.clientX;
        clientY = e.clientY;
      }

      const dx = clientX - dragStartRef.current.x;
      const dy = clientY - dragStartRef.current.y;

      transformRef.current = {
        ...transformRef.current,
        translateX: dragStartRef.current.tx + dx,
        translateY: dragStartRef.current.ty + dy,
      };
      broadcastTransform();
    };

    const handleEnd = () => {
      isDraggingRef.current = false;
      dragStartRef.current = null;
      setIsDragging(false);
    };

    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleEnd);
    document.addEventListener("touchmove", handleMove, { passive: true });
    document.addEventListener("touchend", handleEnd);

    return () => {
      document.removeEventListener("mousemove", handleMove);
      document.removeEventListener("mouseup", handleEnd);
      document.removeEventListener("touchmove", handleMove);
      document.removeEventListener("touchend", handleEnd);
    };
  }, [isActive, broadcastTransform]);

  // Reset zoom handler
  const handleResetZoom = useCallback(() => {
    transformRef.current = { scale: 1, translateX: 0, translateY: 0 };
    broadcastTransform();
  }, [broadcastTransform]);

  // Zoom in/out buttons
  const handleZoomIn = useCallback(() => {
    const t = transformRef.current;
    const newScale = Math.min(MAX_SCALE, t.scale * 1.3);
    transformRef.current = { ...t, scale: newScale };
    broadcastTransform();
  }, [broadcastTransform]);

  const handleZoomOut = useCallback(() => {
    const t = transformRef.current;
    const newScale = Math.max(MIN_SCALE, t.scale / 1.3);
    transformRef.current = { ...t, scale: newScale };
    broadcastTransform();
  }, [broadcastTransform]);

  // Direct layout setters (not toggle) to avoid wrong behavior when clicking the active button
  const setLayout2 = useCallback(() => setLayout("2-panel"), []);
  const setLayout4 = useCallback(() => setLayout("4-panel"), []);

  // Compute grid class based on layout
  const paneCount = layout === "2-panel" ? 2 : 4;
  const gridClass =
    layout === "2-panel"
      ? "grid-cols-2 grid-rows-1"
      : "grid-cols-2 grid-rows-2";

  // Pad products array to fill pane slots
  const paddedProducts: (ComparisonProduct | null)[] = Array.from(
    { length: paneCount },
    (_, i) => products[i] ?? null
  );

  if (!isActive) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-[#0a0e17]">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#1e293b] bg-[#0d1520] shrink-0">
        {/* Left: title + product count */}
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-sky-400">compare</span>
          <h2 className="text-sm font-semibold text-white">
            Comparison Mode
          </h2>
          <span className="text-[10px] text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded-full">
            {products.length} product{products.length !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Center: controls */}
        <div className="flex items-center gap-1.5">
          {/* Layout buttons -- each sets layout directly (not toggle) */}
          <button
            onClick={setLayout2}
            className={`
              flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all border
              ${
                layout === "2-panel"
                  ? "bg-sky-500/10 border-sky-500/30 text-sky-400"
                  : "bg-[#1a2333] border-[#1e293b] text-slate-400 hover:text-slate-200 hover:border-slate-500/50"
              }
            `}
            title="Side by side (2 panels)"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
              view_column_2
            </span>
            2-Panel
          </button>
          <button
            onClick={setLayout4}
            className={`
              flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all border
              ${
                layout === "4-panel"
                  ? "bg-sky-500/10 border-sky-500/30 text-sky-400"
                  : "bg-[#1a2333] border-[#1e293b] text-slate-400 hover:text-slate-200 hover:border-slate-500/50"
              }
            `}
            title="Grid (4 panels)"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
              grid_view
            </span>
            4-Panel
          </button>

          <div className="w-px h-5 bg-[#1e293b] mx-1" />

          {/* Zoom controls */}
          <button
            onClick={handleZoomOut}
            className="w-7 h-7 flex items-center justify-center rounded-md bg-[#1a2333] border border-[#1e293b] text-slate-400 hover:text-white hover:border-slate-500/50 transition-colors"
            title="Zoom out"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              remove
            </span>
          </button>
          <button
            onClick={handleResetZoom}
            className="px-2 h-7 flex items-center justify-center rounded-md bg-[#1a2333] border border-[#1e293b] text-slate-400 hover:text-white hover:border-slate-500/50 transition-colors text-[10px] font-mono"
            title="Reset zoom"
          >
            {zoomPercent}%
          </button>
          <button
            onClick={handleZoomIn}
            className="w-7 h-7 flex items-center justify-center rounded-md bg-[#1a2333] border border-[#1e293b] text-slate-400 hover:text-white hover:border-slate-500/50 transition-colors"
            title="Zoom in"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              add
            </span>
          </button>
        </div>

        {/* Right: close */}
        <button
          onClick={onClose}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-300 transition-all text-xs font-medium"
          title="Exit comparison mode (Esc)"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
            close
          </span>
          Exit
        </button>
      </div>

      {/* Grid of panes */}
      <div className={`flex-1 grid ${gridClass} gap-1 p-1 min-h-0`}>
        {paddedProducts.map((product, i) => (
          <Pane
            key={product?.productId ?? `empty-${i}`}
            product={product}
            paneIndex={i}
            transformRef={transformRef}
            onStartDrag={handleStartDrag}
            onRemove={onRemoveProduct}
            isDragging={isDragging}
          />
        ))}
      </div>

      {/* Status bar */}
      <div className="flex items-center justify-between px-4 py-1.5 border-t border-[#1e293b] bg-[#0d1520] text-[9px] text-slate-600 shrink-0">
        <span>
          Drag to pan &middot; Scroll to zoom &middot; Esc to exit
        </span>
        <span>
          Layout: {layout === "2-panel" ? "Side by Side" : "2x2 Grid"}
        </span>
      </div>
    </div>
  );
}

const ComparisonMode = memo(ComparisonModeInner);
export default ComparisonMode;
