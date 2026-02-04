import { useEffect, useState, useRef, useCallback, useMemo } from "react";

/* =========================================================
 * Types
 * =======================================================*/
type Metadata = {
  product_id: string;
  rows: number;
  range_bins: number;
  lat_range: [number, number];
  lon_range: [number, number];
  alt_range_km: [number, number];
  display: { recommended_downsample: number };
};

type SurfacePoint = { x: number; y: number };

type RadargramMeta = {
  n_traces: number;
  n_bins: number;
  downsample: number;
  lats: number[];
  lons: number[];
};

type DepthResult = {
  trace_idx: number;
  surface_bin: number | null;
  cursor_bin: number;
  delta_bins: number;
  delta_t_us: number;
  depth_m: number | null;
  epsilon_r1: number;
  epsilon_r2: number;
  boundary_m: number;
  lat: number;
  lon: number;
  message?: string;
};

type MolaProfile = {
  distance_km: number[];
  elevation_m: (number | null)[];
  n_traces: number;
  total_distance_km: number;
};

/* =========================================================
 * Constants
 * =======================================================*/
const DOWNSAMPLE = 50;
const MOLA_PANEL_HEIGHT = 120;
const ADJUST_RADIUS = 8;
const HANDLE_HIT_PX = 10;
const DEFAULT_WIDTH = 720;
const MIN_WIDTH = 600;
const MAX_WIDTH_FRACTION = 0.85;
const SPEED_OF_LIGHT = 299792458; // m/s
const BIN_DT_SEC = 0.0375e-6;    // seconds per range bin

/* =========================================================
 * SharadHiresInspector Component
 * =======================================================*/
export default function SharadHiresInspector({
  productId,
  onClose,
}: {
  productId: string;
  onClose: () => void;
}) {
  // Data state
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [radargram, setRadargram] = useState<HTMLImageElement | null>(null);
  const [radargramMeta, setRadargramMeta] = useState<RadargramMeta | null>(null);
  const [surface, setSurface] = useState<SurfacePoint[]>([]);
  const [molaProfile, setMolaProfile] = useState<MolaProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Display controls
  const [useLog, setUseLog] = useState(true);
  const [pmin, setPmin] = useState(1);
  const [pmax, setPmax] = useState(99);
  const [showSurface, setShowSurface] = useState(true);

  // Piecewise depth conversion
  const [epsilonR1, setEpsilonR1] = useState(3.1);
  const [epsilonR2, setEpsilonR2] = useState(3.1);
  const [boundaryM, setBoundaryM] = useState(0);
  const [showBoundaryLine, setShowBoundaryLine] = useState(true);
  const [depthResult, setDepthResult] = useState<DepthResult | null>(null);

  // Surface line adjustment
  const [adjustMode, setAdjustMode] = useState(false);
  const [surfaceOffsets, setSurfaceOffsets] = useState<Map<number, number>>(new Map());
  const adjustDragRef = useRef<{
    traceX: number;
    origBin: number;
    baseOffsets: Map<number, number>;
  } | null>(null);

  const effectiveSurface = useMemo(() => {
    if (surfaceOffsets.size === 0) return surface;
    return surface.map(pt => ({
      x: pt.x,
      y: pt.y + Math.round(surfaceOffsets.get(pt.x) ?? 0),
    }));
  }, [surface, surfaceOffsets]);

  const editCount = surfaceOffsets.size;

  // Shared X/Y view range (normalized 0..1)
  const [viewX, setViewX] = useState({ start: 0, end: 1 });
  const [viewY, setViewY] = useState({ start: 0, end: 1 });

  // Move mode
  const [moveMode, setMoveMode] = useState(false);
  const isDragging = useRef(false);

  // Panel width (resizable)
  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH);

  // Canvas refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const molaCanvasRef = useRef<HTMLCanvasElement>(null);
  const molaContainerRef = useRef<HTMLDivElement>(null);
  const [cursor, setCursor] = useState<{ normX: number; normY: number } | null>(null);

  // ── Data fetching ──────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function loadAll() {
      setLoading(true);
      setError(null);

      try {
        const pid = encodeURIComponent(productId);
        const metaRes = await fetch(`/api/sharad_highres/metadata?product_id=${pid}`);
        if (!metaRes.ok) throw new Error("Failed to load metadata");
        const metaData: Metadata = await metaRes.json();
        if (!cancelled) setMetadata(metaData);

        const [imgRes, rmRes, surfRes, molaRes] = await Promise.all([
          fetch(`/api/sharad_highres/radargram?product_id=${pid}&downsample=${DOWNSAMPLE}&log=${useLog ? 1 : 0}&pmin=${pmin}&pmax=${pmax}`),
          fetch(`/api/sharad_highres/radargram_meta?product_id=${pid}&downsample=${DOWNSAMPLE}`),
          fetch(`/api/sharad_highres/surface?product_id=${pid}&downsample=${DOWNSAMPLE}`),
          fetch(`/api/sharad_highres/mola_profile?product_id=${pid}&downsample=${DOWNSAMPLE}`),
        ]);

        if (!imgRes.ok) throw new Error("Failed to load radargram");
        if (!rmRes.ok) throw new Error("Failed to load radargram meta");

        const blob = await imgRes.blob();
        const img = new Image();
        img.src = URL.createObjectURL(blob);
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("Failed to decode radargram image"));
        });

        const rmData: RadargramMeta = await rmRes.json();

        let surfData: SurfacePoint[] = [];
        let molaData: MolaProfile | null = null;
        if (surfRes.ok) {
          const sj = await surfRes.json();
          surfData = sj.surface || [];
        }
        if (molaRes.ok) {
          molaData = await molaRes.json();
        }

        if (!cancelled) {
          setRadargram(img);
          setRadargramMeta(rmData);
          setSurface(surfData);
          setMolaProfile(molaData);
          setViewX({ start: 0, end: 1 });
          setViewY({ start: 0, end: 1 });
        }
      } catch (e: any) {
        if (!cancelled) setError(e.message || "Unknown error");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadAll();
    return () => { cancelled = true; };
  }, [productId, useLog, pmin, pmax]);

  // ── View helpers ──────────────────────────────────────
  const clampRange = useCallback((start: number, end: number, minSpan = 0.01) => {
    let s = start, e = end;
    if (e - s < minSpan) {
      const mid = (s + e) / 2;
      s = mid - minSpan / 2;
      e = mid + minSpan / 2;
    }
    if (s < 0) { e -= s; s = 0; }
    if (e > 1) { s -= (e - 1); e = 1; }
    return { start: Math.max(0, s), end: Math.min(1, e) };
  }, []);

  // ── Boundary bin offset from surface ────────────────────
  const boundaryBinOffset = useMemo(() => {
    if (boundaryM <= 0) return 0;
    return (2 * boundaryM * Math.sqrt(epsilonR1)) / (SPEED_OF_LIGHT * BIN_DT_SEC);
  }, [boundaryM, epsilonR1]);

  // ── Radargram canvas rendering ─────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !radargram || !radargramMeta) return;

    const container = containerRef.current;
    if (!container) return;

    const W = container.clientWidth;
    const H = container.clientHeight;
    canvas.width = W;
    canvas.height = H;

    ctx.fillStyle = "#0a0f18";
    ctx.fillRect(0, 0, W, H);

    const sx = viewX.start * radargram.width;
    const sy = viewY.start * radargram.height;
    const sw = (viewX.end - viewX.start) * radargram.width;
    const sh = (viewY.end - viewY.start) * radargram.height;

    if (sw > 0 && sh > 0) {
      ctx.drawImage(radargram, sx, sy, sw, sh, 0, 0, W, H);
    }

    const nTraces = radargramMeta.n_traces;
    const nBins = radargramMeta.n_bins;

    const traceToX = (t: number) => ((t / nTraces - viewX.start) / (viewX.end - viewX.start)) * W;
    const binToY = (b: number) => ((b / nBins - viewY.start) / (viewY.end - viewY.start)) * H;

    // Draw surface line
    const surfToDraw = effectiveSurface;
    if (showSurface && surfToDraw.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = adjustMode ? "#4ade80" : "#22c55e";
      ctx.lineWidth = adjustMode ? 2.5 : 1.5;
      let prevX = -Infinity;
      let moved = false;
      for (let i = 0; i < surfToDraw.length; i++) {
        const pt = surfToDraw[i];
        const x = traceToX(pt.x);
        const y = binToY(pt.y);
        if (x < -50 || x > W + 50) { prevX = pt.x; continue; }
        if (moved && pt.x - prevX > 5) {
          ctx.stroke();
          ctx.beginPath();
          moved = false;
        }
        if (!moved) { ctx.moveTo(x, y); moved = true; }
        else ctx.lineTo(x, y);
        prevX = pt.x;
      }
      ctx.stroke();

      // Adjustment handles
      if (adjustMode) {
        for (let i = 0; i < surfToDraw.length; i++) {
          const pt = surfToDraw[i];
          const x = traceToX(pt.x);
          const y = binToY(pt.y);
          if (x < -2 || x > W + 2 || y < -2 || y > H + 2) continue;
          const hasOffset = surfaceOffsets.has(pt.x);
          const isActive = adjustDragRef.current?.traceX === pt.x;
          const r = isActive ? 5 : hasOffset ? 3.5 : 2;
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = isActive ? "#fbbf24" : hasOffset ? "#86efac" : "#22c55e";
          ctx.fill();
          if (isActive) {
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Boundary line (dashed, follows surface offset by boundaryBinOffset)
      if (showBoundaryLine && boundaryBinOffset > 0 && surfToDraw.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(103, 232, 249, 0.7)"; // cyan-300
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        let bMoved = false;
        let bPrevX = -Infinity;
        for (let i = 0; i < surfToDraw.length; i++) {
          const pt = surfToDraw[i];
          const x = traceToX(pt.x);
          const y = binToY(pt.y + boundaryBinOffset);
          if (x < -50 || x > W + 50) { bPrevX = pt.x; continue; }
          if (bMoved && pt.x - bPrevX > 5) {
            ctx.stroke();
            ctx.beginPath();
            bMoved = false;
          }
          if (!bMoved) { ctx.moveTo(x, y); bMoved = true; }
          else ctx.lineTo(x, y);
          bPrevX = pt.x;
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        const midIdx = Math.floor(surfToDraw.length / 2);
        const labelX = traceToX(surfToDraw[midIdx].x);
        const labelY = binToY(surfToDraw[midIdx].y + boundaryBinOffset);
        if (labelX > 40 && labelX < W - 120 && labelY > 10 && labelY < H - 10) {
          ctx.font = "bold 9px monospace";
          ctx.fillStyle = "rgba(103, 232, 249, 0.85)";
          ctx.textAlign = "left";
          ctx.fillText(`ε boundary Z₁=${boundaryM}m`, labelX + 6, labelY - 5);
        }
      }
    }

    // Cursor crosshair
    if (cursor && !adjustDragRef.current && !isDragging.current) {
      const cx = cursor.normX;
      const cy = cursor.normY;
      const px = ((cx - viewX.start) / (viewX.end - viewX.start)) * W;
      const py = ((cy - viewY.start) / (viewY.end - viewY.start)) * H;

      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);

      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, H);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(W, py);
      ctx.stroke();

      ctx.setLineDash([]);
    }
  }, [radargram, radargramMeta, effectiveSurface, showSurface, viewX, viewY, cursor, adjustMode, surfaceOffsets, boundaryBinOffset, boundaryM, showBoundaryLine]);

  useEffect(() => { draw(); }, [draw]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

  // ── MOLA canvas rendering ─────────────────────────────
  const drawMola = useCallback(() => {
    const canvas = molaCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    const container = molaContainerRef.current;
    if (!canvas || !ctx || !container || !molaProfile) return;

    const W = container.clientWidth;
    const H = MOLA_PANEL_HEIGHT;
    canvas.width = W;
    canvas.height = H;

    const pad = { top: 18, bottom: 20, left: 48, right: 12 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = "#0a0f18";
    ctx.fillRect(0, 0, W, H);

    const n = molaProfile.elevation_m.length;
    if (n === 0) return;

    const iStart = Math.floor(viewX.start * n);
    const iEnd = Math.ceil(viewX.end * n);

    const visibleElevs: number[] = [];
    for (let i = iStart; i < iEnd && i < n; i++) {
      const e = molaProfile.elevation_m[i];
      if (e !== null) visibleElevs.push(e);
    }
    if (visibleElevs.length === 0) return;

    const eMin = Math.min(...visibleElevs);
    const eMax = Math.max(...visibleElevs);
    const ePad = Math.max((eMax - eMin) * 0.1, 50);
    const yMin = eMin - ePad;
    const yMax = eMax + ePad;

    const xOf = (i: number) => pad.left + ((i / (n - 1) - viewX.start) / (viewX.end - viewX.start)) * plotW;
    const yOf = (v: number) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Grid
    ctx.strokeStyle = "#1a2236";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#475569";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    const nTicks = 4;
    for (let t = 0; t <= nTicks; t++) {
      const v = yMin + (t / nTicks) * (yMax - yMin);
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(v)}`, pad.left - 4, y + 3);
    }

    // X-axis label
    ctx.fillStyle = "#475569";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    const dStart = molaProfile.distance_km[iStart] ?? 0;
    const dEnd = molaProfile.distance_km[Math.min(iEnd, n - 1)] ?? molaProfile.total_distance_km;
    ctx.fillText(`${dStart.toFixed(0)}\u2013${dEnd.toFixed(0)} km along track`, pad.left + plotW / 2, H - 3);

    // Title
    ctx.fillStyle = "#94a3b8";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("MOLA Elevation (m)", pad.left, 12);

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();

    // Elevation line
    ctx.beginPath();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    let started = false;
    for (let i = Math.max(0, iStart - 1); i <= Math.min(n - 1, iEnd + 1); i++) {
      const e = molaProfile.elevation_m[i];
      if (e === null) { started = false; continue; }
      const x = xOf(i);
      const y = yOf(e);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Cursor vertical line
    if (cursor && !isDragging.current) {
      const cx = pad.left + ((cursor.normX - viewX.start) / (viewX.end - viewX.start)) * plotW;
      if (cx >= pad.left && cx <= pad.left + plotW) {
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cx, pad.top);
        ctx.lineTo(cx, pad.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        const idx = Math.round(cursor.normX * (n - 1));
        if (idx >= 0 && idx < n) {
          const e = molaProfile.elevation_m[idx];
          if (e !== null) {
            ctx.fillStyle = "#38bdf8";
            ctx.font = "bold 9px monospace";
            ctx.textAlign = "left";
            ctx.fillText(`${Math.round(e)} m`, cx + 4, pad.top + 12);
          }
        }
      }
    }

    ctx.restore();
  }, [molaProfile, viewX, cursor]);

  useEffect(() => { drawMola(); }, [drawMola]);

  useEffect(() => {
    const container = molaContainerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => drawMola());
    observer.observe(container);
    return () => observer.disconnect();
  }, [drawMola]);

  // ── Shared zoom handler ────────────────────────────────
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 0.85 : 1 / 0.85;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width;

    const normCursorX = viewX.start + mx * (viewX.end - viewX.start);
    const newSpanX = (viewX.end - viewX.start) * factor;
    const newStartX = normCursorX - mx * newSpanX;
    const newEndX = normCursorX + (1 - mx) * newSpanX;
    setViewX(clampRange(newStartX, newEndX));

    if (e.currentTarget === canvasRef.current) {
      const my = (e.clientY - rect.top) / rect.height;
      const normCursorY = viewY.start + my * (viewY.end - viewY.start);
      const newSpanY = (viewY.end - viewY.start) * factor;
      const newStartY = normCursorY - my * newSpanY;
      const newEndY = normCursorY + (1 - my) * newSpanY;
      setViewY(clampRange(newStartY, newEndY));
    }
  }, [viewX, viewY, clampRange]);

  // ── Surface adjustment helpers ──────────────────────────
  const findNearestSurfaceVertex = useCallback((normX: number, normY: number): number | null => {
    if (!radargramMeta || !containerRef.current) return null;
    const W = containerRef.current.clientWidth;
    const H = containerRef.current.clientHeight;
    const nTraces = radargramMeta.n_traces;
    const nBins = radargramMeta.n_bins;

    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < effectiveSurface.length; i++) {
      const pt = effectiveSurface[i];
      const px = ((pt.x / nTraces - viewX.start) / (viewX.end - viewX.start)) * W;
      const py = ((pt.y / nBins - viewY.start) / (viewY.end - viewY.start)) * H;
      const cx = ((normX - viewX.start) / (viewX.end - viewX.start)) * W;
      const cy = ((normY - viewY.start) / (viewY.end - viewY.start)) * H;
      const dist = Math.sqrt((px - cx) ** 2 + (py - cy) ** 2);
      if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }
    return bestDist <= HANDLE_HIT_PX ? bestIdx : null;
  }, [effectiveSurface, radargramMeta, viewX, viewY]);

  const applyAdjustDrag = useCallback((traceX: number, newBinY: number) => {
    const origPt = surface.find(p => p.x === traceX);
    if (!origPt) return;
    const delta = newBinY - origPt.y;

    setSurfaceOffsets(prev => {
      const result = new Map(adjustDragRef.current?.baseOffsets ?? prev);
      const allTraceXs = surface.map(p => p.x);
      for (const tx of allTraceXs) {
        const dist = Math.abs(tx - traceX);
        if (dist <= ADJUST_RADIUS) {
          const weight = 1 - dist / (ADJUST_RADIUS + 1);
          const base = (adjustDragRef.current?.baseOffsets ?? prev).get(tx) ?? 0;
          result.set(tx, base + delta * weight);
        }
      }
      return result;
    });
  }, [surface]);

  // ── Resize handle (document-level drag) ─────────────────
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = panelWidth;

    const onMove = (ev: MouseEvent) => {
      // Dragging left increases width (handle is on left edge of right panel)
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * MAX_WIDTH_FRACTION);
      setPanelWidth(Math.max(MIN_WIDTH, Math.min(maxW, startWidth + delta)));
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

  // ── Mouse handlers ─────────────────────────────────────
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const isOnRadargram = e.currentTarget === canvasRef.current;

    // Surface adjustment drag start
    if (adjustMode && e.button === 0 && !e.shiftKey && isOnRadargram) {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const mx = (e.clientX - rect.left) / rect.width;
      const my = (e.clientY - rect.top) / rect.height;
      const normX = viewX.start + mx * (viewX.end - viewX.start);
      const normY = viewY.start + my * (viewY.end - viewY.start);

      const vertIdx = findNearestSurfaceVertex(normX, normY);
      if (vertIdx !== null) {
        const pt = effectiveSurface[vertIdx];
        adjustDragRef.current = {
          traceX: pt.x,
          origBin: pt.y,
          baseOffsets: new Map(surfaceOffsets),
        };

        // Document-level listeners for adjust drag
        const onMove = (ev: MouseEvent) => {
          if (!adjustDragRef.current || !radargramMeta || !canvasRef.current) return;
          const r = canvasRef.current.getBoundingClientRect();
          const my2 = (ev.clientY - r.top) / r.height;
          const normY2 = viewY.start + my2 * (viewY.end - viewY.start);
          const newBin = Math.round(normY2 * radargramMeta.n_bins);
          applyAdjustDrag(adjustDragRef.current.traceX, newBin);
        };
        const onUp = () => {
          adjustDragRef.current = null;
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);

        e.preventDefault();
        return;
      }
    }

    // Pan drag (move mode, middle click, shift+click)
    if ((moveMode && !adjustMode) || e.button === 1 || (e.button === 0 && e.shiftKey)) {
      const startX = e.clientX;
      const startY = e.clientY;
      const vxStart = { ...viewX };
      const vyStart = { ...viewY };
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();

      isDragging.current = true;

      const onMove = (ev: MouseEvent) => {
        const dxPx = ev.clientX - startX;
        const dyPx = ev.clientY - startY;
        const dxNorm = -(dxPx / rect.width) * (vxStart.end - vxStart.start);
        setViewX(clampRange(vxStart.start + dxNorm, vxStart.end + dxNorm));

        if (isOnRadargram) {
          const dyNorm = -(dyPx / rect.height) * (vyStart.end - vyStart.start);
          setViewY(clampRange(vyStart.start + dyNorm, vyStart.end + dyNorm));
        }
      };
      const onUp = () => {
        isDragging.current = false;
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.body.style.cursor = "grabbing";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      e.preventDefault();
    }
  }, [moveMode, viewX, viewY, adjustMode, findNearestSurfaceVertex, effectiveSurface, surfaceOffsets, radargramMeta, applyAdjustDrag, clampRange]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width;
    const my = (e.clientY - rect.top) / rect.height;

    const normX = viewX.start + mx * (viewX.end - viewX.start);
    const normY = canvas === canvasRef.current
      ? viewY.start + my * (viewY.end - viewY.start)
      : 0;
    setCursor({ normX, normY });
  }, [viewX, viewY]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (moveMode || adjustMode) return;
    if (!canvasRef.current || !radargramMeta) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width;
    const my = (e.clientY - rect.top) / rect.height;

    const normX = viewX.start + mx * (viewX.end - viewX.start);
    const normY = viewY.start + my * (viewY.end - viewY.start);

    const traceIdx = Math.round(normX * radargramMeta.n_traces);
    const cursorBin = Math.round(normY * radargramMeta.n_bins);
    if (traceIdx < 0 || traceIdx >= radargramMeta.n_traces) return;
    if (cursorBin < 0 || cursorBin >= radargramMeta.n_bins) return;

    const effPt = effectiveSurface.find(p => p.x === traceIdx);
    const surfOverride = effPt && surfaceOffsets.size > 0 ? effPt.y : undefined;

    const pid = encodeURIComponent(productId);
    let url = `/api/sharad_highres/depth_conversion?product_id=${pid}&trace_idx=${traceIdx}&cursor_bin=${cursorBin}&downsample=${DOWNSAMPLE}&epsilon_r1=${epsilonR1}&epsilon_r2=${epsilonR2}&boundary_m=${boundaryM}`;
    if (surfOverride !== undefined) {
      url += `&surface_bin_override=${surfOverride}`;
    }
    fetch(url)
      .then((r) => r.json())
      .then((data) => setDepthResult(data))
      .catch(() => setDepthResult(null));
  }, [radargramMeta, viewX, viewY, epsilonR1, epsilonR2, boundaryM, productId, moveMode, adjustMode, effectiveSurface, surfaceOffsets]);

  // ── Cursor info ────────────────────────────────────────
  const cursorInfo = (() => {
    if (!cursor || !radargramMeta) return null;
    const traceIdx = Math.round(cursor.normX * radargramMeta.n_traces);
    const binIdx = Math.round(cursor.normY * radargramMeta.n_bins);
    if (traceIdx < 0 || traceIdx >= radargramMeta.n_traces) return null;
    if (binIdx < 0 || binIdx >= radargramMeta.n_bins) return null;
    return {
      trace: traceIdx,
      bin: binIdx,
      lat: radargramMeta.lats[traceIdx] ?? null,
      lon: radargramMeta.lons[traceIdx] ?? null,
    };
  })();

  const zoomLevel = (1 / (viewX.end - viewX.start)).toFixed(1);

  // Slider CSS
  const sliderCls = `w-full h-1.5 bg-[#232f48] rounded appearance-none cursor-pointer
    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3
    [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full
    [&::-webkit-slider-thumb]:cursor-pointer`;
  const smallSliderCls = `flex-1 h-1 bg-[#232f48] rounded appearance-none cursor-pointer
    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-2.5
    [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:rounded-full
    [&::-webkit-slider-thumb]:bg-slate-400 [&::-webkit-slider-thumb]:cursor-pointer`;

  // ── Render ─────────────────────────────────────────────
  return (
    <div
      className="flex h-full flex-col border-l border-border-dark bg-surface-dark/40 relative"
      style={{ width: panelWidth }}
    >
      {/* Resize handle (left edge) */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
        onMouseDown={handleResizeStart}
      />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border-dark bg-[#0a0f18]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-amber-400 text-base">radar</span>
          <h3 className="text-white text-xs font-bold uppercase tracking-wider">
            SHARAD High-Res
          </h3>
          {metadata && (
            <span className="text-[10px] text-slate-500 font-mono ml-1">
              {metadata.product_id}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {/* Expand button */}
          <button
            onClick={() => setPanelWidth(Math.floor(window.innerWidth * MAX_WIDTH_FRACTION))}
            className="p-1 text-slate-500 hover:text-white transition-colors"
            title="Expand to 85% viewport"
          >
            <span className="material-symbols-outlined text-sm">open_in_full</span>
          </button>
          {/* Reset width button */}
          <button
            onClick={() => setPanelWidth(DEFAULT_WIDTH)}
            className="p-1 text-slate-500 hover:text-white transition-colors"
            title="Reset panel width"
          >
            <span className="material-symbols-outlined text-sm">width_normal</span>
          </button>
          {/* Close */}
          <button
            onClick={onClose}
            className="p-1 text-slate-500 hover:text-red-400 transition-colors"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left panel: controls */}
        <div className="w-64 border-r border-border-dark overflow-y-auto p-3 space-y-4 shrink-0 scrollbar-dark">
          {/* Display */}
          <Section title="Display">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-400">Scale</span>
              <div className="flex gap-1">
                <MiniButton active={useLog} onClick={() => setUseLog(true)}>Log</MiniButton>
                <MiniButton active={!useLog} onClick={() => setUseLog(false)}>Linear</MiniButton>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-400">Contrast</span>
                <span className="text-[9px] text-slate-500 font-mono">{pmin}-{pmax}%</span>
              </div>
              <div className="flex items-center gap-2">
                <input type="range" min={0} max={20} value={pmin}
                  onChange={(e) => setPmin(Number(e.target.value))}
                  className={smallSliderCls} />
                <input type="range" min={80} max={100} value={pmax}
                  onChange={(e) => setPmax(Number(e.target.value))}
                  className={smallSliderCls} />
              </div>
            </div>
          </Section>

          {/* Overlays */}
          <Section title="Overlays">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={showSurface} onChange={(e) => setShowSurface(e.target.checked)}
                className="rounded border-slate-600 bg-transparent text-green-500 focus:ring-green-500/30" />
              <span className="text-[10px] text-slate-300">Surface line</span>
              <span className="ml-auto w-3 h-0.5 bg-green-500 rounded" />
            </label>

            {showSurface && (
              <div className="space-y-1.5 pl-5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">Adjust</span>
                  <div className="flex gap-1">
                    <MiniButton
                      active={adjustMode}
                      onClick={() => {
                        if (!adjustMode) { setAdjustMode(true); setMoveMode(false); }
                        else setAdjustMode(false);
                      }}
                    >
                      {adjustMode ? "ON" : "OFF"}
                    </MiniButton>
                  </div>
                </div>
                {adjustMode && (
                  <div className="text-[9px] text-slate-600">
                    Click a vertex on the surface line and drag vertically to adjust.
                  </div>
                )}
                {editCount > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-[9px] text-green-400/70">{editCount} edits</span>
                    <button
                      onClick={() => setSurfaceOffsets(new Map())}
                      className="text-[9px] text-slate-500 hover:text-red-400 transition-colors underline"
                    >
                      Reset surface line
                    </button>
                  </div>
                )}
              </div>
            )}

            {boundaryM > 0 && (
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showBoundaryLine} onChange={(e) => setShowBoundaryLine(e.target.checked)}
                  className="rounded border-slate-600 bg-transparent text-cyan-500 focus:ring-cyan-500/30" />
                <span className="text-[10px] text-slate-300">Boundary line</span>
                <span className="ml-auto w-5 h-0 border-t border-dashed border-cyan-400" />
              </label>
            )}
          </Section>

          {/* Navigation */}
          <Section title="Navigation">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-400">Mode</span>
              <div className="flex gap-1">
                <MiniButton active={!moveMode && !adjustMode} onClick={() => { setMoveMode(false); setAdjustMode(false); }}>Query</MiniButton>
                <MiniButton
                  active={moveMode}
                  onClick={() => {
                    if (!moveMode) { setMoveMode(true); setAdjustMode(false); }
                    else setMoveMode(false);
                  }}
                >
                  Move
                </MiniButton>
              </div>
            </div>
            <div className="text-[9px] text-slate-600">
              {adjustMode
                ? "Adjust mode active. Drag surface vertices."
                : moveMode
                  ? "Drag to pan. Scroll to zoom."
                  : "Click for depth. Shift+drag or scroll to navigate."}
            </div>
            <button
              onClick={() => { setViewX({ start: 0, end: 1 }); setViewY({ start: 0, end: 1 }); }}
              className="text-[9px] text-slate-500 hover:text-white transition-colors underline"
            >
              Reset view
            </button>
          </Section>

          {/* Depth Conversion */}
          <Section title="Depth Conversion">
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">Layer 1 εr</span>
                  <span className="text-[10px] text-amber-400 font-mono">{epsilonR1.toFixed(1)}</span>
                </div>
                <input type="range" min={1} max={10} step={0.1} value={epsilonR1}
                  onChange={(e) => setEpsilonR1(Number(e.target.value))}
                  className={`${sliderCls} [&::-webkit-slider-thumb]:bg-amber-400`} />
                <div className="flex justify-between text-[8px] text-slate-600 font-mono">
                  <span>1.0</span><span>3.1 (ice)</span><span>10</span>
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">Boundary Z</span>
                  <span className="text-[10px] text-cyan-400 font-mono">{boundaryM} m</span>
                </div>
                <input type="range" min={0} max={2000} step={10} value={boundaryM}
                  onChange={(e) => setBoundaryM(Number(e.target.value))}
                  className={`${sliderCls} [&::-webkit-slider-thumb]:bg-cyan-400`} />
                <div className="flex justify-between text-[8px] text-slate-600 font-mono">
                  <span>0 (uniform)</span><span>2000 m</span>
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">Layer 2 εr</span>
                  <span className="text-[10px] text-orange-400 font-mono">{epsilonR2.toFixed(1)}</span>
                </div>
                <input type="range" min={1} max={10} step={0.1} value={epsilonR2}
                  onChange={(e) => setEpsilonR2(Number(e.target.value))}
                  className={`${sliderCls} [&::-webkit-slider-thumb]:bg-orange-400`} />
                <div className="flex justify-between text-[8px] text-slate-600 font-mono">
                  <span>1.0</span><span>5.0 (rock)</span><span>10</span>
                </div>
              </div>

              {boundaryM === 0 && (
                <div className="text-[8px] text-slate-600 italic">
                  Boundary=0: uniform εr₁ layer
                </div>
              )}
            </div>

            <div className="text-[9px] text-slate-500 italic mt-2">
              Click below surface line to compute depth
            </div>

            {depthResult && (
              <div className="space-y-1.5 p-2 rounded border border-amber-500/30 bg-amber-500/5 mt-2">
                <div className="flex justify-between text-[10px]">
                  <span className="text-slate-400">Trace</span>
                  <span className="text-white font-mono">{depthResult.trace_idx}</span>
                </div>
                {depthResult.surface_bin !== null && (
                  <div className="flex justify-between text-[10px]">
                    <span className="text-slate-400">Surface bin</span>
                    <span className="text-green-400 font-mono">{depthResult.surface_bin}</span>
                  </div>
                )}
                {depthResult.depth_m !== null && depthResult.depth_m > 0 ? (
                  <>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">Cursor bin</span>
                      <span className="text-amber-400 font-mono">{depthResult.cursor_bin}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">Δt</span>
                      <span className="text-white font-mono">{depthResult.delta_t_us} μs</span>
                    </div>
                    <div className="flex justify-between text-[10px] font-bold">
                      <span className="text-amber-400">Depth</span>
                      <span className="text-amber-300 font-mono">{depthResult.depth_m?.toFixed(1)} m</span>
                    </div>
                    {depthResult.boundary_m > 0 && (
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-400">Model</span>
                        <span className="text-slate-300 font-mono text-[9px]">
                          εr₁={depthResult.epsilon_r1} / εr₂={depthResult.epsilon_r2} @ {depthResult.boundary_m}m
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">Location</span>
                      <span className="text-white font-mono text-[9px]">
                        {depthResult.lat}°, {depthResult.lon}°
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="text-[10px] text-slate-500 italic">
                    {depthResult.message || "Click below the surface line"}
                  </div>
                )}
              </div>
            )}
          </Section>

          {/* Metadata */}
          {metadata && (
            <Section title="Dataset Info">
              <InfoRow label="Traces" value={metadata.rows.toLocaleString()} />
              <InfoRow label="Range bins" value={String(metadata.range_bins)} />
              <InfoRow label="Lat range" value={`${metadata.lat_range[0].toFixed(1)}° – ${metadata.lat_range[1].toFixed(1)}°`} />
              <InfoRow label="Lon range" value={`${metadata.lon_range[0].toFixed(1)}° – ${metadata.lon_range[1].toFixed(1)}°`} />
              <InfoRow label="Alt range" value={`${metadata.alt_range_km[0].toFixed(0)} – ${metadata.alt_range_km[1].toFixed(0)} km`} />
              <InfoRow label="Downsample" value={`${DOWNSAMPLE}×`} />
            </Section>
          )}
        </div>

        {/* Main canvas area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Radargram canvas */}
          <div ref={containerRef} className="flex-1 relative overflow-hidden bg-[#0a0f18]">
            {loading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
                <span className="material-symbols-outlined animate-spin text-3xl text-amber-400 mb-3">
                  progress_activity
                </span>
                <p className="text-xs text-slate-400">Loading radargram...</p>
                <p className="text-[10px] text-slate-600 mt-1">
                  {metadata ? `${metadata.rows.toLocaleString()} traces` : "Loading..."} × 667 range bins
                </p>
              </div>
            )}

            {error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
                <span className="material-symbols-outlined text-3xl text-red-400 mb-3">error</span>
                <p className="text-xs text-red-400">{error}</p>
              </div>
            )}

            <canvas
              ref={canvasRef}
              className={`w-full h-full ${
                adjustMode ? "cursor-cell" : moveMode ? "cursor-grab" : "cursor-crosshair"
              }`}
              onMouseMove={handleMouseMove}
              onMouseDown={handleMouseDown}
              onMouseLeave={() => setCursor(null)}
              onWheel={handleWheel}
              onClick={handleClick}
            />
          </div>

          {/* MOLA elevation profile panel */}
          {molaProfile && (
            <div
              ref={molaContainerRef}
              className="border-t border-border-dark bg-[#0a0f18] shrink-0"
              style={{ height: MOLA_PANEL_HEIGHT }}
            >
              <canvas
                ref={molaCanvasRef}
                className={`w-full h-full ${moveMode ? "cursor-grab" : "cursor-crosshair"}`}
                onMouseMove={handleMouseMove}
                onMouseDown={handleMouseDown}
                onMouseLeave={() => setCursor(null)}
                onWheel={handleWheel}
              />
            </div>
          )}

          {/* Status bar */}
          <div className="flex items-center justify-between px-3 py-1.5 border-t border-border-dark bg-[#0a0f18] text-[10px] font-mono text-slate-500">
            <div className="flex gap-4">
              {cursorInfo && (
                <>
                  <span>Trace: <span className="text-slate-300">{cursorInfo.trace}</span></span>
                  <span>Bin: <span className="text-slate-300">{cursorInfo.bin}</span></span>
                  {cursorInfo.lat !== null && (
                    <span>
                      Lat: <span className="text-slate-300">{cursorInfo.lat?.toFixed(3)}°</span>
                      {" "}Lon: <span className="text-slate-300">{cursorInfo.lon?.toFixed(3)}°</span>
                    </span>
                  )}
                </>
              )}
            </div>
            <div className="flex gap-3">
              <span>Zoom: <span className="text-slate-300">{zoomLevel}×</span></span>
              {moveMode && <span className="text-cyan-400">MOVE</span>}
              {adjustMode && <span className="text-green-400">ADJUST{editCount > 0 ? ` (${editCount})` : ""}</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
 * Sub-components
 * =======================================================*/
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">{title}</h4>
      {children}
    </div>
  );
}

function MiniButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 text-[9px] font-bold uppercase rounded transition-colors ${
        active
          ? "bg-primary/20 text-primary border border-primary/40"
          : "bg-[#0a0f18] text-slate-500 border border-[#232f48] hover:text-slate-300"
      }`}
    >
      {children}
    </button>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-[10px]">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-300 font-mono">{value}</span>
    </div>
  );
}
