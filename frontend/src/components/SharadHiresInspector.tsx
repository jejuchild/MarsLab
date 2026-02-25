import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import type { FieldNote } from "../api/fieldnotes";

/* =========================================================
 * Types
 * =======================================================*/
type Metadata = {
  product_id: string;
  rows: number;
  range_bins: number;
  lat_range: [number, number];
  lon_range: [number, number];
  start_lat: number;
  start_lon: number;
  stop_lat: number;
  stop_lon: number;
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
const MAX_IMAGE_WIDTH = 16384; // Safe browser canvas/image limit
const MOLA_PANEL_HEIGHT = 120;
const ADJUST_RADIUS = 8;
const HANDLE_HIT_PX = 10;
const DEFAULT_WIDTH = 720;
const MIN_WIDTH = 600;
const MAX_WIDTH_FRACTION = 0.85;
const SPEED_OF_LIGHT = 299792458; // m/s
const BIN_DT_SEC = 0.0375e-6;    // 1/26.67 MHz ADC — seconds per range bin (two-way)

/* =========================================================
 * Module-level cache: avoids re-fetching when revisiting a product
 * =======================================================*/
type CachedProductData = {
  metadata: Metadata;
  radargramBlob: Blob;
  radargramMeta: RadargramMeta;
  surface: SurfacePoint[];
  molaProfile: MolaProfile | null;
  clutterAvailable: boolean;
  /** Cache key includes display params that affect the radargram image */
  paramKey: string;
};
const _productCache = new Map<string, CachedProductData>();
const MAX_CACHE_SIZE = 10;

/* =========================================================
 * SharadHiresInspector Component
 * =======================================================*/
export default function SharadHiresInspector({
  productId,
  onClose,
  fieldNotes = [],
  onOpenFieldNote,
  onLocatePoint,
  onOpenRegolith,
  onOpenAttenuation,
}: {
  productId: string;
  onClose: () => void;
  fieldNotes?: FieldNote[];
  onOpenFieldNote?: (productId: string, lat: number, lon: number) => void;
  onLocatePoint?: (lat: number, lon: number) => void;
  onOpenRegolith?: (productId: string) => void;
  onOpenAttenuation?: (productId: string) => void;
}) {
  const hasNote = useMemo(
    () => fieldNotes.some(n => n.product_id === productId),
    [fieldNotes, productId]
  );

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
  const [useAmplitude, setUseAmplitude] = useState(true);
  const [usePerTrace, setUsePerTrace] = useState(true);
  const [showSurface, setShowSurface] = useState(true);

  // Cluttergram overlay
  const [clutterAvailable, setClutterAvailable] = useState(false);
  const [showClutter, setShowClutter] = useState(false);
  const [clutterImage, setClutterImage] = useState<HTMLImageElement | null>(null);
  const [clutterOpacity, setClutterOpacity] = useState(0.5);
  const [clutterLoading, setClutterLoading] = useState(false);

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

  // Smoothed surface for drawing (triangular kernel, radius 3)
  const smoothedSurface = useMemo(() => {
    const pts = effectiveSurface;
    if (pts.length < 5) return pts;
    const R = 3;
    return pts.map((pt, i) => {
      let wSum = 0, ySum = 0;
      for (let d = -R; d <= R; d++) {
        const j = i + d;
        if (j < 0 || j >= pts.length) continue;
        // Skip if there's a gap (non-consecutive traces)
        if (Math.abs(pts[j].x - pt.x) > R + 2) continue;
        const w = R + 1 - Math.abs(d);
        wSum += w;
        ySum += pts[j].y * w;
      }
      return { x: pt.x, y: wSum > 0 ? ySum / wSum : pt.y };
    });
  }, [effectiveSurface]);

  const editCount = surfaceOffsets.size;

  // Shared X/Y view range (normalized 0..1)
  const [viewX, setViewX] = useState({ start: 0, end: 1 });
  const [viewY, setViewY] = useState({ start: 0, end: 1 });

  // Move mode
  const [moveMode, setMoveMode] = useState(false);
  const isDragging = useRef(false);
  const didDragRef = useRef(false);  // Track if mouse moved during drag (suppress click)

  // Panel width (resizable) & collapse
  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH);
  const [collapsed, setCollapsed] = useState(false);

  // MOLA alignment offset (trace index shift)
  const [molaAlignMode, setMolaAlignMode] = useState(false);
  const [molaXOffset, setMolaXOffset] = useState(0);

  // ── Hyperbola εr Tool ──
  type HyperbolaFitResult = {
    v_mps: number; epsr: number; epsr_ci95: [number, number];
    dt_surface_s: number; depth_m: number; depth_ci95: [number, number];
    quality: { snr: number; residual: number; support_traces: number };
    flags: string[];
    overlay_polyline: { trace: number; bin: number }[];
    overlay_ci_band?: { trace: number; bin_lo: number; bin_hi: number }[] | null;
    preset_curves?: Record<string, { trace: number; bin: number }[]> | null;
  };
  const [hyperbolaApex, setHyperbolaApex] = useState<{ trace: number; bin: number } | null>(null);
  const [hyperbolaRoiTraces, setHyperbolaRoiTraces] = useState(60);
  const [hyperbolaRoiBins, setHyperbolaRoiBins] = useState(80);
  const [hyperbolaFit, setHyperbolaFit] = useState<HyperbolaFitResult | null>(null);
  const [hyperbolaLoading, setHyperbolaLoading] = useState(false);
  const [hyperbolaError, setHyperbolaError] = useState<string | null>(null);
  const [showPresetCurves, setShowPresetCurves] = useState(false);
  const [showHyperbolaCi, setShowHyperbolaCi] = useState(true);

  // Canvas refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const molaCanvasRef = useRef<HTMLCanvasElement>(null);
  const molaContainerRef = useRef<HTMLDivElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const plotsWrapperRef = useRef<HTMLDivElement>(null);
  const [cursor, setCursor] = useState<{ normX: number; normY: number } | null>(null);

  // ── Escape key to close ─────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // ── Data fetching (with module-level cache) ────────────
  useEffect(() => {
    let cancelled = false;

    async function loadAll() {
      setLoading(true);
      setError(null);

      try {
        const pid = encodeURIComponent(productId);
        const paramKey = `${useLog ? 1 : 0}_${pmin}_${pmax}_${useAmplitude ? 1 : 0}_${usePerTrace ? 1 : 0}`;
        const cached = _productCache.get(productId);

        let metaData: Metadata;
        let blob: Blob;
        let rmData: RadargramMeta;
        let surfData: SurfacePoint[];
        let molaData: MolaProfile | null;
        let clutterOk = false;

        if (cached && cached.paramKey === paramKey) {
          // Cache hit — use stored data
          metaData = cached.metadata;
          blob = cached.radargramBlob;
          rmData = cached.radargramMeta;
          surfData = cached.surface;
          molaData = cached.molaProfile;
          clutterOk = cached.clutterAvailable;
        } else {
          // Step 1: Fetch metadata first to compute safe downsample
          const metaRes = await fetch(`/api/sharad_highres/metadata?product_id=${pid}`);
          if (!metaRes.ok) throw new Error("Failed to load metadata");
          metaData = await metaRes.json();

          // Dynamic downsample: highest resolution that fits in browser canvas limits
          const ds = Math.max(1, Math.ceil(metaData.rows / MAX_IMAGE_WIDTH));

          // Step 2: Fetch the rest in parallel with computed downsample
          const [imgRes, rmRes, surfRes, molaRes] = await Promise.all([
            fetch(`/api/sharad_highres/radargram?product_id=${pid}&downsample=${ds}&log=${useLog ? 1 : 0}&pmin=${pmin}&pmax=${pmax}&amplitude=${useAmplitude ? 1 : 0}&per_trace=${usePerTrace ? 1 : 0}`),
            fetch(`/api/sharad_highres/radargram_meta?product_id=${pid}&downsample=${ds}`),
            fetch(`/api/sharad_highres/surface?product_id=${pid}&downsample=${ds}`),
            fetch(`/api/sharad_highres/mola_profile?product_id=${pid}&downsample=${ds}`),
          ]);

          if (!imgRes.ok) throw new Error("Failed to load radargram");
          if (!rmRes.ok) throw new Error("Failed to load radargram meta");

          blob = await imgRes.blob();
          rmData = await rmRes.json();

          surfData = [];
          molaData = null;
          if (surfRes.ok) {
            const sj = await surfRes.json();
            surfData = sj.surface || [];
          }
          if (molaRes.ok) {
            molaData = await molaRes.json();
          }

          // Check cluttergram availability (non-blocking)
          fetch(`/api/sharad_highres/cluttergram_meta?product_id=${pid}`)
            .then(r => r.json())
            .then(d => {
              clutterOk = d.available === true;
              // Update cache with clutter info
              const c = _productCache.get(productId);
              if (c) c.clutterAvailable = clutterOk;
              if (!cancelled) setClutterAvailable(clutterOk);
            })
            .catch(() => {});

          // Store in cache (evict oldest if full)
          if (_productCache.size >= MAX_CACHE_SIZE) {
            _productCache.delete(_productCache.keys().next().value!);
          }
          _productCache.set(productId, {
            metadata: metaData, radargramBlob: blob, radargramMeta: rmData,
            surface: surfData, molaProfile: molaData, clutterAvailable: clutterOk,
            paramKey,
          });
        }

        // Decode blob → HTMLImageElement
        const img = new globalThis.Image();
        img.src = URL.createObjectURL(blob);
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("Failed to decode radargram image"));
        });

        if (!cancelled) {
          setMetadata(metaData);
          setClutterAvailable(clutterOk);
          setRadargram(img);
          setRadargramMeta(rmData);
          setSurface(surfData);
          setMolaProfile(molaData);

          // Default to 1:1 pixel scale (native resolution, no horizontal squeeze)
          const cw = containerRef.current?.clientWidth || 450;
          const ch = containerRef.current?.clientHeight || 400;
          setViewX({ start: 0, end: Math.min(1, cw / img.width) });
          setViewY({ start: 0, end: Math.min(1, ch / img.height) });
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        if (!cancelled) setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadAll();
    return () => { cancelled = true; };
  }, [productId, useLog, pmin, pmax, useAmplitude, usePerTrace]);

  // Computed downsample factor based on product size
  const downsample = metadata ? Math.max(1, Math.ceil(metadata.rows / MAX_IMAGE_WIDTH)) : 1;

  // ── Cluttergram image loading ─────────────────────────
  useEffect(() => {
    if (!showClutter || clutterImage) return;
    let cancelled = false;
    setClutterLoading(true);

    const pid = encodeURIComponent(productId);
    fetch(`/api/sharad_highres/cluttergram?product_id=${pid}&downsample=${downsample}&pmin=${pmin}&pmax=${pmax}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to load cluttergram");
        return res.blob();
      })
      .then(blob => {
        const img = new Image();
        img.src = URL.createObjectURL(blob);
        return new Promise<HTMLImageElement>((resolve, reject) => {
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error("Failed to decode cluttergram"));
        });
      })
      .then(img => {
        if (!cancelled) {
          setClutterImage(img);
          setClutterLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setClutterLoading(false);
          setShowClutter(false);
        }
      });

    return () => { cancelled = true; };
  }, [showClutter, productId, clutterImage, pmin, pmax, downsample]);

  // Reset cluttergram when display params change
  useEffect(() => {
    setClutterImage(null);
  }, [productId, pmin, pmax]);

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

    const dpr = window.devicePixelRatio || 1;
    const W = container.clientWidth;
    const H = container.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Crisp pixels when zoomed in (no bilinear blur)
    ctx.imageSmoothingEnabled = false;

    ctx.fillStyle = "#0a0f18";
    ctx.fillRect(0, 0, W, H);

    const sx = viewX.start * radargram.width;
    const sy = viewY.start * radargram.height;
    const sw = (viewX.end - viewX.start) * radargram.width;
    const sh = (viewY.end - viewY.start) * radargram.height;

    if (sw > 0 && sh > 0) {
      ctx.drawImage(radargram, sx, sy, sw, sh, 0, 0, W, H);
    }

    // Cluttergram overlay (drawn on top of radargram with opacity)
    if (showClutter && clutterOpacity > 0) {
      ctx.save();
      ctx.globalAlpha = clutterOpacity;

      if (clutterImage) {
        const csx = viewX.start * clutterImage.width;
        const csy = viewY.start * clutterImage.height;
        const csw = (viewX.end - viewX.start) * clutterImage.width;
        const csh = (viewY.end - viewY.start) * clutterImage.height;
        if (csw > 0 && csh > 0) {
          ctx.drawImage(clutterImage, csx, csy, csw, csh, 0, 0, W, H);
        }
      }

      ctx.globalAlpha = 1;
      ctx.restore();
    }

    const nTraces = radargramMeta.n_traces;
    const nBins = radargramMeta.n_bins;

    const traceToX = (t: number) => ((t / nTraces - viewX.start) / (viewX.end - viewX.start)) * W;
    const binToY = (b: number) => ((b / nBins - viewY.start) / (viewY.end - viewY.start)) * H;

    // Draw surface line (smoothed for display)
    const lineToDraw = smoothedSurface;
    const handlePts = effectiveSurface; // raw points for adjustment handles
    if (showSurface && lineToDraw.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = adjustMode ? "#4ade80" : "#22c55e";
      ctx.lineWidth = adjustMode ? 2.5 : 1.5;
      let prevX = -Infinity;
      let moved = false;
      for (let i = 0; i < lineToDraw.length; i++) {
        const pt = lineToDraw[i];
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

      // Adjustment handles (drawn at raw effectiveSurface positions)
      if (adjustMode) {
        for (let i = 0; i < handlePts.length; i++) {
          const pt = handlePts[i];
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

      // Boundary line (dashed, follows smoothed surface offset by boundaryBinOffset)
      if (showBoundaryLine && boundaryBinOffset > 0 && lineToDraw.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(103, 232, 249, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        let bMoved = false;
        let bPrevX = -Infinity;
        for (let i = 0; i < lineToDraw.length; i++) {
          const pt = lineToDraw[i];
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
        const midIdx = Math.floor(lineToDraw.length / 2);
        const labelX = traceToX(lineToDraw[midIdx].x);
        const labelY = binToY(lineToDraw[midIdx].y + boundaryBinOffset);
        if (labelX > 40 && labelX < W - 120 && labelY > 10 && labelY < H - 10) {
          ctx.font = "bold 9px monospace";
          ctx.fillStyle = "rgba(103, 232, 249, 0.85)";
          ctx.textAlign = "left";
          ctx.fillText(`ε boundary Z₁=${boundaryM}m`, labelX + 6, labelY - 5);
        }
      }
    }

    // ── Hyperbola overlay ──
    if (hyperbolaApex) {
      // Draw apex marker
      const ax = traceToX(hyperbolaApex.trace);
      const ay = binToY(hyperbolaApex.bin);
      if (ax >= 0 && ax <= W && ay >= 0 && ay <= H) {
        ctx.beginPath();
        ctx.arc(ax, ay, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#f59e0b";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.font = "bold 9px monospace";
        ctx.fillStyle = "#f59e0b";
        ctx.textAlign = "left";
        ctx.fillText("APEX", ax + 8, ay - 4);
      }

      // Draw ROI box
      const rT0 = hyperbolaApex.trace - hyperbolaRoiTraces;
      const rT1 = hyperbolaApex.trace + hyperbolaRoiTraces;
      const rB0 = hyperbolaApex.bin - hyperbolaRoiBins;
      const rB1 = hyperbolaApex.bin + hyperbolaRoiBins;
      const rx0 = traceToX(rT0), rx1 = traceToX(rT1);
      const ry0 = binToY(rB0), ry1 = binToY(rB1);
      ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(rx0, ry0, rx1 - rx0, ry1 - ry0);
      ctx.setLineDash([]);
    }

    if (hyperbolaFit && hyperbolaFit.overlay_polyline.length > 0) {
      // Draw fitted curve (red)
      ctx.beginPath();
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      let fMoved = false;
      for (const pt of hyperbolaFit.overlay_polyline) {
        const x = traceToX(pt.trace);
        const y = binToY(pt.bin);
        if (x < -10 || x > W + 10) continue;
        if (!fMoved) { ctx.moveTo(x, y); fMoved = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // CI band
      if (showHyperbolaCi && hyperbolaFit.overlay_ci_band) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.12)";
        ctx.beginPath();
        // Upper edge
        let ciMoved = false;
        for (const pt of hyperbolaFit.overlay_ci_band) {
          const x = traceToX(pt.trace);
          const y = binToY(pt.bin_lo);
          if (!ciMoved) { ctx.moveTo(x, y); ciMoved = true; } else ctx.lineTo(x, y);
        }
        // Lower edge (reverse)
        for (let i = hyperbolaFit.overlay_ci_band.length - 1; i >= 0; i--) {
          const pt = hyperbolaFit.overlay_ci_band[i];
          ctx.lineTo(traceToX(pt.trace), binToY(pt.bin_hi));
        }
        ctx.closePath();
        ctx.fill();
      }

      // Preset curves
      if (showPresetCurves && hyperbolaFit.preset_curves) {
        const presetColors: Record<string, string> = {
          "2.8": "#67e8f9", "3.15": "#4ade80", "4.0": "#facc15",
        };
        for (const [key, curve] of Object.entries(hyperbolaFit.preset_curves)) {
          ctx.beginPath();
          ctx.strokeStyle = presetColors[key] || "#94a3b8";
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 3]);
          let pMoved = false;
          for (const pt of curve) {
            const x = traceToX(pt.trace);
            const y = binToY(pt.bin);
            if (x < -10 || x > W + 10) continue;
            if (!pMoved) { ctx.moveTo(x, y); pMoved = true; } else ctx.lineTo(x, y);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          // Label
          const first = curve[0];
          if (first) {
            const lx = traceToX(first.trace);
            const ly = binToY(first.bin);
            ctx.font = "bold 8px monospace";
            ctx.fillStyle = presetColors[key] || "#94a3b8";
            ctx.textAlign = "right";
            ctx.fillText(`εr=${key}`, lx - 4, ly - 2);
          }
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
  }, [radargram, radargramMeta, effectiveSurface, smoothedSurface, showSurface, viewX, viewY, cursor, adjustMode, surfaceOffsets, boundaryBinOffset, boundaryM, showBoundaryLine, showClutter, clutterImage, clutterOpacity, hyperbolaApex, hyperbolaRoiTraces, hyperbolaRoiBins, hyperbolaFit, showPresetCurves, showHyperbolaCi]);

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

    const dpr = window.devicePixelRatio || 1;
    const W = container.clientWidth;
    const H = MOLA_PANEL_HEIGHT;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { top: 18, bottom: 20, left: 48, right: 12 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = "#0a0f18";
    ctx.fillRect(0, 0, W, H);

    const n = molaProfile.elevation_m.length;
    if (n === 0) return;

    // molaXOffset shifts MOLA left/right relative to SHARAD x-axis
    const iStart = Math.floor(viewX.start * n);
    const iEnd = Math.ceil(viewX.end * n);

    // For visible range calculation, account for offset
    const visibleElevs: number[] = [];
    for (let i = iStart; i < iEnd && i < n; i++) {
      const si = i + molaXOffset;
      if (si < 0 || si >= n) continue;
      const e = molaProfile.elevation_m[si];
      if (e !== null) visibleElevs.push(e);
    }
    if (visibleElevs.length === 0) return;

    let eMin = visibleElevs[0], eMax = visibleElevs[0];
    for (let k = 1; k < visibleElevs.length; k++) {
      if (visibleElevs[k] < eMin) eMin = visibleElevs[k];
      if (visibleElevs[k] > eMax) eMax = visibleElevs[k];
    }
    const ePad = Math.max((eMax - eMin) * 0.1, 50);
    const yMin = eMin - ePad;
    const yMax = eMax + ePad;

    // xOf maps a SHARAD-aligned display index to pixel x
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
    const offsetLabel = molaXOffset !== 0 ? ` [offset: ${molaXOffset > 0 ? "+" : ""}${molaXOffset}]` : "";
    ctx.fillText(`${dStart.toFixed(0)}\u2013${dEnd.toFixed(0)} km along track${offsetLabel}`, pad.left + plotW / 2, H - 3);

    // Title
    ctx.fillStyle = "#94a3b8";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("MOLA Elevation (m)", pad.left, 12);
    if (molaXOffset !== 0) {
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(`  offset: ${molaXOffset > 0 ? "+" : ""}${molaXOffset}`, pad.left + 110, 12);
    }

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();

    // Elevation line (shifted by molaXOffset)
    ctx.beginPath();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    let started = false;
    for (let i = Math.max(0, iStart - 1); i <= Math.min(n - 1, iEnd + 1); i++) {
      const si = i + molaXOffset;
      if (si < 0 || si >= n) { started = false; continue; }
      const e = molaProfile.elevation_m[si];
      if (e === null) { started = false; continue; }
      const x = xOf(i); // x position is SHARAD-aligned (no offset)
      const y = yOf(e);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Cursor vertical line (always tracks SHARAD x, not shifted)
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

        // Show elevation at cursor, accounting for offset
        const idx = Math.round(cursor.normX * (n - 1)) + molaXOffset;
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
  }, [molaProfile, viewX, cursor, molaXOffset]);

  useEffect(() => { drawMola(); }, [drawMola]);

  useEffect(() => {
    const container = molaContainerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => drawMola());
    observer.observe(container);
    return () => observer.disconnect();
  }, [drawMola]);

  // ── Alignment marker overlay (spans both SHARAD and MOLA) ──
  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const wrapper = plotsWrapperRef.current;
    if (!canvas || !wrapper) return;

    const dpr = window.devicePixelRatio || 1;
    const W = wrapper.clientWidth;
    const H = wrapper.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    if (!cursor || isDragging.current) return;

    // Compute the x pixel position based on the SHARAD canvas coordinate system
    // The radargram canvas fills the full width of the wrapper
    const normFrac = (cursor.normX - viewX.start) / (viewX.end - viewX.start);
    const px = normFrac * W;

    if (px < 0 || px > W) return;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [cursor, viewX]);

  useEffect(() => { drawOverlay(); }, [drawOverlay]);

  useEffect(() => {
    const wrapper = plotsWrapperRef.current;
    if (!wrapper) return;
    const observer = new ResizeObserver(() => drawOverlay());
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [drawOverlay]);

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

    // MOLA offset drag (molaAlignMode ON, drag on MOLA panel)
    if (molaAlignMode && !isOnRadargram && e.button === 0 && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();

      const startX = e.clientX;
      const startOffset = molaXOffset;
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const nSamples = molaProfile?.elevation_m.length ?? 1;
      const visibleSpan = viewX.end - viewX.start;

      isDragging.current = true;

      const onMove = (ev: MouseEvent) => {
        ev.preventDefault();
        const dxPx = ev.clientX - startX;
        // Convert pixel movement to trace index offset
        const dxTraces = Math.round(-(dxPx / rect.width) * visibleSpan * nSamples);
        const maxOffset = Math.min(5000, Math.floor(nSamples / 2));
        setMolaXOffset(Math.max(-maxOffset, Math.min(maxOffset, startOffset + dxTraces)));
      };
      const onUp = () => {
        isDragging.current = false;
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.body.style.cursor = "ew-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      return;
    }

    // Pan drag — works in all modes (default left-click, move mode, middle click, shift+click)
    // In Query/default mode: drag to pan, click (no movement) for depth query
    if (e.button === 0 || e.button === 1) {
      e.preventDefault();
      e.stopPropagation();

      const startX = e.clientX;
      const startY = e.clientY;
      const vxStart = { ...viewX };
      const vyStart = { ...viewY };
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const DRAG_THRESHOLD = 3; // px — movement below this counts as click, not drag

      isDragging.current = true;
      didDragRef.current = false;

      const onMove = (ev: MouseEvent) => {
        ev.preventDefault();
        const dxPx = ev.clientX - startX;
        const dyPx = ev.clientY - startY;

        // Only start panning after threshold to distinguish click from drag
        if (!didDragRef.current && Math.abs(dxPx) < DRAG_THRESHOLD && Math.abs(dyPx) < DRAG_THRESHOLD) {
          return;
        }
        didDragRef.current = true;

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
    }
  }, [moveMode, viewX, viewY, adjustMode, molaAlignMode, molaXOffset, molaProfile, findNearestSurfaceVertex, effectiveSurface, surfaceOffsets, radargramMeta, applyAdjustDrag, clampRange]);

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
    if (didDragRef.current) return; // Suppress click after drag
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

    // Shift+Click: set hyperbola apex
    if (e.shiftKey) {
      setHyperbolaApex({ trace: traceIdx, bin: cursorBin });
      setHyperbolaFit(null);
      setHyperbolaError(null);
      return;
    }

    // Show clicked point on the map track
    if (onLocatePoint) {
      const lat = radargramMeta.lats[traceIdx];
      const lon = radargramMeta.lons[traceIdx];
      if (lat != null && lon != null) onLocatePoint(lat, lon);
    }

    const effPt = effectiveSurface.find(p => p.x === traceIdx);
    const surfOverride = effPt && surfaceOffsets.size > 0 ? effPt.y : undefined;

    const pid = encodeURIComponent(productId);
    let url = `/api/sharad_highres/depth_conversion?product_id=${pid}&trace_idx=${traceIdx}&cursor_bin=${cursorBin}&downsample=${downsample}&epsilon_r1=${epsilonR1}&epsilon_r2=${epsilonR2}&boundary_m=${boundaryM}`;
    if (surfOverride !== undefined) {
      url += `&surface_bin_override=${surfOverride}`;
    }
    fetch(url)
      .then((r) => r.json())
      .then((data) => setDepthResult(data))
      .catch(() => setDepthResult(null));
  }, [radargramMeta, viewX, viewY, epsilonR1, epsilonR2, boundaryM, productId, moveMode, adjustMode, effectiveSurface, surfaceOffsets, onLocatePoint, downsample]);

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
      className="flex h-full flex-col border-l border-border-dark bg-surface-dark/40 relative transition-[width] duration-200"
      style={{ width: collapsed ? 48 : panelWidth }}
    >
      {/* Resize handle (left edge) — hidden when collapsed */}
      {!collapsed && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}

      {/* Header */}
      <div className={`flex items-center justify-between border-b border-border-dark bg-[#0a0f18] ${collapsed ? "px-2 py-2.5 flex-col gap-2" : "px-4 py-2.5"}`}>
        {!collapsed && (
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
        )}
        {collapsed && (
          <span className="material-symbols-outlined text-amber-400 text-base">radar</span>
        )}
        <div className={`flex items-center gap-1 ${collapsed ? "flex-col" : ""}`}>
          {/* Field Note */}
          {onOpenFieldNote && (
            <button
              onClick={() => {
                const midLat = metadata ? (metadata.start_lat + metadata.stop_lat) / 2 : 0;
                const midLon = metadata ? (metadata.start_lon + metadata.stop_lon) / 2 : 0;
                onOpenFieldNote(productId, midLat, midLon);
              }}
              className={`p-1 transition-colors ${
                hasNote
                  ? "text-amber-400 hover:text-amber-300"
                  : "text-slate-500 hover:text-amber-400"
              }`}
              title={hasNote ? "Edit field note" : "Add field note"}
            >
              <span className="material-symbols-outlined text-sm">
                {hasNote ? "description" : "note_add"}
              </span>
            </button>
          )}
          {/* Collapse / Expand toggle */}
          <button
            onClick={() => setCollapsed(c => !c)}
            className="p-1 text-slate-500 hover:text-white transition-colors"
            title={collapsed ? "Expand panel (Esc)" : "Collapse panel (Esc)"}
          >
            <span className="material-symbols-outlined text-sm">
              {collapsed ? "left_panel_open" : "left_panel_close"}
            </span>
          </button>
          {/* Expand button */}
          <button
            onClick={() => { setPanelWidth(Math.floor(window.innerWidth * MAX_WIDTH_FRACTION)); setCollapsed(false); }}
            className="p-1 text-slate-500 hover:text-white transition-colors"
            title="Expand to 85% viewport"
          >
            <span className="material-symbols-outlined text-sm">open_in_full</span>
          </button>
          {/* Reset width button */}
          <button
            onClick={() => { setPanelWidth(DEFAULT_WIDTH); setCollapsed(false); }}
            className="p-1 text-slate-500 hover:text-white transition-colors"
            title="Reset to default size"
          >
            <span className="material-symbols-outlined text-sm">width_normal</span>
          </button>
          {/* Close */}
          <button
            onClick={onClose}
            className="p-1 text-slate-500 hover:text-red-400 transition-colors"
            title="Close inspector"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {!collapsed && (
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

            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-400">Value</span>
              <div className="flex gap-1">
                <MiniButton active={useAmplitude} onClick={() => setUseAmplitude(true)}>Amp</MiniButton>
                <MiniButton active={!useAmplitude} onClick={() => setUseAmplitude(false)}>Power</MiniButton>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-400">Normalize</span>
              <div className="flex gap-1">
                <MiniButton active={usePerTrace} onClick={() => setUsePerTrace(true)}>Per-trace</MiniButton>
                <MiniButton active={!usePerTrace} onClick={() => setUsePerTrace(false)}>Global</MiniButton>
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

            {clutterAvailable && (
              <>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={showClutter} onChange={(e) => setShowClutter(e.target.checked)}
                    className="rounded border-slate-600 bg-transparent text-orange-500 focus:ring-orange-500/30" />
                  <span className="text-[10px] text-slate-300">Cluttergram</span>
                  {clutterLoading && (
                    <span className="material-symbols-outlined animate-spin text-[12px] text-orange-400">progress_activity</span>
                  )}
                  <span className="ml-auto w-3 h-3 rounded-sm bg-gradient-to-b from-orange-400/60 to-transparent border border-orange-500/40" />
                </label>
                {showClutter && (
                  <div className="space-y-1 pl-5">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-slate-400">Opacity</span>
                      <span className="text-[9px] text-orange-400 font-mono">{Math.round(clutterOpacity * 100)}%</span>
                    </div>
                    <input type="range" min={0} max={100} value={Math.round(clutterOpacity * 100)}
                      onChange={(e) => setClutterOpacity(Number(e.target.value) / 100)}
                      className={`${smallSliderCls} [&::-webkit-slider-thumb]:bg-orange-400`} />
                    <div className="text-[8px] text-slate-600 italic">
                      Surface clutter simulation (US SHARAD team)
                    </div>
                  </div>
                )}
              </>
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
            <div className="flex gap-2">
              <button
                onClick={() => {
                  const cw = containerRef.current?.clientWidth || 450;
                  const ch = containerRef.current?.clientHeight || 400;
                  const imgW = radargram?.width || 1;
                  const imgH = radargram?.height || 1;
                  setViewX({ start: 0, end: Math.min(1, cw / imgW) });
                  setViewY({ start: 0, end: Math.min(1, ch / imgH) });
                }}
                className="text-[9px] text-slate-500 hover:text-white transition-colors underline"
              >
                1:1
              </button>
              <button
                onClick={() => { setViewX({ start: 0, end: 1 }); setViewY({ start: 0, end: 1 }); }}
                className="text-[9px] text-slate-500 hover:text-white transition-colors underline"
              >
                Fit all
              </button>
            </div>
          </Section>

          {/* MOLA Alignment */}
          {molaProfile && (
            <Section title="MOLA Align">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-400">Adjust Offset</span>
                <MiniButton
                  active={molaAlignMode}
                  onClick={() => setMolaAlignMode(!molaAlignMode)}
                >
                  {molaAlignMode ? "ON" : "OFF"}
                </MiniButton>
              </div>
              {molaAlignMode && (
                <div className="text-[9px] text-slate-600">
                  Drag left/right on MOLA panel to shift offset.
                </div>
              )}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">Offset</span>
                  <span className={`text-[10px] font-mono ${molaXOffset !== 0 ? "text-amber-400" : "text-slate-500"}`}>
                    {molaXOffset > 0 ? "+" : ""}{molaXOffset} traces
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setMolaXOffset(o => Math.max(-5000, o - 10))}
                    className="px-1 py-0.5 text-[9px] text-slate-400 hover:text-white border border-[#232f48] rounded transition-colors"
                  >◀ 10</button>
                  <button
                    onClick={() => setMolaXOffset(o => Math.max(-5000, o - 1))}
                    className="px-1 py-0.5 text-[9px] text-slate-400 hover:text-white border border-[#232f48] rounded transition-colors"
                  >◀</button>
                  <input
                    type="number"
                    value={molaXOffset}
                    onChange={(e) => {
                      const v = parseInt(e.target.value, 10);
                      if (!isNaN(v)) setMolaXOffset(Math.max(-5000, Math.min(5000, v)));
                    }}
                    className="flex-1 min-w-0 bg-[#0a0f18] border border-[#232f48] rounded px-1 py-0.5 text-[10px] text-center text-slate-300 font-mono focus:outline-none focus:border-primary/40"
                  />
                  <button
                    onClick={() => setMolaXOffset(o => Math.min(5000, o + 1))}
                    className="px-1 py-0.5 text-[9px] text-slate-400 hover:text-white border border-[#232f48] rounded transition-colors"
                  >▶</button>
                  <button
                    onClick={() => setMolaXOffset(o => Math.min(5000, o + 10))}
                    className="px-1 py-0.5 text-[9px] text-slate-400 hover:text-white border border-[#232f48] rounded transition-colors"
                  >10 ▶</button>
                </div>
              </div>
              {molaXOffset !== 0 && (
                <button
                  onClick={() => { setMolaXOffset(0); }}
                  className="text-[9px] text-slate-500 hover:text-amber-400 transition-colors underline"
                >
                  Reset to zero
                </button>
              )}
            </Section>
          )}

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

          {/* Hyperbola εr Tool */}
          <Section title="Hyperbola εr">
            <div className="text-[9px] text-slate-500 italic mb-2">
              Shift+Click on radargram to set apex
            </div>

            {hyperbolaApex ? (
              <div className="space-y-2">
                <div className="flex justify-between text-[10px]">
                  <span className="text-slate-400">Apex</span>
                  <span className="text-amber-400 font-mono">T={hyperbolaApex.trace} B={hyperbolaApex.bin}</span>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-400">ROI traces</span>
                    <span className="text-[10px] text-amber-400 font-mono">±{hyperbolaRoiTraces}</span>
                  </div>
                  <input type="range" min={10} max={300} step={5} value={hyperbolaRoiTraces}
                    onChange={(e) => setHyperbolaRoiTraces(Number(e.target.value))}
                    className="w-full h-1 rounded bg-slate-700 appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-400" />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-400">ROI bins</span>
                    <span className="text-[10px] text-amber-400 font-mono">±{hyperbolaRoiBins}</span>
                  </div>
                  <input type="range" min={10} max={300} step={5} value={hyperbolaRoiBins}
                    onChange={(e) => setHyperbolaRoiBins(Number(e.target.value))}
                    className="w-full h-1 rounded bg-slate-700 appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-400" />
                </div>

                <div className="flex flex-wrap gap-1.5 mt-1">
                  <button
                    disabled={hyperbolaLoading}
                    onClick={() => {
                      if (!hyperbolaApex) return;
                      setHyperbolaLoading(true);
                      setHyperbolaError(null);
                      fetch("/api/ice/hyperbola/fit", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          product_id: productId,
                          apex_trace: hyperbolaApex.trace,
                          apex_bin: hyperbolaApex.bin,
                          roi_traces: hyperbolaRoiTraces,
                          roi_bins: hyperbolaRoiBins,
                        }),
                      })
                        .then((r) => {
                          if (!r.ok) throw new Error(`${r.status}`);
                          return r.json();
                        })
                        .then((data) => { setHyperbolaFit(data); setHyperbolaLoading(false); })
                        .catch((err) => { setHyperbolaError(String(err)); setHyperbolaLoading(false); });
                    }}
                    className="px-2 py-0.5 text-[9px] font-bold rounded bg-red-900/40 text-red-300 border border-red-500/30 hover:bg-red-800/50 disabled:opacity-50"
                  >
                    {hyperbolaLoading ? "Fitting..." : "Fit Hyperbola"}
                  </button>
                  <button
                    onClick={() => setShowPresetCurves((p) => !p)}
                    className={`px-2 py-0.5 text-[9px] font-bold rounded border ${
                      showPresetCurves
                        ? "bg-cyan-900/30 text-cyan-300 border-cyan-500/40"
                        : "bg-[#0a0f18] text-slate-500 border-[#232f48] hover:text-slate-300"
                    }`}
                  >
                    Presets
                  </button>
                  <button
                    onClick={() => {
                      setHyperbolaApex(null);
                      setHyperbolaFit(null);
                      setHyperbolaError(null);
                    }}
                    className="px-2 py-0.5 text-[9px] font-bold rounded bg-[#0a0f18] text-slate-500 border border-[#232f48] hover:text-red-400"
                  >
                    Clear
                  </button>
                </div>

                {hyperbolaError && (
                  <div className="text-[9px] text-red-400 mt-1">{hyperbolaError}</div>
                )}

                {hyperbolaFit && (
                  <div className="space-y-1.5 p-2 rounded border border-red-500/30 bg-red-500/5 mt-2">
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">Velocity</span>
                      <span className="text-white font-mono">{(hyperbolaFit.v_mps / 1e6).toFixed(2)} Mm/s</span>
                    </div>
                    <div className="flex justify-between text-[10px] font-bold">
                      <span className="text-red-400">εr</span>
                      <span className="text-red-300 font-mono">{hyperbolaFit.epsr.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">CI95</span>
                      <span className="text-slate-300 font-mono text-[9px]">
                        [{hyperbolaFit.epsr_ci95[0].toFixed(2)}, {hyperbolaFit.epsr_ci95[1].toFixed(2)}]
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">Depth</span>
                      <span className="text-amber-300 font-mono">{hyperbolaFit.depth_m.toFixed(1)} m</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">SNR</span>
                      <span className="text-white font-mono">{hyperbolaFit.quality.snr.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-400">Support</span>
                      <span className="text-white font-mono">{hyperbolaFit.quality.support_traces} traces</span>
                    </div>

                    {/* Flags */}
                    <div className="flex flex-wrap gap-1 mt-1">
                      {hyperbolaFit.flags.map((flag) => (
                        <span
                          key={flag}
                          className={`px-1.5 py-0.5 text-[8px] font-bold rounded ${
                            flag === "ICE_CONSISTENT" ? "bg-green-900/40 text-green-300" :
                            flag.startsWith("CLUTTER_RISK_HIGH") ? "bg-red-900/40 text-red-300" :
                            flag.startsWith("CLUTTER_RISK_MED") ? "bg-amber-900/40 text-amber-300" :
                            flag === "LOW_SNR" ? "bg-orange-900/40 text-orange-300" :
                            "bg-slate-700/40 text-slate-300"
                          }`}
                        >
                          {flag}
                        </span>
                      ))}
                    </div>

                    {/* Overlay controls */}
                    <div className="flex gap-2 mt-1">
                      <label className="flex items-center gap-1 text-[9px] text-slate-400">
                        <input type="checkbox" checked={showHyperbolaCi}
                          onChange={(e) => setShowHyperbolaCi(e.target.checked)}
                          className="w-3 h-3" />
                        CI band
                      </label>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-[9px] text-slate-600">
                No apex set. Shift+Click on a diffraction hyperbola to begin.
              </div>
            )}
          </Section>

          {/* Analysis buttons */}
          {(onOpenRegolith || onOpenAttenuation) && (
            <Section title="Analysis">
              <div className="flex flex-col gap-2">
                {onOpenRegolith && (
                  <button
                    onClick={() => onOpenRegolith(productId)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-500/20 border border-indigo-500/30 rounded text-indigo-400 hover:bg-indigo-500/30 font-medium text-xs transition-colors"
                  >
                    <span className="material-symbols-outlined text-sm">layers</span>
                    Regolith Thickness
                  </button>
                )}
                {onOpenAttenuation && (
                  <button
                    onClick={() => onOpenAttenuation(productId)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-cyan-500/20 border border-cyan-500/30 rounded text-cyan-400 hover:bg-cyan-500/30 font-medium text-xs transition-colors"
                  >
                    <span className="material-symbols-outlined text-sm">radar</span>
                    Radar Attenuation
                  </button>
                )}
              </div>
            </Section>
          )}

          {/* Metadata */}
          {metadata && (
            <Section title="Dataset Info">
              <InfoRow label="Traces" value={metadata.rows.toLocaleString()} />
              <InfoRow label="Range bins" value={String(metadata.range_bins)} />
              <InfoRow label="Start Lon" value={`${metadata.start_lon.toFixed(4)}°`} />
              <InfoRow label="Start Lat" value={`${metadata.start_lat.toFixed(4)}°`} />
              <InfoRow label="Stop Lon" value={`${metadata.stop_lon.toFixed(4)}°`} />
              <InfoRow label="Stop Lat" value={`${metadata.stop_lat.toFixed(4)}°`} />
              <InfoRow label="Alt range" value={`${metadata.alt_range_km[0].toFixed(0)} – ${metadata.alt_range_km[1].toFixed(0)} km`} />
              <InfoRow label="Downsample" value={`${downsample}×`} />
            </Section>
          )}
        </div>

        {/* Main canvas area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Plots wrapper — overlay canvas sits on top for alignment marker */}
          <div ref={plotsWrapperRef} className="flex-1 flex flex-col relative overflow-hidden">
            {/* Alignment marker overlay (pointer-events: none so clicks pass through) */}
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 z-20 pointer-events-none"
              style={{ width: "100%", height: "100%" }}
            />

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
                  adjustMode ? "cursor-cell" : "cursor-grab"
                }`}
                draggable={false}
                onDragStart={(e) => e.preventDefault()}
                onMouseMove={handleMouseMove}
                onMouseDown={handleMouseDown}
                onMouseLeave={() => { setCursor(null); }}
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
                  className={`w-full h-full ${molaAlignMode ? "cursor-ew-resize" : "cursor-grab"}`}
                  draggable={false}
                  onDragStart={(e) => e.preventDefault()}
                  onMouseMove={handleMouseMove}
                  onMouseDown={handleMouseDown}
                  onMouseLeave={() => setCursor(null)}
                  onWheel={handleWheel}
                />
              </div>
            )}

          </div>

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
              {molaAlignMode && <span className="text-amber-400">MOLA ALIGN</span>}
              {molaXOffset !== 0 && <span className="text-amber-300">MOLA: {molaXOffset > 0 ? "+" : ""}{molaXOffset}</span>}
              {showClutter && <span className="text-orange-400">CLUTTER {Math.round(clutterOpacity * 100)}%</span>}
              {hyperbolaApex && <span className="text-red-400">HYPERBOLA T={hyperbolaApex.trace}</span>}
              {hyperbolaFit && <span className="text-red-300">εr={hyperbolaFit.epsr.toFixed(2)}</span>}
            </div>
          </div>
        </div>
      </div>
      )}
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
