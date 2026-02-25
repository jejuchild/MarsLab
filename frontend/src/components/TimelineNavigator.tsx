/**
 * TimelineNavigator - Temporal visualization and filtering for Mars products.
 *
 * Shows a horizontal timeline bar at the bottom of the map with:
 * - Colored dots for each product, positioned by acquisition date
 * - Instrument legend with color coding
 * - Draggable handles to filter by time range
 * - Play/animate button for chronological scrubbing
 * - Mars solar longitude (Ls) secondary axis
 * - Collapsible (minimized = 4px bar, expanded = ~80px)
 *
 * Performance notes:
 * - SVG dots capped at MAX_RENDERED_DOTS via uniform sampling
 * - Drag updates are visual-only; parent notified only on mouseup
 * - Playback uses requestAnimationFrame with throttled parent callbacks
 * - All date-to-pixel computations memoized
 */

import { useState, useRef, useCallback, useEffect, useMemo, memo } from "react";
import { getInstrumentColor } from "../config/instrumentRegistry";
import { earthToLs, earthToMarsYear, formatMarsDateRange, getLsTicksInRange } from "../utils/marsTime";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface TimelineProduct {
  productId: string;
  instrument: string;
  acquisitionDate: Date;
  label: string;
}

export interface TimelineNavigatorProps {
  products: TimelineProduct[];
  onTimeRangeChange: (start: Date | null, end: Date | null) => void;
  isExpanded: boolean;
  onToggleExpand: () => void;
}

// ─── Constants ────────────────────────────────────────────────────────────────

/** Horizontal padding inside the SVG timeline area (px) */
const TRACK_PAD_LEFT = 48;
const TRACK_PAD_RIGHT = 16;

/** Handle dimensions */
const HANDLE_WIDTH = 8;
const HANDLE_HEIGHT = 32;

/** Dot radius for product markers */
const DOT_RADIUS = 3;

/** Playback: how many Earth-milliseconds advance per real second */
const PLAYBACK_RATE = 30 * 86400000; // 30 Earth days per second

/** Minimum time range span (1 day) to prevent zero-width selections */
const MIN_RANGE_MS = 86400000;

/** Collapsed bar height */
const COLLAPSED_HEIGHT = 4;

/** Expanded content height */
const EXPANDED_HEIGHT = 84;

/** Maximum number of product dots to render (performance guard) */
const MAX_RENDERED_DOTS = 2000;

/** Minimum interval (ms) between parent range-change callbacks during playback */
const PLAYBACK_EMIT_THROTTLE_MS = 100;

/** Instrument display order and colors for the legend */
const LEGEND_INSTRUMENTS = [
  { key: "crism",         label: "CRISM",      color: "#00FFFF" },
  { key: "hirise",        label: "HiRISE",     color: "#FFFF00" },
  { key: "sharad",        label: "SHARAD",     color: "#FFA500" },
  { key: "ctx",           label: "CTX",        color: "#FF69B4" },
  { key: "hirise_dtm",    label: "DTM",        color: "#8B4513" },
  { key: "sharad_highres",label: "SHARAD HR",  color: "#FFD700" },
  { key: "crism_trr3",    label: "TRR3",       color: "#00CED1" },
] as const;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Map instrument name (case-insensitive) to its hex color */
function instrumentColor(instrument: string): string {
  return getInstrumentColor(instrument) || "#FFFFFF";
}

/** Clamp a value between min and max */
function clamp(val: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, val));
}

/** Format a date for axis labels: "2008 Jul" */
function formatAxisDate(date: Date): string {
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${date.getUTCFullYear()} ${months[date.getUTCMonth()]}`;
}

/** Compute nice axis tick dates spanning a range */
function computeDateTicks(start: Date, end: Date, maxTicks: number): Date[] {
  const rangeMs = end.getTime() - start.getTime();
  if (rangeMs <= 0) return [];

  const rangeDays = rangeMs / 86400000;
  let stepDays: number;

  if (rangeDays < 30)   stepDays = 7;    // weekly
  else if (rangeDays < 120)  stepDays = 14;   // bi-weekly
  else if (rangeDays < 365)  stepDays = 30;   // monthly
  else if (rangeDays < 730)  stepDays = 60;   // bi-monthly
  else if (rangeDays < 1825) stepDays = 182;  // semi-annual
  else if (rangeDays < 3650) stepDays = 365;  // annual
  else stepDays = 730;                        // bi-annual

  const ticks: Date[] = [];
  const firstTick = new Date(start);
  firstTick.setUTCDate(1);
  firstTick.setUTCHours(0, 0, 0, 0);
  if (stepDays >= 365) {
    firstTick.setUTCMonth(0);
  }

  let current = firstTick.getTime();
  const endMs = end.getTime();
  const startMs = start.getTime();

  while (current <= endMs && ticks.length < maxTicks) {
    if (current >= startMs) {
      ticks.push(new Date(current));
    }
    current += stepDays * 86400000;
  }

  // Fallback: if no ticks generated (narrow range), place ticks uniformly
  if (ticks.length === 0 && rangeMs > 0) {
    const count = Math.min(maxTicks, Math.max(2, Math.floor(rangeDays / 3)));
    const step = rangeMs / (count + 1);
    for (let i = 1; i <= count; i++) {
      ticks.push(new Date(startMs + step * i));
    }
  }

  return ticks;
}

// ─── Component ────────────────────────────────────────────────────────────────

function TimelineNavigatorRaw({
  products,
  onTimeRangeChange,
  isExpanded,
  onToggleExpand,
}: TimelineNavigatorProps) {
  // ── Refs ──
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  // Stable ref for the range-change callback (avoids recreating playback closure)
  const onTimeRangeChangeRef = useRef(onTimeRangeChange);
  onTimeRangeChangeRef.current = onTimeRangeChange;

  // ── State ──
  const [containerWidth, setContainerWidth] = useState(800);
  const [isPlaying, setIsPlaying] = useState(false);

  // Range selection: null means "no filter" (show all)
  const [rangeStart, setRangeStart] = useState<Date | null>(null);
  const [rangeEnd, setRangeEnd] = useState<Date | null>(null);

  // Dragging state
  const [dragging, setDragging] = useState<"start" | "end" | "range" | null>(null);
  const dragStartXRef = useRef(0);
  const dragStartValRef = useRef<{ start: number; end: number }>({ start: 0, end: 0 });

  // Tooltip
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    product: TimelineProduct;
  } | null>(null);

  // ── Derived: time domain ──
  const { domainStart, domainEnd, validProducts } = useMemo(() => {
    const valid = products.filter(
      (p) => p.acquisitionDate instanceof Date && !isNaN(p.acquisitionDate.getTime())
    );

    if (valid.length === 0) {
      const now = new Date();
      return {
        domainStart: new Date(now.getTime() - 365 * 86400000),
        domainEnd: now,
        validProducts: [] as TimelineProduct[],
      };
    }

    let minMs = Infinity;
    let maxMs = -Infinity;
    for (const p of valid) {
      const t = p.acquisitionDate.getTime();
      if (t < minMs) minMs = t;
      if (t > maxMs) maxMs = t;
    }

    // Add 2% padding on each side (minimum 1 day)
    const padding = Math.max((maxMs - minMs) * 0.02, 86400000);
    return {
      domainStart: new Date(minMs - padding),
      domainEnd: new Date(maxMs + padding),
      validProducts: valid,
    };
  }, [products]);

  // Keep domain in refs so playback can read latest values without closure staleness
  const domainStartRef = useRef(domainStart);
  const domainEndRef = useRef(domainEnd);
  domainStartRef.current = domainStart;
  domainEndRef.current = domainEnd;

  // ── Track geometry ──
  const trackLeft = TRACK_PAD_LEFT;
  const trackRight = containerWidth - TRACK_PAD_RIGHT;
  const trackWidth = Math.max(trackRight - trackLeft, 1);

  // ── Scale functions ──
  const dateToX = useCallback(
    (date: Date): number => {
      const domainMs = domainEnd.getTime() - domainStart.getTime();
      if (domainMs <= 0) return trackLeft;
      const frac = (date.getTime() - domainStart.getTime()) / domainMs;
      return trackLeft + frac * trackWidth;
    },
    [domainStart, domainEnd, trackLeft, trackWidth]
  );

  const xToDate = useCallback(
    (x: number): Date => {
      const domainMs = domainEnd.getTime() - domainStart.getTime();
      const frac = clamp((x - trackLeft) / trackWidth, 0, 1);
      return new Date(domainStart.getTime() + frac * domainMs);
    },
    [domainStart, domainEnd, trackLeft, trackWidth]
  );

  // ── Resize observer ──
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // ── Drag handlers ──
  const handleMouseDown = useCallback(
    (handle: "start" | "end" | "range", e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();

      // Initialize range if not set
      if (rangeStart === null || rangeEnd === null) {
        const s = domainStart;
        const en = domainEnd;
        setRangeStart(s);
        setRangeEnd(en);
        dragStartValRef.current = { start: s.getTime(), end: en.getTime() };
      } else {
        dragStartValRef.current = {
          start: rangeStart.getTime(),
          end: rangeEnd.getTime(),
        };
      }

      dragStartXRef.current = e.clientX;
      setDragging(handle);
    },
    [rangeStart, rangeEnd, domainStart, domainEnd]
  );

  useEffect(() => {
    if (!dragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const dx = e.clientX - dragStartXRef.current;
      const domainMs = domainEnd.getTime() - domainStart.getTime();
      const dtMs = (dx / trackWidth) * domainMs;

      if (dragging === "start") {
        const newStart = new Date(
          clamp(
            dragStartValRef.current.start + dtMs,
            domainStart.getTime(),
            (rangeEnd?.getTime() ?? domainEnd.getTime()) - MIN_RANGE_MS
          )
        );
        setRangeStart(newStart);
      } else if (dragging === "end") {
        const newEnd = new Date(
          clamp(
            dragStartValRef.current.end + dtMs,
            (rangeStart?.getTime() ?? domainStart.getTime()) + MIN_RANGE_MS,
            domainEnd.getTime()
          )
        );
        setRangeEnd(newEnd);
      } else if (dragging === "range") {
        const origSpan =
          dragStartValRef.current.end - dragStartValRef.current.start;
        let newStart = dragStartValRef.current.start + dtMs;
        let newEnd = dragStartValRef.current.end + dtMs;

        // Clamp to domain
        if (newStart < domainStart.getTime()) {
          newStart = domainStart.getTime();
          newEnd = newStart + origSpan;
        }
        if (newEnd > domainEnd.getTime()) {
          newEnd = domainEnd.getTime();
          newStart = newEnd - origSpan;
        }

        setRangeStart(new Date(newStart));
        setRangeEnd(new Date(newEnd));
      }
    };

    const handleMouseUp = () => {
      setDragging(null);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragging, domainStart, domainEnd, trackWidth, rangeStart, rangeEnd]);

  // Emit range changes when drag ends (not during -- avoids expensive parent re-renders)
  useEffect(() => {
    if (dragging === null && rangeStart !== null && rangeEnd !== null) {
      onTimeRangeChangeRef.current(rangeStart, rangeEnd);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging]);

  // ── Playback via requestAnimationFrame ──
  const stopPlayback = useCallback(() => {
    setIsPlaying(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
  }, []);

  const startPlayback = useCallback(() => {
    if (validProducts.length === 0) return;

    stopPlayback();
    setIsPlaying(true);

    const ds = domainStartRef.current;
    const de = domainEndRef.current;
    const totalMs = de.getTime() - ds.getTime();
    const windowMs = totalMs * 0.1; // 10% window
    let currentStartMs = ds.getTime();
    let lastEmitTime = 0;
    let lastFrameTime = 0;

    const tick = (timestamp: number) => {
      if (lastFrameTime === 0) lastFrameTime = timestamp;
      const elapsed = timestamp - lastFrameTime;
      lastFrameTime = timestamp;

      // Advance by elapsed real-time scaled to playback rate
      currentStartMs += (elapsed / 1000) * PLAYBACK_RATE;

      // Read latest domain from refs (handles product changes during playback)
      const domEnd = domainEndRef.current.getTime();
      const domStart = domainStartRef.current.getTime();

      if (currentStartMs > domEnd) {
        currentStartMs = domStart;
      }

      const s = new Date(currentStartMs);
      const e = new Date(Math.min(currentStartMs + windowMs, domEnd));
      setRangeStart(s);
      setRangeEnd(e);

      // Throttle parent callbacks
      if (timestamp - lastEmitTime > PLAYBACK_EMIT_THROTTLE_MS) {
        lastEmitTime = timestamp;
        onTimeRangeChangeRef.current(s, e);
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
  }, [validProducts.length, stopPlayback]);

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  }, [isPlaying, stopPlayback, startPlayback]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  // ── Clear range ──
  const clearRange = useCallback(() => {
    stopPlayback();
    setRangeStart(null);
    setRangeEnd(null);
    onTimeRangeChangeRef.current(null, null);
  }, [stopPlayback]);

  // ── Compute rendered dots (capped for performance) ──
  const renderedDots = useMemo(() => {
    if (validProducts.length <= MAX_RENDERED_DOTS) {
      return validProducts;
    }
    // Uniform sampling to stay under the cap
    const step = validProducts.length / MAX_RENDERED_DOTS;
    const sampled: TimelineProduct[] = [];
    for (let i = 0; i < MAX_RENDERED_DOTS; i++) {
      sampled.push(validProducts[Math.floor(i * step)]!);
    }
    return sampled;
  }, [validProducts]);

  // ── Date axis ticks ──
  const dateTicks = useMemo(
    () => computeDateTicks(domainStart, domainEnd, 8),
    [domainStart, domainEnd]
  );

  // ── Ls axis ticks ──
  const lsTicks = useMemo(
    () => getLsTicksInRange(domainStart, domainEnd, 8),
    [domainStart, domainEnd]
  );

  // ── Range display text ──
  const rangeText = useMemo(() => {
    if (rangeStart && rangeEnd) {
      return formatMarsDateRange(rangeStart, rangeEnd);
    }
    if (validProducts.length > 0) {
      return formatMarsDateRange(domainStart, domainEnd);
    }
    return "No products loaded";
  }, [rangeStart, rangeEnd, validProducts, domainStart, domainEnd]);

  // ── Count products in range ──
  const productsInRange = useMemo(() => {
    if (!rangeStart || !rangeEnd) return validProducts.length;
    const s = rangeStart.getTime();
    const e = rangeEnd.getTime();
    let count = 0;
    for (const p of validProducts) {
      const t = p.acquisitionDate.getTime();
      if (t >= s && t <= e) count++;
    }
    return count;
  }, [validProducts, rangeStart, rangeEnd]);

  // ── Instrument counts for legend badges ──
  const instrumentCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const p of validProducts) {
      const key = p.instrument.toLowerCase();
      counts[key] = (counts[key] || 0) + 1;
    }
    return counts;
  }, [validProducts]);

  // ── SVG dimensions ──
  const svgHeight = 56;
  const dotY = 16;
  const axisY = 36;
  const lsAxisY = 48;

  // ── Handle positions ──
  const handleStartX = rangeStart ? dateToX(rangeStart) : trackLeft;
  const handleEndX = rangeEnd ? dateToX(rangeEnd) : trackRight;

  // ── Tooltip handlers ──
  const handleDotEnter = useCallback(
    (product: TimelineProduct, e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      setTooltip({
        x: clamp(e.clientX - rect.left, 80, rect.width - 80),
        y: e.clientY - rect.top,
        product,
      });
    },
    []
  );

  const handleDotLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  // ── Track click to position range ──
  const handleTrackClick = useCallback(
    (e: React.MouseEvent<SVGRectElement>) => {
      if (dragging) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const clickedDate = xToDate(x);

      // Create a 10% window centered on click
      const windowMs = (domainEnd.getTime() - domainStart.getTime()) * 0.1;
      const newStart = new Date(
        Math.max(clickedDate.getTime() - windowMs / 2, domainStart.getTime())
      );
      const newEnd = new Date(
        Math.min(clickedDate.getTime() + windowMs / 2, domainEnd.getTime())
      );

      setRangeStart(newStart);
      setRangeEnd(newEnd);
      onTimeRangeChangeRef.current(newStart, newEnd);
    },
    [dragging, xToDate, domainStart, domainEnd]
  );

  // ── Render ──
  if (!isExpanded) {
    // Collapsed: thin clickable bar
    return (
      <div
        className="fixed bottom-0 left-0 right-0 z-40 cursor-pointer"
        style={{ height: COLLAPSED_HEIGHT }}
        onClick={onToggleExpand}
        role="button"
        aria-label="Expand timeline"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onToggleExpand(); }}
      >
        <div
          className="h-full w-full"
          style={{
            background: "linear-gradient(90deg, #00FFFF33, #FFFF0033, #FFA50033, #FF69B433)",
          }}
        />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="fixed bottom-0 left-0 right-0 z-40 select-none"
      style={{ height: EXPANDED_HEIGHT }}
      role="region"
      aria-label="Timeline navigator"
    >
      {/* Semi-transparent background */}
      <div
        className="absolute inset-0"
        style={{
          background: "linear-gradient(180deg, rgba(10,14,23,0.88) 0%, rgba(10,14,23,0.96) 100%)",
          backdropFilter: "blur(8px)",
          borderTop: "1px solid rgba(45,58,84,0.6)",
        }}
      />

      {/* Content */}
      <div className="relative h-full flex flex-col">
        {/* Top bar: controls + legend + range text */}
        <div className="flex items-center gap-2 px-3 py-1" style={{ height: 24 }}>
          {/* Collapse button */}
          <button
            onClick={onToggleExpand}
            className="flex items-center justify-center w-5 h-5 rounded hover:bg-white/10 transition-colors"
            title="Collapse timeline"
            aria-label="Collapse timeline"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 14, color: "#94a3b8" }}
            >
              expand_more
            </span>
          </button>

          {/* Play/Pause */}
          <button
            onClick={togglePlayback}
            className="flex items-center justify-center w-5 h-5 rounded hover:bg-white/10 transition-colors"
            title={isPlaying ? "Pause" : "Play animation"}
            aria-label={isPlaying ? "Pause animation" : "Play animation"}
            disabled={validProducts.length === 0}
          >
            <span
              className="material-symbols-outlined"
              style={{
                fontSize: 14,
                color: validProducts.length === 0 ? "#4a5568" : isPlaying ? "#22c55e" : "#94a3b8",
              }}
            >
              {isPlaying ? "pause" : "play_arrow"}
            </span>
          </button>

          {/* Clear filter */}
          {rangeStart !== null && (
            <button
              onClick={clearRange}
              className="flex items-center justify-center w-5 h-5 rounded hover:bg-white/10 transition-colors"
              title="Clear time filter"
              aria-label="Clear time filter"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 14, color: "#ef4444" }}
              >
                close
              </span>
            </button>
          )}

          {/* Range text */}
          <span
            className="text-xs font-medium ml-1 truncate"
            style={{ color: "#94a3b8", maxWidth: 340, fontFamily: "'Space Grotesk', monospace" }}
            aria-live="polite"
          >
            {rangeText}
          </span>

          {/* Product count */}
          <span
            className="text-xs ml-auto shrink-0"
            style={{ color: "#64748b" }}
          >
            {productsInRange}/{validProducts.length} products
          </span>

          {/* Legend */}
          <div className="flex items-center gap-2 ml-2 shrink-0" role="list" aria-label="Instrument legend">
            {LEGEND_INSTRUMENTS.map(({ key, label, color }) => {
              const count = instrumentCounts[key] || 0;
              if (count === 0) return null;
              return (
                <div key={key} className="flex items-center gap-1" role="listitem">
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ backgroundColor: color }}
                    aria-hidden="true"
                  />
                  <span className="text-[10px]" style={{ color: "#64748b" }}>
                    {label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* SVG Timeline */}
        <svg
          ref={svgRef}
          width={containerWidth}
          height={svgHeight}
          className="flex-1"
          style={{ overflow: "visible" }}
          aria-hidden="true"
        >
          {/* Track background (clickable) */}
          <rect
            x={trackLeft}
            y={dotY - 8}
            width={trackWidth}
            height={16}
            fill="transparent"
            className="cursor-crosshair"
            onClick={handleTrackClick}
          />

          {/* Track line */}
          <line
            x1={trackLeft}
            y1={dotY}
            x2={trackRight}
            y2={dotY}
            stroke="#2d3a54"
            strokeWidth={1}
          />

          {/* Selected range highlight */}
          {rangeStart !== null && rangeEnd !== null && (
            <rect
              x={handleStartX}
              y={dotY - 8}
              width={Math.max(handleEndX - handleStartX, 0)}
              height={16}
              fill="rgba(19,91,236,0.15)"
              stroke="rgba(19,91,236,0.3)"
              strokeWidth={0.5}
              className="cursor-grab"
              onMouseDown={(e) => handleMouseDown("range", e)}
              style={{ cursor: dragging === "range" ? "grabbing" : "grab" }}
            />
          )}

          {/* Product dots */}
          {renderedDots.map((product) => {
            const x = dateToX(product.acquisitionDate);
            const color = instrumentColor(product.instrument);
            const inRange =
              !rangeStart ||
              !rangeEnd ||
              (product.acquisitionDate.getTime() >= rangeStart.getTime() &&
                product.acquisitionDate.getTime() <= rangeEnd.getTime());

            return (
              <circle
                key={product.productId}
                cx={x}
                cy={dotY}
                r={DOT_RADIUS}
                fill={color}
                opacity={inRange ? 0.85 : 0.15}
                className="cursor-pointer"
                onMouseEnter={(e) => handleDotEnter(product, e)}
                onMouseLeave={handleDotLeave}
              />
            );
          })}

          {/* Drag handles — rendered after dots so they sit on top */}
          {rangeStart !== null && rangeEnd !== null && (
            <>
              {/* Start handle */}
              <rect
                x={handleStartX - HANDLE_WIDTH / 2}
                y={dotY - HANDLE_HEIGHT / 2}
                width={HANDLE_WIDTH}
                height={HANDLE_HEIGHT}
                rx={2}
                fill="#135bec"
                stroke="#2563eb"
                strokeWidth={1}
                className="cursor-ew-resize"
                onMouseDown={(e) => handleMouseDown("start", e)}
              />
              <line
                x1={handleStartX - 1}
                y1={dotY - 4}
                x2={handleStartX - 1}
                y2={dotY + 4}
                stroke="white"
                strokeWidth={0.5}
                opacity={0.6}
                pointerEvents="none"
              />
              <line
                x1={handleStartX + 1}
                y1={dotY - 4}
                x2={handleStartX + 1}
                y2={dotY + 4}
                stroke="white"
                strokeWidth={0.5}
                opacity={0.6}
                pointerEvents="none"
              />

              {/* End handle */}
              <rect
                x={handleEndX - HANDLE_WIDTH / 2}
                y={dotY - HANDLE_HEIGHT / 2}
                width={HANDLE_WIDTH}
                height={HANDLE_HEIGHT}
                rx={2}
                fill="#135bec"
                stroke="#2563eb"
                strokeWidth={1}
                className="cursor-ew-resize"
                onMouseDown={(e) => handleMouseDown("end", e)}
              />
              <line
                x1={handleEndX - 1}
                y1={dotY - 4}
                x2={handleEndX - 1}
                y2={dotY + 4}
                stroke="white"
                strokeWidth={0.5}
                opacity={0.6}
                pointerEvents="none"
              />
              <line
                x1={handleEndX + 1}
                y1={dotY - 4}
                x2={handleEndX + 1}
                y2={dotY + 4}
                stroke="white"
                strokeWidth={0.5}
                opacity={0.6}
                pointerEvents="none"
              />
            </>
          )}

          {/* Earth date axis ticks */}
          {dateTicks.map((date, i) => {
            const x = dateToX(date);
            if (x < trackLeft + 10 || x > trackRight - 10) return null;
            return (
              <g key={`dt-${i}`}>
                <line
                  x1={x}
                  y1={dotY + 8}
                  x2={x}
                  y2={axisY - 2}
                  stroke="#2d3a54"
                  strokeWidth={0.5}
                />
                <text
                  x={x}
                  y={axisY + 1}
                  textAnchor="middle"
                  fill="#64748b"
                  fontSize={9}
                  fontFamily="'Space Grotesk', monospace"
                >
                  {formatAxisDate(date)}
                </text>
              </g>
            );
          })}

          {/* Ls secondary axis ticks */}
          {lsTicks.map((tick, i) => {
            const x = dateToX(tick.date);
            if (x < trackLeft + 10 || x > trackRight - 10) return null;
            return (
              <text
                key={`ls-${i}`}
                x={x}
                y={lsAxisY + 2}
                textAnchor="middle"
                fill="#4a5568"
                fontSize={8}
                fontFamily="'Space Grotesk', monospace"
              >
                Ls {tick.ls}
              </text>
            );
          })}

          {/* Left axis label */}
          <text
            x={4}
            y={dotY + 1}
            fill="#4a5568"
            fontSize={8}
            fontFamily="'Space Grotesk', monospace"
            dominantBaseline="middle"
          >
            Time
          </text>
        </svg>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none z-50"
          style={{
            left: tooltip.x,
            top: tooltip.y - 52,
            transform: "translateX(-50%)",
          }}
        >
          <div
            className="rounded px-2 py-1 text-xs whitespace-nowrap"
            style={{
              background: "rgba(22,30,45,0.95)",
              border: "1px solid rgba(45,58,84,0.8)",
              color: "#f1f5f9",
              fontFamily: "'Space Grotesk', monospace",
              boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
            }}
          >
            <span style={{ color: instrumentColor(tooltip.product.instrument) }}>
              {tooltip.product.instrument.toUpperCase()}
            </span>
            {" \u2014 "}
            <span>{tooltip.product.label || tooltip.product.productId}</span>
            <br />
            <span style={{ color: "#94a3b8" }}>
              {tooltip.product.acquisitionDate.toISOString().slice(0, 10)}
              {" \u2022 "}
              MY{Math.floor(earthToMarsYear(tooltip.product.acquisitionDate))}
              {" Ls "}
              {Math.round(earthToLs(tooltip.product.acquisitionDate))}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

const TimelineNavigator = memo(TimelineNavigatorRaw);
export default TimelineNavigator;
