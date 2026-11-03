/**
 * Axis / scale-bar / colour-bar overlay for the SHARAD radargram canvas.
 *
 *  - left:   two-way time inside the receive window (µs) — 37.5 ns per bin
 *  - right:  depth below the surface pick at the view-centre trace (m) for
 *            the chosen (layered) εr
 *  - bottom: along-track distance (km) + horizontal scale bar
 *  - top-right: colour bar (dB, uncalibrated relative power, or "relative")
 */
import { BIN_DT_SEC, BIN_DT_US, niceStep, twtFromDepth } from "../../utils/sharad";

export type ValueScale = {
  unit: string;
  vmin: number | null;
  vmax: number | null;
  per_trace?: boolean;
  log?: boolean;
};

export type AxesParams = {
  W: number;
  H: number;
  viewX: { start: number; end: number };
  viewY: { start: number; end: number };
  nTraces: number;
  nBins: number;
  distanceKm?: number[];
  /** Surface bin at the centre trace of the view (null → no depth axis). */
  surfaceBin: number | null;
  eps1: number;
  eps2: number;
  boundaryM: number;
  valueScale?: ValueScale | null;
};

const AXIS_BG = "rgba(10, 15, 24, 0.72)";
const AXIS_FG = "#cbd5e1";
const TICK_FG = "#94a3b8";

export function drawRadargramAxes(ctx: CanvasRenderingContext2D, p: AxesParams): void {
  const { W, H, viewX, viewY, nTraces, nBins } = p;
  const binToY = (b: number) => ((b / nBins - viewY.start) / (viewY.end - viewY.start)) * H;
  const yToBin = (y: number) => (viewY.start + (y / H) * (viewY.end - viewY.start)) * nBins;
  const traceToX = (t: number) => ((t / nTraces - viewX.start) / (viewX.end - viewX.start)) * W;

  ctx.save();
  ctx.font = "9px monospace";
  ctx.lineWidth = 1;

  // ── Left: two-way time in window (µs) ──
  const b0 = yToBin(0), b1 = yToBin(H);
  const t0 = b0 * BIN_DT_US, t1 = b1 * BIN_DT_US;
  const tStep = niceStep(t1 - t0, Math.max(3, Math.floor(H / 60)));
  ctx.fillStyle = AXIS_BG;
  ctx.fillRect(0, 0, 34, H);
  ctx.fillStyle = TICK_FG;
  ctx.strokeStyle = TICK_FG;
  ctx.textAlign = "left";
  for (let t = Math.ceil(t0 / tStep) * tStep; t <= t1; t += tStep) {
    const y = binToY(t / BIN_DT_US);
    ctx.beginPath(); ctx.moveTo(28, y); ctx.lineTo(34, y); ctx.stroke();
    ctx.fillText(formatNum(t, tStep), 2, y + 3);
  }
  ctx.save();
  ctx.translate(9, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = AXIS_FG;
  ctx.fillText("TWT in window (µs)", 0, 0);
  ctx.restore();

  // ── Right: depth below surface at centre trace ──
  if (p.surfaceBin != null && p.surfaceBin >= 0) {
    const sBin = p.surfaceBin;
    const depthAtBin = (b: number) => {
      // invert numerically via twtFromDepth monotonicity (two-layer safe)
      const t = (b - sBin) * BIN_DT_SEC;
      if (t <= 0) return 0;
      let lo = 0, hi = 1e5;
      for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        if (twtFromDepth(mid, p.eps1, p.eps2, p.boundaryM) < t) lo = mid; else hi = mid;
      }
      return (lo + hi) / 2;
    };
    const dMax = depthAtBin(b1);
    if (dMax > 0) {
      ctx.fillStyle = AXIS_BG;
      ctx.fillRect(W - 40, 0, 40, H);
      const dStep = niceStep(dMax, Math.max(3, Math.floor(H / 60)));
      ctx.fillStyle = TICK_FG;
      ctx.strokeStyle = TICK_FG;
      ctx.textAlign = "right";
      for (let d = 0; d <= dMax; d += dStep) {
        const y = binToY(sBin + twtFromDepth(d, p.eps1, p.eps2, p.boundaryM) / BIN_DT_SEC);
        if (y < 0 || y > H) continue;
        ctx.beginPath(); ctx.moveTo(W - 40, y); ctx.lineTo(W - 34, y); ctx.stroke();
        ctx.fillText(formatNum(d, dStep), W - 2, y + 3);
      }
      ctx.save();
      ctx.translate(W - 3, H / 2);
      ctx.rotate(Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = AXIS_FG;
      const epsLabel = p.boundaryM > 0 && p.eps2 !== p.eps1
        ? `εr ${p.eps1}/${p.eps2} @${p.boundaryM} m`
        : `εr ${p.eps1}`;
      ctx.fillText(`Depth (m), ${epsLabel}, below centre-trace surface`, 0, 0);
      ctx.restore();
    }
  }

  // ── Bottom: along-track km + scale bar ──
  const dk = p.distanceKm;
  if (dk && dk.length > 1) {
    const iA = Math.max(0, Math.floor(viewX.start * nTraces));
    const iB = Math.min(dk.length - 1, Math.ceil(viewX.end * nTraces) - 1);
    const kmA = dk[iA] ?? 0, kmB = dk[iB] ?? 0;
    const span = kmB - kmA;
    if (span > 0) {
      ctx.fillStyle = AXIS_BG;
      ctx.fillRect(34, H - 14, W - 74, 14);
      const step = niceStep(span, Math.max(3, Math.floor(W / 110)));
      ctx.fillStyle = TICK_FG;
      ctx.strokeStyle = TICK_FG;
      ctx.textAlign = "center";
      // ticks: find trace index for each km value (distance is monotonic)
      let j = iA;
      for (let km = Math.ceil(kmA / step) * step; km <= kmB; km += step) {
        while (j < iB && (dk[j] ?? 0) < km) j++;
        const x = traceToX(j);
        if (x < 40 || x > W - 46) continue;
        ctx.beginPath(); ctx.moveTo(x, H - 14); ctx.lineTo(x, H - 10); ctx.stroke();
        ctx.fillText(`${formatNum(km, step)}`, x, H - 2);
      }
      ctx.textAlign = "left";
      ctx.fillStyle = AXIS_FG;
      ctx.fillText("km", W - 70, H - 2);

      // scale bar (~1/5 of the width)
      const pxPerKm = W / span;
      const barKm = niceStep(span / 5, 1);
      const barPx = barKm * pxPerKm;
      const bx = 44, by = H - 22;
      ctx.fillStyle = AXIS_BG;
      ctx.fillRect(bx - 4, by - 12, barPx + 8 + 40, 16);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + barPx, by); ctx.stroke();
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(bx, by - 4); ctx.lineTo(bx, by + 2);
      ctx.moveTo(bx + barPx, by - 4); ctx.lineTo(bx + barPx, by + 2); ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(`${formatNum(barKm, barKm)} km`, bx + barPx + 4, by + 3);
    }
  }

  // ── Colour bar ──
  const vs = p.valueScale;
  if (vs) {
    const cw = 90, ch = 8, cx = W - 40 - cw - 8, cy = 6;
    ctx.fillStyle = AXIS_BG;
    ctx.fillRect(cx - 4, cy - 3, cw + 8, ch + 26);
    const g = ctx.createLinearGradient(cx, 0, cx + cw, 0);
    g.addColorStop(0, "#000000");
    g.addColorStop(1, "#ffffff");
    ctx.fillStyle = g;
    ctx.fillRect(cx, cy, cw, ch);
    ctx.fillStyle = AXIS_FG;
    ctx.font = "8px monospace";
    if (vs.vmin != null && vs.vmax != null) {
      ctx.textAlign = "left";
      ctx.fillText(formatScale(vs.vmin), cx, cy + ch + 9);
      ctx.textAlign = "right";
      ctx.fillText(formatScale(vs.vmax), cx + cw, cy + ch + 9);
    }
    ctx.textAlign = "center";
    ctx.fillText(shortUnit(vs.unit), cx + cw / 2, cy + ch + 19);
  }
  ctx.restore();
}

function formatNum(v: number, step: number): string {
  const decimals = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)));
  return v.toFixed(decimals);
}

function formatScale(v: number): string {
  return Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(1);
}

function shortUnit(u: string): string {
  if (u.startsWith("dB")) return "dB, uncalibrated rel. power";
  if (u.startsWith("relative")) return "per-trace stretch (relative)";
  return u.replace(" (uncalibrated, linear)", ", linear (uncal.)");
}
