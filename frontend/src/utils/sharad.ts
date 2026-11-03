/**
 * Pure SHARAD radargram helpers shared by the viewer (unit-tested).
 *
 * Trace indices: the viewer works in *display* traces (full-resolution trace
 * index / downsample). Every request for one product view uses the same
 * downsample factor, so image columns, surface points, MOLA samples and
 * depth/hyperbola queries all refer to the same trace axis.
 */

export const MAX_IMAGE_WIDTH = 16384; // safe browser canvas/image width
export const SPEED_OF_LIGHT = 299792458; // m/s
export const BIN_DT_SEC = 0.0375e-6; // 37.5 ns two-way per range bin
export const BIN_DT_US = 0.0375;
export const FREE_SPACE_RESOLUTION_M = 15; // c / (2B), B = 10 MHz
const MARS_MEAN_RADIUS_M = (2 * 3396190 + 3376200) / 3;

/** Single along-track downsample factor for a product with `rows` traces. */
export function computeDownsample(rows: number, maxWidth = MAX_IMAGE_WIDTH): number {
  if (!Number.isFinite(rows) || rows <= 0) return 1;
  return Math.max(1, Math.ceil(rows / maxWidth));
}

/** Great-circle distance on Mars (m), planetocentric lat/lon in degrees. */
export function haversineM(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const r = Math.PI / 180;
  const dLat = (lat2 - lat1) * r;
  const dLon = (lon2 - lon1) * r;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dLon / 2) ** 2;
  return 2 * MARS_MEAN_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Index of the trace closest to (lat, lon) and its distance in metres. */
export function nearestTrace(
  lats: number[], lons: number[], lat: number, lon: number,
): { index: number; distanceM: number } | null {
  let best = -1;
  let bestD = Infinity;
  const n = Math.min(lats.length, lons.length);
  for (let i = 0; i < n; i++) {
    const la = lats[i]!;
    const lo = lons[i]!;
    if (la == null || lo == null) continue;
    const d = haversineM(la, lo, lat, lon);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best < 0 ? null : { index: best, distanceM: bestD };
}

/** Depth below the surface (m) for a two-way time (s), piecewise two-layer. */
export function depthFromTwt(twtS: number, eps1: number, eps2 = eps1, boundaryM = 0): number {
  const t = Math.max(0, twtS);
  const v1 = SPEED_OF_LIGHT / Math.sqrt(eps1);
  if (boundaryM <= 0 || eps2 === eps1) return (v1 * t) / 2;
  const v2 = SPEED_OF_LIGHT / Math.sqrt(eps2);
  const t1 = (2 * boundaryM) / v1;
  return t <= t1 ? (v1 * t) / 2 : boundaryM + (v2 * (t - t1)) / 2;
}

/** Inverse of depthFromTwt: depth (m) → two-way time (s). */
export function twtFromDepth(depthM: number, eps1: number, eps2 = eps1, boundaryM = 0): number {
  const d = Math.max(0, depthM);
  const v1 = SPEED_OF_LIGHT / Math.sqrt(eps1);
  if (boundaryM <= 0 || eps2 === eps1 || d <= boundaryM) return (2 * d) / v1;
  const v2 = SPEED_OF_LIGHT / Math.sqrt(eps2);
  return (2 * boundaryM) / v1 + (2 * (d - boundaryM)) / v2;
}

/** "Nice" tick step (1/2/5 × 10^k) for roughly `target` ticks over `span`. */
export function niceStep(span: number, target = 5): number {
  if (!(span > 0)) return 1;
  const raw = span / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/** Build a CSV string (RFC-4180 quoting) from header + rows. */
export function toCsv(header: string[], rows: (string | number | null | undefined)[][]): string {
  const esc = (v: string | number | null | undefined) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header.map(esc).join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n") + "\n";
}

/** Read the most useful error text from a failed fetch Response. */
export async function describeHttpError(res: Response, what: string): Promise<string> {
  let detail = "";
  try {
    const text = await res.text();
    try {
      const j = JSON.parse(text);
      detail = j.detail ?? j.error ?? text;
      if (typeof detail !== "string") detail = JSON.stringify(detail);
    } catch {
      detail = text;
    }
  } catch { /* ignore */ }
  detail = (detail || res.statusText || "").slice(0, 300);
  return `${what} failed (HTTP ${res.status}${detail ? `: ${detail}` : ""})`;
}
