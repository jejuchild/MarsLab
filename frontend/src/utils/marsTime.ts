/**
 * Mars Time Utilities
 *
 * Converts between Earth dates and Mars calendar coordinates (Mars Year + Ls).
 *
 * References:
 * - Mars Year 1 starts April 11, 1955 (Ls=0), per Clancy et al. (2000)
 * - 1 Mars Year = 686.9713 Earth days (tropical year)
 * - Ls = areocentric solar longitude (0=N. vernal equinox, 90=N. summer solstice)
 *
 * The Ls calculation uses the Allison & McEwen (2000) algorithm adapted
 * for a simplified orbital model. Mars eccentricity (0.0934) causes Ls
 * to advance non-uniformly; perihelion occurs near Ls~251.
 */

/** J2000 epoch: January 1, 2000, 12:00 TT */
const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);

/** Mars Year 1 epoch: April 11, 1955, 00:00 UTC (Ls=0) */
const MY1_EPOCH_MS = Date.UTC(1955, 3, 11, 0, 0, 0);

/** Mean Mars tropical year in Earth days */
const MARS_YEAR_DAYS = 686.9713;

/** Mars orbital eccentricity */
const ECCENTRICITY = 0.0934;

/** Areocentric longitude of perihelion (degrees) */
const PERIHELION_LS = 251.0;

/** Degrees to radians */
const DEG2RAD = Math.PI / 180;

/** Radians to degrees */
const RAD2DEG = 180 / Math.PI;

/**
 * Days since J2000.0 for a given Date.
 */
function daysSinceJ2000(date: Date): number {
  return (date.getTime() - J2000_MS) / 86400000;
}

/**
 * Convert Earth date to fractional Mars Year.
 *
 * MY1 starts April 11, 1955. Each Mars Year is 686.9713 Earth days.
 * Returns e.g. 29.42 meaning Mars Year 29, ~42% through the year.
 */
export function earthToMarsYear(date: Date): number {
  const daysSinceEpoch = (date.getTime() - MY1_EPOCH_MS) / 86400000;
  return 1 + daysSinceEpoch / MARS_YEAR_DAYS;
}

/**
 * Compute areocentric solar longitude (Ls) from an Earth date.
 *
 * Uses the method from Allison & McEwen (2000):
 * 1. Compute mean anomaly from days since J2000
 * 2. Solve Kepler's equation iteratively for eccentric anomaly
 * 3. Compute true anomaly
 * 4. Convert to Ls via perihelion longitude offset
 *
 * Accuracy: within ~0.5 degrees of precise ephemeris calculations.
 */
export function earthToLs(date: Date): number {
  const d = daysSinceJ2000(date);

  // Mean anomaly (degrees) - linear advance from J2000
  // At J2000, Mars mean anomaly was ~19.387 degrees
  // Mean motion = 360 / 686.9713 = 0.52403 deg/day
  const M_deg = (19.3871 + 0.52402075 * d) % 360;
  const M = ((M_deg % 360) + 360) % 360; // Normalize to [0, 360)
  const M_rad = M * DEG2RAD;

  // Solve Kepler's equation: E - e*sin(E) = M
  // Newton-Raphson iteration
  let E = M_rad; // Initial guess
  for (let i = 0; i < 10; i++) {
    const dE = (M_rad - (E - ECCENTRICITY * Math.sin(E))) / (1 - ECCENTRICITY * Math.cos(E));
    E += dE;
    if (Math.abs(dE) < 1e-10) break;
  }

  // True anomaly from eccentric anomaly
  const sinNu = Math.sqrt(1 - ECCENTRICITY * ECCENTRICITY) * Math.sin(E) / (1 - ECCENTRICITY * Math.cos(E));
  const cosNu = (Math.cos(E) - ECCENTRICITY) / (1 - ECCENTRICITY * Math.cos(E));
  let nu = Math.atan2(sinNu, cosNu) * RAD2DEG;

  // Ls = true anomaly + longitude of perihelion
  let ls = (nu + PERIHELION_LS) % 360;
  if (ls < 0) ls += 360;

  return ls;
}

/**
 * Convert Mars Year + Ls back to an approximate Earth date.
 *
 * This is the inverse of earthToMarsYear + earthToLs.
 * Since Ls is non-linear, we use iterative refinement:
 * 1. Start with a linear estimate based on MY + Ls/360
 * 2. Compute actual Ls at that date
 * 3. Adjust by the Ls error
 * 4. Repeat until convergence
 */
export function marsYearToEarth(marsYear: number, ls: number): Date {
  // Initial linear estimate: start of the Mars Year + Ls fraction
  const myStart = MY1_EPOCH_MS + (marsYear - 1) * MARS_YEAR_DAYS * 86400000;
  let estimate = new Date(myStart + (ls / 360) * MARS_YEAR_DAYS * 86400000);

  // Iterative refinement (typically converges in 3-4 iterations)
  for (let i = 0; i < 8; i++) {
    const currentLs = earthToLs(estimate);
    let lsError = ls - currentLs;

    // Handle wraparound at 0/360 boundary
    if (lsError > 180) lsError -= 360;
    if (lsError < -180) lsError += 360;

    if (Math.abs(lsError) < 0.1) break;

    // Adjust: ~1.909 days per degree of Ls (686.97/360)
    const adjustDays = lsError * (MARS_YEAR_DAYS / 360);
    estimate = new Date(estimate.getTime() + adjustDays * 86400000);
  }

  return estimate;
}

/**
 * Format a date as a Mars date string.
 *
 * Returns e.g. "MY29 Ls 142" or "MY34 Ls 0"
 */
export function formatMarsDate(date: Date): string {
  const my = Math.floor(earthToMarsYear(date));
  const ls = Math.round(earthToLs(date));
  return `MY${my} Ls ${ls}`;
}

/**
 * Format a Mars date range for display.
 *
 * Returns e.g. "MY29 Ls 120-180 (2008 Jul - 2008 Oct)"
 */
export function formatMarsDateRange(start: Date, end: Date): string {
  const myStart = Math.floor(earthToMarsYear(start));
  const myEnd = Math.floor(earthToMarsYear(end));
  const lsStart = Math.round(earthToLs(start));
  const lsEnd = Math.round(earthToLs(end));

  const earthStart = start.toLocaleDateString("en-US", { year: "numeric", month: "short" });
  const earthEnd = end.toLocaleDateString("en-US", { year: "numeric", month: "short" });

  if (myStart === myEnd) {
    return `MY${myStart} Ls ${lsStart}\u2013${lsEnd} (${earthStart} \u2013 ${earthEnd})`;
  }
  return `MY${myStart} Ls ${lsStart} \u2013 MY${myEnd} Ls ${lsEnd} (${earthStart} \u2013 ${earthEnd})`;
}

/**
 * Get the Ls value at evenly spaced intervals for axis tick generation.
 * Returns an array of { date, ls, marsYear } objects spanning the range.
 */
export function getLsTicksInRange(
  start: Date,
  end: Date,
  maxTicks: number = 12,
): Array<{ date: Date; ls: number; marsYear: number }> {
  const ticks: Array<{ date: Date; ls: number; marsYear: number }> = [];
  const rangeMs = end.getTime() - start.getTime();

  if (rangeMs <= 0) return ticks;

  const step = rangeMs / (maxTicks + 1);

  for (let i = 1; i <= maxTicks; i++) {
    const d = new Date(start.getTime() + step * i);
    ticks.push({
      date: d,
      ls: Math.round(earthToLs(d)),
      marsYear: Math.floor(earthToMarsYear(d)),
    });
  }

  return ticks;
}
