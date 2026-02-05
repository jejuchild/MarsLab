/**
 * Coordinate normalization utilities for Mars mapping.
 *
 * Single source of truth for longitude conversions used by:
 * - Fly-To navigation
 * - View Bound selection
 * - ODE search bounding boxes
 * - Footprint loading
 */

/**
 * Normalize longitude to -180 to 180 range (for Cesium/map display).
 * Handles any input value including those beyond ±180.
 */
export function normalizeLonForMap(lon: number): number {
  let normalized = lon;
  while (normalized > 180) normalized -= 360;
  while (normalized < -180) normalized += 360;
  return normalized;
}

/**
 * Normalize longitude to 0 to 360 range (positive East, for ODE queries).
 * This is the standard planetocentric longitude system.
 */
export function normalizeLonForODE(lon: number): number {
  let normalized = lon;
  while (normalized < 0) normalized += 360;
  while (normalized >= 360) normalized -= 360;
  return normalized;
}

/**
 * Clamp latitude to valid range (-90 to 90).
 */
export function clampLatitude(lat: number): number {
  return Math.max(-90, Math.min(90, lat));
}

/**
 * Parse and validate a coordinate input string.
 * Returns the parsed number or null if invalid.
 */
export function parseCoordinate(value: string): number | null {
  const num = parseFloat(value);
  return isNaN(num) ? null : num;
}

/**
 * Format coordinate for display with specified decimal places.
 */
export function formatCoordinate(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}
