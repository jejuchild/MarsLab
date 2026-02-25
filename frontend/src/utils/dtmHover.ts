/**
 * DTM Hover System - Fast elevation lookup for HiRISE DTM
 *
 * Performance-first design:
 * - Elevation grid loaded once and cached
 * - O(1) lookup via lat/lon → pixel index
 * - No API calls on hover
 * - Throttled to 25Hz (40ms)
 */

// Cached elevation grid per product
export interface DTMElevationGrid {
  productId: string;
  rows: number;
  cols: number;
  bounds: {
    west: number;
    south: number;
    east: number;
    north: number;
  };
  elevations: (number | null)[];
  minElevation: number;
  maxElevation: number;
}

const gridCache = new Map<string, DTMElevationGrid>();
const loadingPromises = new Map<string, Promise<DTMElevationGrid | null>>();

/**
 * Load elevation grid for a DTM product (cached)
 */
export async function loadDTMElevationGrid(
  productId: string
): Promise<DTMElevationGrid | null> {
  // Return cached
  if (gridCache.has(productId)) {
    return gridCache.get(productId)!;
  }

  // Return in-flight promise
  if (loadingPromises.has(productId)) {
    return loadingPromises.get(productId)!;
  }

  // Start loading
  const promise = (async () => {
    try {
      const res = await fetch(
        `/api/terrain/hirise_dtm_elevation_grid?product_id=${encodeURIComponent(productId)}&max_size=256`
      );
      if (!res.ok) {
        console.error(`[DTMHover] Failed to load grid for ${productId}`);
        return null;
      }

      const data = await res.json();
      const grid: DTMElevationGrid = {
        productId: data.product_id,
        rows: data.rows,
        cols: data.cols,
        bounds: data.bounds,
        elevations: data.elevations,
        minElevation: data.min_elevation,
        maxElevation: data.max_elevation,
      };

      gridCache.set(productId, grid);

      return grid;
    } catch (e) {
      console.error(`[DTMHover] Error loading grid:`, e);
      return null;
    } finally {
      loadingPromises.delete(productId);
    }
  })();

  loadingPromises.set(productId, promise);
  return promise;
}

/**
 * Get elevation at lat/lon from cached grid (O(1))
 */
export function getElevationFromGrid(
  grid: DTMElevationGrid,
  lat: number,
  lon: number
): number | null {
  const { bounds, rows, cols, elevations } = grid;

  // Check if point is within bounds
  if (
    lat < bounds.south ||
    lat > bounds.north ||
    lon < bounds.west ||
    lon > bounds.east
  ) {
    return null;
  }

  // Calculate pixel indices
  const lonRange = bounds.east - bounds.west;
  const latRange = bounds.north - bounds.south;

  // Column: west to east
  const colFrac = (lon - bounds.west) / lonRange;
  const col = Math.floor(colFrac * cols);

  // Row: north to south (row 0 is north)
  const rowFrac = (bounds.north - lat) / latRange;
  const row = Math.floor(rowFrac * rows);

  // Clamp to valid range
  const clampedRow = Math.max(0, Math.min(rows - 1, row));
  const clampedCol = Math.max(0, Math.min(cols - 1, col));

  // Get elevation (row-major order)
  const index = clampedRow * cols + clampedCol;
  return elevations[index];
}

/**
 * Check if a point is within DTM bounds
 */
export function isWithinDTMBounds(
  grid: DTMElevationGrid,
  lat: number,
  lon: number
): boolean {
  const { bounds } = grid;
  return (
    lat >= bounds.south &&
    lat <= bounds.north &&
    lon >= bounds.west &&
    lon <= bounds.east
  );
}

/**
 * Clear cached grid for a product
 */
export function clearDTMGrid(productId: string): void {
  gridCache.delete(productId);
}

/**
 * Clear all cached grids
 */
export function clearAllDTMGrids(): void {
  gridCache.clear();
}

/**
 * Create a throttled function (for hover handlers)
 */
export function throttle<T extends (...args: any[]) => void>(
  fn: T,
  intervalMs: number
): T {
  let lastCall = 0;
  let timeoutId: number | null = null;

  return ((...args: Parameters<T>) => {
    const now = performance.now();
    const elapsed = now - lastCall;

    if (elapsed >= intervalMs) {
      lastCall = now;
      fn(...args);
    } else if (!timeoutId) {
      // Schedule trailing call
      timeoutId = window.setTimeout(() => {
        lastCall = performance.now();
        timeoutId = null;
        fn(...args);
      }, intervalMs - elapsed);
    }
  }) as T;
}

/**
 * Hover state for UI display
 */
export interface DTMHoverState {
  active: boolean;
  lat: number | null;
  lon: number | null;
  elevation: number | null;
  productId: string | null;
}

export const EMPTY_HOVER_STATE: DTMHoverState = {
  active: false,
  lat: null,
  lon: null,
  elevation: null,
  productId: null,
};
