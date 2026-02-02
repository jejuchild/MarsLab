/**
 * Search & Download API client
 *
 * Provides functions to:
 * - Search ODE for products by ID or coordinates
 * - Check local existence of products
 * - Start and monitor downloads
 */

import {
  detectInstrument as detectInstrumentFromRegistry,
  type InstrumentId,
} from "../config/instrumentRegistry";

// =============================================================================
// Types
// =============================================================================

export type Instrument = InstrumentId;

export interface SearchResult {
  product_id: string;
  instrument: Instrument;
  base_key: string;
  lat: number | null;
  lon: number | null;
  exists: boolean;  // True only if ALL required files are downloaded
  has_core: boolean;  // True if core files (.img, .lbl) exist
  has_browse: boolean;  // True if at least one browse file exists
  missing_files: string[];  // List of missing file types
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  count: number;
}

export interface FileStatus {
  filename: string;
  status: "pending" | "queued" | "downloading" | "processing" | "completed" | "failed";
  bytes_downloaded: number;
  bytes_total: number | null;
  progress_percent: number;
  error: string | null;
}

export interface DownloadTask {
  task_id: string;
  product_id: string;
  base_key: string;
  instrument: Instrument;
  status: "pending" | "queued" | "downloading" | "processing" | "completed" | "failed";
  files: FileStatus[];
  target_dir: string;
  progress_percent: number;
  total_bytes: number;
  downloaded_bytes: number;
  created_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface ExistsResponse {
  exists: boolean;
  product_id: string;
  base_key: string;
  instrument: Instrument;
}

// =============================================================================
// Search API
// =============================================================================

/**
 * Search ODE for products by product ID (typeahead).
 *
 * @param query - Search string (partial or full product ID)
 * @param instrument - Optional filter by instrument
 * @param limit - Maximum results (default 10)
 */
export async function searchProducts(
  query: string,
  instrument?: Instrument,
  limit: number = 10
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: query,
    limit: limit.toString(),
  });

  if (instrument) {
    params.set("instrument", instrument);
  }

  const res = await fetch(`/api/search?${params}`);

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Search failed" }));
    throw new Error(error.detail || "Search failed");
  }

  return res.json();
}

/**
 * Bounding box for spatial search.
 */
export interface BoundingBox {
  minlat: number;  // Southern latitude boundary
  maxlat: number;  // Northern latitude boundary
  westernlon: number;  // Western longitude boundary
  easternlon: number;  // Eastern longitude boundary
}

/**
 * Search ODE for products within a bounding box.
 *
 * @param bbox - Bounding box with minlat, maxlat, westernlon, easternlon
 * @param instrument - Optional filter by instrument
 * @param limit - Maximum results (default 10)
 */
export async function searchSpatial(
  bbox: BoundingBox,
  instrument?: Instrument,
  limit: number = 10
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    minlat: bbox.minlat.toString(),
    maxlat: bbox.maxlat.toString(),
    westernlon: bbox.westernlon.toString(),
    easternlon: bbox.easternlon.toString(),
    limit: limit.toString(),
  });

  if (instrument) {
    params.set("instrument", instrument);
  }

  const res = await fetch(`/api/search/spatial?${params}`);

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Spatial search failed" }));
    throw new Error(error.detail || "Spatial search failed");
  }

  return res.json();
}

// =============================================================================
// Existence Check API
// =============================================================================

/**
 * Check if a product already exists locally.
 *
 * @param productId - Product identifier
 * @param instrument - CRISM or HiRISE
 */
export async function checkExists(
  productId: string,
  instrument: Instrument
): Promise<ExistsResponse> {
  const res = await fetch(`/api/exists/${instrument}/${encodeURIComponent(productId)}`);

  if (!res.ok) {
    throw new Error("Existence check failed");
  }

  return res.json();
}

// =============================================================================
// Download API
// =============================================================================

/**
 * Start downloading a product bundle (or missing files only).
 *
 * @param productId - Product identifier
 * @param instrument - CRISM or HiRISE
 * @param lat - Optional latitude for index
 * @param lon - Optional longitude for index
 * @param fileTypes - Optional list of file types to download ("core", "header", "wavelength", "browse")
 */
export async function startDownload(
  productId: string,
  instrument: Instrument,
  lat?: number,
  lon?: number,
  fileTypes?: string[]
): Promise<DownloadTask> {
  const res = await fetch("/api/download", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      product_id: productId,
      instrument,
      lat,
      lon,
      file_types: fileTypes,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Download failed" }));
    throw new Error(error.detail || "Download failed");
  }

  return res.json();
}

/**
 * Get the status of a download task.
 *
 * @param taskId - Task identifier
 */
export async function getDownloadStatus(taskId: string): Promise<DownloadTask> {
  const res = await fetch(`/api/download/${taskId}`);

  if (!res.ok) {
    throw new Error("Failed to get download status");
  }

  return res.json();
}

/**
 * List all download tasks.
 */
export async function listDownloads(): Promise<DownloadTask[]> {
  const res = await fetch("/api/download");

  if (!res.ok) {
    throw new Error("Failed to list downloads");
  }

  return res.json();
}

/**
 * Poll download status until completion or failure.
 *
 * @param taskId - Task identifier
 * @param onProgress - Callback for progress updates
 * @param intervalMs - Polling interval in milliseconds (default 1000)
 */
export async function pollDownloadStatus(
  taskId: string,
  onProgress?: (task: DownloadTask) => void,
  intervalMs: number = 1000
): Promise<DownloadTask> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const task = await getDownloadStatus(taskId);

        if (onProgress) {
          onProgress(task);
        }

        if (task.status === "completed") {
          resolve(task);
        } else if (task.status === "failed") {
          reject(new Error(task.error || "Download failed"));
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (e) {
        reject(e);
      }
    };

    poll();
  });
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format bytes to human-readable string.
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";

  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Parse CRISM base_key from product ID.
 */
export function parseCrismBaseKey(productId: string): string {
  const parts = productId.toLowerCase().split("_");
  if (parts.length >= 2) {
    return `${parts[0]}_${parts[1]}`;
  }
  return productId.toLowerCase();
}

/**
 * Detect instrument from product ID using the registry.
 */
export function detectInstrument(productId: string): Instrument | null {
  const config = detectInstrumentFromRegistry(productId);
  return config ? (config.id as Instrument) : null;
}

/**
 * Check if a string looks like a CRISM product ID.
 * @deprecated Use detectInstrument() instead
 */
export function isCrismProductId(productId: string): boolean {
  return detectInstrument(productId) === "crism";
}

/**
 * Check if a string looks like a HiRISE product ID.
 * @deprecated Use detectInstrument() instead
 */
export function isHiriseProductId(productId: string): boolean {
  return detectInstrument(productId) === "hirise";
}

/**
 * Check if a string looks like a SHARAD product ID.
 * @deprecated Use detectInstrument() instead
 */
export function isSharadProductId(productId: string): boolean {
  return detectInstrument(productId) === "sharad";
}
