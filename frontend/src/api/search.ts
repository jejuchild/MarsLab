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

// =============================================================================
// Download Cancellation API
// =============================================================================

/**
 * Cancel a specific download task.
 *
 * @param taskId - Task identifier
 */
export async function cancelDownload(taskId: string): Promise<{ status: string; task_id: string }> {
  const res = await fetch(`/api/download/${taskId}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    throw new Error("Failed to cancel download");
  }

  return res.json();
}

/**
 * Retry a failed download task.
 */
export async function retryDownload(taskId: string): Promise<DownloadTask> {
  const res = await fetch(`/api/download/${taskId}/retry`, {
    method: "POST",
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Retry failed" }));
    throw new Error(error.detail || "Retry failed");
  }
  return res.json();
}

/**
 * Cancel all active download tasks.
 */
export async function cancelAllDownloads(): Promise<{ status: string; cancelled_count: number }> {
  const res = await fetch("/api/download", {
    method: "DELETE",
  });

  if (!res.ok) {
    throw new Error("Failed to cancel downloads");
  }

  return res.json();
}

/**
 * Clear all completed and failed download tasks from history.
 */
export async function clearDownloadHistory(): Promise<{ status: string; removed_count: number }> {
  const res = await fetch("/api/download/history", {
    method: "DELETE",
  });

  if (!res.ok) {
    throw new Error("Failed to clear download history");
  }

  return res.json();
}

// =============================================================================
// AI Search API
// =============================================================================

/**
 * Parsed intent from natural language query.
 */
export interface ParsedIntent {
  instruments: string[];
  region_name: string | null;
  region_lat: number | null;
  region_lon: number | null;
  radius_deg: number;
  raw_query: string;
}

/**
 * Single AI search result.
 */
export interface AISearchResult {
  product_id: string;
  instrument: string;
  lat: number | null;
  lon: number | null;
  distance_km: number | null;
  exists: boolean;
  has_core: boolean;
  has_browse: boolean;
  missing_files: string[];
}

/**
 * Response from AI search.
 */
export interface AISearchResponse {
  parsed_intent: ParsedIntent;
  total_count: number;
  shown_count: number;
  truncated: boolean;
  results: AISearchResult[];
  message: string;
}

/**
 * AI-assisted natural language search.
 *
 * @param query - Natural language query (e.g., "Download CRISM data over Arcadia Planitia")
 * @param maxResults - Maximum number of results to return (default 10)
 */
export async function aiSearch(query: string, maxResults: number = 10): Promise<AISearchResponse> {
  const res = await fetch("/api/ai/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      max_results: maxResults,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "AI search failed" }));
    throw new Error(error.detail || "AI search failed");
  }

  return res.json();
}

// =============================================================================
// Gemini AI Search API (Two-step: Preview + Execute)
// =============================================================================

/**
 * Preview response from Gemini parsing.
 */
export interface GeminiSearchPreview {
  success: boolean;
  instruments: string[];
  region_type: string;
  region_name: string | null;
  region_lat: number | null;
  region_lon: number | null;
  region_bbox: { minLat: number; maxLat: number; minLon: number; maxLon: number } | null;
  radius_km: number;
  max_results: number;
  spatial_predicate: string;
  distribution: string;
  terrain_hint: string | null;
  cross_instrument: string | null;
  cross_title_filter: string | null;
  message: string;
  error: string | null;
}

/**
 * Single result from Gemini search execution.
 */
export interface GeminiSearchResult {
  product_id: string;
  instrument: string;
  lat: number | null;
  lon: number | null;
  distance_km: number | null;
  exists: boolean;
  has_core: boolean;
  has_browse: boolean;
  missing_files: string[];
}

/**
 * Full response from Gemini search execution.
 */
export interface GeminiSearchResponse {
  preview: GeminiSearchPreview;
  total_count: number;
  shown_count: number;
  truncated: boolean;
  results: GeminiSearchResult[];
}

/**
 * Get Gemini API status.
 */
export async function getGeminiStatus(): Promise<{ available: boolean; api_key_set: boolean; models: string[] }> {
  const res = await fetch("/api/ai/gemini/status");
  if (!res.ok) {
    throw new Error("Failed to get Gemini status");
  }
  return res.json();
}

/**
 * Step 1: Parse query with Gemini and get a preview (no search execution).
 * User should confirm before calling geminiExecute().
 */
export async function geminiPreview(query: string, maxResults: number = 20): Promise<GeminiSearchPreview> {
  const res = await fetch("/api/ai/gemini/preview", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      max_results: maxResults,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Gemini preview failed" }));
    throw new Error(error.detail || "Gemini preview failed");
  }

  return res.json();
}

/**
 * Step 2: Parse query with Gemini AND execute the search.
 * Should only be called after user confirms the preview.
 * Pass the `plan` from preview to avoid re-parsing with Gemini.
 */
export async function geminiExecute(query: string, maxResults: number = 20, plan?: Record<string, unknown>): Promise<GeminiSearchResponse> {
  const res = await fetch("/api/ai/gemini/execute", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      max_results: maxResults,
      plan: plan ?? null,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Gemini search failed" }));
    throw new Error(error.detail || "Gemini search failed");
  }

  return res.json();
}

// =============================================================================
// Point Search API (Coordinate-based)
// =============================================================================

/**
 * Result from point-based coordinate search (ODE).
 */
export interface PointSearchResult {
  product_id: string;
  instrument: string;
  lat: number | null;
  lon: number | null;
  distance_km: number | null;  // Distance from query point in km
  exists: boolean;
  has_core: boolean;
  has_browse: boolean;
  missing_files: string[];
}

/**
 * Response from point search endpoint.
 */
export interface PointSearchResponse {
  query: {
    lat: number;
    lon: number;
    lon_360: number;
    radius_deg: number;
  };
  total_count: number;
  results: {
    CRISM: PointSearchResult[];
    HIRISE: PointSearchResult[];
    SHARAD: PointSearchResult[];
    SHARAD_HIGHRES: PointSearchResult[];
    HIRISE_DTM: PointSearchResult[];
  };
}

/**
 * Search ODE for all datasets that cover a given coordinate.
 *
 * Creates a bounding box around the point and queries ODE for each instrument type.
 *
 * @param lat - Latitude in degrees (-90 to 90)
 * @param lon - Longitude in degrees (any range, normalized internally)
 * @param radius - Search radius in degrees (default 1.0, roughly 60km)
 */
export async function searchByPoint(
  lat: number,
  lon: number,
  radius: number = 1
): Promise<PointSearchResponse> {
  const params = new URLSearchParams({
    lat: lat.toString(),
    lon: lon.toString(),
    radius: radius.toString(),
  });

  const res = await fetch(`/api/search/point?${params}`);

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Point search failed" }));
    throw new Error(error.detail || error.error || "Point search failed");
  }

  return res.json();
}

// =============================================================================
// Product Proximity / Overlap Search API
// =============================================================================

export interface ProximityResult {
  product_id: string;
  instrument: string;
  lat: number | null;
  lon: number | null;
  distance_km: number | null;
  overlap: boolean;
}

export interface ProximityResponse {
  source_product_id: string;
  source_instrument: string;
  source_bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null;
  query_mode: string;
  target_instruments: string[];
  results: ProximityResult[];
  total_count: number;
}

/**
 * Search for products overlapping with or near a given product.
 */
export async function searchProximity(
  productId: string,
  instrument: string,
  mode: "overlap" | "nearest" = "overlap",
  targetInstruments: string = "all",
  limit: number = 50,
  searchRadiusDeg: number = 5.0,
): Promise<ProximityResponse> {
  const params = new URLSearchParams({
    product_id: productId,
    instrument,
    mode,
    target_instruments: targetInstruments,
    limit: limit.toString(),
    search_radius_deg: searchRadiusDeg.toString(),
  });

  const res = await fetch(`/api/proximity/search?${params}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Proximity search failed" }));
    throw new Error(error.detail || "Proximity search failed");
  }
  return res.json();
}

// =============================================================================
// Region Lookup API
// =============================================================================

export interface MarsRegionInfo {
  region_id: string;
  display_name: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  description: string;
  tags: string[];
}

/**
 * List all Mars regions with bounding boxes.
 */
export async function listRegions(
  tag?: string,
  query?: string,
): Promise<MarsRegionInfo[]> {
  const params = new URLSearchParams();
  if (tag) params.set("tag", tag);
  if (query) params.set("query", query);

  const res = await fetch(`/api/proximity/regions?${params}`);
  if (!res.ok) {
    throw new Error("Failed to list regions");
  }
  return res.json();
}

/**
 * Get a specific region by ID.
 */
export async function getRegion(regionId: string): Promise<MarsRegionInfo> {
  const res = await fetch(`/api/proximity/regions/${regionId}`);
  if (!res.ok) {
    throw new Error(`Region '${regionId}' not found`);
  }
  return res.json();
}

// =============================================================================
// Smart AI Search API (Llama-powered, single-step)
// =============================================================================

/**
 * A product selected by Llama with reasoning.
 */
export interface SmartProductSelection {
  product_id: string;
  instrument: string;
  lat: number | null;
  lon: number | null;
  distance_km: number | null;
  reason: string;
  already_local: boolean;
  download_task_id: string | null;
}

/**
 * Full response from smart search.
 */
export interface SmartSearchResponse {
  session_id: string;
  query: string;
  reasoning: string;
  selected_products: SmartProductSelection[];
  download_tasks: string[];
  search_summary: string;
  total_found: number;
  total_selected: number;
  total_downloading: number;
  total_already_local: number;
}

/**
 * SSE event from smart search stream.
 */
export interface SmartSearchEvent {
  event: string;
  data: Record<string, unknown>;
}

/**
 * Check Llama/Ollama availability for smart search.
 */
export async function getSmartSearchStatus(): Promise<{
  ollama_available: boolean;
  model: string;
  message: string;
}> {
  const res = await fetch("/api/ai/smart/status");
  if (!res.ok) {
    throw new Error("Failed to get smart search status");
  }
  return res.json();
}

/**
 * Start a smart search session. Returns session_id immediately.
 * Use streamSmartSearch() to get real-time events.
 */
export async function startSmartSearch(
  query: string,
  maxResults: number = 20
): Promise<{ session_id: string; status: string }> {
  const res = await fetch("/api/ai/smart/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, max_results: maxResults }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Smart search failed" }));
    throw new Error(error.detail || "Smart search failed");
  }
  return res.json();
}

/**
 * Stream SSE events from a smart search session.
 * Calls onEvent for each event, onDone when stream ends.
 */
export function streamSmartSearch(
  sessionId: string,
  onEvent: (event: SmartSearchEvent) => void,
  onDone: (response: SmartSearchResponse | null) => void,
  onError: (error: string) => void,
): () => void {
  const eventSource = new EventSource(`/api/ai/smart/stream/${sessionId}`);
  let finalResponse: SmartSearchResponse | null = null;
  let finished = false;

  const finish = () => {
    if (finished) return;
    finished = true;
    eventSource.close();
    onDone(finalResponse);
  };

  eventSource.onmessage = (msg) => {
    try {
      const parsed = JSON.parse(msg.data) as SmartSearchEvent;

      if (parsed.event === "stream_end" || parsed.event === "timeout") {
        finish();
        return;
      }

      if (parsed.event === "done") {
        finalResponse = parsed.data as unknown as SmartSearchResponse;
      }

      onEvent(parsed);
    } catch {
      // ignore parse errors
    }
  };

  eventSource.onerror = () => {
    if (!finished && !finalResponse) {
      onError("Connection lost to search stream");
    }
    finish();
  };

  // Return cleanup function
  return () => {
    finished = true;
    eventSource.close();
  };
}

/**
 * Get smart search session state (polling fallback).
 */
export async function getSmartSearchSession(
  sessionId: string,
): Promise<SmartSearchResponse> {
  const res = await fetch(`/api/ai/smart/session/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to get session");
  }
  return res.json();
}
