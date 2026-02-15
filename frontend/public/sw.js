/**
 * MarsLab Service Worker — Offline-First with Intelligent Caching
 *
 * Strategies:
 *   - App shell (HTML, JS, CSS): CacheFirst
 *   - Map tiles (NASA Trek, CTX tiles): CacheFirst with network fallback
 *   - API data (footprints, GeoJSON indices): NetworkFirst with cache fallback
 *   - Images (quickviews, thumbnails, PNGs): StaleWhileRevalidate
 *   - Never cached: POST requests, SSE streams, chrome-extension, ws://
 *
 * Cache names are versioned. Bumping CACHE_VERSION invalidates everything.
 * Each cache has a max-entry limit enforced by LRU eviction.
 */

const CACHE_VERSION = "v1";
const CACHE_SHELL   = `marslab-${CACHE_VERSION}-shell`;
const CACHE_TILES   = `marslab-${CACHE_VERSION}-tiles`;
const CACHE_API     = `marslab-${CACHE_VERSION}-api`;
const CACHE_IMAGES  = `marslab-${CACHE_VERSION}-images`;

const ALL_CACHES = [CACHE_SHELL, CACHE_TILES, CACHE_API, CACHE_IMAGES];

// Max entries per cache bucket
const MAX_ENTRIES = {
  [CACHE_SHELL]:  80,
  [CACHE_TILES]:  2000,  // tiles are small JPGs (~20KB each), 2000 * 20KB = ~40MB
  [CACHE_API]:    100,
  [CACHE_IMAGES]: 300,
};

// Global max cache size in bytes (200MB)
const MAX_TOTAL_BYTES = 200 * 1024 * 1024;

// NetworkFirst timeout (ms) — fall back to cache after this
const NETWORK_TIMEOUT_MS = 8000;

// ---------------------------------------------------------------------------
// INSTALL — precache app shell essentials
// ---------------------------------------------------------------------------
const PRECACHE_URLS = [
  "/",
  "/manifest.json",
  "/favicon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_SHELL).then((cache) => {
      return cache.addAll(PRECACHE_URLS);
    }).then(() => {
      // Activate immediately without waiting for old SW to die
      return self.skipWaiting();
    })
  );
});

// ---------------------------------------------------------------------------
// ACTIVATE — clean up old versioned caches
// ---------------------------------------------------------------------------
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => key.startsWith("marslab-") && !ALL_CACHES.includes(key))
          .map((key) => caches.delete(key))
      );
    }).then(() => {
      // Take control of all open tabs immediately
      return self.clients.claim();
    })
  );
});

// ---------------------------------------------------------------------------
// FETCH — route to appropriate caching strategy
// ---------------------------------------------------------------------------
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // ------- NEVER CACHE -------
  // Skip non-GET requests (POST for SSE streams, form submissions, etc.)
  if (request.method !== "GET") return;

  // Skip chrome-extension and non-http(s) schemes
  if (!url.protocol.startsWith("http")) return;

  // Skip SSE endpoints (these return text/event-stream and must not be cached)
  if (url.pathname.includes("/agent/") ||
      url.pathname.includes("/report/generate") ||
      url.pathname.includes("/mineral-cnn/classify")) {
    return;
  }

  // Skip websocket upgrade requests
  if (request.headers.get("upgrade") === "websocket") return;

  // ------- ROUTE TO STRATEGY -------
  const strategy = classifyRequest(url, request);

  switch (strategy) {
    case "cache-first-shell":
      event.respondWith(cacheFirst(request, CACHE_SHELL));
      break;
    case "cache-first-tile":
      event.respondWith(cacheFirst(request, CACHE_TILES));
      break;
    case "network-first-api":
      event.respondWith(networkFirst(request, CACHE_API));
      break;
    case "stale-while-revalidate-image":
      event.respondWith(staleWhileRevalidate(request, CACHE_IMAGES));
      break;
    case "network-first-page":
      event.respondWith(networkFirst(request, CACHE_SHELL));
      break;
    default:
      // Let the browser handle it normally
      return;
  }
});

// ---------------------------------------------------------------------------
// Request classifier
// ---------------------------------------------------------------------------
function classifyRequest(url, request) {
  const { pathname, hostname } = url;
  const accept = request.headers.get("accept") || "";

  // Map tiles from NASA Trek or local tile servers
  if (hostname === "trek.nasa.gov" ||
      pathname.includes("/tiles/") ||
      pathname.includes("/world_tiles/") ||
      pathname.match(/\/\d+\/\d+\/\d+\.(jpg|png|jpeg|webp)$/)) {
    return "cache-first-tile";
  }

  // Static assets — JS, CSS, fonts, WASM
  if (pathname.match(/\.(js|css|woff2?|wasm)$/) ||
      pathname.includes("/assets/")) {
    return "cache-first-shell";
  }

  // Cesium assets (workers, static data)
  if (pathname.startsWith("/cesium/") || pathname.startsWith("/Cesium/")) {
    return "cache-first-shell";
  }

  // API calls — GeoJSON indices, footprint data, search results
  if (pathname.startsWith("/api/") ||
      pathname.endsWith(".geojson") ||
      pathname.startsWith("/terrain/") ||
      pathname.startsWith("/sharad/") ||
      pathname.startsWith("/hirise/") ||
      pathname.startsWith("/crism/") ||
      pathname.startsWith("/world_meta/")) {
    // Image responses from API (quickviews, maps, PNGs)
    if (pathname.match(/\.(png|jpg|jpeg|gif|webp)$/)) {
      return "stale-while-revalidate-image";
    }
    return "network-first-api";
  }

  // Images (thumbnails, quickviews, browse products)
  if (pathname.match(/\.(png|jpg|jpeg|gif|webp|svg)$/)) {
    return "stale-while-revalidate-image";
  }

  // HTML page navigation — network first so updates are picked up
  if (accept.includes("text/html") || pathname === "/") {
    return "network-first-page";
  }

  // Everything else — no caching (let browser handle)
  return null;
}

// ---------------------------------------------------------------------------
// CacheFirst: Serve from cache if available, otherwise fetch and cache.
// ---------------------------------------------------------------------------
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    // Cache successful responses AND opaque responses (cross-origin no-cors).
    // Opaque responses have status 0 and ok=false, but are still usable.
    if (response.ok || response.type === "opaque") {
      cache.put(request, response.clone());
      enforceCacheLimit(cacheName);
    }
    return response;
  } catch (err) {
    // Network completely unavailable — return a basic offline fallback
    return new Response("Offline — resource not cached", {
      status: 503,
      statusText: "Service Unavailable",
      headers: { "Content-Type": "text/plain" },
    });
  }
}

// ---------------------------------------------------------------------------
// NetworkFirst: Try network with timeout, fall back to cache.
// ---------------------------------------------------------------------------
async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);

  try {
    const response = await fetchWithTimeout(request, NETWORK_TIMEOUT_MS);
    if (response.ok) {
      cache.put(request, response.clone());
      enforceCacheLimit(cacheName);
    }
    return response;
  } catch (err) {
    // Network failed or timed out — serve from cache
    const cached = await cache.match(request);
    if (cached) return cached;

    // For page navigations, try returning the cached root page (SPA fallback)
    if (request.headers.get("accept")?.includes("text/html")) {
      const fallback = await cache.match("/");
      if (fallback) return fallback;
    }

    return new Response("Offline — no cached data available", {
      status: 503,
      statusText: "Service Unavailable",
      headers: { "Content-Type": "text/plain" },
    });
  }
}

// ---------------------------------------------------------------------------
// StaleWhileRevalidate: Return cache immediately, update in background.
// ---------------------------------------------------------------------------
async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  // Fire off revalidation in background (non-blocking)
  const revalidate = fetch(request).then((response) => {
    if (response.ok || response.type === "opaque") {
      cache.put(request, response.clone());
      enforceCacheLimit(cacheName);
    }
    return response;
  }).catch(() => {
    // Network failed — that's fine, we'll serve stale
    return null;
  });

  // If we have a cached version, return it immediately
  if (cached) return cached;

  // No cached version — must wait for network
  try {
    const freshResponse = await revalidate;
    if (freshResponse) return freshResponse;
  } catch (_) {
    // Fall through
  }

  return new Response("Offline — image not cached", {
    status: 503,
    statusText: "Service Unavailable",
    headers: { "Content-Type": "text/plain" },
  });
}

// ---------------------------------------------------------------------------
// Fetch with timeout — rejects if network takes too long
// ---------------------------------------------------------------------------
function fetchWithTimeout(request, timeoutMs) {
  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      controller.abort();
      reject(new Error("Network timeout"));
    }, timeoutMs);

    fetch(request, { signal: controller.signal })
      .then((response) => {
        clearTimeout(timer);
        resolve(response);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

// ---------------------------------------------------------------------------
// LRU Cache Eviction — trim cache to max entries
// ---------------------------------------------------------------------------
async function enforceCacheLimit(cacheName) {
  const maxEntries = MAX_ENTRIES[cacheName];
  if (!maxEntries) return;

  const cache = await caches.open(cacheName);
  const keys = await cache.keys();

  if (keys.length > maxEntries) {
    // Evict oldest entries (FIFO — Cache API returns keys in insertion order)
    const excess = keys.length - maxEntries;
    for (let i = 0; i < excess; i++) {
      await cache.delete(keys[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// MESSAGE HANDLER — for cache management commands from the app
// ---------------------------------------------------------------------------
self.addEventListener("message", (event) => {
  const { type, payload } = event.data || {};

  switch (type) {
    case "GET_CACHE_STATS":
      getCacheStats().then((stats) => {
        event.source.postMessage({ type: "CACHE_STATS", payload: stats });
      });
      break;

    case "CLEAR_ALL_CACHES":
      Promise.all(ALL_CACHES.map((name) => caches.delete(name))).then(() => {
        event.source.postMessage({ type: "CACHES_CLEARED" });
      });
      break;

    case "PRECACHE_TILES":
      // Pre-cache tile URLs for a specific region (for "Save for Offline")
      if (payload && Array.isArray(payload.urls)) {
        precacheTileUrls(payload.urls).then((result) => {
          event.source.postMessage({
            type: "PRECACHE_COMPLETE",
            payload: result,
          });
        });
      }
      break;

    case "SKIP_WAITING":
      self.skipWaiting();
      break;

    default:
      break;
  }
});

// ---------------------------------------------------------------------------
// Cache statistics — used by the OfflineIndicator tooltip
// ---------------------------------------------------------------------------
async function getCacheStats() {
  const stats = {};
  let totalSize = 0;

  for (const cacheName of ALL_CACHES) {
    try {
      const cache = await caches.open(cacheName);
      const keys = await cache.keys();
      let cacheSize = 0;

      // Sample up to 50 entries to estimate average size, then extrapolate
      const sampleSize = Math.min(keys.length, 50);
      for (let i = 0; i < sampleSize; i++) {
        try {
          const response = await cache.match(keys[i]);
          if (response) {
            const blob = await response.clone().blob();
            cacheSize += blob.size;
          }
        } catch (_) {
          // Skip unreadable entries
        }
      }

      // Extrapolate if we sampled
      if (sampleSize > 0 && sampleSize < keys.length) {
        cacheSize = Math.round((cacheSize / sampleSize) * keys.length);
      }

      stats[cacheName] = { entries: keys.length, estimatedBytes: cacheSize };
      totalSize += cacheSize;
    } catch (_) {
      stats[cacheName] = { entries: 0, estimatedBytes: 0 };
    }
  }

  stats.total = {
    entries: Object.values(stats).reduce((sum, s) => sum + (s.entries || 0), 0),
    estimatedBytes: totalSize,
    maxBytes: MAX_TOTAL_BYTES,
  };

  return stats;
}

// ---------------------------------------------------------------------------
// Pre-cache tile URLs for offline region support
// ---------------------------------------------------------------------------
async function precacheTileUrls(urls) {
  const cache = await caches.open(CACHE_TILES);
  let cached = 0;
  let failed = 0;

  for (const url of urls) {
    try {
      // Skip if already cached
      const existing = await cache.match(url);
      if (existing) {
        cached++;
        continue;
      }

      // Cross-origin tiles may require no-cors mode (returns opaque response)
      const parsedUrl = new URL(url, self.location.origin);
      const isCrossOrigin = parsedUrl.origin !== self.location.origin;
      const response = await fetch(url, isCrossOrigin ? { mode: "cors" } : {});

      if (response.ok || response.type === "opaque") {
        await cache.put(url, response);
        cached++;
      } else {
        failed++;
      }
    } catch (_) {
      failed++;
    }
  }

  await enforceCacheLimit(CACHE_TILES);
  return { cached, failed, total: urls.length };
}
