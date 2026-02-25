import { useState, useEffect, useCallback, useRef } from "react";

export interface CacheBucketStats {
  entries: number;
  estimatedBytes: number;
}

export interface CacheStats {
  [key: string]: CacheBucketStats;
  total: CacheBucketStats & { maxBytes: number };
}

interface ServiceWorkerState {
  /** Whether the browser is currently online */
  isOnline: boolean;
  /** Whether a new service worker version is waiting to activate */
  isUpdateAvailable: boolean;
  /** Whether the service worker is registered and active */
  isReady: boolean;
  /** Trigger the waiting SW to skip waiting and reload */
  updateApp: () => void;
  /** Current cache size statistics (null until first fetch) */
  cacheStats: CacheStats | null;
  /** Manually refresh cache stats */
  refreshCacheStats: () => void;
  /** Clear all SW caches */
  clearAllCaches: () => Promise<void>;
  /** Pre-cache tile URLs for a region */
  precacheTiles: (urls: string[]) => Promise<{ cached: number; failed: number; total: number }>;
}

export default function useServiceWorker(): ServiceWorkerState {
  const [isOnline, setIsOnline] = useState(() => navigator.onLine);
  const [isUpdateAvailable, setIsUpdateAvailable] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);

  const registrationRef = useRef<ServiceWorkerRegistration | null>(null);

  // ----- Online / Offline listeners -----
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // ----- Service Worker registration -----
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;

    let cancelled = false;

    navigator.serviceWorker
      .register("/sw.js", { scope: "/" })
      .then((registration) => {
        if (cancelled) return;

        registrationRef.current = registration;

        // Check if SW is already active
        if (registration.active) {
          setIsReady(true);
        }

        // Listen for new SW installations (update available)
        registration.addEventListener("updatefound", () => {
          const newWorker = registration.installing;
          if (!newWorker) return;

          newWorker.addEventListener("statechange", () => {
            // A new SW is installed and waiting — prompt user to update
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              setIsUpdateAvailable(true);
            }
            if (newWorker.state === "activated") {
              setIsReady(true);
            }
          });
        });

        // If there's already a waiting worker on page load
        if (registration.waiting && navigator.serviceWorker.controller) {
          setIsUpdateAvailable(true);
        }
      })
      .catch(() => {});

    // When the new SW takes over, reload the page for a clean state
    let refreshing = false;
    const handleControllerChange = () => {
      if (refreshing) return;
      refreshing = true;
      window.location.reload();
    };
    navigator.serviceWorker.addEventListener(
      "controllerchange",
      handleControllerChange
    );

    // Listen for messages from the SW
    const handleMessage = (event: MessageEvent) => {
      const { type, payload } = event.data || {};
      if (type === "CACHE_STATS") {
        setCacheStats(payload);
      }
      if (type === "CACHES_CLEARED") {
        setCacheStats(null);
      }
    };
    navigator.serviceWorker.addEventListener("message", handleMessage);

    return () => {
      cancelled = true;
      navigator.serviceWorker.removeEventListener(
        "controllerchange",
        handleControllerChange
      );
      navigator.serviceWorker.removeEventListener("message", handleMessage);
    };
  }, []);

  // ----- Fetch cache stats on mount + periodically -----
  const refreshCacheStats = useCallback(() => {
    if (navigator.serviceWorker?.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: "GET_CACHE_STATS",
      });
    }
  }, []);

  useEffect(() => {
    if (!isReady) return;

    // Initial fetch with slight delay to let SW settle
    const initial = setTimeout(refreshCacheStats, 1000);

    // Refresh every 60s
    const interval = setInterval(refreshCacheStats, 60_000);

    return () => {
      clearTimeout(initial);
      clearInterval(interval);
    };
  }, [isReady, refreshCacheStats]);

  // ----- Actions -----
  const updateApp = useCallback(() => {
    const waiting = registrationRef.current?.waiting;
    if (waiting) {
      waiting.postMessage({ type: "SKIP_WAITING" });
    }
  }, []);

  const clearAllCaches = useCallback(async () => {
    if (navigator.serviceWorker?.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: "CLEAR_ALL_CACHES",
      });
    }
  }, []);

  const precacheTiles = useCallback(
    (urls: string[]): Promise<{ cached: number; failed: number; total: number }> => {
      return new Promise((resolve) => {
        if (!navigator.serviceWorker?.controller) {
          resolve({ cached: 0, failed: urls.length, total: urls.length });
          return;
        }

        let resolved = false;

        const cleanup = () => {
          navigator.serviceWorker.removeEventListener("message", handleMessage);
        };

        const handleMessage = (event: MessageEvent) => {
          if (event.data?.type === "PRECACHE_COMPLETE" && !resolved) {
            resolved = true;
            cleanup();
            resolve(event.data.payload);
          }
        };

        navigator.serviceWorker.addEventListener("message", handleMessage);
        navigator.serviceWorker.controller.postMessage({
          type: "PRECACHE_TILES",
          payload: { urls },
        });

        // Timeout after 2 minutes — guard against double resolution
        setTimeout(() => {
          if (!resolved) {
            resolved = true;
            cleanup();
            resolve({ cached: 0, failed: urls.length, total: urls.length });
          }
        }, 120_000);
      });
    },
    []
  );

  return {
    isOnline,
    isUpdateAvailable,
    isReady,
    updateApp,
    cacheStats,
    refreshCacheStats,
    clearAllCaches,
    precacheTiles,
  };
}
