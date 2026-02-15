import { useState, useRef, useEffect, useCallback } from "react";
import type { CacheStats } from "../hooks/useServiceWorker";

interface OfflineIndicatorProps {
  isOnline: boolean;
  isUpdateAvailable: boolean;
  cacheStats: CacheStats | null;
  onUpdate: () => void;
}

/** Format bytes to a human-readable string (e.g. "14.3 MB") */
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}

/** Map cache bucket names to friendly display labels */
const BUCKET_LABELS: Record<string, string> = {
  "marslab-v1-shell":  "App Shell",
  "marslab-v1-tiles":  "Map Tiles",
  "marslab-v1-api":    "API Data",
  "marslab-v1-images": "Images",
};

export default function OfflineIndicator({
  isOnline,
  isUpdateAvailable,
  cacheStats,
  onUpdate,
}: OfflineIndicatorProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [visible, setVisible] = useState(false);
  const [animateIn, setAnimateIn] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Determine if the banner should be shown
  const shouldShow = !isOnline || isUpdateAvailable;

  // Animate in/out
  useEffect(() => {
    if (shouldShow) {
      setVisible(true);
      // Trigger animation on next frame
      requestAnimationFrame(() => {
        requestAnimationFrame(() => setAnimateIn(true));
      });
    } else {
      setAnimateIn(false);
      // Wait for exit animation, then hide
      const timer = setTimeout(() => setVisible(false), 300);
      return () => clearTimeout(timer);
    }
  }, [shouldShow]);

  // Close tooltip on outside click
  useEffect(() => {
    if (!showTooltip) return;
    const handler = (e: MouseEvent) => {
      if (tooltipRef.current && !tooltipRef.current.contains(e.target as Node)) {
        setShowTooltip(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showTooltip]);

  const toggleTooltip = useCallback(() => {
    setShowTooltip((prev) => !prev);
  }, []);

  if (!visible) return null;

  const totalStats = cacheStats?.total;
  const usedPercent = totalStats
    ? Math.round((totalStats.estimatedBytes / totalStats.maxBytes) * 100)
    : 0;

  return (
    <div
      className="fixed top-0 left-0 right-0 z-[9999] transition-transform duration-300 ease-out"
      style={{ transform: animateIn ? "translateY(0)" : "translateY(-100%)" }}
    >
      {/* Offline Banner */}
      {!isOnline && (
        <div className="flex items-center justify-center gap-3 bg-amber-600/95 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur-sm">
          <span className="material-symbols-outlined text-[18px]">
            cloud_off
          </span>
          <span>You're offline — cached data available</span>
          {totalStats && totalStats.entries > 0 && (
            <button
              onClick={toggleTooltip}
              className="ml-2 rounded bg-amber-700/60 px-2 py-0.5 text-xs font-normal text-amber-100 hover:bg-amber-700/80 transition-colors"
            >
              {totalStats.entries} cached items ({formatBytes(totalStats.estimatedBytes)})
            </button>
          )}
        </div>
      )}

      {/* Update Available Banner */}
      {isUpdateAvailable && isOnline && (
        <div className="flex items-center justify-center gap-3 bg-blue-600/95 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur-sm">
          <span className="material-symbols-outlined text-[18px]">
            system_update
          </span>
          <span>A new version of MarsLab is available</span>
          <button
            onClick={onUpdate}
            className="ml-2 rounded bg-white/20 px-3 py-0.5 text-xs font-semibold text-white hover:bg-white/30 transition-colors"
          >
            Update Now
          </button>
        </div>
      )}

      {/* Cache Stats Tooltip */}
      {showTooltip && cacheStats && (
        <div
          ref={tooltipRef}
          className="mx-auto mt-1 max-w-xs rounded-lg border border-amber-500/30 bg-[#1a1f2e] p-4 shadow-xl"
        >
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-amber-400">
            Cached Data
          </h4>
          <div className="space-y-2">
            {Object.entries(BUCKET_LABELS).map(([key, label]) => {
              const bucket = cacheStats[key];
              if (!bucket || bucket.entries === 0) return null;
              return (
                <div
                  key={key}
                  className="flex items-center justify-between text-xs text-slate-300"
                >
                  <span>{label}</span>
                  <span className="tabular-nums text-slate-400">
                    {bucket.entries} items · {formatBytes(bucket.estimatedBytes)}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Usage bar */}
          {totalStats && (
            <div className="mt-3 pt-3 border-t border-slate-700">
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                <span>Cache usage</span>
                <span>{formatBytes(totalStats.estimatedBytes)} / {formatBytes(totalStats.maxBytes)}</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min(usedPercent, 100)}%`,
                    backgroundColor:
                      usedPercent > 80
                        ? "#ef4444"
                        : usedPercent > 50
                          ? "#f59e0b"
                          : "#22c55e",
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
