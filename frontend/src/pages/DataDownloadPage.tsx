import { useState, useEffect, useCallback, useRef } from "react";
import {
  searchProducts,
  searchSpatial,
  startDownload,
  getDownloadStatus,
  type SearchResult,
  type DownloadTask,
  type Instrument,
  type BoundingBox,
  formatBytes,
} from "../api/search";

// =============================================================================
// Types
// =============================================================================

type SearchMode = "id" | "spatial";

interface SpatialSearch {
  minlat: string;  // Southern boundary
  maxlat: string;  // Northern boundary
  westernlon: string;  // Western boundary
  easternlon: string;  // Eastern boundary
}

// =============================================================================
// Components
// =============================================================================

function SearchModeToggle({
  mode,
  onModeChange,
}: {
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
}) {
  return (
    <div className="flex items-center bg-bg-dark rounded-lg p-1 mr-2 border border-border-dark">
      <button
        onClick={() => onModeChange("id")}
        className={`px-4 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
          mode === "id"
            ? "bg-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        ID Search
      </button>
      <button
        onClick={() => onModeChange("spatial")}
        className={`px-4 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
          mode === "spatial"
            ? "bg-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        Spatial Search
      </button>
    </div>
  );
}

function SearchResultItem({
  result,
  isSelected,
  onClick,
}: {
  result: SearchResult;
  isSelected: boolean;
  onClick: () => void;
}) {
  // Determine download status
  const isPartial = !result.exists && (result.has_core || result.has_browse);
  const isComplete = result.exists;

  return (
    <div
      onClick={onClick}
      className={`rounded-lg p-3 cursor-pointer transition-colors ${
        isSelected
          ? "bg-primary/20 border border-primary/50"
          : "hover:bg-surface-dark border border-transparent"
      }`}
    >
      <div className="flex justify-between items-start mb-1">
        <p className="text-white font-bold font-mono text-sm uppercase">
          {result.product_id}
        </p>
        {isComplete ? (
          <span className="bg-emerald-500/20 text-emerald-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-emerald-500/30">
            <span className="material-symbols-outlined text-[10px]">check</span>
            Complete
          </span>
        ) : isPartial ? (
          <span className="bg-amber-500/20 text-amber-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-amber-500/30">
            <span className="material-symbols-outlined text-[10px]">warning</span>
            Partial
          </span>
        ) : (
          <span className="border border-border-dark text-[10px] px-2 py-0.5 rounded text-slate-500 font-bold uppercase">
            Remote
          </span>
        )}
      </div>
      <p className="text-slate-400 text-xs">
        {result.instrument.toUpperCase()} | {result.base_key}
      </p>
      {(result.lat !== null || result.lon !== null) && (
        <div className="flex items-center gap-2 mt-1.5 text-[10px] font-mono text-slate-500">
          <span className="material-symbols-outlined text-xs">location_on</span>
          <span>
            LAT: {result.lat?.toFixed(2) ?? "N/A"} | LON:{" "}
            {result.lon?.toFixed(2) ?? "N/A"}
          </span>
        </div>
      )}
      {isPartial && result.missing_files.length > 0 && (
        <div className="mt-1.5 text-[10px] text-amber-400">
          Missing: {result.missing_files.join(", ")}
        </div>
      )}
      <div className="mt-2 flex gap-2">
        <span className="text-[10px] bg-surface-dark px-2 py-0.5 rounded text-slate-400">
          {result.instrument.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

function FileStatusRow({ file }: { file: DownloadTask["files"][0] }) {
  const getStatusIcon = () => {
    switch (file.status) {
      case "completed":
        return (
          <span className="material-symbols-outlined text-emerald-500 text-sm">
            check_circle
          </span>
        );
      case "failed":
        return (
          <span className="material-symbols-outlined text-red-500 text-sm">
            error
          </span>
        );
      case "downloading":
        return (
          <span className="text-primary text-[10px] font-bold uppercase tracking-tight">
            Downloading
          </span>
        );
      case "processing":
        return (
          <span className="text-amber-400 text-[10px] font-bold uppercase tracking-tight">
            Processing
          </span>
        );
      default:
        return (
          <span className="text-slate-500 text-[10px] font-bold uppercase tracking-tight">
            Queued
          </span>
        );
    }
  };

  return (
    <tr>
      <td className="px-4 py-4 font-mono text-sm text-slate-400">
        {file.filename}
      </td>
      <td className="px-4 py-4 text-slate-400">
        {file.bytes_total ? formatBytes(file.bytes_total) : "-"}
      </td>
      <td className="px-4 py-4 w-1/3">
        <div className="w-full bg-surface-dark rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-full transition-all ${
              file.status === "completed"
                ? "bg-emerald-500"
                : file.status === "failed"
                ? "bg-red-500"
                : "bg-primary shadow-[0_0_8px_rgba(19,91,236,0.6)]"
            }`}
            style={{ width: `${file.progress_percent}%` }}
          />
        </div>
      </td>
      <td className="px-4 py-4 text-right">{getStatusIcon()}</td>
    </tr>
  );
}

function ProductPreview({
  result,
  onDownload,
  onDownloadMissing,
  isDownloading,
}: {
  result: SearchResult;
  onDownload: () => void;
  onDownloadMissing: (fileTypes: string[]) => void;
  isDownloading: boolean;
}) {
  const isPartial = !result.exists && (result.has_core || result.has_browse);
  const isComplete = result.exists;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-border-dark bg-surface-dark/50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-lg">
            {result.instrument === "crism" ? "satellite_alt" : result.instrument === "sharad" ? "radar" : "photo_camera"}
          </span>
          <span className="text-sm font-medium text-white">
            {result.instrument.toUpperCase()}
          </span>
        </div>
        {isComplete ? (
          <span className="bg-emerald-500/20 text-emerald-400 text-[10px] px-2 py-0.5 rounded flex items-center gap-1">
            <span className="material-symbols-outlined text-xs">check_circle</span>
            Complete
          </span>
        ) : isPartial ? (
          <span className="bg-amber-500/20 text-amber-400 text-[10px] px-2 py-0.5 rounded flex items-center gap-1">
            <span className="material-symbols-outlined text-xs">warning</span>
            Partial
          </span>
        ) : null}
      </div>

      {/* Content - scrollable */}
      <div className="p-4 flex-1 overflow-y-auto flex flex-col gap-3">
        {/* Title */}
        <div>
          <h2 className="text-lg font-bold text-white font-mono break-all">
            {result.product_id}
          </h2>
          <p className="text-slate-400 text-xs mt-1">
            Base: <span className="text-primary font-mono">{result.base_key}</span>
          </p>
        </div>

        {/* Details - compact grid */}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-bg-dark rounded p-2 border border-border-dark">
            <p className="text-[9px] uppercase text-slate-500">Instrument</p>
            <p className="text-white">{result.instrument.toUpperCase()}</p>
          </div>
          <div className="bg-bg-dark rounded p-2 border border-border-dark">
            <p className="text-[9px] uppercase text-slate-500">Status</p>
            <p className={isComplete ? "text-emerald-400" : isPartial ? "text-amber-400" : "text-slate-400"}>
              {isComplete ? "Complete" : isPartial ? "Partial" : "Remote"}
            </p>
          </div>
          {(result.lat !== null || result.lon !== null) && (
            <div className="bg-bg-dark rounded p-2 border border-border-dark col-span-2">
              <p className="text-[9px] uppercase text-slate-500">Location</p>
              <p className="text-white font-mono text-xs">
                {result.lat?.toFixed(3)}°, {result.lon?.toFixed(3)}°
              </p>
            </div>
          )}
        </div>

        {/* Bundle Info - compact */}
        <div className="bg-bg-dark rounded p-2 border border-border-dark">
          <p className="text-[9px] uppercase text-slate-500 mb-1">Bundle Contents</p>
          {result.instrument === "crism" ? (
            <div className="grid grid-cols-2 gap-1 text-xs">
              <div className={result.has_core ? "text-emerald-400" : "text-slate-500"}>
                {result.has_core ? "✓" : "○"} Core (.img, .lbl)
              </div>
              <div className={!result.missing_files.includes("hdr") ? "text-emerald-400" : "text-slate-500"}>
                {!result.missing_files.includes("hdr") ? "✓" : "○"} Header (.hdr)
              </div>
              <div className={!result.missing_files.includes("tab") ? "text-emerald-400" : "text-slate-500"}>
                {!result.missing_files.includes("tab") ? "✓" : "○"} Wavelength (.tab)
              </div>
              <div className={result.has_browse ? "text-emerald-400" : "text-slate-500"}>
                {result.has_browse ? "✓" : "○"} Browse images
              </div>
            </div>
          ) : result.instrument === "sharad" ? (
            <div className="grid grid-cols-2 gap-1 text-xs">
              <div className={result.has_browse ? "text-emerald-400" : "text-slate-500"}>
                {result.has_browse ? "✓" : "○"} Quickview (.jpg)
              </div>
              <div className="text-slate-500">
                ○ High-res (.tif)
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-300">.JP2, .lbl → .tif (GDAL converted)</p>
          )}
        </div>

        {/* Missing files info */}
        {isPartial && result.missing_files.length > 0 && (
          <div className="bg-amber-500/10 rounded p-2 border border-amber-500/30">
            <p className="text-[9px] uppercase text-amber-400 mb-1">Missing Files</p>
            <p className="text-xs text-amber-300">{result.missing_files.join(", ")}</p>
          </div>
        )}
      </div>

      {/* Download Buttons - fixed at bottom */}
      <div className="p-3 border-t border-border-dark shrink-0 space-y-2">
        {isComplete ? (
          <button
            disabled
            className="w-full py-3 px-4 rounded-lg font-bold flex items-center justify-center gap-2 bg-emerald-500/20 text-emerald-400 cursor-not-allowed border border-emerald-500/30"
          >
            <span className="material-symbols-outlined text-lg">check_circle</span>
            <span>All Files Downloaded</span>
          </button>
        ) : isPartial ? (
          <>
            <button
              onClick={() => onDownloadMissing(result.missing_files)}
              disabled={isDownloading}
              className={`w-full py-3 px-4 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${
                isDownloading
                  ? "bg-primary/50 text-white cursor-wait"
                  : "bg-primary hover:bg-primary/80 text-white"
              }`}
            >
              {isDownloading ? (
                <>
                  <span className="material-symbols-outlined text-lg animate-spin">progress_activity</span>
                  <span>Starting...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-lg">download</span>
                  <span>Download Missing ({result.missing_files.length})</span>
                </>
              )}
            </button>
            <button
              onClick={onDownload}
              disabled={isDownloading}
              className="w-full py-2 px-4 rounded-lg text-sm flex items-center justify-center gap-2 transition-all border border-border-dark text-slate-400 hover:text-white hover:border-slate-500"
            >
              <span className="material-symbols-outlined text-sm">refresh</span>
              <span>Re-download All</span>
            </button>
          </>
        ) : (
          <button
            onClick={onDownload}
            disabled={isDownloading}
            className={`w-full py-3 px-4 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${
              isDownloading
                ? "bg-primary/50 text-white cursor-wait"
                : "bg-primary hover:bg-primary/80 text-white"
            }`}
          >
            {isDownloading ? (
              <>
                <span className="material-symbols-outlined text-lg animate-spin">progress_activity</span>
                <span>Starting...</span>
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-lg">download</span>
                <span>Download Bundle</span>
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

function DownloadManifest({
  task,
  onClose,
}: {
  task: DownloadTask | null;
  onClose: () => void;
}) {
  if (!task) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2">
            download
          </span>
          <p>Select a product to download</p>
        </div>
      </div>
    );
  }

  const statusColor =
    task.status === "completed"
      ? "bg-emerald-500/10 border-emerald-500/20"
      : task.status === "failed"
      ? "bg-red-500/10 border-red-500/20"
      : "bg-primary/10 border-primary/20";

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div
        className={`px-6 py-3 border-b ${statusColor} flex items-center justify-between`}
      >
        <div className="flex items-center gap-3">
          {task.status === "downloading" || task.status === "processing" ? (
            <div className="size-2 bg-primary rounded-full animate-pulse" />
          ) : task.status === "completed" ? (
            <div className="size-2 bg-emerald-500 rounded-full" />
          ) : task.status === "failed" ? (
            <div className="size-2 bg-red-500 rounded-full" />
          ) : (
            <div className="size-2 bg-slate-500 rounded-full" />
          )}
          <span className="text-sm font-medium text-white">
            Target: <span className="text-primary font-mono">{task.target_dir}</span>
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined text-sm">close</span>
        </button>
      </div>

      {/* Content */}
      <div className="p-6 flex-1 flex flex-col gap-6 overflow-hidden">
        {/* Title */}
        <div className="flex justify-between items-end">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">
              {task.base_key.toUpperCase()} Manifest
            </h2>
            <p className="text-slate-400 text-sm italic">
              {task.instrument.toUpperCase()} |{" "}
              {task.status === "completed"
                ? "Download complete"
                : task.status === "failed"
                ? task.error
                : `${task.progress_percent.toFixed(0)}% complete`}
            </p>
          </div>
          <div className="text-xs text-slate-400">
            {task.total_bytes > 0 && (
              <>
                {formatBytes(task.downloaded_bytes)} / {formatBytes(task.total_bytes)}
              </>
            )}
          </div>
        </div>

        {/* File table */}
        <div className="flex-1 border border-border-dark rounded-lg overflow-hidden bg-bg-dark/50">
          <div className="overflow-y-auto max-h-[400px]">
            <table className="w-full text-left border-collapse">
              <thead className="bg-surface-dark text-slate-400 text-[10px] uppercase tracking-widest sticky top-0">
                <tr>
                  <th className="px-4 py-3 font-semibold">File Entity</th>
                  <th className="px-4 py-3 font-semibold">Size</th>
                  <th className="px-4 py-3 font-semibold">Progress</th>
                  <th className="px-4 py-3 font-semibold text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-dark text-sm">
                {task.files.map((file) => (
                  <FileStatusRow key={file.filename} file={file} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function DataDownloadPage() {
  // Search state
  const [searchMode, setSearchMode] = useState<SearchMode>("id");
  const [query, setQuery] = useState("");
  const [spatialSearch, setSpatialSearch] = useState<SpatialSearch>({
    minlat: "",
    maxlat: "",
    westernlon: "",
    easternlon: "",
  });
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Selection state
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

  // Download state
  const [downloadTask, setDownloadTask] = useState<DownloadTask | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  // Polling interval ref
  const pollIntervalRef = useRef<number | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Handle search
  const handleSearch = useCallback(async () => {
    setIsSearching(true);
    setSearchError(null);
    setSearchResults([]);

    try {
      if (searchMode === "id") {
        if (!query.trim()) {
          setSearchError("Enter a product ID to search");
          return;
        }
        const response = await searchProducts(query.trim(), undefined, 12);
        setSearchResults(response.results);
      } else {
        const minlat = parseFloat(spatialSearch.minlat);
        const maxlat = parseFloat(spatialSearch.maxlat);
        const westernlon = parseFloat(spatialSearch.westernlon);
        const easternlon = parseFloat(spatialSearch.easternlon);

        if (isNaN(minlat) || isNaN(maxlat) || isNaN(westernlon) || isNaN(easternlon)) {
          setSearchError("Enter valid latitude and longitude boundaries");
          return;
        }

        if (minlat > maxlat) {
          setSearchError("South latitude must be less than North latitude");
          return;
        }

        const bbox: BoundingBox = { minlat, maxlat, westernlon, easternlon };
        const response = await searchSpatial(bbox, undefined, 20);
        setSearchResults(response.results);
      }
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setIsSearching(false);
    }
  }, [searchMode, query, spatialSearch]);

  // Poll for download status
  const startPolling = useCallback((taskId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const task = await getDownloadStatus(taskId);
        setDownloadTask(task);

        if (task.status === "completed" || task.status === "failed") {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setIsDownloading(false);

          // Refresh search results to update existence status
          if (task.status === "completed") {
            handleSearch();
          }
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 1000);
  }, [handleSearch]);

  // Handle download (all files)
  const handleDownload = useCallback(async () => {
    if (!selectedResult) return;

    setIsDownloading(true);

    try {
      const task = await startDownload(
        selectedResult.product_id,
        selectedResult.instrument as Instrument,
        selectedResult.lat ?? undefined,
        selectedResult.lon ?? undefined
      );
      setDownloadTask(task);
      startPolling(task.task_id);
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Download failed");
      setIsDownloading(false);
    }
  }, [selectedResult, startPolling]);

  // Handle download missing files only
  const handleDownloadMissing = useCallback(async (missingTypes: string[]) => {
    if (!selectedResult) return;

    setIsDownloading(true);

    // Map missing file type names to API file_types
    const fileTypesMap: Record<string, string> = {
      "img": "core",
      "lbl": "core",
      "hdr": "header",
      "tab": "wavelength",
      "browse": "browse",
    };

    const fileTypes = [...new Set(missingTypes.map(t => fileTypesMap[t] || t))];

    try {
      const task = await startDownload(
        selectedResult.product_id,
        selectedResult.instrument as Instrument,
        selectedResult.lat ?? undefined,
        selectedResult.lon ?? undefined,
        fileTypes
      );
      setDownloadTask(task);
      startPolling(task.task_id);
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Download failed");
      setIsDownloading(false);
    }
  }, [selectedResult, startPolling]);

  // Handle Enter key in search
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="h-screen bg-bg-dark text-white flex flex-col overflow-hidden">
      {/* Header - compact */}
      <header className="flex items-center justify-between border-b border-border-dark px-6 py-2 shrink-0">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-white">
            <span className="material-symbols-outlined text-primary text-2xl">
              rocket_launch
            </span>
            <h2 className="text-white text-base font-bold">MarsLab</h2>
          </div>
          <div className="h-5 w-px bg-border-dark" />
          <nav className="flex items-center gap-4">
            <a
              href="/"
              className="text-slate-400 hover:text-white text-sm font-medium transition-colors"
            >
              Workbench
            </a>
            <a
              href="/download"
              className="text-primary text-sm font-medium border-b-2 border-primary pb-0.5"
            >
              Data Download
            </a>
          </nav>
        </div>
      </header>

      {/* Main content - scrollable */}
      <main className="flex-1 flex flex-col overflow-auto max-w-[1600px] mx-auto w-full px-4 md:px-6 py-4 gap-4">
        {/* Search section */}
        <section className="w-full flex flex-col gap-4">
          <div className="flex items-center gap-4">
            <div className="flex-1 bg-surface-dark rounded-xl p-1.5 flex items-center shadow-lg border border-border-dark">
              <SearchModeToggle mode={searchMode} onModeChange={setSearchMode} />

              <div className="flex-1 flex items-center px-2 gap-3">
                <span className="material-symbols-outlined text-slate-400">
                  search
                </span>

                {searchMode === "id" ? (
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="bg-transparent border-none text-white focus:ring-0 w-full text-lg placeholder:text-slate-500 font-mono focus:outline-none"
                    placeholder="Enter Product ID (e.g. frt00009312, ESP_045857_2350, S_00195401)"
                  />
                ) : (
                  <div className="flex items-center gap-3 flex-1 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">S:</span>
                      <input
                        type="text"
                        value={spatialSearch.minlat}
                        onChange={(e) =>
                          setSpatialSearch((s) => ({ ...s, minlat: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="35"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">N:</span>
                      <input
                        type="text"
                        value={spatialSearch.maxlat}
                        onChange={(e) =>
                          setSpatialSearch((s) => ({ ...s, maxlat: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="70"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">W:</span>
                      <input
                        type="text"
                        value={spatialSearch.westernlon}
                        onChange={(e) =>
                          setSpatialSearch((s) => ({ ...s, westernlon: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="-130"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">E:</span>
                      <input
                        type="text"
                        value={spatialSearch.easternlon}
                        onChange={(e) =>
                          setSpatialSearch((s) => ({ ...s, easternlon: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="150"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2 px-4 border-l border-border-dark group relative">
                <span className="material-symbols-outlined text-slate-500 cursor-help">
                  info
                </span>
                <div className="absolute bottom-full mb-2 right-0 w-72 p-3 bg-bg-dark border border-border-dark rounded-lg text-[10px] leading-relaxed text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                  {searchMode === "id"
                    ? "Search by CRISM, HiRISE, or SHARAD product ID. Partial matches are supported."
                    : "Search by bounding box. S/N = latitude range (-90 to 90), W/E = longitude range (-180 to 360). Example: Arcadia Planitia = S:35, N:70, W:-130, E:150"}
                </div>
              </div>

              <button
                onClick={handleSearch}
                disabled={isSearching}
                className="bg-primary hover:bg-primary/80 disabled:opacity-50 text-white font-bold py-3 px-8 rounded-lg transition-all flex items-center gap-2 shrink-0"
              >
                {isSearching ? (
                  <>
                    <span className="animate-spin material-symbols-outlined text-sm">
                      progress_activity
                    </span>
                    <span>Searching...</span>
                  </>
                ) : (
                  <>
                    <span>SCAN ODE</span>
                    <span className="material-symbols-outlined text-sm">
                      database
                    </span>
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* Results & Manifest panels */}
        <div className="flex-1 flex gap-6 overflow-hidden min-h-0">
          {/* Left panel - Search Results */}
          <aside className="w-1/3 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden">
            <div className="p-4 border-b border-border-dark flex justify-between items-center">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">
                Search Results ({searchResults.length})
              </h3>
              <span className="material-symbols-outlined text-slate-400 cursor-pointer">
                filter_list
              </span>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-2">
              {searchError && (
                <div className="p-4 text-red-400 text-sm">{searchError}</div>
              )}
              {searchResults.length === 0 && !searchError && !isSearching && (
                <div className="p-4 text-slate-500 text-sm text-center">
                  Enter a search query and click "SCAN ODE"
                </div>
              )}
              {searchResults.map((result) => (
                <SearchResultItem
                  key={result.product_id}
                  result={result}
                  isSelected={selectedResult?.product_id === result.product_id}
                  onClick={() => {
                    setSelectedResult(result);
                    // Clear download task to show new product preview
                    if (downloadTask?.status === "completed" || downloadTask?.status === "failed") {
                      setDownloadTask(null);
                    }
                  }}
                />
              ))}
            </div>

          </aside>

          {/* Right panel - Product Preview or Download Manifest */}
          <section className="flex-1 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden">
            {downloadTask ? (
              <DownloadManifest
                task={downloadTask}
                onClose={() => setDownloadTask(null)}
              />
            ) : selectedResult ? (
              <ProductPreview
                result={selectedResult}
                onDownload={handleDownload}
                onDownloadMissing={handleDownloadMissing}
                isDownloading={isDownloading}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-500">
                <div className="text-center">
                  <span className="material-symbols-outlined text-5xl mb-3 text-slate-600">
                    search
                  </span>
                  <p className="text-lg">Search for a product</p>
                  <p className="text-sm mt-1">Enter a product ID and click "SCAN ODE"</p>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
