import { useState, useEffect, useCallback, useRef } from "react";
import { Link, useSearchParams } from "react-router-dom";
import toast from "react-hot-toast";
import useIsMobile from "../hooks/useIsMobile";
import {
  searchProducts,
  searchSpatial,
  searchByPoint,
  searchProximity,
  startDownload,
  getDownloadStatus,
  listDownloads,
  cancelDownload,
  cancelAllDownloads,
  clearDownloadHistory,
  retryDownload,
  startSmartSearch,
  streamSmartSearch,
  listLocalFiles,
  downloadLocalFile,
  downloadLocalZip,
  type SearchResult,
  type DownloadTask,
  type Instrument,
  type BoundingBox,
  type PointSearchResponse,
  type PointSearchResult,
  type SmartSearchResponse,
  type SmartSearchEvent,
  type SmartProductSelection,
  type ProximityResponse,
  type ProximityResult,
  type LocalFileInfo,
  formatBytes,
} from "../api/search";

// =============================================================================
// Types
// =============================================================================

type PageTab = "search" | "downloads";
type SearchMode = "id" | "spatial" | "point" | "ai" | "product";

// Dataset types for selection
type DatasetType = "crism" | "crism_trr3" | "hirise" | "sharad" | "sharad_highres" | "hirise_dtm";

interface DatasetSelection {
  crism: boolean;
  crism_trr3: boolean;
  hirise: boolean;
  sharad: boolean;
  sharad_highres: boolean;
  hirise_dtm: boolean;
}

interface SpatialSearch {
  minlat: string;  // Southern boundary
  maxlat: string;  // Northern boundary
  westernlon: string;  // Western boundary
  easternlon: string;  // Eastern boundary
}

interface PointSearch {
  lat: string;
  lon: string;
  radius: string;  // Search radius in degrees
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
    <div className="flex items-center bg-bg-dark rounded-lg p-1 mr-2 border border-border-dark overflow-x-auto shrink-0">
      <button
        onClick={() => onModeChange("id")}
        className={`px-2 md:px-3 py-1 md:py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all whitespace-nowrap ${
          mode === "id"
            ? "bg-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        ID
      </button>
      <button
        onClick={() => onModeChange("spatial")}
        className={`px-2 md:px-3 py-1 md:py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all whitespace-nowrap ${
          mode === "spatial"
            ? "bg-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        Spatial
      </button>
      <button
        onClick={() => onModeChange("point")}
        className={`px-2 md:px-3 py-1 md:py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all whitespace-nowrap ${
          mode === "point"
            ? "bg-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        Coord
      </button>
      <button
        onClick={() => onModeChange("product")}
        className={`px-2 md:px-3 py-1 md:py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all flex items-center gap-1 whitespace-nowrap ${
          mode === "product"
            ? "bg-purple-600 text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        <span className="material-symbols-outlined text-xs">hub</span>
        Product
      </button>
      <button
        onClick={() => onModeChange("ai")}
        className={`px-2 md:px-3 py-1 md:py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all flex items-center gap-1 whitespace-nowrap ${
          mode === "ai"
            ? "bg-gradient-to-r from-purple-500 to-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        <span className="material-symbols-outlined text-xs">auto_awesome</span>
        AI
      </button>
    </div>
  );
}

function DatasetSelector({
  selection,
  onSelectionChange,
}: {
  selection: DatasetSelection;
  onSelectionChange: (selection: DatasetSelection) => void;
}) {
  const datasets: { key: DatasetType; label: string; icon: string }[] = [
    { key: "crism", label: "CRISM", icon: "satellite_alt" },
    { key: "crism_trr3", label: "CRISM TRR3", icon: "science" },
    { key: "hirise", label: "HiRISE", icon: "photo_camera" },
    { key: "sharad", label: "SHARAD", icon: "radar" },
    { key: "sharad_highres", label: "SHARAD Hi-Res", icon: "science" },
    { key: "hirise_dtm", label: "HiRISE DTM", icon: "terrain" },
  ];

  const toggleDataset = (key: DatasetType) => {
    onSelectionChange({
      ...selection,
      [key]: !selection[key],
    });
  };

  const selectedCount = Object.values(selection).filter(Boolean).length;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-slate-500 text-[10px] font-bold uppercase tracking-wider mr-1">
        Datasets:
      </span>
      {datasets.map(({ key, label, icon }) => (
        <button
          key={key}
          onClick={() => toggleDataset(key)}
          className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-all border ${
            selection[key]
              ? "bg-primary/20 text-primary border-primary/50"
              : "bg-bg-dark text-slate-500 border-border-dark hover:text-white hover:border-slate-500"
          }`}
        >
          <span className="material-symbols-outlined text-xs">{icon}</span>
          {label}
        </button>
      ))}
      {selectedCount === 0 && (
        <span className="text-amber-400 text-[10px] ml-2">
          Select at least one dataset
        </span>
      )}
    </div>
  );
}

/**
 * Save to Local PC button — fetches PDS URL on click, then triggers browser download.
 * Works for any product (Remote or Local) as long as it's CRISM or HiRISE.
 */
function SaveToLocalButton({ productId, instrument }: { productId: string; instrument: string }) {
  const [loading, setLoading] = useState(false);
  const inst = instrument.toLowerCase();
  if (inst !== "crism" && inst !== "hirise" && inst !== "crism_trr3") return null;

  const handleClick = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      const endpoint = inst === "hirise"
        ? `/api/product-urls/hirise/${productId}`
        : `/api/product-urls/crism/${productId}`;
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error("Failed");
      const data = await res.json() as PdsUrls;
      // Pick the main file URL
      const url = data.img_url || data.jp2_url;
      if (url) {
        const a = document.createElement("a");
        a.href = url;
        a.download = data.img_filename || data.jp2_filename || productId;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      } else {
        toast.error("No download URL found");
      }
    } catch {
      toast.error("Failed to get download URL");
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={loading}
      className="flex items-center gap-0.5 px-2 py-1 rounded text-[10px] font-bold uppercase border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10 transition-colors disabled:opacity-50"
    >
      <span className={`material-symbols-outlined text-xs ${loading ? "animate-spin" : ""}`}>
        {loading ? "progress_activity" : "save_alt"}
      </span>
      <span>Save to Local</span>
    </button>
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
      <div className="mt-2 flex items-center justify-between">
        <span className="text-[10px] bg-surface-dark px-2 py-0.5 rounded text-slate-400">
          {result.instrument.toUpperCase()}
        </span>
        <SaveToLocalButton productId={result.product_id} instrument={result.instrument} />
      </div>
    </div>
  );
}

function PointResultItem({
  result,
  instrument,
  isSelected,
  onClick,
  onDownload,
  isDownloading,
  downloadable = true,
}: {
  result: PointSearchResult;
  instrument: string;
  isSelected: boolean;
  onClick: () => void;
  onDownload: () => void;
  isDownloading: boolean;
  downloadable?: boolean;
}) {
  const displayInstrument = instrument === "SHARAD_HIGHRES" ? "SHARAD Hi-Res" : instrument;
  const isComplete = result.exists;
  const isPartial = !result.exists && (result.has_core || result.has_browse);

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
            Local
          </span>
        ) : isPartial ? (
          <span className="bg-amber-500/20 text-amber-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase border border-amber-500/30">
            Partial
          </span>
        ) : (
          <span className="border border-border-dark text-[10px] px-2 py-0.5 rounded text-slate-500 font-bold uppercase">
            Remote
          </span>
        )}
      </div>
      <div className="flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="bg-surface-dark px-2 py-0.5 rounded">{displayInstrument}</span>
          {result.distance_km !== null && (
            <span className="text-primary font-mono text-[10px] font-bold">
              {result.distance_km.toFixed(1)} km
            </span>
          )}
        </div>
        {isSelected && downloadable && !isComplete && (
          <div className="flex items-center gap-1.5">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDownload();
              }}
              disabled={isDownloading}
              className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold uppercase transition-all ${
                isDownloading
                  ? "bg-primary/50 text-white cursor-wait"
                  : "bg-primary hover:bg-primary/80 text-white"
              }`}
            >
              {isDownloading ? (
                <>
                  <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                  <span>Starting...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-xs">download</span>
                  <span>Download</span>
                </>
              )}
            </button>
            <SaveToLocalButton productId={result.product_id} instrument={instrument} />
          </div>
        )}
        {isSelected && (isComplete || isPartial) && (
          <div className="flex items-center gap-1.5">
            <button
              onClick={(e) => {
                e.stopPropagation();
                downloadLocalZip(result.product_id, instrument.toLowerCase() as Instrument);
              }}
              className="text-[10px] text-cyan-400 hover:text-cyan-300 font-bold uppercase flex items-center gap-0.5 px-2 py-0.5 rounded border border-cyan-500/30 hover:bg-cyan-500/10 transition-colors"
            >
              <span className="material-symbols-outlined text-xs">folder_zip</span>
              ZIP
            </button>
            <SaveToLocalButton productId={result.product_id} instrument={instrument} />
          </div>
        )}
      </div>
    </div>
  );
}

function PointSearchResults({
  results,
  query,
  error,
  onDownload,
  onBatchDownload,
  isDownloading,
  downloadingProductId,
  batchProgress,
}: {
  results: PointSearchResponse | null;
  query: { lat: number; lon: number } | null;
  error?: string | null;
  onDownload: (productId: string, instrument: string) => void;
  onBatchDownload: (items: Array<{ product_id: string; instrument: string; lat?: number | null; lon?: number | null; exists?: boolean }>, instrumentLabel: string) => void;
  isDownloading: boolean;
  downloadingProductId: string | null;
  batchProgress: { completed: number; total: number; instrument: string } | null;
}) {
  const [selectedItem, setSelectedItem] = useState<{ productId: string; instrument: string } | null>(null);

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2 text-red-400">
            error
          </span>
          <p className="text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  if (!results || !query) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2">
            location_on
          </span>
          <p>Enter a coordinate and search</p>
        </div>
      </div>
    );
  }

  const totalCount = results.total_count;
  const instruments: { key: keyof typeof results.results; label: string; icon: string; apiKey: string; downloadable: boolean }[] = [
    { key: "CRISM", label: "CRISM", icon: "satellite_alt", apiKey: "crism", downloadable: true },
    { key: "HIRISE", label: "HiRISE", icon: "photo_camera", apiKey: "hirise", downloadable: true },
    { key: "SHARAD", label: "SHARAD", icon: "radar", apiKey: "sharad", downloadable: true },
    { key: "SHARAD_HIGHRES", label: "SHARAD Hi-Res", icon: "science", apiKey: "sharad_highres", downloadable: true },
    { key: "HIRISE_DTM", label: "HiRISE DTM", icon: "terrain", apiKey: "hirise_dtm", downloadable: false },
  ];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-border-dark bg-surface-dark/50 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-lg">
              location_on
            </span>
            <span className="text-sm font-medium text-white">
              Coordinate Search Results
            </span>
          </div>
          <span className="text-slate-400 text-sm">
            {totalCount} dataset{totalCount !== 1 ? "s" : ""} found
          </span>
        </div>
        <div className="text-xs text-slate-500 mt-1 font-mono">
          Query: {query.lat.toFixed(4)}° lat, {results.query.lon.toFixed(4)}° lon | Radius: {results.query.radius_deg}°
        </div>
      </div>

      {/* Results grouped by instrument */}
      <div className="p-4 flex-1 overflow-y-auto space-y-4">
        {totalCount === 0 ? (
          <div className="text-center text-slate-500 py-8">
            <span className="material-symbols-outlined text-4xl mb-2">
              search_off
            </span>
            <p>No datasets found at this coordinate</p>
            <p className="text-xs mt-1">
              Try increasing the search radius or search a different location
            </p>
          </div>
        ) : (
          instruments.map(({ key, label, icon, apiKey, downloadable }) => {
            const instrumentResults = results.results[key];
            if (!instrumentResults || instrumentResults.length === 0) return null;

            return (
              <div key={key} className="bg-bg-dark rounded-lg border border-border-dark overflow-hidden">
                <div className="px-3 py-2 border-b border-border-dark flex items-center gap-2 bg-surface-dark/50">
                  <span className="material-symbols-outlined text-primary text-sm">{icon}</span>
                  <span className="text-sm font-bold text-white">{label}</span>
                  <span className="text-xs text-slate-500">({instrumentResults.length})</span>
                  {!downloadable && (
                    <span className="text-[9px] text-amber-400 ml-auto">Download not supported</span>
                  )}
                  {downloadable && instrumentResults.some(r => !r.exists) && (() => {
                    const notDownloaded = instrumentResults.filter(r => !r.exists).length;
                    const isActive = batchProgress?.instrument === apiKey;
                    return (
                      <button
                        onClick={(e) => { e.stopPropagation(); onBatchDownload(instrumentResults.map(r => ({ ...r, instrument: apiKey })), apiKey); }}
                        disabled={isActive || batchProgress !== null}
                        className="ml-auto flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-all disabled:opacity-50"
                      >
                        {isActive ? (
                          <><span className="material-symbols-outlined text-xs animate-spin">progress_activity</span><span>{batchProgress.completed}/{batchProgress.total}</span></>
                        ) : (
                          <><span className="material-symbols-outlined text-xs">download</span><span>All ({notDownloaded})</span></>
                        )}
                      </button>
                    );
                  })()}
                </div>
                <div className="max-h-48 overflow-y-auto divide-y divide-border-dark/50">
                  {instrumentResults.map((r) => (
                    <PointResultItem
                      key={r.product_id}
                      result={r}
                      instrument={key}
                      isSelected={selectedItem?.productId === r.product_id && selectedItem?.instrument === key}
                      onClick={() => setSelectedItem({ productId: r.product_id, instrument: key })}
                      onDownload={() => onDownload(r.product_id, apiKey)}
                      isDownloading={isDownloading && downloadingProductId === r.product_id}
                      downloadable={downloadable}
                    />
                  ))}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Smart AI Search Results Components
// =============================================================================

function SmartSearchPanel({
  response,
  stageMessage,
  error,
  isRunning,
  onCancel,
}: {
  response: SmartSearchResponse | null;
  stageMessage: string;
  error: string | null;
  isRunning: boolean;
  onCancel: () => void;
}) {
  // Running / streaming state
  if (isRunning) {
    const stageIcons: Record<string, string> = {
      parsing: "psychology",
      searching: "search",
      analyzing: "auto_awesome",
      downloading: "download",
    };
    const stageName = stageMessage.toLowerCase();
    const icon = Object.entries(stageIcons).find(([k]) => stageName.includes(k))?.[1] || "progress_activity";

    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center max-w-sm">
          <span className="material-symbols-outlined text-4xl mb-3 animate-spin text-purple-400">
            {icon}
          </span>
          <p className="text-purple-400 font-bold">{stageMessage || "Starting..."}</p>
          <p className="text-xs text-slate-500 mt-2">Llama is thinking...</p>
          <button
            onClick={onCancel}
            className="mt-4 px-4 py-1.5 rounded-lg text-xs font-bold text-slate-400 border border-border-dark hover:text-white hover:border-slate-500 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  // Error state
  if (error && !response) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center max-w-sm">
          <span className="material-symbols-outlined text-4xl mb-2 text-red-400">error</span>
          <p className="text-red-400 text-sm">{error}</p>
          {error.includes("Ollama") && (
            <p className="text-xs text-slate-500 mt-2 font-mono">ollama serve && ollama pull llama3.3</p>
          )}
        </div>
      </div>
    );
  }

  // Initial state
  if (!response) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-5xl mb-3 bg-gradient-to-r from-purple-500 to-primary text-transparent bg-clip-text">
            psychology
          </span>
          <p className="text-lg">Smart AI Search</p>
          <p className="text-sm mt-1 max-w-md">
            Describe what data you want. Llama will reason, search, pick the best products, and download them automatically.
          </p>
          <div className="mt-4 text-xs text-slate-600 space-y-1">
            <p>Examples:</p>
            <p className="text-slate-500">"Download CRISM that intersects SHARAD high-res and HiRISE DTM"</p>
            <p className="text-slate-500">"Get SHARAD tracks in Arcadia Planitia"</p>
            <p className="text-slate-500">"HiRISE imagery over Jezero Crater"</p>
          </div>
        </div>
      </div>
    );
  }

  // Results state
  const instrumentConfig: Record<string, { label: string; icon: string }> = {
    crism: { label: "CRISM", icon: "satellite_alt" },
    hirise: { label: "HiRISE", icon: "photo_camera" },
    sharad: { label: "SHARAD", icon: "radar" },
    sharad_highres: { label: "SHARAD Hi-Res", icon: "science" },
    ctx: { label: "CTX", icon: "camera" },
    hirise_dtm: { label: "HiRISE DTM", icon: "terrain" },
  };

  // Group products by instrument
  const grouped = response.selected_products.reduce((acc, p) => {
    const key = p.instrument.toLowerCase();
    if (!acc[key]) acc[key] = [];
    acc[key].push(p);
    return acc;
  }, {} as Record<string, SmartProductSelection[]>);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header: Reasoning + Summary */}
      <div className="px-4 py-3 border-b border-border-dark bg-gradient-to-r from-purple-500/10 to-primary/10 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-purple-400 text-lg">psychology</span>
            <span className="text-sm font-bold text-white">Llama Smart Search</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <span>{response.total_found} found</span>
            <span className="text-primary font-bold">{response.total_selected} selected</span>
            {response.total_downloading > 0 && (
              <span className="text-amber-400">{response.total_downloading} downloading</span>
            )}
            {response.total_already_local > 0 && (
              <span className="text-emerald-400">{response.total_already_local} local</span>
            )}
          </div>
        </div>

        {/* Reasoning from Llama */}
        {response.reasoning && (
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3 mt-2">
            <div className="flex items-start gap-2">
              <span className="material-symbols-outlined text-purple-400 text-sm mt-0.5 shrink-0">auto_awesome</span>
              <p className="text-sm text-slate-300 leading-relaxed">{response.reasoning}</p>
            </div>
          </div>
        )}
      </div>

      {/* Product List */}
      <div className="p-4 flex-1 overflow-y-auto space-y-4">
        {response.selected_products.length === 0 ? (
          <div className="text-center text-slate-500 py-8">
            <span className="material-symbols-outlined text-4xl mb-2">search_off</span>
            <p>No products found</p>
            <p className="text-xs mt-1">Try a different region or instruments</p>
          </div>
        ) : (
          Object.entries(grouped).map(([instrument, items]) => {
            const config = instrumentConfig[instrument] || { label: instrument.toUpperCase(), icon: "science" };
            return (
              <div key={instrument} className="bg-bg-dark rounded-lg border border-border-dark overflow-hidden">
                <div className="px-3 py-2 border-b border-border-dark flex items-center gap-2 bg-surface-dark/50">
                  <span className="material-symbols-outlined text-primary text-sm">{config.icon}</span>
                  <span className="text-sm font-bold text-white">{config.label}</span>
                  <span className="text-xs text-slate-500">({items.length})</span>
                </div>
                <div className="max-h-80 overflow-y-auto divide-y divide-border-dark/50">
                  {items.map((p) => (
                    <SmartResultItem key={p.product_id} product={p} />
                  ))}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function SmartResultItem({ product }: { product: SmartProductSelection }) {
  const isLocal = product.already_local;
  const isDownloading = !!product.download_task_id;
  const instrumentLabel = product.instrument === "SHARAD_HIGHRES" ? "SHARAD Hi-Res" : product.instrument;

  return (
    <div className="px-3 py-2.5">
      <div className="flex justify-between items-start mb-1">
        <p className="text-white font-bold font-mono text-sm uppercase">{product.product_id}</p>
        {isLocal ? (
          <span className="bg-emerald-500/20 text-emerald-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-emerald-500/30 shrink-0">
            <span className="material-symbols-outlined text-[10px]">check</span>
            Local
          </span>
        ) : isDownloading ? (
          <span className="bg-amber-500/20 text-amber-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-amber-500/30 shrink-0">
            <span className="material-symbols-outlined text-[10px] animate-spin">progress_activity</span>
            Downloading
          </span>
        ) : (
          <span className="border border-border-dark text-[10px] px-2 py-0.5 rounded text-slate-500 font-bold uppercase shrink-0">
            Remote
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 text-xs text-slate-400 mb-1.5">
        <span className="bg-surface-dark px-2 py-0.5 rounded">{instrumentLabel}</span>
        {product.distance_km !== null && (
          <span className="text-primary font-mono text-[10px] font-bold">
            {product.distance_km.toFixed(1)} km
          </span>
        )}
      </div>
      {/* Per-product reasoning from Llama */}
      {product.reason && (
        <p className="text-[11px] text-slate-500 leading-snug italic">{product.reason}</p>
      )}
      {isLocal && (
        <button
          onClick={() => downloadLocalZip(product.product_id, product.instrument.toLowerCase() as Instrument)}
          className="mt-1.5 text-[10px] text-cyan-400 hover:text-cyan-300 font-bold uppercase flex items-center gap-0.5 px-2 py-0.5 rounded border border-cyan-500/30 hover:bg-cyan-500/10 transition-colors w-fit"
        >
          <span className="material-symbols-outlined text-xs">save_alt</span>
          Save to Local
        </button>
      )}
    </div>
  );
}

// =============================================================================
// Product Proximity Search Results Component
// =============================================================================

function ProximitySearchResults({
  response,
  error,
  isSearching,
  onDownload,
  onBatchDownload,
  downloadingProductId,
  onClearError,
  alreadyDownloadedProducts,
  batchProgress,
}: {
  response: ProximityResponse | null;
  error: string | null;
  isSearching: boolean;
  onDownload: (productId: string, instrument: string) => void;
  onBatchDownload: (items: Array<{ product_id: string; instrument: string; lat?: number | null; lon?: number | null; exists?: boolean }>, instrumentLabel: string) => void;
  downloadingProductId: string | null;
  onClearError?: () => void;
  alreadyDownloadedProducts?: Set<string>;
  batchProgress: { completed: number; total: number; instrument: string } | null;
}) {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [_queryMode, _setQueryMode] = useState<"overlap" | "nearest">("overlap");
  const downloadedProducts = alreadyDownloadedProducts ?? new Set<string>();

  if (isSearching) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2 animate-spin text-purple-400">
            hub
          </span>
          <p className="text-purple-400">Searching for related products...</p>
        </div>
      </div>
    );
  }

  // Only show full-page error if there are no results to fall back to
  if (error && !response) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2 text-red-400">error</span>
          <p className="text-red-400">{error}</p>
          {onClearError && (
            <button
              onClick={onClearError}
              className="mt-3 px-3 py-1.5 rounded text-xs font-bold text-slate-400 bg-surface-dark border border-border-dark hover:text-white transition-colors"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-5xl mb-3 text-purple-400/50">hub</span>
          <p className="text-lg">Product Proximity Search</p>
          <p className="text-sm mt-1 max-w-md">
            Enter a product ID to find overlapping and nearby products across all instruments.
          </p>
          <div className="mt-4 text-xs text-slate-600 space-y-1">
            <p>Examples:</p>
            <p className="text-slate-500">"ESP_045857_2350" - Find data near this HiRISE image</p>
            <p className="text-slate-500">"frt00009312" - Find products overlapping this CRISM observation</p>
          </div>
        </div>
      </div>
    );
  }

  // Group results by instrument
  const grouped = response.results.reduce((acc, r) => {
    const key = r.instrument.toLowerCase();
    if (!acc[key]) acc[key] = [];
    acc[key].push(r);
    return acc;
  }, {} as Record<string, ProximityResult[]>);

  const instrumentConfig: Record<string, { label: string; icon: string }> = {
    crism: { label: "CRISM", icon: "satellite_alt" },
    hirise: { label: "HiRISE", icon: "photo_camera" },
    sharad: { label: "SHARAD", icon: "radar" },
    sharad_highres: { label: "SHARAD Hi-Res", icon: "science" },
    ctx: { label: "CTX", icon: "camera" },
    hirise_dtm: { label: "HiRISE DTM", icon: "terrain" },
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border-dark bg-purple-500/10 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-purple-400 text-lg">hub</span>
            <span className="text-sm font-bold text-white">
              Related Products
            </span>
          </div>
          <span className="text-slate-400 text-sm">
            {response.total_count} product{response.total_count !== 1 ? "s" : ""} found
          </span>
        </div>
        <div className="text-xs text-slate-400 space-y-1">
          <p>
            <span className="text-slate-500">Source:</span>{" "}
            <span className="text-purple-400 font-mono">{response.source_product_id}</span>
            <span className="text-slate-500 ml-2">({response.source_instrument})</span>
          </p>
          {response.source_bbox && (
            <p className="text-slate-500 font-mono text-[10px]">
              BBox: [{response.source_bbox.lat_min.toFixed(2)}, {response.source_bbox.lat_max.toFixed(2)}] lat,
              [{response.source_bbox.lon_min.toFixed(2)}, {response.source_bbox.lon_max.toFixed(2)}] lon
            </p>
          )}
          <p>
            <span className="text-slate-500">Mode:</span>{" "}
            <span className={response.query_mode === "overlap" ? "text-emerald-400" : "text-primary"}>
              {response.query_mode === "overlap" ? "Overlapping" : "Nearest"}
            </span>
          </p>
        </div>
      </div>

      {/* Dismissible error banner (shown over results) */}
      {error && response && (
        <div className="mx-4 mt-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
          <span className="material-symbols-outlined text-sm">error</span>
          <span className="flex-1">{error}</span>
          <button onClick={onClearError} className="text-red-400 hover:text-white transition-colors">
            <span className="material-symbols-outlined text-sm">close</span>
          </button>
        </div>
      )}

      {/* Results */}
      <div className="p-4 flex-1 overflow-y-auto space-y-4">
        {response.total_count === 0 ? (
          <div className="text-center text-slate-500 py-8">
            <span className="material-symbols-outlined text-4xl mb-2">search_off</span>
            <p>No related products found</p>
            <p className="text-xs mt-1">Try the "nearest" mode or check the product ID</p>
          </div>
        ) : (
          Object.entries(grouped).map(([instrument, items]) => {
            const config = instrumentConfig[instrument] || { label: instrument.toUpperCase(), icon: "science" };
            return (
              <div key={instrument} className="bg-bg-dark rounded-lg border border-border-dark overflow-hidden">
                <div className="px-3 py-2 border-b border-border-dark flex items-center gap-2 bg-surface-dark/50">
                  <span className="material-symbols-outlined text-primary text-sm">{config.icon}</span>
                  <span className="text-sm font-bold text-white">{config.label}</span>
                  <span className="text-xs text-slate-500">({items.length})</span>
                  {instrument !== "hirise_dtm" && instrument !== "ctx" && (() => {
                    const notDownloaded = items.filter(r => !downloadedProducts.has(r.product_id)).length;
                    if (notDownloaded === 0) return null;
                    const isActive = batchProgress?.instrument === instrument;
                    return (
                      <button
                        onClick={(e) => { e.stopPropagation(); onBatchDownload(items.map(r => ({ ...r, exists: downloadedProducts.has(r.product_id) })), instrument); }}
                        disabled={isActive || batchProgress !== null}
                        className="ml-auto flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-all disabled:opacity-50"
                      >
                        {isActive ? (
                          <><span className="material-symbols-outlined text-xs animate-spin">progress_activity</span><span>{batchProgress.completed}/{batchProgress.total}</span></>
                        ) : (
                          <><span className="material-symbols-outlined text-xs">download</span><span>All ({notDownloaded})</span></>
                        )}
                      </button>
                    );
                  })()}
                </div>
                <div className="max-h-64 overflow-y-auto divide-y divide-border-dark/50">
                  {items.map((r) => (
                    <div
                      key={r.product_id}
                      onClick={() => setSelectedItem(r.product_id)}
                      className={`p-3 cursor-pointer transition-colors ${
                        selectedItem === r.product_id
                          ? "bg-primary/20 border-l-2 border-primary"
                          : "hover:bg-surface-dark"
                      }`}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <div className="flex items-center gap-1.5">
                          <p className="text-white font-bold font-mono text-sm uppercase">
                            {r.product_id}
                          </p>
                          <button
                            title="Show on map"
                            onClick={(e) => {
                              e.stopPropagation();
                              window.location.href = `/?flyTo=${encodeURIComponent(r.product_id)}&instrument=${encodeURIComponent(r.instrument)}`;
                            }}
                            className="text-slate-400 hover:text-purple-400 transition-colors"
                          >
                            <span className="material-symbols-outlined text-sm">map</span>
                          </button>
                        </div>
                        <div className="flex items-center gap-1">
                          {r.overlap && (
                            <span className="bg-emerald-500/20 text-emerald-400 text-[9px] px-1.5 py-0.5 rounded font-bold">
                              OVERLAP
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <div className="flex items-center gap-2">
                          {r.distance_km !== null && (
                            <span className="text-primary font-mono text-[10px] font-bold">
                              {r.distance_km.toFixed(1)} km
                            </span>
                          )}
                        </div>
                        {selectedItem === r.product_id && (() => {
                          const isLocal = r.instrument.toLowerCase() === "hirise_dtm" || r.instrument.toLowerCase() === "ctx";
                          const isDownloaded = downloadedProducts.has(r.product_id);
                          if (isLocal || isDownloaded) {
                            return (
                              <span className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                <span className="material-symbols-outlined text-xs">check_circle</span>
                                {isLocal ? "Local" : "Downloaded"}
                              </span>
                            );
                          }
                          return (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onDownload(r.product_id, r.instrument.toLowerCase());
                              }}
                              disabled={downloadingProductId === r.product_id}
                              className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold uppercase transition-all ${
                                downloadingProductId === r.product_id
                                  ? "bg-primary/50 text-white cursor-wait"
                                  : "bg-primary hover:bg-primary/80 text-white"
                              }`}
                            >
                              {downloadingProductId === r.product_id ? (
                                <>
                                  <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
                                  <span>Starting...</span>
                                </>
                              ) : (
                                <>
                                  <span className="material-symbols-outlined text-xs">download</span>
                                  <span>Download</span>
                                </>
                              )}
                            </button>
                          );
                        })()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })
        )}
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

type PdsUrls = {
  img_url?: string;
  img_filename?: string;
  jp2_url?: string;
  jp2_filename?: string;
  jp2_size_bytes?: number;
  lbl_url?: string;
  browse_urls?: Record<string, string>;
};

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

  // Fetch PDS direct download URLs for Save to Local PC
  const [pdsUrls, setPdsUrls] = useState<PdsUrls | null>(null);
  useEffect(() => {
    setPdsUrls(null);
    const inst = result.instrument.toLowerCase();
    if (inst !== "crism" && inst !== "hirise") return;
    const endpoint = inst === "hirise"
      ? `/api/product-urls/hirise/${result.product_id}`
      : `/api/product-urls/crism/${result.product_id}`;
    fetch(endpoint)
      .then((r) => r.ok ? r.json() as Promise<PdsUrls> : null)
      .then((d) => setPdsUrls(d))
      .catch(() => {});
  }, [result.product_id, result.instrument]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-border-dark bg-surface-dark/50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-lg">
            {result.instrument === "crism" ? "satellite_alt" :
             result.instrument === "sharad" ? "radar" :
             result.instrument === "sharad_highres" ? "science" :
             "photo_camera"}
          </span>
          <span className="text-sm font-medium text-white">
            {result.instrument === "sharad_highres" ? "SHARAD HI-RES" : result.instrument.toUpperCase()}
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
              <div className={!result.missing_files.includes("lbl") ? "text-emerald-400" : "text-slate-500"}>
                {!result.missing_files.includes("lbl") ? "✓" : "○"} Label (.lbl)
              </div>
            </div>
          ) : result.instrument === "sharad_highres" ? (
            <div className="grid grid-cols-2 gap-1 text-xs">
              <div className={!result.missing_files.includes("dat") ? "text-emerald-400" : "text-slate-500"}>
                {!result.missing_files.includes("dat") ? "✓" : "○"} RDR data (.dat)
              </div>
              <div className={!result.missing_files.includes("lbl") ? "text-emerald-400" : "text-slate-500"}>
                {!result.missing_files.includes("lbl") ? "✓" : "○"} Label (.lbl)
              </div>
            </div>
          ) : result.instrument === "hirise" ? (
            <p className="text-xs text-slate-300">.JP2, .lbl → .tif (GDAL converted)</p>
          ) : (
            <p className="text-xs text-slate-300">Unknown instrument</p>
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

        {/* Save to Local PC — direct PDS download (no MarsLab needed) */}
        {pdsUrls && (
          <div className="border-t border-cyan-500/20 pt-2 space-y-1.5">
            <p className="text-[9px] uppercase text-cyan-400/60 font-bold tracking-wider">
              Save to Local PC (direct from PDS)
            </p>
            {pdsUrls.img_url && (
              <a
                href={pdsUrls.img_url}
                download={pdsUrls.img_filename ?? true}
                className="w-full py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-2 border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10 transition-colors"
              >
                <span className="material-symbols-outlined text-sm">save_alt</span>
                {pdsUrls.img_filename ?? "Download IMG"}
              </a>
            )}
            {pdsUrls.jp2_url && (
              <a
                href={pdsUrls.jp2_url}
                download={pdsUrls.jp2_filename ?? true}
                className="w-full py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-2 border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10 transition-colors"
              >
                <span className="material-symbols-outlined text-sm">save_alt</span>
                {pdsUrls.jp2_filename ?? "Download JP2"}
                {pdsUrls.jp2_size_bytes != null && (
                  <span className="text-[10px] text-cyan-400/60">({formatBytes(pdsUrls.jp2_size_bytes)})</span>
                )}
              </a>
            )}
            {pdsUrls.lbl_url && (
              <a
                href={pdsUrls.lbl_url}
                download
                className="w-full py-1.5 px-3 rounded-lg text-[10px] font-bold flex items-center justify-center gap-2 border border-border-dark text-slate-400 hover:text-cyan-400 hover:border-cyan-500/30 transition-colors"
              >
                <span className="material-symbols-outlined text-xs">save_alt</span>
                Download Label (.lbl)
              </a>
            )}
          </div>
        )}
        {/* Save to Local PC for already-downloaded products (from MarsLab server) */}
        {(isComplete || isPartial) && (
          <button
            onClick={() => downloadLocalZip(result.product_id, result.instrument)}
            className="w-full py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-2 border border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/10 transition-colors"
          >
            <span className="material-symbols-outlined text-sm">folder_zip</span>
            Save to Local (ZIP from MarsLab)
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

        {/* Save to Local section */}
        {task.status === "completed" && (
          <SaveToLocalSection productId={task.product_id} instrument={task.instrument} />
        )}
      </div>
    </div>
  );
}

/**
 * SaveToLocalSection: Shows local files and provides download buttons.
 */
function SaveToLocalSection({ productId, instrument }: { productId: string; instrument: Instrument }) {
  const [files, setFiles] = useState<LocalFileInfo[]>([]);
  const [totalSize, setTotalSize] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
    listLocalFiles(productId, instrument)
      .then((res) => {
        setFiles(res.files);
        setTotalSize(res.total_size);
      })
      .catch(() => {
        setFiles([]);
      })
      .finally(() => setIsLoading(false));
  }, [productId, instrument]);

  if (isLoading) {
    return (
      <div className="border border-cyan-500/30 rounded-lg p-4 bg-cyan-500/5">
        <div className="flex items-center gap-2 text-cyan-400 text-sm">
          <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
          Loading local files...
        </div>
      </div>
    );
  }

  if (files.length === 0) {
    return null;
  }

  return (
    <div className="border border-cyan-500/30 rounded-lg overflow-hidden bg-cyan-500/5">
      <div className="px-4 py-3 flex items-center justify-between border-b border-cyan-500/20">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-cyan-400 text-lg">save_alt</span>
          <span className="text-sm font-bold text-cyan-400">Save to Local</span>
          <span className="text-[10px] text-cyan-400/60 ml-1">
            {files.length} files | {formatBytes(totalSize)}
          </span>
        </div>
        <button
          onClick={() => downloadLocalZip(productId, instrument)}
          className="px-3 py-1.5 rounded-md text-xs font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 hover:bg-cyan-500/30 transition-colors flex items-center gap-1.5"
        >
          <span className="material-symbols-outlined text-sm">folder_zip</span>
          Download ZIP ({formatBytes(totalSize)})
        </button>
      </div>
      <div className="max-h-[200px] overflow-y-auto">
        <table className="w-full text-left border-collapse">
          <thead className="bg-cyan-500/10 text-cyan-400/80 text-[10px] uppercase tracking-widest sticky top-0">
            <tr>
              <th className="px-4 py-2 font-semibold">File</th>
              <th className="px-4 py-2 font-semibold">Size</th>
              <th className="px-4 py-2 font-semibold text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-cyan-500/10 text-sm">
            {files.map((file) => (
              <tr key={file.filename} className="hover:bg-cyan-500/5">
                <td className="px-4 py-2 font-mono text-xs text-white truncate max-w-[280px]">
                  {file.filename}
                </td>
                <td className="px-4 py-2 text-xs text-slate-400">
                  {formatBytes(file.size)}
                </td>
                <td className="px-4 py-2 text-right">
                  <button
                    onClick={() => downloadLocalFile(productId, instrument, file.filename)}
                    className="text-[10px] text-cyan-400 hover:text-cyan-300 font-bold uppercase flex items-center gap-0.5 ml-auto"
                  >
                    <span className="material-symbols-outlined text-xs">download</span>
                    Save
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =============================================================================
// Downloads Tab Component
// =============================================================================

function DownloadTaskCard({
  task,
  isSelected,
  onClick,
  onCancel,
  onRetry,
}: {
  task: DownloadTask;
  isSelected: boolean;
  onClick: () => void;
  onCancel?: () => void;
  onRetry?: () => void;
}) {
  const isActive = task.status === "downloading" || task.status === "processing" || task.status === "queued";
  const isComplete = task.status === "completed";
  const isFailed = task.status === "failed";

  const getStatusBadge = () => {
    if (isActive) {
      return (
        <span className="bg-primary/20 text-primary text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-primary/30">
          <span className="material-symbols-outlined text-[10px] animate-spin">progress_activity</span>
          {task.status === "processing" ? "Processing" : `${task.progress_percent.toFixed(0)}%`}
        </span>
      );
    }
    if (isComplete) {
      return (
        <span className="bg-emerald-500/20 text-emerald-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-emerald-500/30">
          <span className="material-symbols-outlined text-[10px]">check</span>
          Complete
        </span>
      );
    }
    if (isFailed) {
      return (
        <span className="bg-red-500/20 text-red-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-red-500/30">
          <span className="material-symbols-outlined text-[10px]">error</span>
          Failed
        </span>
      );
    }
    return (
      <span className="border border-border-dark text-[10px] px-2 py-0.5 rounded text-slate-500 font-bold uppercase">
        Pending
      </span>
    );
  };

  const instrumentLabel = task.instrument === "sharad_highres" ? "SHARAD Hi-Res" : task.instrument.toUpperCase();

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
          {task.product_id}
        </p>
        {getStatusBadge()}
      </div>
      <p className="text-slate-400 text-xs mb-2">
        {instrumentLabel} | {task.base_key}
      </p>

      {/* Progress bar for active downloads */}
      {isActive && (
        <div className="mt-2">
          <div className="w-full bg-bg-dark rounded-full h-1.5 overflow-hidden">
            <div
              className="h-full bg-primary transition-all shadow-[0_0_8px_rgba(19,91,236,0.6)]"
              style={{ width: `${task.progress_percent}%` }}
            />
          </div>
          <div className="flex justify-between items-center mt-1">
            <div className="text-[10px] text-slate-500">
              <span>{formatBytes(task.downloaded_bytes)} / {formatBytes(task.total_bytes)}</span>
              <span className="mx-2">|</span>
              <span>{task.files.filter(f => f.status === "completed").length} / {task.files.length} files</span>
            </div>
            {onCancel && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onCancel();
                }}
                className="text-[10px] text-red-400 hover:text-red-300 font-bold uppercase flex items-center gap-0.5"
              >
                <span className="material-symbols-outlined text-xs">close</span>
                Cancel
              </button>
            )}
          </div>
        </div>
      )}

      {/* Completed info + Save to Local */}
      {isComplete && (
        <div className="mt-1 flex items-center justify-between">
          {task.completed_at && (
            <div className="text-[10px] text-slate-500">
              Completed {new Date(task.completed_at).toLocaleString()}
            </div>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              downloadLocalZip(task.product_id, task.instrument);
            }}
            className="text-[10px] text-cyan-400 hover:text-cyan-300 font-bold uppercase flex items-center gap-0.5 shrink-0 px-2 py-0.5 rounded border border-cyan-500/30 hover:bg-cyan-500/10 transition-colors"
          >
            <span className="material-symbols-outlined text-xs">save_alt</span>
            Save to Local
          </button>
        </div>
      )}

      {/* Error info + retry */}
      {isFailed && (
        <div className="mt-1 flex items-center justify-between gap-2">
          <div className="text-[10px] text-red-400 truncate flex-1">
            {task.error}
          </div>
          {onRetry && (
            <button
              onClick={(e) => { e.stopPropagation(); onRetry(); }}
              className="text-[10px] text-primary hover:text-blue-300 font-bold uppercase flex items-center gap-0.5 shrink-0"
            >
              <span className="material-symbols-outlined text-xs">refresh</span>
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function DownloadsTab({
  onSelectTask,
  selectedTaskId,
  refreshTrigger,
}: {
  onSelectTask: (task: DownloadTask) => void;
  selectedTaskId: string | null;
  refreshTrigger: number;
}) {
  const [tasks, setTasks] = useState<DownloadTask[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  // Cancel a specific download
  const handleCancel = useCallback(async (taskId: string) => {
    setIsCancelling(taskId);
    try {
      await cancelDownload(taskId);
    } catch (e) {
      console.error("Failed to cancel download:", e);
    } finally {
      setIsCancelling(null);
    }
  }, []);

  // Cancel all downloads
  const handleCancelAll = useCallback(async () => {
    setIsCancelling("all");
    try {
      await cancelAllDownloads();
    } catch (e) {
      console.error("Failed to cancel all downloads:", e);
    } finally {
      setIsCancelling(null);
    }
  }, []);

  // Retry a failed download
  const handleRetry = useCallback(async (taskId: string) => {
    try {
      await retryDownload(taskId);
    } catch (e) {
      console.error("Failed to retry download:", e);
    }
  }, []);

  // Clear completed/failed history
  const handleClearHistory = useCallback(async () => {
    setIsCancelling("clear");
    try {
      await clearDownloadHistory();
    } catch (e) {
      console.error("Failed to clear history:", e);
    } finally {
      setIsCancelling(null);
    }
  }, []);

  // Fetch downloads
  const fetchDownloads = useCallback(async () => {
    try {
      const downloadTasks = await listDownloads();
      // Sort: active first, then by created_at descending
      downloadTasks.sort((a, b) => {
        const aActive = ["downloading", "processing", "queued", "pending"].includes(a.status);
        const bActive = ["downloading", "processing", "queued", "pending"].includes(b.status);
        if (aActive && !bActive) return -1;
        if (!aActive && bActive) return 1;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      setTasks(downloadTasks);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch downloads");
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial fetch and polling
  useEffect(() => {
    fetchDownloads();

    // Poll for updates every 2 seconds
    pollIntervalRef.current = window.setInterval(fetchDownloads, 2000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [fetchDownloads, refreshTrigger]);

  // Group tasks
  const activeTasks = tasks.filter(t =>
    ["downloading", "processing", "queued", "pending"].includes(t.status)
  );
  const completedTasks = tasks.filter(t => t.status === "completed");
  const failedTasks = tasks.filter(t => t.status === "failed");

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <span className="material-symbols-outlined text-4xl animate-spin">progress_activity</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-red-400">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2">error</span>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (tasks.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl mb-2">download</span>
          <p>No downloads yet</p>
          <p className="text-xs mt-1">Search for products and start downloading</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Clear History bar */}
      {(completedTasks.length > 0 || failedTasks.length > 0) && (
        <div className="px-4 py-1.5 border-b border-border-dark flex items-center justify-end">
          <button
            onClick={handleClearHistory}
            disabled={isCancelling === "clear"}
            className="text-[10px] text-slate-400 hover:text-slate-200 font-bold uppercase flex items-center gap-1 px-2 py-1 rounded border border-slate-500/30 hover:bg-slate-500/10 transition-colors disabled:opacity-50"
          >
            {isCancelling === "clear" ? (
              <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
            ) : (
              <span className="material-symbols-outlined text-xs">delete_sweep</span>
            )}
            Clear History
          </button>
        </div>
      )}

      {/* Active Downloads Section */}
      {activeTasks.length > 0 && (
        <div className="border-b border-border-dark">
          <div className="px-4 py-2 bg-primary/10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-sm animate-pulse">downloading</span>
              <span className="text-sm font-bold text-primary">
                Downloading ({activeTasks.length})
              </span>
            </div>
            <button
              onClick={handleCancelAll}
              disabled={isCancelling === "all"}
              className="text-[10px] text-red-400 hover:text-red-300 font-bold uppercase flex items-center gap-1 px-2 py-1 rounded border border-red-400/30 hover:bg-red-400/10 transition-colors disabled:opacity-50"
            >
              {isCancelling === "all" ? (
                <span className="material-symbols-outlined text-xs animate-spin">progress_activity</span>
              ) : (
                <span className="material-symbols-outlined text-xs">cancel</span>
              )}
              Cancel All
            </button>
          </div>
          <div className="p-2 space-y-2 max-h-64 overflow-y-auto">
            {activeTasks.map(task => (
              <DownloadTaskCard
                key={task.task_id}
                task={task}
                isSelected={selectedTaskId === task.task_id}
                onClick={() => onSelectTask(task)}
                onCancel={() => handleCancel(task.task_id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Completed Downloads Section */}
      {completedTasks.length > 0 && (
        <div className="flex-1 overflow-hidden flex flex-col">
          <div className="px-4 py-2 bg-emerald-500/10 flex items-center gap-2 shrink-0">
            <span className="material-symbols-outlined text-emerald-400 text-sm">check_circle</span>
            <span className="text-sm font-bold text-emerald-400">
              Completed ({completedTasks.length})
            </span>
          </div>
          <div className="p-2 space-y-2 flex-1 overflow-y-auto">
            {completedTasks.map(task => (
              <DownloadTaskCard
                key={task.task_id}
                task={task}
                isSelected={selectedTaskId === task.task_id}
                onClick={() => onSelectTask(task)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Failed Downloads Section */}
      {failedTasks.length > 0 && (
        <div className="border-t border-border-dark">
          <div className="px-4 py-2 bg-red-500/10 flex items-center gap-2">
            <span className="material-symbols-outlined text-red-400 text-sm">error</span>
            <span className="text-sm font-bold text-red-400">
              Failed ({failedTasks.length})
            </span>
          </div>
          <div className="p-2 space-y-2 max-h-48 overflow-y-auto">
            {failedTasks.map(task => (
              <DownloadTaskCard
                key={task.task_id}
                task={task}
                isSelected={selectedTaskId === task.task_id}
                onClick={() => onSelectTask(task)}
                onRetry={() => handleRetry(task.task_id)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function DataDownloadPage() {
  const isMobile = useIsMobile();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Page tab state
  const [pageTab, setPageTab] = useState<PageTab>("search");
  const [downloadRefreshTrigger, setDownloadRefreshTrigger] = useState(0);

  // Search state
  const [searchMode, setSearchMode] = useState<SearchMode>("id");

  // Smart AI Search state (single-step: Llama reasons + downloads)
  const [aiQuery, setAiQuery] = useState("");
  const [aiMaxResults, setAiMaxResults] = useState(20);
  const [smartSearchResponse, setSmartSearchResponse] = useState<SmartSearchResponse | null>(null);
  const [smartSearchStage, setSmartSearchStage] = useState("");
  const [isSmartSearchRunning, setIsSmartSearchRunning] = useState(false);
  const [smartSearchError, setSmartSearchError] = useState<string | null>(null);
  const smartSearchCleanup = useRef<(() => void) | null>(null);
  const [_aiBatchDownloading, _setAiBatchDownloading] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{
    completed: number; total: number; instrument: string;
  } | null>(null);
  const [_aiDownloadingProductId, _setAiDownloadingProductId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [spatialSearch, setSpatialSearch] = useState<SpatialSearch>({
    minlat: "",
    maxlat: "",
    westernlon: "",
    easternlon: "",
  });
  const [pointSearch, setPointSearch] = useState<PointSearch>({
    lat: "",
    lon: "",
    radius: "1",
  });
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [pointSearchResults, setPointSearchResults] = useState<PointSearchResponse | null>(null);
  const [pointSearchQuery, setPointSearchQuery] = useState<{ lat: number; lon: number } | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Dataset selection state - all enabled by default
  const [datasetSelection, setDatasetSelection] = useState<DatasetSelection>({
    crism: true,
    crism_trr3: true,
    hirise: true,
    sharad: true,
    sharad_highres: true,
    hirise_dtm: true,
  });

  // Selection state
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

  // Download state
  const [downloadTask, setDownloadTask] = useState<DownloadTask | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [pointDownloadingProductId, setPointDownloadingProductId] = useState<string | null>(null);

  // Product proximity search state
  const [productSearchId, setProductSearchId] = useState("");
  const [productSearchMode, setProductSearchMode] = useState<"overlap" | "nearest">("overlap");
  const [proximityResults, setProximityResults] = useState<ProximityResponse | null>(null);
  const [isProximitySearching, setIsProximitySearching] = useState(false);
  const [proximityError, setProximityError] = useState<string | null>(null);
  const [proximityDownloadingId, setProximityDownloadingId] = useState<string | null>(null);
  const [alreadyDownloadedProducts, setAlreadyDownloadedProducts] = useState<Set<string>>(new Set());

  // URL search params for deep-linking from Inspector
  const [searchParams] = useSearchParams();

  // Polling interval ref
  const pollIntervalRef = useRef<number | null>(null);
  const panelDismissedRef = useRef(false);
  // Track latest polling task for the background indicator
  const [backgroundTask, setBackgroundTask] = useState<DownloadTask | null>(null);

  // Dismiss download detail panel (polling continues in background)
  const dismissDownloadPanel = useCallback(() => {
    panelDismissedRef.current = true;
    setDownloadTask(null);
  }, []);

  // Cleanup polling + smart search stream on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      smartSearchCleanup.current?.();
    };
  }, []);

  // Handle deep-link from Inspector
  useEffect(() => {
    const tab = searchParams.get("tab");
    const pid = searchParams.get("product_id");
    const inst = searchParams.get("instrument");
    const autoDownload = searchParams.get("autoDownload") === "true";

    if (!pid) return;

    if (autoDownload && inst) {
      // Auto-download: start download immediately and switch to downloads tab
      (async () => {
        try {
          const task = await startDownload(pid, inst.toLowerCase() as Instrument);
          setDownloadTask(task);
          setPageTab("downloads");
          setDownloadRefreshTrigger(t => t + 1);
          startPolling(task.task_id);
          toast.success(`Downloading ${pid}`);
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Download failed";
          if (msg.toLowerCase().includes("already")) {
            toast.success(`${pid} is already downloaded`);
            setPageTab("downloads");
          } else {
            toast.error(msg);
          }
        }
      })();
    } else if (tab === "product") {
      // Proximity search mode
      setSearchMode("product");
      setProductSearchId(pid);
      (async () => {
        setIsProximitySearching(true);
        setProximityError(null);
        setProximityResults(null);
        try {
          const detected = inst || "HIRISE";
          const response = await searchProximity(pid, detected, "overlap", "all", 50);
          setProximityResults(response);
        } catch (e) {
          setProximityError(e instanceof Error ? e.message : "Proximity search failed");
        } finally {
          setIsProximitySearching(false);
        }
      })();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  // Escape key to close detail/manifest panels → then back to search tab
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (downloadTask) {
          dismissDownloadPanel();
        } else if (pageTab === "downloads") {
          setPageTab("search");
        } else if (selectedResult) {
          setSelectedResult(null);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedResult, downloadTask, pageTab]);

  // Get list of selected datasets
  const getSelectedDatasets = useCallback((): DatasetType[] => {
    return (Object.entries(datasetSelection) as [DatasetType, boolean][])
      .filter(([, selected]) => selected)
      .map(([key]) => key);
  }, [datasetSelection]);

  // Handle search
  const handleSearch = useCallback(async () => {
    setIsSearching(true);
    setSearchError(null);
    setSearchResults([]);
    setPointSearchResults(null);
    setPointSearchQuery(null);

    try {
      if (searchMode === "point") {
        // Point-based coordinate search via ODE
        const lat = parseFloat(pointSearch.lat);
        const lon = parseFloat(pointSearch.lon);
        const radius = parseFloat(pointSearch.radius) || 1;

        if (isNaN(lat) || isNaN(lon)) {
          setSearchError("Enter valid latitude and longitude values");
          return;
        }

        if (lat < -90 || lat > 90) {
          setSearchError("Latitude must be between -90 and 90");
          return;
        }

        const response = await searchByPoint(lat, lon, radius);
        setPointSearchResults(response);
        setPointSearchQuery({ lat, lon });
      } else {
        // Dataset selection only applies to ID and spatial search
        const selectedDatasets = getSelectedDatasets();

        if (selectedDatasets.length === 0) {
          setSearchError("Select at least one dataset to search");
          return;
        }

        if (searchMode === "id") {
          if (!query.trim()) {
            setSearchError("Enter a product ID to search");
            return;
          }

          // Search each selected dataset and combine results
          const allResults: SearchResult[] = [];

          for (const dataset of selectedDatasets) {
            try {
              const response = await searchProducts(query.trim(), dataset as Instrument, 12);
              allResults.push(...response.results);
            } catch (e) {
              console.error(`Search error for ${dataset}:`, e);
            }
          }

          setSearchResults(allResults);
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

          // Search each selected dataset for spatial
          const allResults: SearchResult[] = [];

          for (const dataset of selectedDatasets) {
            try {
              const response = await searchSpatial(bbox, dataset as Instrument, 20);
              allResults.push(...response.results);
            } catch (e) {
              console.error(`Spatial search error for ${dataset}:`, e);
            }
          }

          setSearchResults(allResults);
        }
      }
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setIsSearching(false);
    }
  }, [searchMode, query, spatialSearch, pointSearch, getSelectedDatasets]);

  // Handle Smart AI search — single step: Llama reasons + searches + downloads
  const handleSmartSearch = useCallback(async () => {
    if (!aiQuery.trim()) {
      setSearchError("Enter a search query");
      return;
    }

    // Cleanup any previous stream
    if (smartSearchCleanup.current) {
      smartSearchCleanup.current();
      smartSearchCleanup.current = null;
    }

    setIsSmartSearchRunning(true);
    setSmartSearchResponse(null);
    setSmartSearchError(null);
    setSmartSearchStage("Starting...");
    setSearchError(null);

    try {
      const { session_id } = await startSmartSearch(aiQuery.trim(), aiMaxResults);

      const cleanup = streamSmartSearch(
        session_id,
        // onEvent
        (event: SmartSearchEvent) => {
          if (event.event === "stage") {
            setSmartSearchStage((event.data as { message?: string }).message || "Processing...");
          } else if (event.event === "error") {
            setSmartSearchError((event.data as { error?: string }).error || "Search failed");
            setIsSmartSearchRunning(false);
          } else if (event.event === "done") {
            setSmartSearchResponse(event.data as unknown as SmartSearchResponse);
            setIsSmartSearchRunning(false);
            setDownloadRefreshTrigger(t => t + 1);
          }
        },
        // onDone
        (response) => {
          if (response) {
            setSmartSearchResponse(response);
          }
          setIsSmartSearchRunning(false);
        },
        // onError
        (error) => {
          setSmartSearchError(error);
          setIsSmartSearchRunning(false);
        },
      );

      smartSearchCleanup.current = cleanup;
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Smart search failed");
      setIsSmartSearchRunning(false);
    }
  }, [aiQuery, aiMaxResults]);

  // Handle cancel smart search
  const handleSmartSearchCancel = useCallback(() => {
    if (smartSearchCleanup.current) {
      smartSearchCleanup.current();
      smartSearchCleanup.current = null;
    }
    setSmartSearchResponse(null);
    setSmartSearchError(null);
    setSmartSearchStage("");
    setIsSmartSearchRunning(false);
    setSearchError(null);
  }, []);

  // Handle batch download for any search result set (per-instrument or all)
  const handleBatchDownload = useCallback(async (
    items: Array<{ product_id: string; instrument: string; lat?: number | null; lon?: number | null; exists?: boolean }>,
    instrumentLabel: string
  ) => {
    const toDownload = items.filter(r => {
      if (r.exists) return false;
      const inst = r.instrument.toLowerCase();
      if (inst === "hirise_dtm" || inst === "ctx") return false;
      return true;
    });
    if (toDownload.length === 0) return;

    setBatchProgress({ completed: 0, total: toDownload.length, instrument: instrumentLabel });
    setDownloadRefreshTrigger(t => t + 1);

    for (let i = 0; i < toDownload.length; i++) {
      const r = toDownload[i]!;
      try {
        await startDownload(
          r.product_id,
          r.instrument.toLowerCase() as Instrument,
          r.lat ?? undefined,
          r.lon ?? undefined
        );
      } catch (e) {
        console.error(`Batch download failed for ${r.product_id}:`, e);
      }
      setBatchProgress(prev => prev ? { ...prev, completed: i + 1 } : null);
    }

    setBatchProgress(null);
    setDownloadRefreshTrigger(t => t + 1);
    setPageTab("downloads");
  }, []);

  // Poll for download status
  const startPolling = useCallback((taskId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    panelDismissedRef.current = false;

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const task = await getDownloadStatus(taskId);

        // Only show panel if user hasn't dismissed it
        if (!panelDismissedRef.current) {
          setDownloadTask(task);
        }
        // Always track background state for the indicator
        setBackgroundTask(task);

        if (task.status === "completed" || task.status === "failed") {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setIsDownloading(false);
          setPointDownloadingProductId(null);
          panelDismissedRef.current = false;
          setBackgroundTask(null);

          // Refresh search results to update existence status
          if (task.status === "completed") {
            handleSearch();
          }

          // Auto-dismiss download detail panel after 3 seconds
          setTimeout(() => {
            setDownloadTask(null);
          }, 3000);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 1000);
  }, [handleSearch]);

  // Handle single download from Gemini search results
  // Smart search handles downloads automatically — no single download handler needed

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
      setPageTab("downloads");  // Switch to downloads tab
      setDownloadRefreshTrigger(t => t + 1);  // Refresh downloads list
      startPolling(task.task_id);
      toast.success(`Downloading ${selectedResult.product_id}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Download failed";
      setSearchError(msg);
      toast.error(msg);
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
      setPageTab("downloads");  // Switch to downloads tab
      setDownloadRefreshTrigger(t => t + 1);  // Refresh downloads list
      startPolling(task.task_id);
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Download failed");
      setIsDownloading(false);
    }
  }, [selectedResult, startPolling]);

  // Handle download from point search results
  const handlePointDownload = useCallback(async (productId: string, instrument: string) => {
    setIsDownloading(true);
    setPointDownloadingProductId(productId);

    try {
      // Use the query coordinates for the download
      const lat = pointSearchQuery?.lat;
      const lon = pointSearchQuery?.lon;

      const task = await startDownload(
        productId,
        instrument as Instrument,
        lat,
        lon
      );
      setDownloadTask(task);
      setPageTab("downloads");  // Switch to downloads tab
      setDownloadRefreshTrigger(t => t + 1);  // Refresh downloads list
      startPolling(task.task_id);
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Download failed");
      setIsDownloading(false);
      setPointDownloadingProductId(null);
    }
  }, [pointSearchQuery, startPolling]);

  // Handle product proximity search
  const handleProximitySearch = useCallback(async () => {
    if (!productSearchId.trim()) {
      setProximityError("Enter a product ID to search");
      return;
    }
    setIsProximitySearching(true);
    setProximityError(null);
    setProximityResults(null);

    try {
      // Auto-detect instrument from product ID
      const pid = productSearchId.trim();
      let instrument = "HIRISE";
      const pidLower = pid.toLowerCase();
      if (pidLower.startsWith("frt")) {
        instrument = "CRISM";
      } else if (pidLower.startsWith("s_")) {
        instrument = "SHARAD";
      } else if (pidLower.startsWith("r_")) {
        instrument = "SHARAD_HIGHRES";
      } else if (pidLower.startsWith("dte") || pidLower.startsWith("dteec")) {
        instrument = "HIRISE_DTM";
      } else if (pidLower.includes("ctx") || pidLower.startsWith("b")) {
        instrument = "CTX";
      }

      const response = await searchProximity(pid, instrument, productSearchMode, "all", 50);
      setProximityResults(response);
    } catch (e) {
      setProximityError(e instanceof Error ? e.message : "Proximity search failed");
    } finally {
      setIsProximitySearching(false);
    }
  }, [productSearchId, productSearchMode]);

  // Handle download from proximity results
  const handleProximityDownload = useCallback(async (productId: string, instrument: string) => {
    setProximityDownloadingId(productId);
    try {
      const task = await startDownload(productId, instrument as Instrument);
      setDownloadTask(task);
      // Don't switch to downloads tab — let user keep browsing results
      setDownloadRefreshTrigger(t => t + 1);
      startPolling(task.task_id);
      setAlreadyDownloadedProducts(prev => new Set(prev).add(productId));
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Download failed";
      if (msg.toLowerCase().includes("already")) {
        setAlreadyDownloadedProducts(prev => new Set(prev).add(productId));
      } else {
        setProximityError(msg);
      }
    } finally {
      setProximityDownloadingId(null);
    }
  }, [startPolling]);

  // Handle Enter key in search
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      if (searchMode === "product") {
        handleProximitySearch();
      } else {
        handleSearch();
      }
    }
  };

  return (
    <div className="h-screen bg-bg-dark text-white flex flex-col overflow-hidden">
      {/* Header */}
      {isMobile ? (
        <>
          <header className="flex h-12 items-center justify-between border-b border-border-dark bg-bg-dark px-4 shrink-0">
            <Link to="/" className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-xl">rocket_launch</span>
              <h2 className="text-lg font-bold tracking-tight">MarsLab</h2>
            </Link>
            <div className="flex items-center gap-2">
              {/* Page Tabs inline on mobile header */}
              <div className="flex items-center bg-bg-dark rounded-lg p-0.5 border border-border-dark">
                <button
                  onClick={() => setPageTab("search")}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold uppercase transition-all ${
                    pageTab === "search" ? "bg-primary text-white" : "text-slate-500"
                  }`}
                >
                  <span className="material-symbols-outlined text-xs">search</span>
                  Search
                </button>
                <button
                  onClick={() => setPageTab("downloads")}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold uppercase transition-all ${
                    pageTab === "downloads" ? "bg-primary text-white" : "text-slate-500"
                  }`}
                >
                  <span className="material-symbols-outlined text-xs">download</span>
                  DL
                </button>
              </div>
              <button
                onClick={() => setMobileMenuOpen(p => !p)}
                className="flex items-center justify-center w-9 h-9 rounded-lg hover:bg-white/10 text-slate-300"
              >
                <span className="material-symbols-outlined text-2xl">
                  {mobileMenuOpen ? "close" : "menu"}
                </span>
              </button>
            </div>
          </header>
          {mobileMenuOpen && (
            <div className="absolute top-12 left-0 right-0 z-50 border-b border-border-dark bg-bg-dark p-4 flex flex-col gap-2 shadow-xl">
              <Link to="/" className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors">Workbench</Link>
              <Link to="/download" className="text-sm font-medium text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors">Data Download</Link>
              <Link to="/upload" className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors">Data Upload</Link>
              <Link to="/suggestions" onClick={() => setMobileMenuOpen(false)} className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors flex items-center gap-2">
                <span className="material-symbols-outlined text-sm">lightbulb</span>
                Suggest Feature
              </Link>
            </div>
          )}
        </>
      ) : (
        <header className="flex items-center justify-between border-b border-border-dark px-6 py-2 shrink-0">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-white">
              <span className="material-symbols-outlined text-primary text-2xl">rocket_launch</span>
              <h2 className="text-white text-base font-bold">MarsLab</h2>
            </div>
            <div className="h-5 w-px bg-border-dark" />
            <nav className="flex items-center gap-4">
              <a href="/" className="text-slate-400 hover:text-white text-sm font-medium transition-colors">Workbench</a>
              <a href="/download" className="text-primary text-sm font-medium border-b-2 border-primary pb-0.5">Data Download</a>
              <a href="/upload" className="text-slate-400 hover:text-white text-sm font-medium transition-colors">Data Upload</a>
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-bg-dark rounded-lg p-1 border border-border-dark">
              <button
                onClick={() => setPageTab("search")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-bold uppercase tracking-wider transition-all ${
                  pageTab === "search" ? "bg-primary text-white" : "text-slate-500 hover:text-white"
                }`}
              >
                <span className="material-symbols-outlined text-sm">search</span>
                Search
              </button>
              <button
                onClick={() => setPageTab("downloads")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-bold uppercase tracking-wider transition-all ${
                  pageTab === "downloads" ? "bg-primary text-white" : "text-slate-500 hover:text-white"
                }`}
              >
                <span className="material-symbols-outlined text-sm">download</span>
                Downloads
              </button>
            </div>
            <Link
              to="/suggestions"
              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium text-slate-400 hover:text-white border border-border-dark rounded-md hover:bg-white/5 transition-colors shrink-0"
            >
              <span className="material-symbols-outlined text-sm">lightbulb</span>
              Suggest Feature
            </Link>
          </div>
        </header>
      )}

      {/* Main content - scrollable */}
      <main className="flex-1 flex flex-col overflow-auto max-w-[1600px] mx-auto w-full px-4 md:px-6 py-4 gap-4">
        {/* Downloads Tab Content */}
        {pageTab === "downloads" ? (
          <div className="flex-1 flex flex-col md:flex-row gap-4 md:gap-6 overflow-auto md:overflow-hidden min-h-0">
            {/* Left panel - Downloads List */}
            <aside className="w-full md:w-1/3 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden max-h-[50vh] md:max-h-none">
              <div className="p-4 border-b border-border-dark flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPageTab("search")}
                    className="text-slate-400 hover:text-white transition-colors"
                    title="Back to Search (Esc)"
                  >
                    <span className="material-symbols-outlined text-sm">arrow_back</span>
                  </button>
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">
                    Download History
                  </h3>
                </div>
                <button
                  onClick={() => setDownloadRefreshTrigger(t => t + 1)}
                  className="text-slate-400 hover:text-white transition-colors"
                  title="Refresh"
                >
                  <span className="material-symbols-outlined text-sm">refresh</span>
                </button>
              </div>
              <DownloadsTab
                onSelectTask={(task) => setDownloadTask(task)}
                selectedTaskId={downloadTask?.task_id ?? null}
                refreshTrigger={downloadRefreshTrigger}
              />
            </aside>

            {/* Right panel - Download Details */}
            <section className="flex-1 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden">
              <DownloadManifest
                task={downloadTask}
                onClose={dismissDownloadPanel}
              />
            </section>
          </div>
        ) : (
        <>
        {/* Search section */}
        <section className="w-full flex flex-col gap-3">
          {/* Dataset selection (hidden in point/ai/product mode - these search all datasets) */}
          {searchMode !== "point" && searchMode !== "ai" && searchMode !== "product" && (
            <div className="bg-surface-dark rounded-xl p-3 border border-border-dark">
              <DatasetSelector
                selection={datasetSelection}
                onSelectionChange={setDatasetSelection}
              />
            </div>
          )}

          {/* Search bar */}
          <div className="flex flex-col md:flex-row items-stretch gap-2 md:gap-4">
            <div className="flex-1 bg-surface-dark rounded-xl p-1.5 flex flex-col md:flex-row md:items-center shadow-lg border border-border-dark">
              <SearchModeToggle mode={searchMode} onModeChange={setSearchMode} />

              <div className="flex-1 flex items-center px-2 gap-3 min-w-0">
                <span className="material-symbols-outlined text-slate-400 hidden md:block">
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
                ) : searchMode === "spatial" ? (
                  <div className="grid grid-cols-2 md:flex md:items-center gap-2 md:gap-3 flex-1 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">S:</span>
                      <input
                        type="text"
                        value={spatialSearch.minlat}
                        onChange={(e) =>
                          setSpatialSearch((s) => ({ ...s, minlat: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-full md:w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
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
                        className="bg-transparent border-none text-white focus:ring-0 w-full md:w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
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
                        className="bg-transparent border-none text-white focus:ring-0 w-full md:w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
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
                        className="bg-transparent border-none text-white focus:ring-0 w-full md:w-16 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="150"
                      />
                    </div>
                  </div>
                ) : searchMode === "point" ? (
                  <div className="flex flex-wrap items-center gap-2 md:gap-4 flex-1 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">Lat:</span>
                      <input
                        type="text"
                        value={pointSearch.lat}
                        onChange={(e) =>
                          setPointSearch((s) => ({ ...s, lat: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-16 md:w-20 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="18.5"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">Lon:</span>
                      <input
                        type="text"
                        value={pointSearch.lon}
                        onChange={(e) =>
                          setPointSearch((s) => ({ ...s, lon: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-16 md:w-20 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="77.4"
                      />
                    </div>
                    <div className="flex items-center gap-1 md:border-l md:border-border-dark md:pl-4">
                      <span className="text-slate-500 text-xs">Radius:</span>
                      <input
                        type="text"
                        value={pointSearch.radius}
                        onChange={(e) =>
                          setPointSearch((s) => ({ ...s, radius: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-12 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="1"
                      />
                      <span className="text-slate-500 text-xs">deg</span>
                    </div>
                  </div>
                ) : searchMode === "product" ? (
                  <div className="flex items-center gap-3 flex-1">
                    <input
                      type="text"
                      value={productSearchId}
                      onChange={(e) => setProductSearchId(e.target.value)}
                      onKeyDown={handleKeyDown}
                      className="bg-transparent border-none text-white focus:ring-0 flex-1 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                      placeholder="Enter product ID (e.g. ESP_045857_2350, frt00009312, DTEEC_060706_2195)"
                    />
                    <div className="flex items-center gap-1 border-l border-border-dark pl-3">
                      <button
                        onClick={() => setProductSearchMode("overlap")}
                        className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${
                          productSearchMode === "overlap"
                            ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                            : "text-slate-500 hover:text-white"
                        }`}
                      >
                        Overlap
                      </button>
                      <button
                        onClick={() => setProductSearchMode("nearest")}
                        className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${
                          productSearchMode === "nearest"
                            ? "bg-primary/20 text-primary border border-primary/30"
                            : "text-slate-500 hover:text-white"
                        }`}
                      >
                        Nearest
                      </button>
                    </div>
                  </div>
                ) : searchMode === "ai" ? (
                  <div className="flex items-center gap-3 flex-1">
                    <input
                      type="text"
                      value={aiQuery}
                      onChange={(e) => setAiQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && !isSmartSearchRunning && handleSmartSearch()}
                      className="bg-transparent border-none text-white focus:ring-0 flex-1 text-sm placeholder:text-slate-500 focus:outline-none"
                      placeholder="Download CRISM that intersects SHARAD high-res"
                    />
                    <div className="flex items-center gap-1 border-l border-border-dark pl-3">
                      <span className="text-slate-500 text-xs">Max:</span>
                      <input
                        type="number"
                        value={aiMaxResults}
                        onChange={(e) => setAiMaxResults(Math.min(50, Math.max(1, parseInt(e.target.value) || 10)))}
                        className="bg-transparent border-none text-white focus:ring-0 w-12 text-sm font-mono focus:outline-none text-center"
                        min="1"
                        max="50"
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="hidden md:flex items-center gap-2 px-4 border-l border-border-dark group relative">
                <span className="material-symbols-outlined text-slate-500 cursor-help">
                  info
                </span>
                <div className="absolute bottom-full mb-2 right-0 w-72 p-3 bg-bg-dark border border-border-dark rounded-lg text-[10px] leading-relaxed text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                  {searchMode === "id"
                    ? "Search by CRISM, HiRISE, or SHARAD product ID. Partial matches are supported."
                    : searchMode === "spatial"
                    ? "Search by bounding box. S/N = latitude range (-90 to 90), W/E = longitude range (-180 to 360). Example: Arcadia Planitia = S:35, N:70, W:-130, E:150"
                    : searchMode === "point"
                    ? "Search ODE by coordinate. Enter latitude (-90 to 90) and longitude. Radius controls the search area in degrees."
                    : searchMode === "product"
                    ? "Find products that overlap with or are near a given product. Enter a product ID from any instrument. Toggle between 'Overlap' (geometry intersection) and 'Nearest' (ranked by distance)."
                    : "Smart AI Search: Describe what data you want. Llama will reason about your request, search, pick the best products, and auto-download. Powered by Llama 3.3 (local)."}
                </div>
              </div>

              <button
                onClick={searchMode === "ai" ? handleSmartSearch : searchMode === "product" ? handleProximitySearch : handleSearch}
                disabled={isSearching || isSmartSearchRunning || isProximitySearching}
                className={`${searchMode === "ai" ? "bg-gradient-to-r from-purple-500 to-primary" : searchMode === "product" ? "bg-purple-600" : "bg-primary"} hover:opacity-80 disabled:opacity-50 text-white font-bold py-3 px-4 md:px-8 rounded-lg transition-all flex items-center justify-center gap-2 shrink-0 w-full md:w-auto`}
              >
                {isSearching || isSmartSearchRunning || isProximitySearching ? (
                  <>
                    <span className="animate-spin material-symbols-outlined text-sm">
                      progress_activity
                    </span>
                    <span>{isSmartSearchRunning ? "Running..." : isProximitySearching ? "Finding..." : "Searching..."}</span>
                  </>
                ) : searchMode === "product" ? (
                  <>
                    <span className="material-symbols-outlined text-sm">hub</span>
                    <span>FIND</span>
                  </>
                ) : searchMode === "ai" ? (
                  <>
                    <span className="material-symbols-outlined text-sm">psychology</span>
                    <span>AI GO</span>
                  </>
                ) : searchMode === "point" ? (
                  <>
                    <span>SEARCH</span>
                    <span className="material-symbols-outlined text-sm">
                      location_on
                    </span>
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
        <div className="flex-1 flex flex-col md:flex-row gap-4 md:gap-6 overflow-auto md:overflow-hidden min-h-0">
          {/* Left panel - Search Results (hidden in point, ai, and product mode) */}
          {searchMode !== "point" && searchMode !== "ai" && searchMode !== "product" && (
            <aside className="w-full md:w-1/3 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden max-h-[50vh] md:max-h-none shrink-0">
              <div className="p-4 border-b border-border-dark">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">
                    Search Results ({searchResults.length})
                  </h3>
                  <span className="material-symbols-outlined text-slate-400 cursor-pointer">
                    filter_list
                  </span>
                </div>
                {searchResults.filter(r => !r.exists && r.instrument !== "hirise_dtm" && r.instrument !== "ctx").length > 0 && (
                  <button
                    onClick={() => handleBatchDownload(searchResults, "all")}
                    disabled={batchProgress !== null}
                    className="mt-2 w-full py-1.5 rounded-lg text-[10px] font-bold uppercase flex items-center justify-center gap-1 bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-all disabled:opacity-50"
                  >
                    {batchProgress?.instrument === "all" ? (
                      <><span className="material-symbols-outlined text-xs animate-spin">progress_activity</span><span>{batchProgress.completed}/{batchProgress.total}</span></>
                    ) : (
                      <><span className="material-symbols-outlined text-xs">download</span><span>Download All ({searchResults.filter(r => !r.exists && r.instrument !== "hirise_dtm" && r.instrument !== "ctx").length})</span></>
                    )}
                  </button>
                )}
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
                      // Clear download detail panel to show product preview
                      if (downloadTask) {
                        dismissDownloadPanel();
                      }
                    }}
                  />
                ))}
              </div>

            </aside>
          )}

          {/* Right panel - Product Preview, Download Manifest, Point Search Results, or AI Search Results */}
          <section className="flex-1 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden relative">
            {/* Background download indicator (visible when panel dismissed but download active) */}
            {!downloadTask && backgroundTask && (
              <button
                onClick={() => { panelDismissedRef.current = false; setDownloadTask(backgroundTask); }}
                className="absolute top-3 right-3 z-10 flex items-center gap-2 px-3 py-1.5 bg-primary/20 border border-primary/30 rounded-lg text-xs text-primary hover:bg-primary/30 transition-colors backdrop-blur-sm"
              >
                <span className="material-symbols-outlined text-sm animate-pulse">downloading</span>
                <span className="font-medium">
                  {backgroundTask.progress_percent.toFixed(0)}%
                </span>
                <span className="text-primary/70">{backgroundTask.base_key}</span>
              </button>
            )}
            {downloadTask ? (
              <DownloadManifest
                task={downloadTask}
                onClose={dismissDownloadPanel}
              />
            ) : searchMode === "product" ? (
              <ProximitySearchResults
                response={proximityResults}
                error={proximityError}
                isSearching={isProximitySearching}
                onDownload={handleProximityDownload}
                onBatchDownload={handleBatchDownload}
                downloadingProductId={proximityDownloadingId}
                onClearError={() => setProximityError(null)}
                alreadyDownloadedProducts={alreadyDownloadedProducts}
                batchProgress={batchProgress}
              />
            ) : searchMode === "point" ? (
              <PointSearchResults
                results={pointSearchResults}
                query={pointSearchQuery}
                error={searchError}
                onDownload={handlePointDownload}
                onBatchDownload={handleBatchDownload}
                isDownloading={isDownloading}
                downloadingProductId={pointDownloadingProductId}
                batchProgress={batchProgress}
              />
            ) : searchMode === "ai" ? (
              <SmartSearchPanel
                response={smartSearchResponse}
                stageMessage={smartSearchStage}
                error={smartSearchError || searchError}
                isRunning={isSmartSearchRunning}
                onCancel={handleSmartSearchCancel}
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
        </>
        )}
      </main>
    </div>
  );
}
