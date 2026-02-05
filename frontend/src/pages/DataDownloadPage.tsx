import { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import {
  searchProducts,
  searchSpatial,
  searchByPoint,
  startDownload,
  getDownloadStatus,
  type SearchResult,
  type DownloadTask,
  type Instrument,
  type BoundingBox,
  type PointSearchResponse,
  type PointSearchResult,
  formatBytes,
} from "../api/search";

// =============================================================================
// Types
// =============================================================================

type SearchMode = "id" | "spatial" | "point";

// Dataset types for selection
type DatasetType = "crism" | "hirise" | "sharad" | "sharad_highres";

interface DatasetSelection {
  crism: boolean;
  hirise: boolean;
  sharad: boolean;
  sharad_highres: boolean;
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
      <button
        onClick={() => onModeChange("point")}
        className={`px-4 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
          mode === "point"
            ? "bg-primary text-white"
            : "text-slate-500 hover:text-white"
        }`}
      >
        Coordinate
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
    { key: "hirise", label: "HiRISE", icon: "photo_camera" },
    { key: "sharad", label: "SHARAD", icon: "radar" },
    { key: "sharad_highres", label: "SHARAD Hi-Res", icon: "science" },
  ];

  const toggleDataset = (key: DatasetType) => {
    onSelectionChange({
      ...selection,
      [key]: !selection[key],
    });
  };

  const selectedCount = Object.values(selection).filter(Boolean).length;

  return (
    <div className="flex items-center gap-2">
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
          {result.lat !== null && result.lon !== null && (
            <span className="text-slate-500 font-mono text-[10px]">
              {result.lat.toFixed(2)}°, {result.lon.toFixed(2)}°
            </span>
          )}
        </div>
        {isSelected && downloadable && !isComplete && (
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
  isDownloading,
  downloadingProductId,
}: {
  results: PointSearchResponse | null;
  query: { lat: number; lon: number } | null;
  error?: string | null;
  onDownload: (productId: string, instrument: string) => void;
  isDownloading: boolean;
  downloadingProductId: string | null;
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
    hirise: true,
    sharad: true,
    sharad_highres: true,
  });

  // Selection state
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

  // Download state
  const [downloadTask, setDownloadTask] = useState<DownloadTask | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [pointDownloadingProductId, setPointDownloadingProductId] = useState<string | null>(null);

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
          setPointDownloadingProductId(null);

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
      startPolling(task.task_id);
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : "Download failed");
      setIsDownloading(false);
      setPointDownloadingProductId(null);
    }
  }, [pointSearchQuery, startPolling]);

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
            <a
              href="/upload"
              className="text-slate-400 hover:text-white text-sm font-medium transition-colors"
            >
              Data Upload
            </a>
          </nav>
        </div>
        {/* Suggest Feature */}
        <Link
          to="/suggestions"
          className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium text-slate-400 hover:text-white border border-border-dark rounded-md hover:bg-white/5 transition-colors shrink-0"
        >
          <span className="material-symbols-outlined text-sm">lightbulb</span>
          Suggest Feature
        </Link>
      </header>

      {/* Main content - scrollable */}
      <main className="flex-1 flex flex-col overflow-auto max-w-[1600px] mx-auto w-full px-4 md:px-6 py-4 gap-4">
        {/* Search section */}
        <section className="w-full flex flex-col gap-3">
          {/* Dataset selection (hidden in point mode - point search checks all datasets) */}
          {searchMode !== "point" && (
            <div className="bg-surface-dark rounded-xl p-3 border border-border-dark">
              <DatasetSelector
                selection={datasetSelection}
                onSelectionChange={setDatasetSelection}
              />
            </div>
          )}

          {/* Search bar */}
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
                ) : searchMode === "spatial" ? (
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
                ) : (
                  <div className="flex items-center gap-4 flex-1 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500 text-xs">Lat:</span>
                      <input
                        type="text"
                        value={pointSearch.lat}
                        onChange={(e) =>
                          setPointSearch((s) => ({ ...s, lat: e.target.value }))
                        }
                        onKeyDown={handleKeyDown}
                        className="bg-transparent border-none text-white focus:ring-0 w-20 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
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
                        className="bg-transparent border-none text-white focus:ring-0 w-20 text-sm placeholder:text-slate-500 font-mono focus:outline-none"
                        placeholder="77.4"
                      />
                    </div>
                    <div className="flex items-center gap-1 border-l border-border-dark pl-4">
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
                )}
              </div>

              <div className="flex items-center gap-2 px-4 border-l border-border-dark group relative">
                <span className="material-symbols-outlined text-slate-500 cursor-help">
                  info
                </span>
                <div className="absolute bottom-full mb-2 right-0 w-72 p-3 bg-bg-dark border border-border-dark rounded-lg text-[10px] leading-relaxed text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                  {searchMode === "id"
                    ? "Search by CRISM, HiRISE, or SHARAD product ID. Partial matches are supported."
                    : searchMode === "spatial"
                    ? "Search by bounding box. S/N = latitude range (-90 to 90), W/E = longitude range (-180 to 360). Example: Arcadia Planitia = S:35, N:70, W:-130, E:150"
                    : "Search ODE by coordinate. Enter latitude (-90 to 90) and longitude. Radius controls the search area in degrees. Returns all CRISM, HiRISE, SHARAD, and SHARAD Hi-Res datasets in that area."}
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
        <div className="flex-1 flex gap-6 overflow-hidden min-h-0">
          {/* Left panel - Search Results (hidden in point mode) */}
          {searchMode !== "point" && (
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
          )}

          {/* Right panel - Product Preview, Download Manifest, or Point Search Results */}
          <section className="flex-1 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden">
            {downloadTask ? (
              <DownloadManifest
                task={downloadTask}
                onClose={() => setDownloadTask(null)}
              />
            ) : searchMode === "point" ? (
              <PointSearchResults
                results={pointSearchResults}
                query={pointSearchQuery}
                error={searchError}
                onDownload={handlePointDownload}
                isDownloading={isDownloading}
                downloadingProductId={pointDownloadingProductId}
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
