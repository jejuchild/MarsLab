import { useState, useEffect, useRef, useCallback } from "react";
import { Link } from "react-router-dom";
import useIsMobile from "../hooks/useIsMobile";

// =============================================================================
// Types
// =============================================================================

interface DatasetBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

interface DatasetMetadata {
  id: string;
  name: string;
  bounds: DatasetBounds;
  crs: string;
  crs_valid: boolean;
  crs_warning?: string | null;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  nodata: number | null;
  created_at: string;
  original_filename: string;
}

interface ValidationResult {
  valid: boolean;
  filename: string;
  filesize: number;
  crs: string;
  crs_valid: boolean;
  crs_warning?: string | null;
  crs_error?: string | null;
  bounds: DatasetBounds;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  nodata: number | null;
}

// =============================================================================
// Helpers
// =============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// =============================================================================
// Disclaimer / Info Section
// =============================================================================

function GeoTIFFDisclaimer() {
  return (
    <div className="bg-surface-dark rounded-xl border border-border-dark p-5">
      <div className="flex items-start gap-3 mb-3">
        <span className="material-symbols-outlined text-primary text-xl mt-0.5">info</span>
        <div>
          <h3 className="text-sm font-bold text-white mb-1">Accepted GeoTIFF Requirements</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Upload Mars GeoTIFF files to overlay on the interactive map alongside CRISM, HiRISE, and SHARAD data.
            Files are validated before upload to ensure compatibility.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
        <div className="bg-bg-dark rounded-lg p-3 border border-border-dark">
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 mb-2 flex items-center gap-1.5">
            <span className="material-symbols-outlined text-xs">check_circle</span>
            Accepted
          </h4>
          <ul className="space-y-1.5 text-xs text-slate-300">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>GeoTIFF format (.tif, .tiff)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>Mars CRS (IAU:49900 series, IAU:49910, IAU:49990)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>Mars-radius ellipsoid (~3,396,190 m semi-major axis)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>No CRS (treated as raw Mars lat/lon degrees)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>Single-band or multi-band (RGB) rasters</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>0-360 or -180 to 180 longitude convention</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">+</span>
              <span>Maximum file size: 500 MB</span>
            </li>
          </ul>
        </div>

        <div className="bg-bg-dark rounded-lg p-3 border border-border-dark">
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-red-400 mb-2 flex items-center gap-1.5">
            <span className="material-symbols-outlined text-xs">cancel</span>
            Rejected
          </h4>
          <ul className="space-y-1.5 text-xs text-slate-300">
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              <span>Earth CRS (EPSG:4326, WGS84, a=6,378,137 m)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              <span>Non-GeoTIFF formats (PNG, JPEG, NetCDF, etc.)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              <span>Files exceeding 500 MB</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              <span>Latitude outside -90 to 90 range</span>
            </li>
          </ul>

          <div className="mt-3 pt-3 border-t border-border-dark">
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">Processing</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Uploaded GeoTIFFs are downsampled to a max 2048px overlay PNG with percentile stretch (2nd-98th)
              for map rendering. Nodata pixels become transparent. Original files are preserved on the server.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Dataset List Item
// =============================================================================

function DatasetItem({
  dataset,
  isSelected,
  onClick,
}: {
  dataset: DatasetMetadata;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={`rounded-lg p-3 cursor-pointer transition-colors ${
        isSelected
          ? "bg-fuchsia-500/20 border border-fuchsia-500/50"
          : "hover:bg-surface-dark border border-transparent"
      }`}
    >
      <div className="flex justify-between items-start mb-1">
        <p className="text-white font-bold text-sm truncate flex-1">{dataset.name}</p>
        {dataset.crs_valid ? (
          <span className="bg-emerald-500/20 text-emerald-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-emerald-500/30 shrink-0 ml-2">
            <span className="material-symbols-outlined text-[10px]">check</span>
            Valid
          </span>
        ) : (
          <span className="bg-amber-500/20 text-amber-400 text-[10px] px-2 py-0.5 rounded font-bold uppercase flex items-center gap-1 border border-amber-500/30 shrink-0 ml-2">
            <span className="material-symbols-outlined text-[10px]">warning</span>
            Warning
          </span>
        )}
      </div>
      <p className="text-slate-500 text-[11px] font-mono truncate">{dataset.original_filename}</p>
      <div className="flex items-center gap-3 mt-1.5 text-[10px] text-slate-500">
        <span>{dataset.width} x {dataset.height}px</span>
        <span className="text-slate-600">|</span>
        <span>{dataset.bands} band{dataset.bands > 1 ? "s" : ""}</span>
        <span className="text-slate-600">|</span>
        <span>{dataset.dtype}</span>
      </div>
      <div className="flex items-center gap-1 mt-1 text-[10px] text-slate-500">
        <span className="material-symbols-outlined text-[11px]">schedule</span>
        <span>{formatDate(dataset.created_at)}</span>
      </div>
    </div>
  );
}

// =============================================================================
// Dataset Detail Panel
// =============================================================================

function DatasetDetail({
  dataset,
  onDelete,
  isDeleting,
}: {
  dataset: DatasetMetadata;
  onDelete: () => void;
  isDeleting: boolean;
}) {
  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-border-dark bg-surface-dark/50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-fuchsia-400 text-lg">layers</span>
          <span className="text-sm font-medium text-white">Custom Dataset</span>
        </div>
        {dataset.crs_warning && (
          <span className="bg-amber-500/20 text-amber-400 text-[10px] px-2 py-0.5 rounded flex items-center gap-1">
            <span className="material-symbols-outlined text-xs">warning</span>
            CRS Warning
          </span>
        )}
      </div>

      {/* Content */}
      <div className="p-4 flex-1 overflow-y-auto flex flex-col gap-3">
        <div>
          <h2 className="text-lg font-bold text-white break-all">{dataset.name}</h2>
          <p className="text-slate-400 text-xs mt-1 font-mono">{dataset.original_filename}</p>
        </div>

        {/* Metadata grid */}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-bg-dark rounded p-2 border border-border-dark">
            <p className="text-[9px] uppercase text-slate-500">CRS</p>
            <p className="text-white font-mono text-xs">{dataset.crs}</p>
          </div>
          <div className="bg-bg-dark rounded p-2 border border-border-dark">
            <p className="text-[9px] uppercase text-slate-500">Dimensions</p>
            <p className="text-white font-mono text-xs">{dataset.width} x {dataset.height}</p>
          </div>
          <div className="bg-bg-dark rounded p-2 border border-border-dark">
            <p className="text-[9px] uppercase text-slate-500">Bands</p>
            <p className="text-white">{dataset.bands} ({dataset.dtype})</p>
          </div>
          <div className="bg-bg-dark rounded p-2 border border-border-dark">
            <p className="text-[9px] uppercase text-slate-500">Nodata</p>
            <p className="text-white font-mono text-xs">{dataset.nodata ?? "None"}</p>
          </div>
        </div>

        {/* Bounds */}
        <div className="bg-bg-dark rounded p-3 border border-border-dark">
          <p className="text-[9px] uppercase text-slate-500 mb-2">Geographic Bounds</p>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div>
              <span className="text-slate-500">North: </span>
              <span className="text-white">{dataset.bounds.north.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate-500">South: </span>
              <span className="text-white">{dataset.bounds.south.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate-500">West: </span>
              <span className="text-white">{dataset.bounds.west.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate-500">East: </span>
              <span className="text-white">{dataset.bounds.east.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* CRS Warning */}
        {dataset.crs_warning && (
          <div className="bg-amber-500/10 rounded p-2 border border-amber-500/30">
            <p className="text-[9px] uppercase text-amber-400 mb-1">CRS Warning</p>
            <p className="text-xs text-amber-300">{dataset.crs_warning}</p>
          </div>
        )}

        {/* Upload date */}
        <div className="bg-bg-dark rounded p-2 border border-border-dark">
          <p className="text-[9px] uppercase text-slate-500">Uploaded</p>
          <p className="text-white text-xs">{formatDate(dataset.created_at)}</p>
        </div>

        {/* Overlay preview */}
        <div className="bg-bg-dark rounded border border-border-dark overflow-hidden">
          <p className="text-[9px] uppercase text-slate-500 px-3 pt-2 mb-1">Overlay Preview</p>
          <img
            src={`/api/custom/${dataset.id}/overlay.png`}
            alt={`${dataset.name} overlay`}
            className="w-full h-auto max-h-48 object-contain bg-black/50 p-2"
          />
        </div>
      </div>

      {/* Delete button */}
      <div className="p-3 border-t border-border-dark shrink-0">
        <button
          onClick={onDelete}
          disabled={isDeleting}
          className={`w-full py-2.5 px-4 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${
            isDeleting
              ? "bg-red-500/20 text-red-400/50 cursor-wait border border-red-500/20"
              : "bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30"
          }`}
        >
          <span className="material-symbols-outlined text-lg">
            {isDeleting ? "progress_activity" : "delete"}
          </span>
          <span>{isDeleting ? "Deleting..." : "Delete Dataset"}</span>
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// Upload Flow Panel (Validation → Confirm)
// =============================================================================

type UploadStage = "idle" | "validating" | "validated" | "uploading" | "done" | "error";

function UploadFlow({
  onUploadComplete,
}: {
  onUploadComplete: (dataset: DatasetMetadata) => void;
}) {
  const [stage, setStage] = useState<UploadStage>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setError(null);
    setValidation(null);
    setStage("validating");
    setDatasetName(selected.name.replace(/\.(tif|tiff)$/i, ""));

    try {
      const formData = new FormData();
      formData.append("file", selected);

      const response = await fetch("/api/custom/validate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Validation failed" }));
        throw new Error(errData.detail || `Validation failed (${response.status})`);
      }

      const result: ValidationResult = await response.json();
      setValidation(result);

      if (result.valid) {
        setStage("validated");
      } else {
        setError(result.crs_error || "File is not a valid Mars GeoTIFF.");
        setStage("error");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Validation failed");
      setStage("error");
    }

    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleConfirmUpload = async () => {
    if (!file) return;

    setStage("uploading");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("name", datasetName || file.name.replace(/\.(tif|tiff)$/i, ""));

      const response = await fetch("/api/custom/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(errData.detail || `Upload failed (${response.status})`);
      }

      const metadata: DatasetMetadata = await response.json();
      setStage("done");
      onUploadComplete(metadata);

      // Reset after a moment
      setTimeout(() => {
        setStage("idle");
        setFile(null);
        setValidation(null);
        setDatasetName("");
      }, 2000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStage("error");
    }
  };

  const handleReset = () => {
    setStage("idle");
    setFile(null);
    setValidation(null);
    setError(null);
    setDatasetName("");
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-border-dark bg-surface-dark/50 flex items-center gap-2 shrink-0">
        <span className="material-symbols-outlined text-fuchsia-400 text-lg">upload_file</span>
        <span className="text-sm font-medium text-white">Upload GeoTIFF</span>
      </div>

      <div className="p-4 flex-1 overflow-y-auto flex flex-col gap-4">
        {/* Step indicator */}
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider">
          <span className={`px-2 py-0.5 rounded ${
            stage === "idle" ? "bg-primary/20 text-primary" : "bg-slate-700 text-slate-500"
          }`}>1. Select File</span>
          <span className="text-slate-600">→</span>
          <span className={`px-2 py-0.5 rounded ${
            stage === "validating" || stage === "validated" || stage === "error"
              ? "bg-primary/20 text-primary" : "bg-slate-700 text-slate-500"
          }`}>2. Validate</span>
          <span className="text-slate-600">→</span>
          <span className={`px-2 py-0.5 rounded ${
            stage === "uploading" || stage === "done"
              ? "bg-primary/20 text-primary" : "bg-slate-700 text-slate-500"
          }`}>3. Upload</span>
        </div>

        {/* Stage: Idle - File selection */}
        {stage === "idle" && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <div className="text-center">
              <span className="material-symbols-outlined text-5xl text-slate-600 mb-3">cloud_upload</span>
              <p className="text-lg text-white">Select a Mars GeoTIFF</p>
              <p className="text-sm text-slate-400 mt-1">The file will be validated before upload</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              onChange={handleFileSelect}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 hover:bg-fuchsia-500/30 font-bold transition-colors"
            >
              <span className="material-symbols-outlined">folder_open</span>
              Choose File
            </button>
          </div>
        )}

        {/* Stage: Validating */}
        {stage === "validating" && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <span className="material-symbols-outlined text-4xl text-primary animate-spin">progress_activity</span>
            <div className="text-center">
              <p className="text-white">Validating {file?.name}...</p>
              <p className="text-sm text-slate-400 mt-1">Checking CRS, bounds, and metadata</p>
            </div>
          </div>
        )}

        {/* Stage: Validated - Show results and confirm */}
        {stage === "validated" && validation && (
          <>
            {/* Validation success banner */}
            <div className="bg-emerald-500/10 rounded-lg p-3 border border-emerald-500/30 flex items-center gap-3">
              <span className="material-symbols-outlined text-emerald-400 text-2xl">check_circle</span>
              <div>
                <p className="text-emerald-400 font-bold text-sm">Validation Passed</p>
                <p className="text-emerald-400/70 text-xs">This file is a valid Mars GeoTIFF and ready for upload.</p>
              </div>
            </div>

            {/* CRS warning if any */}
            {validation.crs_warning && (
              <div className="bg-amber-500/10 rounded-lg p-3 border border-amber-500/30 flex items-center gap-3">
                <span className="material-symbols-outlined text-amber-400">warning</span>
                <p className="text-amber-300 text-xs">{validation.crs_warning}</p>
              </div>
            )}

            {/* File info */}
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-bg-dark rounded p-2 border border-border-dark">
                <p className="text-[9px] uppercase text-slate-500">Filename</p>
                <p className="text-white text-xs font-mono truncate">{validation.filename}</p>
              </div>
              <div className="bg-bg-dark rounded p-2 border border-border-dark">
                <p className="text-[9px] uppercase text-slate-500">File Size</p>
                <p className="text-white text-xs">{formatBytes(validation.filesize)}</p>
              </div>
              <div className="bg-bg-dark rounded p-2 border border-border-dark">
                <p className="text-[9px] uppercase text-slate-500">CRS</p>
                <p className="text-white text-xs font-mono">{validation.crs}</p>
              </div>
              <div className="bg-bg-dark rounded p-2 border border-border-dark">
                <p className="text-[9px] uppercase text-slate-500">Dimensions</p>
                <p className="text-white text-xs font-mono">{validation.width} x {validation.height}</p>
              </div>
              <div className="bg-bg-dark rounded p-2 border border-border-dark">
                <p className="text-[9px] uppercase text-slate-500">Bands</p>
                <p className="text-white">{validation.bands} ({validation.dtype})</p>
              </div>
              <div className="bg-bg-dark rounded p-2 border border-border-dark">
                <p className="text-[9px] uppercase text-slate-500">Nodata</p>
                <p className="text-white text-xs font-mono">{validation.nodata ?? "None"}</p>
              </div>
            </div>

            {/* Bounds */}
            <div className="bg-bg-dark rounded p-3 border border-border-dark">
              <p className="text-[9px] uppercase text-slate-500 mb-2">Detected Bounds</p>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div><span className="text-slate-500">North: </span><span className="text-white">{validation.bounds.north.toFixed(4)}</span></div>
                <div><span className="text-slate-500">South: </span><span className="text-white">{validation.bounds.south.toFixed(4)}</span></div>
                <div><span className="text-slate-500">West: </span><span className="text-white">{validation.bounds.west.toFixed(4)}</span></div>
                <div><span className="text-slate-500">East: </span><span className="text-white">{validation.bounds.east.toFixed(4)}</span></div>
              </div>
            </div>

            {/* Dataset name input */}
            <div>
              <label className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">Dataset Name</label>
              <input
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="mt-1 w-full bg-bg-dark border border-border-dark rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-fuchsia-500/50"
                placeholder="Enter a name for this dataset"
              />
            </div>
          </>
        )}

        {/* Stage: Uploading */}
        {stage === "uploading" && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <span className="material-symbols-outlined text-4xl text-fuchsia-400 animate-spin">progress_activity</span>
            <div className="text-center">
              <p className="text-white">Uploading and processing...</p>
              <p className="text-sm text-slate-400 mt-1">Generating overlay for map rendering</p>
            </div>
          </div>
        )}

        {/* Stage: Done */}
        {stage === "done" && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <span className="material-symbols-outlined text-5xl text-emerald-400">check_circle</span>
            <div className="text-center">
              <p className="text-lg text-white font-bold">Upload Complete</p>
              <p className="text-sm text-slate-400 mt-1">Dataset is now available on the map</p>
            </div>
          </div>
        )}

        {/* Stage: Error */}
        {stage === "error" && (
          <>
            <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/30 flex items-start gap-3">
              <span className="material-symbols-outlined text-red-400 text-2xl mt-0.5">error</span>
              <div>
                <p className="text-red-400 font-bold text-sm">Validation Failed</p>
                <p className="text-red-400/70 text-xs mt-1">{error}</p>
              </div>
            </div>

            {/* Show partial validation results if available */}
            {validation && (
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-bg-dark rounded p-2 border border-border-dark">
                  <p className="text-[9px] uppercase text-slate-500">Filename</p>
                  <p className="text-white text-xs font-mono truncate">{validation.filename}</p>
                </div>
                <div className="bg-bg-dark rounded p-2 border border-border-dark">
                  <p className="text-[9px] uppercase text-slate-500">CRS Detected</p>
                  <p className="text-red-400 text-xs font-mono">{validation.crs}</p>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Bottom action buttons */}
      <div className="p-3 border-t border-border-dark shrink-0 space-y-2">
        {stage === "validated" && (
          <button
            onClick={handleConfirmUpload}
            className="w-full py-3 px-4 rounded-lg font-bold flex items-center justify-center gap-2 bg-fuchsia-500/20 text-fuchsia-400 hover:bg-fuchsia-500/30 border border-fuchsia-500/30 transition-all"
          >
            <span className="material-symbols-outlined text-lg">cloud_upload</span>
            <span>Confirm Upload</span>
          </button>
        )}
        {(stage === "validated" || stage === "error") && (
          <button
            onClick={handleReset}
            className="w-full py-2 px-4 rounded-lg text-sm flex items-center justify-center gap-2 border border-border-dark text-slate-400 hover:text-white hover:border-slate-500 transition-all"
          >
            <span className="material-symbols-outlined text-sm">restart_alt</span>
            <span>Choose Different File</span>
          </button>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function DataUploadPage() {
  const isMobile = useIsMobile();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetMetadata | null>(null);
  const [showUploadFlow, setShowUploadFlow] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [loading, setLoading] = useState(true);

  // Load datasets on mount
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await fetch("/api/custom/datasets");
      if (response.ok) {
        const data = await response.json();
        setDatasets(data.datasets || []);
      }
    } catch (e) {
      console.error("Failed to load datasets:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = useCallback(async () => {
    if (!selectedDataset) return;

    setIsDeleting(true);
    try {
      const response = await fetch(`/api/custom/${selectedDataset.id}`, { method: "DELETE" });
      if (response.ok) {
        setDatasets((prev) => prev.filter((d) => d.id !== selectedDataset.id));
        setSelectedDataset(null);
      }
    } catch (e) {
      console.error("Failed to delete dataset:", e);
    } finally {
      setIsDeleting(false);
    }
  }, [selectedDataset]);

  const handleUploadComplete = useCallback((dataset: DatasetMetadata) => {
    setDatasets((prev) => [...prev, dataset]);
    setSelectedDataset(dataset);
    setShowUploadFlow(false);
  }, []);

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
            <button
              onClick={() => setMobileMenuOpen(p => !p)}
              className="flex items-center justify-center w-9 h-9 rounded-lg hover:bg-white/10 text-slate-300"
            >
              <span className="material-symbols-outlined text-2xl">
                {mobileMenuOpen ? "close" : "menu"}
              </span>
            </button>
          </header>
          {mobileMenuOpen && (
            <div className="absolute top-12 left-0 right-0 z-50 border-b border-border-dark bg-bg-dark p-4 flex flex-col gap-2 shadow-xl">
              <Link to="/" className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors">Workbench</Link>
              <Link to="/download" className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors">Data Download</Link>
              <Link to="/upload" className="text-sm font-medium text-fuchsia-400 px-2 py-2 rounded-lg hover:bg-white/5 transition-colors">Data Upload</Link>
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
              <a href="/download" className="text-slate-400 hover:text-white text-sm font-medium transition-colors">Data Download</a>
              <a href="/upload" className="text-fuchsia-400 text-sm font-medium border-b-2 border-fuchsia-400 pb-0.5">Data Upload</a>
            </nav>
          </div>
          <Link
            to="/suggestions"
            className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium text-slate-400 hover:text-white border border-border-dark rounded-md hover:bg-white/5 transition-colors shrink-0"
          >
            <span className="material-symbols-outlined text-sm">lightbulb</span>
            Suggest Feature
          </Link>
        </header>
      )}

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-auto max-w-[1600px] mx-auto w-full px-4 md:px-6 py-4 gap-4">
        {/* Disclaimer section */}
        <GeoTIFFDisclaimer />

        {/* Dataset panels */}
        <div className="flex-1 flex flex-col md:flex-row gap-4 md:gap-6 overflow-auto md:overflow-hidden min-h-0">
          {/* Left panel - Dataset list */}
          <aside className="w-full md:w-1/3 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden max-h-[40vh] md:max-h-none shrink-0">
            <div className="p-4 border-b border-border-dark flex justify-between items-center">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">
                Uploaded Datasets ({datasets.length})
              </h3>
              <button
                onClick={() => {
                  setShowUploadFlow(true);
                  setSelectedDataset(null);
                }}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-fuchsia-500/20 text-fuchsia-400 text-[11px] font-bold border border-fuchsia-500/30 hover:bg-fuchsia-500/30 transition-colors"
              >
                <span className="material-symbols-outlined text-sm">add</span>
                Upload
              </button>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-2">
              {loading && (
                <div className="flex items-center justify-center p-8 text-slate-500">
                  <span className="material-symbols-outlined animate-spin mr-2">progress_activity</span>
                  Loading...
                </div>
              )}
              {!loading && datasets.length === 0 && (
                <div className="p-6 text-center">
                  <span className="material-symbols-outlined text-3xl text-slate-600 mb-2">cloud_upload</span>
                  <p className="text-slate-500 text-sm">No datasets uploaded yet</p>
                  <p className="text-slate-600 text-xs mt-1">
                    Click "Upload" to add a Mars GeoTIFF
                  </p>
                </div>
              )}
              {datasets.map((dataset) => (
                <DatasetItem
                  key={dataset.id}
                  dataset={dataset}
                  isSelected={selectedDataset?.id === dataset.id && !showUploadFlow}
                  onClick={() => {
                    setSelectedDataset(dataset);
                    setShowUploadFlow(false);
                  }}
                />
              ))}
            </div>
          </aside>

          {/* Right panel - Detail or Upload flow */}
          <section className="flex-1 flex flex-col bg-surface-dark rounded-xl border border-border-dark overflow-hidden">
            {showUploadFlow ? (
              <UploadFlow onUploadComplete={handleUploadComplete} />
            ) : selectedDataset ? (
              <DatasetDetail
                dataset={selectedDataset}
                onDelete={handleDelete}
                isDeleting={isDeleting}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-500">
                <div className="text-center">
                  <span className="material-symbols-outlined text-5xl mb-3 text-slate-600">layers</span>
                  <p className="text-lg">Select a dataset or upload a new one</p>
                  <p className="text-sm mt-1">Upload Mars GeoTIFF files to visualize on the map</p>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
