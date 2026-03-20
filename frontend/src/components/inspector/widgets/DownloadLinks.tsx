import { useState, useEffect } from "react";
import type { InstrumentType } from "../types";
import { formatBytes } from "../types";
import {
  listLocalFiles,
  downloadLocalZip,
  type LocalFileInfo,
  type Instrument,
} from "../../../api/search";

type DownloadLinksProps = {
  productId: string;
  instrument: InstrumentType;
};

type ProductUrls = {
  jp2_url?: string;
  jp2_size_bytes?: number;
  jp2_filename?: string;
  lbl_url?: string;
  img_url?: string;
  img_filename?: string;
  browse_urls?: Record<string, string>;
  product_type?: string;
};

const BROWSE_LABELS: Record<string, { label: string; color: string }> = {
  vna: { label: "VNIR", color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 hover:bg-emerald-500/30" },
  hyd: { label: "HYD", color: "bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/40 hover:bg-fuchsia-500/30" },
  ice: { label: "ICE", color: "bg-blue-500/20 text-blue-400 border-blue-500/40 hover:bg-blue-500/30" },
  ic2: { label: "CO₂", color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40 hover:bg-cyan-500/30" },
};

const SUPPORTED_INSTRUMENTS = new Set<string>(["HIRISE", "CRISM", "CRISM_TRR3"]);

// Map InstrumentType to lowercase Instrument for API calls
const INSTRUMENT_MAP: Record<string, Instrument> = {
  HIRISE: "hirise",
  CRISM: "crism",
  CRISM_TRR3: "crism",
  SHARAD: "sharad",
  SHARAD_HIGHRES: "sharad_highres",
};

export default function DownloadLinks({ productId, instrument }: DownloadLinksProps) {
  const [urls, setUrls] = useState<ProductUrls | null>(null);
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");
  const [localFiles, setLocalFiles] = useState<LocalFileInfo[]>([]);
  const [localTotalSize, setLocalTotalSize] = useState(0);

  // Fetch external URLs
  useEffect(() => {
    if (!SUPPORTED_INSTRUMENTS.has(instrument)) {
      setStatus("error");
      return;
    }

    setStatus("loading");
    const endpoint =
      instrument === "HIRISE"
        ? `/api/product-urls/hirise/${productId}`
        : `/api/product-urls/crism/${productId}`;

    fetch(endpoint)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<ProductUrls>;
      })
      .then((data) => {
        setUrls(data);
        setStatus("loaded");
      })
      .catch(() => {
        setStatus("error");
      });
  }, [productId, instrument]);

  // Fetch local files for save-to-local
  useEffect(() => {
    const inst = INSTRUMENT_MAP[instrument];
    if (!inst) return;
    listLocalFiles(productId, inst)
      .then((res) => {
        setLocalFiles(res.files);
        setLocalTotalSize(res.total_size);
      })
      .catch(() => {
        setLocalFiles([]);
      });
  }, [productId, instrument]);

  if (!SUPPORTED_INSTRUMENTS.has(instrument)) return null;

  if (status === "loading") {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border-dark bg-bg-dark/60 p-3">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-blue-400" />
        <span className="text-xs text-slate-500">Loading download links…</span>
      </div>
    );
  }

  if (status === "error" || !urls) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border-dark bg-bg-dark/60 p-3">
        <span className="material-symbols-outlined text-sm text-slate-500">
          link_off
        </span>
        <span className="text-xs text-slate-500">No download links available</span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* HiRISE JP2 download */}
      {instrument === "HIRISE" && urls.jp2_url && (
        <a
          href={urls.jp2_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-lg border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-sky-400 transition-colors hover:bg-sky-500/20"
        >
          <span className="material-symbols-outlined text-[18px]">
            cloud_download
          </span>
          <span className="flex-1 truncate font-mono text-xs">
            {urls.jp2_filename ?? "Download JP2"}
          </span>
          {urls.jp2_size_bytes != null && (
            <span className="rounded-full bg-sky-500/20 px-2 py-0.5 text-[10px] font-medium">
              {formatBytes(urls.jp2_size_bytes)}
            </span>
          )}
        </a>
      )}

      {/* CRISM browse downloads */}
      {urls.browse_urls && Object.keys(urls.browse_urls).length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {Object.entries(urls.browse_urls).map(([key, url]) => {
            const config = BROWSE_LABELS[key] ?? {
              label: key.toUpperCase(),
              color: "bg-slate-500/20 text-slate-400 border-slate-500/40 hover:bg-slate-500/30",
            };
            return (
              <a
                key={key}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center justify-center gap-1.5 rounded-md border px-2 py-1.5 text-xs font-medium transition-colors ${config.color}`}
              >
                <span className="material-symbols-outlined text-[14px]">
                  download
                </span>
                {config.label}
              </a>
            );
          })}
        </div>
      )}

      {/* CRISM IMG download */}
      {(instrument === "CRISM" || instrument === "CRISM_TRR3") && urls.img_url && (
        <a
          href={urls.img_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-lg border border-violet-500/40 bg-violet-500/10 px-3 py-2 text-violet-400 transition-colors hover:bg-violet-500/20"
        >
          <span className="material-symbols-outlined text-[18px]">
            cloud_download
          </span>
          <span className="flex-1 truncate font-mono text-xs">
            {urls.img_filename ?? "Download IMG"}
          </span>
        </a>
      )}

      {/* Save to Local — server files to browser */}
      {localFiles.length > 0 && (
        <button
          onClick={() => {
            const inst = INSTRUMENT_MAP[instrument];
            if (inst) downloadLocalZip(productId, inst);
          }}
          className="flex w-full items-center gap-2 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-cyan-400 transition-colors hover:bg-cyan-500/20"
        >
          <span className="material-symbols-outlined text-[18px]">save_alt</span>
          <span className="flex-1 text-left text-xs font-medium">
            Save to Local ({localFiles.length} files)
          </span>
          <span className="rounded-full bg-cyan-500/20 px-2 py-0.5 text-[10px] font-medium">
            {formatBytes(localTotalSize)}
          </span>
        </button>
      )}
    </div>
  );
}
