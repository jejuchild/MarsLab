import { useState, useEffect } from "react";
import type { InstrumentType } from "../types";
import { formatBytes } from "../types";
import {
  listLocalFiles,
  downloadLocalZip,
  type LocalFileInfo,
  type Instrument,
} from "../../../api/search";

type SaveToLocalButtonProps = {
  productId: string;
  instrument: InstrumentType;
};

const INSTRUMENT_MAP: Record<string, Instrument> = {
  HIRISE: "hirise",
  CRISM: "crism",
  CRISM_TRR3: "crism",
  SHARAD: "sharad",
  SHARAD_HIGHRES: "sharad_highres",
};

// DownloadLinks already shows Save to Local for these instruments
const HANDLED_BY_DOWNLOAD_LINKS = new Set(["HIRISE", "CRISM", "CRISM_TRR3"]);

export default function SaveToLocalButton({ productId, instrument }: SaveToLocalButtonProps) {
  const [localFiles, setLocalFiles] = useState<LocalFileInfo[]>([]);
  const [localTotalSize, setLocalTotalSize] = useState(0);
  const [loading, setLoading] = useState(true);

  const skip = HANDLED_BY_DOWNLOAD_LINKS.has(instrument);

  useEffect(() => {
    if (skip) { setLoading(false); return; }
    const inst = INSTRUMENT_MAP[instrument];
    if (!inst) {
      setLoading(false);
      return;
    }
    setLoading(true);
    listLocalFiles(productId, inst)
      .then((res) => {
        setLocalFiles(res.files);
        setLocalTotalSize(res.total_size);
      })
      .catch(() => {
        setLocalFiles([]);
      })
      .finally(() => setLoading(false));
  }, [productId, instrument]);

  if (loading || skip) return null;
  if (localFiles.length === 0) return null;

  return (
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
  );
}
