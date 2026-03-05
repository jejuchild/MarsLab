import { useState, useCallback } from "react";
import type { InstrumentType } from "../types";

type ProductIdBadgeProps = {
  productId: string;
  instrument: InstrumentType;
};

const INSTRUMENT_COLORS: Record<string, string> = {
  HIRISE: "bg-yellow-500/20 text-yellow-400 border-yellow-500/40",
  CRISM: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40",
  CRISM_TRR3: "bg-teal-500/20 text-teal-400 border-teal-500/40",
  HIRISE_DTM: "bg-amber-500/20 text-amber-400 border-amber-500/40",
  CTX: "bg-pink-500/20 text-pink-400 border-pink-500/40",
  SHARAD: "bg-green-500/20 text-green-400 border-green-500/40",
};

const DEFAULT_COLOR = "bg-slate-500/20 text-slate-400 border-slate-500/40";

export default function ProductIdBadge({ productId, instrument }: ProductIdBadgeProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(productId).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [productId]);

  const badgeColor = INSTRUMENT_COLORS[instrument] ?? DEFAULT_COLOR;

  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase ${badgeColor}`}
      >
        {instrument}
      </span>
      <span className="font-mono text-xs text-slate-300 select-all">
        {productId}
      </span>
      <button
        onClick={handleCopy}
        className="flex items-center justify-center rounded p-0.5 text-slate-500 transition-colors hover:bg-surface-dark hover:text-slate-300"
        title="Copy product ID"
      >
        <span className="material-symbols-outlined text-[16px]">
          {copied ? "check" : "content_copy"}
        </span>
      </button>
    </div>
  );
}
