import { useState, useCallback } from "react";

type CoordinateDisplayProps = {
  lat: number;
  lon: number;
};

export default function CoordinateDisplay({ lat, lon }: CoordinateDisplayProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(`${lat}, ${lon}`).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [lat, lon]);

  return (
    <div
      className="flex cursor-pointer items-center gap-3 rounded-md border border-border-dark/50 bg-bg-dark/40 px-2.5 py-1.5 transition-colors hover:border-border-dark"
      onClick={handleCopy}
      title="Click to copy coordinates"
    >
      <div className="flex items-center gap-1">
        <span className="text-[10px] uppercase text-slate-500">Lat</span>
        <span className="font-mono text-xs text-slate-300">
          {lat.toFixed(4)}°
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-[10px] uppercase text-slate-500">Lon</span>
        <span className="font-mono text-xs text-slate-300">
          {lon.toFixed(4)}°
        </span>
      </div>
      <span className="material-symbols-outlined text-[14px] text-slate-500">
        {copied ? "check" : "content_copy"}
      </span>
    </div>
  );
}
