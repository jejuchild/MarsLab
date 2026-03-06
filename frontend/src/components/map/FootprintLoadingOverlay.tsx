interface FootprintLoadingOverlayProps {
  loading: boolean;
  instrument?: string;
  count?: number;
}

export default function FootprintLoadingOverlay({
  loading,
  instrument,
  count,
}: FootprintLoadingOverlayProps) {
  if (!loading) return null;

  return (
    <div className="absolute bottom-6 right-6 z-20 transition-opacity duration-300">
      <div className="flex items-center gap-2.5 rounded-full border border-border-dark bg-bg-dark/90 px-4 py-2 backdrop-blur-md">
        {/* Pulsing dot */}
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-500" />
        </span>

        <div className="flex items-center gap-1.5 text-xs text-slate-300">
          <span>Loading footprints</span>
          {instrument && (
            <span className="font-medium text-cyan-400">
              · {instrument}
            </span>
          )}
          {count !== undefined && count > 0 && (
            <span className="tabular-nums text-slate-500">
              ({count.toLocaleString()})
            </span>
          )}
          <span className="text-slate-500">…</span>
        </div>
      </div>
    </div>
  );
}
