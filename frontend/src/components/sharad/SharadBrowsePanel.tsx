import PanZoomSurface from "../viewer/PanZoomSurface";

/**
 * Standard SHARAD (THM browse) radargram, docked beside the Inspector like the
 * hi-res viewer — it used to be a modal that hid the map.
 */
export default function SharadBrowsePanel({
  productId,
  quickviewUrl,
  start,
  stop,
  onClose,
}: {
  productId: string;
  quickviewUrl: string;
  start: { lat: number; lon: number };
  stop: { lat: number; lon: number };
  onClose: () => void;
}) {
  return (
    <section className="flex h-full w-[min(640px,45vw)] flex-col bg-[#0a0f18]" aria-label={`SHARAD ${productId}`}>
      <header className="flex shrink-0 items-center gap-2 border-b border-border-dark px-3 py-2">
        <span className="material-symbols-outlined text-[18px] text-amber-400">radar</span>
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-xs font-bold uppercase tracking-wider text-white">SHARAD browse · {productId}</h2>
          <p className="truncate font-mono text-[10px] text-slate-400">
            {start.lat.toFixed(2)}°, {start.lon.toFixed(2)}°E → {stop.lat.toFixed(2)}°, {stop.lon.toFixed(2)}°E
          </p>
        </div>
        <a
          href={quickviewUrl}
          download={`${productId}.jpg`}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-300 hover:bg-white/10"
          aria-label="Download browse image"
          title="Download"
        >
          <span className="material-symbols-outlined text-[18px]">download</span>
        </a>
        <button
          type="button"
          onClick={onClose}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-300 hover:bg-white/10"
          aria-label="Close SHARAD browse"
          title="Close (Esc)"
        >
          <span className="material-symbols-outlined text-[20px]">close</span>
        </button>
      </header>
      <div className="relative min-h-0 flex-1">
        <PanZoomSurface src={quickviewUrl} alt={`SHARAD browse radargram ${productId}`} />
      </div>
      <p className="shrink-0 border-t border-border-dark px-3 py-1.5 text-[10px] text-slate-500">
        Browse product (uncalibrated, no per-trace geolocation). Hi-res (R_) tracks open the full radargram with map linking.
      </p>
    </section>
  );
}
