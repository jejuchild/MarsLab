import { useEffect, useMemo, useState } from "react";

export type HiriseCrossing = {
  product_id: string;
  start_trace: number;
  end_trace: number;
  lat: number;
  lon: number;
};

type Zone = { start: number; end: number; items: HiriseCrossing[] };

/** Merge overlapping crossings (several HiRISE images of one site) into zones. */
export function mergeCrossings(crossings: HiriseCrossing[]): Zone[] {
  const sorted = [...crossings].sort((a, b) => a.start_trace - b.start_trace);
  const zones: Zone[] = [];
  for (const c of sorted) {
    const last = zones[zones.length - 1];
    if (last && c.start_trace <= last.end) {
      last.end = Math.max(last.end, c.end_trace);
      last.items.push(c);
    } else {
      zones.push({ start: c.start_trace, end: c.end_trace, items: [c] });
    }
  }
  return zones;
}

/**
 * Shaded bands over the radargram where its ground track passes over HiRISE
 * images; the chip on each band lists the images and shows one on the map.
 */
export default function CrossingBands({
  productId,
  viewX,
  onPick,
}: {
  productId: string;
  /** Visible fraction of the full trace range. */
  viewX: { start: number; end: number };
  onPick: (c: HiriseCrossing) => void;
}) {
  const [data, setData] = useState<{ total: number; crossings: HiriseCrossing[] } | null>(null);
  const [open, setOpen] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setOpen(null);
    fetch(`/api/crossings/sharad/${encodeURIComponent(productId)}/hirise`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled && d) setData({ total: d.total_traces, crossings: d.crossings ?? [] });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [productId]);

  const zones = useMemo(() => (data ? mergeCrossings(data.crossings) : []), [data]);
  if (!data || zones.length === 0) return null;

  const span = viewX.end - viewX.start || 1;
  return (
    <div className="pointer-events-none absolute inset-0 z-[15]" aria-label="HiRISE crossings">
      {zones.map((z, i) => {
        const a = (z.start / data.total - viewX.start) / span;
        const b = ((z.end + 1) / data.total - viewX.start) / span;
        if (b < 0 || a > 1) return null;
        const left = Math.max(0, a) * 100;
        const width = Math.max(0.4, (Math.min(1, b) - Math.max(0, a)) * 100);
        return (
          <div key={i} className="absolute top-0 bottom-0" style={{ left: `${left}%`, width: `${width}%` }}>
            <div className="absolute inset-0 border-x border-emerald-400/70 bg-emerald-400/10" />
            <button
              type="button"
              onClick={() => setOpen(open === i ? null : i)}
              className="pointer-events-auto absolute left-1/2 top-1 -translate-x-1/2 whitespace-nowrap rounded bg-emerald-600/90 px-1.5 py-0.5 text-[10px] font-bold text-white shadow hover:bg-emerald-500"
              title="This part of the track passes over HiRISE images"
              aria-expanded={open === i}
            >
              HiRISE{z.items.length > 1 ? ` ×${z.items.length}` : ""}
            </button>
            {open === i && (
              <div className="pointer-events-auto absolute left-1/2 top-7 z-10 w-56 -translate-x-1/2 rounded-md border border-border-dark bg-surface-dark/95 p-1 shadow-xl">
                {z.items.map((c) => (
                  <button
                    key={`${c.product_id}-${c.start_trace}`}
                    type="button"
                    onClick={() => {
                      setOpen(null);
                      onPick(c);
                    }}
                    className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left font-mono text-[11px] text-slate-200 hover:bg-white/10"
                  >
                    <span className="material-symbols-outlined text-[14px] text-emerald-400">photo_camera</span>
                    {c.product_id}
                  </button>
                ))}
                <p className="px-2 pb-1 pt-0.5 text-[10px] text-slate-500">Shows the image on the map at the crossing</p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
