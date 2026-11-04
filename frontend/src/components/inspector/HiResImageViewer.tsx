import { useEffect, useState } from "react";
import ViewerShell, { ViewerAction } from "../viewer/ViewerShell";
import PanZoomSurface from "../viewer/PanZoomSurface";
import { getProductBounds } from "../../utils/productBounds";

type HiResImageViewerProps = {
  productId: string;
  /** Inspected point, marked on the image when it falls inside it. */
  lat?: number;
  lon?: number;
  /** Open a SHARAD radargram that crosses this image, centred on the clicked point. */
  onOpenRadargram?: (productId: string, point: { lat: number; lon: number }) => void;
  onClose: () => void;
};

type SharadCrossing = { product_id: string; start_trace: number; coords: [number, number][] };

const fmt = (lat: number, lon: number) => `${lat.toFixed(4)}°, ${lon.toFixed(4)}°E`;

/**
 * Fullscreen HiRISE image. Shows the same north-up overlay image that is
 * draped on the map (on its product bounds), so the inspected point and the
 * cursor position map to lat/lon with the same geometry as the map.
 */
export default function HiResImageViewer({ productId, lat, lon, onOpenRadargram, onClose }: HiResImageViewerProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [isJP2, setIsJP2] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bounds, setBounds] = useState<{ west: number; east: number; south: number; north: number } | null>(null);
  const [hover, setHover] = useState<{ fx: number; fy: number } | null>(null);
  const [sharad, setSharad] = useState<SharadCrossing[]>([]);

  // SHARAD hi-res ground tracks that cross this image.
  useEffect(() => {
    let cancelled = false;
    setSharad([]);
    fetch(`/api/crossings/hirise/${encodeURIComponent(productId)}/sharad`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled && d?.tracks) setSharad(d.tracks); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [productId]);

  // JP2-backed high-res when downloaded, else a smaller browse image.
  useEffect(() => {
    let revoke: string | null = null;
    let cancelled = false;
    setSrc(null);
    setError(null);
    (async () => {
      try {
        const ex = await fetch(`/api/exists/hirise/${encodeURIComponent(productId)}`);
        const hasJP2 = ex.ok && (await ex.json()).has_core;
        const res = await fetch(`/hirise/overlay/${encodeURIComponent(productId)}.png?max_size=${hasJP2 ? 8192 : 2048}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        if (cancelled) return;
        revoke = URL.createObjectURL(blob);
        setSrc(revoke);
        setIsJP2(hasJP2);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    getProductBounds(productId)
      .then((b) => { if (!cancelled) setBounds(b); })
      .catch(() => {});
    return () => {
      cancelled = true;
      if (revoke) URL.revokeObjectURL(revoke);
    };
  }, [productId]);

  const span = bounds ? { lon: bounds.east - bounds.west, lat: bounds.north - bounds.south } : null;
  const marker =
    bounds && span && span.lon > 0 && span.lat > 0 && lat !== undefined && lon !== undefined
      ? { fx: (lon - bounds.west) / span.lon, fy: (bounds.north - lat) / span.lat }
      : null;
  const markerInside = marker && marker.fx >= 0 && marker.fx <= 1 && marker.fy >= 0 && marker.fy <= 1;
  const toFrac = (lo: number, la: number) =>
    bounds && span ? { fx: (lo - bounds.west) / span.lon, fy: (bounds.north - la) / span.lat } : null;
  const lines =
    bounds && span
      ? sharad.map((t) => ({
          id: `${t.product_id}#${t.start_trace}`,
          label: `SHARAD ${t.product_id} — click to open the radargram here`,
          points: t.coords.map(([lo, la]) => toFrac(lo, la)!),
        }))
      : [];
  const nTracks = new Set(sharad.map((t) => t.product_id)).size;

  const cursor =
    hover && bounds && span
      ? { lat: bounds.north - hover.fy * span.lat, lon: bounds.west + hover.fx * span.lon }
      : null;

  return (
    <ViewerShell
      variant="fullscreen"
      icon="photo_camera"
      title={`HiRISE ${productId}`}
      subtitle={lat !== undefined && lon !== undefined ? `Inspected point ${fmt(lat, lon)}` : undefined}
      badge={
        src && (
          <span
            className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
              isJP2 ? "bg-emerald-500/20 text-emerald-300" : "bg-amber-500/20 text-amber-300"
            }`}
            title={isJP2 ? "Full-resolution data is downloaded" : "Reduced browse image — download the product for full resolution"}
          >
            {isJP2 ? "Full resolution" : "Browse preview"}
          </span>
        )
      }
      actions={src && <ViewerAction icon="download" label="Download PNG" href={src} download={`${productId}.png`} />}
      footer={
        <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
          <span>{cursor ? `Cursor ${fmt(cursor.lat, cursor.lon)}` : "Move over the image for coordinates"}</span>
          {marker && !markerInside && <span className="text-amber-300">Inspected point is outside this image</span>}
          {nTracks > 0 && (
            <span className="text-amber-300">
              {nTracks} SHARAD track{nTracks > 1 ? "s" : ""} cross this image{onOpenRadargram ? " — click a line to open its radargram there" : ""}
            </span>
          )}
          <span className="ml-auto hidden md:inline">Wheel / double-click zoom · drag pan · 0 fit · 1 actual · Esc close</span>
        </div>
      }
      onClose={onClose}
    >
      {error ? (
        <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-red-300">
          <span className="material-symbols-outlined text-3xl">error</span>
          Could not load the image ({error})
        </div>
      ) : src ? (
        <PanZoomSurface
          src={src}
          alt={`HiRISE ${productId}`}
          marker={markerInside ? { fx: marker.fx, fy: marker.fy, label: "Inspected point" } : null}
          onHover={bounds ? setHover : undefined}
          lines={lines}
          onLineClick={
            onOpenRadargram && bounds && span
              ? (id, pos) => {
                  onOpenRadargram(id.split("#")[0]!, {
                    lat: bounds.north - pos.fy * span.lat,
                    lon: bounds.west + pos.fx * span.lon,
                  });
                  onClose();
                }
              : undefined
          }
        />
      ) : (
        <div className="flex h-full items-center justify-center gap-3 text-sm text-slate-300">
          <span className="material-symbols-outlined animate-spin text-2xl text-primary">progress_activity</span>
          Loading HiRISE image…
        </div>
      )}
    </ViewerShell>
  );
}
