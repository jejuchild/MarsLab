import { useMemo, useState } from "react";
import HiResImageViewer from "./HiResImageViewer";
import QuickviewImage from "./widgets/QuickviewImage";
import useSpectrumData from "./hooks/useSpectrumData";
import usePixelAtPoint from "./hooks/usePixelAtPoint";
import { LANES, LANE_CONFIG, instrumentFor } from "./laneConfig";
import type { ProductDetailHandlers } from "./ProductDetail";
import type { Lane, LaneProduct } from "../../api/inspector";
import type { InspectorContext, SpectrumData } from "./types";

/** Lanes whose quickview can be draped on the map. */
const OVERLAYABLE: Lane[] = ["HIRISE", "CRISM", "CTX"];

export interface ComparePanelProps {
  lanes: Record<Lane, LaneProduct[]>;
  context: InspectorContext;
  handlers: ProductDetailHandlers;
  /** Switch to the lane's detail view with this product active. */
  onOpenLane: (lane: Lane, productId: string) => void;
  onOpenRadargram?: (productId: string) => void;
}

/**
 * "Compare at this point": one card per instrument with the best product
 * covering the point, side by side in a single scroll — the core MarsLab
 * workflow (docs/identity.md), instead of flipping between lane tabs.
 */
export default function ComparePanel({ lanes, context, handlers, onOpenLane, onOpenRadargram }: ComparePanelProps) {
  const best = useMemo(() => {
    const out: Partial<Record<Lane, LaneProduct>> = {};
    for (const lane of LANES) {
      const covering = (lanes[lane] ?? []).filter((p) => p.covers);
      // Prefer the clicked product when it is in this lane.
      out[lane] = covering.find((p) => p.product_id === context.productId) ?? covering[0];
    }
    return out;
  }, [lanes, context.productId]);

  const overlayTargets = OVERLAYABLE.map((lane) => best[lane]?.product_id).filter((id): id is string => !!id);
  const allShown = overlayTargets.length > 0 && overlayTargets.every((id) => handlers.getActiveOverlay(id));

  const toggleAll = () => {
    for (const id of overlayTargets) {
      const active = handlers.getActiveOverlay(id);
      if (allShown) handlers.onSetOverlay(id, null);
      else if (!active) handlers.onSetOverlay(id, "quickview");
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="flex items-center justify-between gap-2 px-4 py-2 border-b border-border-dark/60">
        <p className="text-[10px] text-slate-500">
          Best product covering this point, per instrument
        </p>
        {overlayTargets.length > 0 && (
          <button
            type="button"
            onClick={toggleAll}
            className="shrink-0 px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider bg-primary/15 border border-primary/40 text-primary hover:bg-primary/25 transition-colors"
            title="Drape the quickview of every covering image product on the map"
          >
            {allShown ? "Hide all from map" : "Show all on map"}
          </button>
        )}
      </div>

      {LANES.map((lane) => (
        <CompareCard
          key={lane}
          lane={lane}
          product={best[lane]}
          coverCount={(lanes[lane] ?? []).filter((p) => p.covers).length}
          nearby={(lanes[lane] ?? []).filter((p) => !p.covers)}
          context={context}
          handlers={handlers}
          onOpenLane={onOpenLane}
          onOpenRadargram={onOpenRadargram}
        />
      ))}
    </div>
  );
}

function CompareCard({
  lane,
  product,
  coverCount,
  nearby,
  context,
  handlers,
  onOpenLane,
  onOpenRadargram,
}: {
  lane: Lane;
  product: LaneProduct | undefined;
  coverCount: number;
  nearby: LaneProduct[];
  context: InspectorContext;
  handlers: ProductDetailHandlers;
  onOpenLane: (lane: Lane, productId: string) => void;
  onOpenRadargram?: (productId: string) => void;
}) {
  const cfg = LANE_CONFIG[lane];
  const instrument = product ? instrumentFor(lane, product.variant) : null;
  const overlay = product ? handlers.getActiveOverlay(product.product_id) : null;
  const [viewerOpen, setViewerOpen] = useState(false);

  return (
    <section className="px-4 py-3 border-b border-border-dark/60" aria-label={`${cfg.label} at this point`}>
      <header className="flex items-center gap-2 mb-2">
        <span className="w-1.5 h-4 rounded-sm" style={{ backgroundColor: cfg.accent }} aria-hidden />
        <h3 className="text-[11px] font-bold uppercase tracking-wider text-white">{cfg.label}</h3>
        <span className="ml-auto text-[9px] font-mono text-slate-500">
          {coverCount} cover{nearby.length > 0 ? ` · ${nearby.length} near` : ""}
        </span>
      </header>

      {!product || !instrument ? (
        <p className="text-[11px] text-slate-500">
          No {cfg.label} coverage at this point
          {nearby[0]?.distance_km != null ? ` — nearest is ${nearby[0].distance_km.toFixed(1)} km away.` : "."}
        </p>
      ) : (
        <div className="flex gap-3">
          {instrument === "HIRISE" ? (
            <button
              type="button"
              onClick={() => setViewerOpen(true)}
              className="group relative w-28 h-28 shrink-0 rounded overflow-hidden border border-border-dark bg-black hover:border-primary focus-visible:border-primary"
              aria-label={`Open ${product.product_id} full screen`}
              title="Open full screen with the inspected point marked"
            >
              <QuickviewImage productId={product.product_id} instrument={instrument} />
              <span className="material-symbols-outlined absolute bottom-1 right-1 rounded bg-black/70 p-0.5 text-[16px] text-white opacity-80 group-hover:opacity-100">
                open_in_full
              </span>
            </button>
          ) : (
            <div className="w-28 h-28 shrink-0 rounded overflow-hidden border border-border-dark bg-black">
              <QuickviewImage productId={product.product_id} instrument={instrument} />
            </div>
          )}
          <div className="min-w-0 flex-1 flex flex-col gap-1.5">
            <p className="text-[10px] font-mono text-slate-300 truncate" title={product.product_id}>
              {product.product_id}
            </p>
            <p className="text-[10px] text-slate-500">
              {cfg.variants.find((v) => v.value === product.variant)?.description}
            </p>

            {(instrument === "CRISM" || instrument === "CRISM_TRR3") && (
              <SpectrumAtPoint
                product={{
                  instrument,
                  productId: product.product_id,
                  lat: context.lat,
                  lon: context.lon,
                  ...(product.product_id === context.productId
                    ? { pixelLine: context.pixelLine, pixelSample: context.pixelSample }
                    : {}),
                }}
              />
            )}

            <div className="mt-auto flex flex-wrap gap-1.5">
              {OVERLAYABLE.includes(lane) && (
                <button
                  type="button"
                  onClick={() => handlers.onSetOverlay(product.product_id, overlay ? null : "quickview")}
                  className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider border transition-colors ${
                    overlay
                      ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300"
                      : "border-border-dark text-slate-300 hover:text-white hover:border-slate-500"
                  }`}
                >
                  {overlay ? "On map ✓" : "Show on map"}
                </button>
              )}
              {instrument === "SHARAD_HIGHRES" && onOpenRadargram && (
                <button
                  type="button"
                  onClick={() => onOpenRadargram(product.product_id)}
                  className="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider border border-red-500/40 text-red-300 hover:bg-red-500/15 transition-colors"
                >
                  Radargram
                </button>
              )}
              <button
                type="button"
                onClick={() => onOpenLane(lane, product.product_id)}
                className="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider border border-border-dark text-slate-300 hover:text-white hover:border-slate-500 transition-colors"
              >
                Details →
              </button>
            </div>
          </div>
        </div>
      )}
      {viewerOpen && product && (
        <HiResImageViewer
          productId={product.product_id}
          lat={context.lat}
          lon={context.lon}
          onClose={() => setViewerOpen(false)}
        />
      )}
    </section>
  );
}

/** Reflectance spectrum of a CRISM product at the inspected point (sparkline). */
function SpectrumAtPoint({ product }: { product: InspectorContext }) {
  const pixel = usePixelAtPoint(product);
  const { spectrumData, loading, unavailableReason } = useSpectrumData(
    product.productId, product.instrument, pixel.line, pixel.sample, product.lat, product.lon,
  );
  if (loading) return <p className="text-[10px] text-slate-500">Loading spectrum…</p>;
  if (pixel.line === undefined) return <p className="text-[10px] text-slate-600">Spectrum unavailable at this point</p>;
  if (!spectrumData) {
    return unavailableReason ? (
      <p className="text-[10px] text-slate-500" title={unavailableReason}>Spectrum not available for this cube type</p>
    ) : null;
  }
  return <Sparkline data={spectrumData} />;
}

function Sparkline({ data }: { data: SpectrumData }) {
  const pts = data.wavelengths
    .map((w, i) => [w, data.reflectance[i]] as const)
    .filter((p): p is readonly [number, number] => p[1] != null && Number.isFinite(p[1]));
  if (pts.length < 2) return null;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const W = 160, H = 40;
  const path = pts
    .map(([x, y], i) => `${i ? "L" : "M"}${(((x - x0) / (x1 - x0 || 1)) * W).toFixed(1)},${(H - ((y - y0) / (y1 - y0 || 1)) * H).toFixed(1)}`)
    .join("");
  return (
    <figure className="m-0">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" role="img" aria-label="Reflectance spectrum at this point">
        <path d={path} fill="none" stroke="#fbbf24" strokeWidth={1.2} />
      </svg>
      <figcaption className="text-[9px] font-mono text-slate-500">
        I/F {y0.toFixed(3)}–{y1.toFixed(3)} · {x0.toFixed(2)}–{x1.toFixed(2)} µm
      </figcaption>
    </figure>
  );
}
