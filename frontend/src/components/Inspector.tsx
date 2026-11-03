import { useCallback, useEffect, useMemo, useState } from "react";
import LaneTabs, { type LaneTabConfig } from "./inspector/lanes/LaneTabs";
import Lane from "./inspector/lanes/Lane";
import CrossSection, { type CrossActions } from "./inspector/cross/CrossSection";
import ProductDetail, { type ProductDetailHandlers } from "./inspector/ProductDetail";
import ComparePanel from "./inspector/ComparePanel";
import { LANES, LANE_CONFIG, laneForInstrument } from "./inspector/laneConfig";
import { fetchAtPoint, pickDefaultLane, type AtPointResponse, type Lane as LaneId, type LaneVariant } from "../api/inspector";
import type { InspectorContext, RecentProduct } from "./inspector/types";

export interface InspectorProps {
  /** Clicked point or product. When null, the panel is hidden. */
  context: InspectorContext | null;
  onClose: () => void;
  onCollapse?: () => void;
  isMobile?: boolean;
  handlers: ProductDetailHandlers;
  recentProducts?: RecentProduct[];
  onSelectRecent?: (product: RecentProduct) => void;
  onRemoveRecent?: (productId: string) => void;
  /** Open the full SHARAD hi-res radargram viewer (drill-in; closing returns here).
   *  `point` is the inspected map point (planetocentric) so the viewer can
   *  centre on and mark the nearest trace. */
  onOpenRadargram?: (productId: string, point?: { lat: number; lon: number }) => void;
  /** Launchers for the Cross-Analysis tools at the bottom of the panel. */
  crossActions?: CrossActions;
}

type InspectorView = "compare" | "lanes";
const VIEW_KEY = "marslab.inspectorView";
function loadView(): InspectorView {
  try {
    return localStorage.getItem(VIEW_KEY) === "lanes" ? "lanes" : "compare";
  } catch {
    return "compare";
  }
}

const DEFAULT_VARIANTS: Record<LaneId, LaneVariant> = {
  SHARAD: "standard",
  CRISM: "standard",
  HIRISE: "image",
  CTX: "image",
};

/**
 * MarsLab Inspector — one panel, four instrument lanes (docs/identity.md).
 *
 * Fetches `/api/inspector/at-point` for the selected coordinate and shows
 * SHARAD / CRISM / HiRISE / CTX as tabs. Each lane renders full product
 * detail in place; deep viewers (radargram, 3D DTM, analysis panels) open
 * as drill-ins from MainPage and return here when closed.
 */
export default function Inspector({
  context,
  onClose,
  onCollapse,
  isMobile = false,
  handlers,
  recentProducts = [],
  onSelectRecent,
  onRemoveRecent,
  onOpenRadargram,
  crossActions,
}: InspectorProps) {
  // Pass the inspected point along so the radargram opens on the nearest trace.
  const openRadargramAtPoint = onOpenRadargram
    ? (productId: string) => onOpenRadargram(productId, context ? { lat: context.lat, lon: context.lon } : undefined)
    : undefined;
  const [response, setResponse] = useState<AtPointResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeLane, setActiveLane] = useState<LaneId>("HIRISE");
  const [activeProductByLane, setActiveProductByLane] = useState<Partial<Record<LaneId, string>>>({});
  const [variantByLane, setVariantByLane] = useState<Record<LaneId, LaneVariant>>(DEFAULT_VARIANTS);
  const [view, setViewState] = useState<InspectorView>(loadView);
  const setView = useCallback((v: InspectorView) => {
    setViewState(v);
    try {
      localStorage.setItem(VIEW_KEY, v);
    } catch {
      /* storage unavailable — view just isn't remembered */
    }
  }, []);

  const clicked = useMemo(() => (context ? laneForInstrument(context.instrument) : null), [context]);
  const isCustom = context?.instrument === "CUSTOM";

  // Fetch nearby products whenever the selection changes.
  useEffect(() => {
    if (!context || isCustom) {
      setResponse(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchAtPoint(context.lat, context.lon, 10)
      .then((res) => {
        if (cancelled) return;
        setResponse(res);
        const target = laneForInstrument(context.instrument);
        if (target) {
          setActiveLane(target.lane);
          setVariantByLane((prev) => ({ ...prev, [target.lane]: target.variant }));
          setActiveProductByLane({ [target.lane]: context.productId });
        } else {
          setActiveLane(pickDefaultLane(res));
          setActiveProductByLane({});
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [context, isCustom]);

  // The clicked product may lie outside the at-point radius (large CTX/CRISM
  // footprints are matched by center). Make sure it is always listed.
  const lanes = useMemo(() => {
    if (!response) return null;
    if (!context || !clicked) return response.lanes;
    const list = response.lanes[clicked.lane] ?? [];
    if (list.some((p) => p.product_id === context.productId)) return response.lanes;
    return {
      ...response.lanes,
      [clicked.lane]: [
        { product_id: context.productId, title: context.title ?? null, lat: context.lat, lon: context.lon, variant: clicked.variant, distance_km: 0, covers: true },
        ...list,
      ],
    };
  }, [response, context, clicked]);

  // Escape is handled centrally in MainPage (closes the top-most panel).

  // ── Resizable width (desktop) ──
  const [panelWidth, setPanelWidth] = useState(440);
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = panelWidth;
      const onMove = (ev: MouseEvent) => {
        const maxW = Math.floor(window.innerWidth * 0.6);
        setPanelWidth(Math.max(320, Math.min(maxW, startW + (startX - ev.clientX))));
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [panelWidth],
  );

  const openLane = (lane: LaneId, productId: string) => {
    const variant = lanes?.[lane]?.find((p) => p.product_id === productId)?.variant;
    setActiveLane(lane);
    if (variant) setVariantByLane((prev) => ({ ...prev, [lane]: variant }));
    setActiveProductByLane((prev) => ({ ...prev, [lane]: productId }));
    setView("lanes");
  };

  if (!context) return null;

  const tabs: LaneTabConfig[] = LANES.map((lane) => ({
    lane,
    label: LANE_CONFIG[lane].label,
    accent: LANE_CONFIG[lane].accent,
    count: lanes?.[lane]?.filter((p) => p.covers).length ?? 0,
    nearby: lanes?.[lane]?.filter((p) => !p.covers).length ?? 0,
  }));

  return (
    <aside
      className={`relative flex flex-col bg-[#0a0f18] ${isMobile ? "w-full h-full" : "h-full border-l border-border-dark"}`}
      style={isMobile ? undefined : { width: panelWidth }}
      role="complementary"
      aria-label="Inspector"
    >
      {!isMobile && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}

      <header className="flex items-center justify-between px-4 py-3 border-b border-border-dark">
        <div className="flex items-center gap-2 min-w-0">
          <span className="material-symbols-outlined text-primary text-lg">my_location</span>
          <div className="min-w-0">
            <h1 className="text-sm font-semibold text-white">Inspector</h1>
            <p className="text-[10px] font-mono text-slate-500 truncate">
              {context.lat.toFixed(3)}°, {context.lon.toFixed(3)}°
            </p>
          </div>
        </div>
        <div className="flex items-center">
          {onCollapse && (
            <button
              type="button"
              onClick={onCollapse}
              className="p-1 rounded hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              title="Collapse panel"
              aria-label="Collapse inspector panel"
            >
              <span className="material-symbols-outlined text-lg">chevron_right</span>
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
            title="Close inspector"
            aria-label="Close inspector"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </header>

      {!isCustom && (
        <div className="flex gap-1 px-3 py-1.5 border-b border-border-dark" role="tablist" aria-label="Inspector view">
          {(["compare", "lanes"] as InspectorView[]).map((v) => (
            <button
              key={v}
              type="button"
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={`flex-1 px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-colors ${
                view === v ? "bg-primary/20 text-primary" : "text-slate-500 hover:text-white"
              }`}
            >
              {v === "compare" ? "Compare at point" : "By instrument"}
            </button>
          ))}
        </div>
      )}

      {recentProducts.length > 1 && onSelectRecent && (
        <RecentChips
          products={recentProducts}
          currentId={context.productId}
          onSelect={onSelectRecent}
          onRemove={onRemoveRecent}
        />
      )}

      {isCustom ? (
        <div className="flex-1 overflow-y-auto">
          <ProductDetail product={context} handlers={handlers} />
        </div>
      ) : (
        <>
          {view === "lanes" && <LaneTabs tabs={tabs} active={activeLane} onChange={setActiveLane} />}
          <div className="flex-1 overflow-hidden flex flex-col">
            {loading && (
              <div className="flex-1 flex items-center justify-center text-xs text-slate-500">
                <span className="material-symbols-outlined animate-spin text-base mr-2">progress_activity</span>
                Loading nearby products…
              </div>
            )}
            {error && !loading && (
              <div className="flex-1 flex items-center justify-center text-xs text-red-400 px-4 text-center">
                Failed to load: {error}
              </div>
            )}
            {!loading && !error && lanes && view === "compare" && (
              <ComparePanel
                lanes={lanes}
                context={context}
                handlers={handlers}
                onOpenLane={openLane}
                onOpenRadargram={openRadargramAtPoint}
              />
            )}
            {!loading && !error && lanes && view === "lanes" && (
              <Lane
                key={activeLane}
                lane={activeLane}
                products={lanes[activeLane] ?? []}
                activeProductId={activeProductByLane[activeLane] ?? null}
                onSelectProduct={(id) => setActiveProductByLane((prev) => ({ ...prev, [activeLane]: id }))}
                activeVariant={variantByLane[activeLane]}
                onVariantChange={(v) => {
                  setVariantByLane((prev) => ({ ...prev, [activeLane]: v }));
                  setActiveProductByLane((prev) => ({ ...prev, [activeLane]: undefined }));
                }}
                context={context}
                handlers={handlers}
                onOpenRadargram={openRadargramAtPoint}
              />
            )}
          </div>
          {response && <CrossSection response={response} {...crossActions} />}
        </>
      )}
    </aside>
  );
}

function RecentChips({
  products,
  currentId,
  onSelect,
  onRemove,
}: {
  products: RecentProduct[];
  currentId: string;
  onSelect: (p: RecentProduct) => void;
  onRemove?: (productId: string) => void;
}) {
  return (
    <div className="flex gap-1.5 px-3 py-2 border-b border-border-dark overflow-x-auto scrollbar-dark">
      {products.map((p) => (
        <div
          key={p.productId}
          className={`flex-shrink-0 flex items-center rounded-full text-[10px] font-medium transition-all ${
            p.productId === currentId
              ? "bg-primary/20 border border-primary/50 text-primary"
              : "bg-surface-dark border border-border-dark text-slate-400 hover:text-white hover:border-slate-500"
          }`}
        >
          <button onClick={() => onSelect(p)} className="pl-2.5 pr-1 py-1" title={`${p.instrument} — ${p.productId}`}>
            {p.productId.length > 16 ? p.productId.slice(0, 14) + "…" : p.productId}
          </button>
          {onRemove && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove(p.productId);
              }}
              className="pr-1.5 pl-0.5 py-1 opacity-50 hover:opacity-100 hover:text-red-400 transition-all"
              title="Remove from history"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 12 }}>close</span>
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
