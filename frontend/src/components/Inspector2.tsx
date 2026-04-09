import { useCallback, useEffect, useState } from "react";
import LaneTabs, { type LaneTabConfig } from "./inspector/lanes/LaneTabs";
import SharadLane from "./inspector/lanes/SharadLane";
import CrismLane from "./inspector/lanes/CrismLane";
import HiriseLane from "./inspector/lanes/HiriseLane";
import CtxLane from "./inspector/lanes/CtxLane";
import CrossSection from "./inspector/cross/CrossSection";
import { fetchAtPoint, pickDefaultLane, type AtPointResponse, type Lane, type LaneVariant } from "../api/inspector";
import type { InspectorContext } from "./inspector/types";

export interface Inspector2Props {
  /** Currently selected coordinate or product. When null, panel is hidden. */
  context: InspectorContext | null;
  /** Notified when the user dismisses the inspector. */
  onClose: () => void;
  /** Hand-off into the legacy inspector for a specific product. */
  onOpenLegacyProduct?: (productId: string, instrument: string) => void;
}

const LANE_LABELS: Record<Lane, { label: string; accent: string }> = {
  SHARAD: { label: "SHARAD", accent: "#ef4444" },
  CRISM: { label: "CRISM", accent: "#fbbf24" },
  HIRISE: { label: "HiRISE", accent: "#22c55e" },
  CTX: { label: "CTX", accent: "#60a5fa" },
};

/**
 * Phase 3 Inspector — 4-lane Inspector container.
 *
 * On mount/context-change, fetches `/api/inspector/at-point` for the current
 * lat/lon and renders SHARAD/CRISM/HiRISE/CTX as tabs. Each lane component
 * shows the active product's quickview and offers a "open legacy inspector"
 * fallback for sub-panels not yet migrated.
 *
 * The Cross-Analysis section appears at the bottom (collapsed by default,
 * per Q3=B decision).
 */
export default function Inspector2({ context, onClose, onOpenLegacyProduct }: Inspector2Props) {
  const [response, setResponse] = useState<AtPointResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeLane, setActiveLane] = useState<Lane>("HIRISE");
  const [activeProductByLane, setActiveProductByLane] = useState<Partial<Record<Lane, string>>>({});
  const [variantByLane, setVariantByLane] = useState<Record<Lane, LaneVariant>>({
    SHARAD: "standard",
    CRISM: "standard",
    HIRISE: "image",
    CTX: "image",
  });

  // Fetch products when context changes
  useEffect(() => {
    if (!context) {
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
        // Pick default active lane (lane with most products), or follow context
        const ctxLane = context.instrument as Lane | undefined;
        if (ctxLane && res.counts[ctxLane] > 0) {
          setActiveLane(ctxLane);
        } else {
          setActiveLane(pickDefaultLane(res));
        }
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [context]);

  const handleSelectProduct = useCallback((lane: Lane, productId: string) => {
    setActiveProductByLane((prev) => ({ ...prev, [lane]: productId }));
  }, []);

  const handleVariantChange = useCallback((lane: Lane, variant: LaneVariant) => {
    setVariantByLane((prev) => ({ ...prev, [lane]: variant }));
    // Clear active product for this lane since variant filter will change the list
    setActiveProductByLane((prev) => ({ ...prev, [lane]: undefined }));
  }, []);

  const handleOpenLegacy = useCallback(
    (lane: Lane, productId: string) => {
      onOpenLegacyProduct?.(productId, lane);
    },
    [onOpenLegacyProduct]
  );

  if (!context) return null;

  const tabs: LaneTabConfig[] = (["SHARAD", "CRISM", "HIRISE", "CTX"] as Lane[]).map((lane) => ({
    lane,
    label: LANE_LABELS[lane].label,
    accent: LANE_LABELS[lane].accent,
    count: response?.counts[lane] ?? 0,
  }));

  const products = response?.lanes[activeLane] ?? [];
  const activeProductId = activeProductByLane[activeLane] ?? null;
  const activeVariant = variantByLane[activeLane];

  return (
    <aside className="w-[440px] h-full flex flex-col bg-[#0a0f18] border-l border-border-dark">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border-dark">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-lg">my_location</span>
          <div>
            <h1 className="text-sm font-semibold text-white">Inspector</h1>
            <p className="text-[10px] font-mono text-slate-500">
              {context.lat.toFixed(3)}°, {context.lon.toFixed(3)}°
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-1 rounded hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
          title="Close inspector"
          aria-label="Close inspector"
        >
          <span className="material-symbols-outlined text-lg">close</span>
        </button>
      </header>

      {/* Lane tabs */}
      <LaneTabs tabs={tabs} active={activeLane} onChange={setActiveLane} />

      {/* Loading / error / lane body */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {loading && (
          <div className="flex-1 flex items-center justify-center text-xs text-slate-500">
            <span className="material-symbols-outlined animate-spin text-base mr-2">progress_activity</span>
            Loading nearby products…
          </div>
        )}
        {error && (
          <div className="flex-1 flex items-center justify-center text-xs text-red-400 px-4 text-center">
            Failed to load: {error}
          </div>
        )}
        {!loading && !error && response && (
          <>
            {activeLane === "SHARAD" && (
              <SharadLane
                products={products}
                activeProductId={activeProductId}
                onSelectProduct={(id) => handleSelectProduct("SHARAD", id)}
                activeVariant={activeVariant}
                onVariantChange={(v) => handleVariantChange("SHARAD", v)}
                onOpenLegacy={(id) => handleOpenLegacy("SHARAD", id)}
              />
            )}
            {activeLane === "CRISM" && (
              <CrismLane
                products={products}
                activeProductId={activeProductId}
                onSelectProduct={(id) => handleSelectProduct("CRISM", id)}
                activeVariant={activeVariant}
                onVariantChange={(v) => handleVariantChange("CRISM", v)}
                onOpenLegacy={(id) => handleOpenLegacy("CRISM", id)}
              />
            )}
            {activeLane === "HIRISE" && (
              <HiriseLane
                products={products}
                activeProductId={activeProductId}
                onSelectProduct={(id) => handleSelectProduct("HIRISE", id)}
                activeVariant={activeVariant}
                onVariantChange={(v) => handleVariantChange("HIRISE", v)}
                onOpenLegacy={(id) => handleOpenLegacy("HIRISE", id)}
              />
            )}
            {activeLane === "CTX" && (
              <CtxLane
                products={products}
                activeProductId={activeProductId}
                onSelectProduct={(id) => handleSelectProduct("CTX", id)}
                onOpenLegacy={(id) => handleOpenLegacy("CTX", id)}
              />
            )}
          </>
        )}
      </div>

      {/* Cross-Analysis (bottom, collapsed by default) */}
      {response && <CrossSection response={response} />}
    </aside>
  );
}
