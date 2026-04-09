import { useMemo } from "react";
import LaneShell from "./LaneShell";
import QuickviewImage from "../widgets/QuickviewImage";
import type { LaneProduct, LaneVariant } from "../../../api/inspector";

export interface CrismLaneProps {
  products: LaneProduct[];
  activeProductId: string | null;
  onSelectProduct: (productId: string) => void;
  activeVariant: LaneVariant;
  onVariantChange: (v: LaneVariant) => void;
  /** Hand off to the legacy CRISM inspector for spectrum / band math. */
  onOpenLegacy?: (productId: string) => void;
}

/**
 * CRISM Lane — variant: Standard | TRR3.
 *
 * MVP shows a quickview thumbnail. Spectrum / Bands / Mineral CNN
 * are accessed via legacy inspector hand-off in this iteration.
 */
export default function CrismLane({
  products,
  activeProductId,
  onSelectProduct,
  activeVariant,
  onVariantChange,
  onOpenLegacy,
}: CrismLaneProps) {
  const visibleProducts = useMemo(
    () => products.filter((p) => p.variant === activeVariant),
    [products, activeVariant]
  );

  const variants = useMemo(() => {
    const hasStandard = products.some((p) => p.variant === "standard");
    const hasTrr3 = products.some((p) => p.variant === "trr3");
    return [
      { value: "standard" as const, label: "Standard", disabled: !hasStandard },
      { value: "trr3" as const, label: "TRR3", disabled: !hasTrr3 },
    ];
  }, [products]);

  const activeProduct = visibleProducts.find((p) => p.product_id === activeProductId)
    ?? visibleProducts[0];

  // CRISM uses obs_id (no _07 suffix) for TRR3
  const variantInstrument = activeVariant === "trr3" ? "CRISM_TRR3" : "CRISM";

  return (
    <LaneShell
      title="CRISM"
      accent="#fbbf24"
      products={visibleProducts}
      activeProductId={activeProduct?.product_id ?? null}
      onSelectProduct={onSelectProduct}
      variants={variants}
      activeVariant={activeVariant}
      onVariantChange={onVariantChange}
    >
      {activeProduct && (
        <div className="p-4 space-y-3">
          <div className="rounded-lg overflow-hidden border border-border-dark bg-black">
            <QuickviewImage productId={activeProduct.product_id} instrument={variantInstrument} />
          </div>
          <div className="text-[11px] text-slate-400">
            <p>{activeVariant === "trr3" ? "TRR3 mineral classification" : "Standard CRISM cube"}</p>
          </div>
          {onOpenLegacy && (
            <button
              type="button"
              onClick={() => onOpenLegacy(activeProduct.product_id)}
              className="w-full px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-amber-500/15 border border-amber-500/40 text-amber-300 hover:bg-amber-500/25 transition-colors"
            >
              Open spectral inspector →
            </button>
          )}
        </div>
      )}
    </LaneShell>
  );
}
