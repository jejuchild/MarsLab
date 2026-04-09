import { useMemo } from "react";
import LaneShell from "./LaneShell";
import QuickviewImage from "../widgets/QuickviewImage";
import type { LaneProduct, LaneVariant } from "../../../api/inspector";

export interface SharadLaneProps {
  products: LaneProduct[];
  activeProductId: string | null;
  onSelectProduct: (productId: string) => void;
  activeVariant: LaneVariant;
  onVariantChange: (v: LaneVariant) => void;
  /** When user wants to dive deeper, open the legacy SHARAD radargram inspector. */
  onOpenLegacy?: (productId: string) => void;
}

/**
 * SHARAD Lane — variant: Standard | Hi-res.
 *
 * MVP shows a quickview thumbnail and metadata. Deeper sub-panels
 * (radargram picker, regolith, attenuation) are accessed via legacy
 * inspector hand-off, until they are migrated in a follow-up.
 */
export default function SharadLane({
  products,
  activeProductId,
  onSelectProduct,
  activeVariant,
  onVariantChange,
  onOpenLegacy,
}: SharadLaneProps) {
  // Filter products by variant for the visible list
  const visibleProducts = useMemo(
    () => products.filter((p) => p.variant === activeVariant),
    [products, activeVariant]
  );

  const variants = useMemo(() => {
    const hasStandard = products.some((p) => p.variant === "standard");
    const hasHighres = products.some((p) => p.variant === "highres");
    return [
      { value: "standard" as const, label: "Standard", disabled: !hasStandard },
      { value: "highres" as const, label: "Hi-res", disabled: !hasHighres },
    ];
  }, [products]);

  const activeProduct = visibleProducts.find((p) => p.product_id === activeProductId)
    ?? visibleProducts[0];

  return (
    <LaneShell
      title="SHARAD"
      accent="#ef4444"
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
            <QuickviewImage productId={activeProduct.product_id} instrument="SHARAD" />
          </div>
          <div className="text-[11px] text-slate-400">
            <p>SHARAD radargram trace coverage</p>
          </div>
          {onOpenLegacy && (
            <button
              type="button"
              onClick={() => onOpenLegacy(activeProduct.product_id)}
              className="w-full px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-red-500/15 border border-red-500/40 text-red-300 hover:bg-red-500/25 transition-colors"
            >
              Open subsurface inspector →
            </button>
          )}
        </div>
      )}
    </LaneShell>
  );
}
