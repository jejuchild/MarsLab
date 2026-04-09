import { useMemo } from "react";
import LaneShell from "./LaneShell";
import QuickviewImage from "../widgets/QuickviewImage";
import type { LaneProduct, LaneVariant } from "../../../api/inspector";

export interface HiriseLaneProps {
  products: LaneProduct[];
  activeProductId: string | null;
  onSelectProduct: (productId: string) => void;
  activeVariant: LaneVariant;
  onVariantChange: (v: LaneVariant) => void;
  /** Hand off to the legacy HiRISE inspector (image / DTM / slope / etc.). */
  onOpenLegacy?: (productId: string) => void;
}

/**
 * HiRISE Lane — variant: Image | DTM.
 */
export default function HiriseLane({
  products,
  activeProductId,
  onSelectProduct,
  activeVariant,
  onVariantChange,
  onOpenLegacy,
}: HiriseLaneProps) {
  const visibleProducts = useMemo(
    () => products.filter((p) => p.variant === activeVariant),
    [products, activeVariant]
  );

  const variants = useMemo(() => {
    const hasImage = products.some((p) => p.variant === "image");
    const hasDtm = products.some((p) => p.variant === "dtm");
    return [
      { value: "image" as const, label: "Image", disabled: !hasImage },
      { value: "dtm" as const, label: "DTM", disabled: !hasDtm },
    ];
  }, [products]);

  const activeProduct = visibleProducts.find((p) => p.product_id === activeProductId)
    ?? visibleProducts[0];

  const variantInstrument = activeVariant === "dtm" ? "HIRISE_DTM" : "HIRISE";

  return (
    <LaneShell
      title="HiRISE"
      accent="#22c55e"
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
            <p>{activeVariant === "dtm" ? "Digital terrain model (elevation)" : "High-resolution image"}</p>
          </div>
          {onOpenLegacy && (
            <button
              type="button"
              onClick={() => onOpenLegacy(activeProduct.product_id)}
              className="w-full px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/25 transition-colors"
            >
              Open high-res inspector →
            </button>
          )}
        </div>
      )}
    </LaneShell>
  );
}
