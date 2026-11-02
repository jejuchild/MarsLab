import { useMemo } from "react";
import LaneShell from "./LaneShell";
import QuickviewImage from "../widgets/QuickviewImage";
import ProductDetail, { type ProductDetailHandlers } from "../ProductDetail";
import { LANE_CONFIG, instrumentFor } from "../laneConfig";
import type { Lane as LaneId, LaneProduct, LaneVariant } from "../../../api/inspector";
import type { InspectorContext } from "../types";

export interface LaneProps {
  lane: LaneId;
  products: LaneProduct[];
  activeProductId: string | null;
  onSelectProduct: (productId: string) => void;
  activeVariant: LaneVariant;
  onVariantChange: (v: LaneVariant) => void;
  /** The clicked point/product; supplies pixel coordinates for its own product. */
  context: InspectorContext;
  handlers: ProductDetailHandlers;
  /** SHARAD hi-res only: open the full radargram viewer. */
  onOpenRadargram?: (productId: string) => void;
}

/**
 * One Inspector lane (SHARAD / CRISM / HiRISE / CTX), driven by LANE_CONFIG.
 * Shows the variant toggle, product picker, quickview, and the full
 * product detail for the active product.
 */
export default function Lane({
  lane,
  products,
  activeProductId,
  onSelectProduct,
  activeVariant,
  onVariantChange,
  context,
  handlers,
  onOpenRadargram,
}: LaneProps) {
  const cfg = LANE_CONFIG[lane];

  const variants = useMemo(
    () => cfg.variants.map((v) => ({
      value: v.value,
      label: v.label,
      disabled: !products.some((p) => p.variant === v.value),
    })),
    [cfg, products],
  );

  // If the requested variant has no products but another does, show that one
  // instead of an empty lane with no way to switch.
  const effectiveVariant = variants.find((v) => v.value === activeVariant && !v.disabled)
    ? activeVariant
    : variants.find((v) => !v.disabled)?.value ?? activeVariant;

  const visibleProducts = useMemo(
    () => products.filter((p) => p.variant === effectiveVariant),
    [products, effectiveVariant],
  );
  const activeProduct = visibleProducts.find((p) => p.product_id === activeProductId) ?? visibleProducts[0];
  const variantCfg = cfg.variants.find((v) => v.value === effectiveVariant);
  const instrument = instrumentFor(lane, effectiveVariant);

  // Pixel coordinates only belong to the product the user actually clicked.
  const detailProduct: InspectorContext | null = activeProduct
    ? activeProduct.product_id === context.productId
      ? { ...context, instrument }
      : {
          instrument,
          productId: activeProduct.product_id,
          lat: context.lat,
          lon: context.lon,
          title: activeProduct.title ?? undefined,
        }
    : null;

  return (
    <LaneShell
      title={cfg.label}
      accent={cfg.accent}
      products={visibleProducts}
      activeProductId={activeProduct?.product_id ?? null}
      onSelectProduct={onSelectProduct}
      variants={variants}
      activeVariant={effectiveVariant}
      onVariantChange={onVariantChange}
    >
      {activeProduct && detailProduct && (
        <>
          <div className="p-4 space-y-3">
            <div className="rounded-lg overflow-hidden border border-border-dark bg-black">
              <QuickviewImage productId={activeProduct.product_id} instrument={instrument} />
            </div>
            {variantCfg && <p className="text-[11px] text-slate-400">{variantCfg.description}</p>}
            {instrument === "SHARAD_HIGHRES" && onOpenRadargram && (
              <button
                type="button"
                onClick={() => onOpenRadargram(activeProduct.product_id)}
                className="w-full px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-red-500/15 border border-red-500/40 text-red-300 hover:bg-red-500/25 transition-colors"
              >
                Open radargram →
              </button>
            )}
          </div>
          <ProductDetail product={detailProduct} handlers={handlers} />
        </>
      )}
    </LaneShell>
  );
}
