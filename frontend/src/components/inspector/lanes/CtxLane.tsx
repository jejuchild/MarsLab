import LaneShell from "./LaneShell";
import QuickviewImage from "../widgets/QuickviewImage";
import type { LaneProduct } from "../../../api/inspector";

export interface CtxLaneProps {
  products: LaneProduct[];
  activeProductId: string | null;
  onSelectProduct: (productId: string) => void;
  /** Hand off to the legacy CTX inspector. */
  onOpenLegacy?: (productId: string) => void;
}

/**
 * CTX Lane — wide-area context imagery.
 *
 * No variant toggle (CTX images and Murray Lab mosaic share the same lane
 * but mosaic shows up as a single overlay).
 */
export default function CtxLane({
  products,
  activeProductId,
  onSelectProduct,
  onOpenLegacy,
}: CtxLaneProps) {
  const activeProduct = products.find((p) => p.product_id === activeProductId) ?? products[0];

  return (
    <LaneShell
      title="CTX"
      accent="#60a5fa"
      products={products}
      activeProductId={activeProduct?.product_id ?? null}
      onSelectProduct={onSelectProduct}
    >
      {activeProduct && (
        <div className="p-4 space-y-3">
          <div className="rounded-lg overflow-hidden border border-border-dark bg-black">
            <QuickviewImage productId={activeProduct.product_id} instrument="CTX" />
          </div>
          <div className="text-[11px] text-slate-400">
            <p>Wide-area context image</p>
          </div>
          {onOpenLegacy && (
            <button
              type="button"
              onClick={() => onOpenLegacy(activeProduct.product_id)}
              className="w-full px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-blue-500/15 border border-blue-500/40 text-blue-300 hover:bg-blue-500/25 transition-colors"
            >
              Open CTX inspector →
            </button>
          )}
        </div>
      )}
    </LaneShell>
  );
}
