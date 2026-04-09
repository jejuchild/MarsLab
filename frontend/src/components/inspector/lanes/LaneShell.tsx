import { type ReactNode } from "react";
import type { LaneProduct, LaneVariant } from "../../../api/inspector";

/**
 * Shared shell for the 4 instrument lanes.
 *
 * Provides:
 * - Product picker (visible only when more than one product is available)
 * - Optional variant toggle
 * - Body content (whatever the lane wants to render for the active product)
 *
 * Each lane composes this with its own variant labels and body rendering.
 */

export interface LaneVariantOption {
  value: LaneVariant;
  label: string;
  /** Disable the option (e.g. no products available for this variant). */
  disabled?: boolean;
}

export interface LaneShellProps {
  /** Title shown in the lane header (e.g. "SHARAD"). */
  title: string;
  /** Lane accent color (hex or tailwind class). */
  accent?: string;
  /** All products available for this lane. */
  products: LaneProduct[];
  /** Currently active product id. */
  activeProductId: string | null;
  /** Notified when the user picks a different product. */
  onSelectProduct: (productId: string) => void;
  /** Variant toggle options (e.g. Standard | Hi-res). Omit for no toggle. */
  variants?: LaneVariantOption[];
  /** Active variant value. */
  activeVariant?: LaneVariant;
  /** Notified when the user changes variant. */
  onVariantChange?: (v: LaneVariant) => void;
  /** Body content. */
  children?: ReactNode;
}

export default function LaneShell({
  title,
  accent = "#4f9cf7",
  products,
  activeProductId,
  onSelectProduct,
  variants,
  activeVariant,
  onVariantChange,
  children,
}: LaneShellProps) {
  if (products.length === 0) {
    return (
      <div className="flex flex-col h-full p-4">
        <header className="flex items-center gap-2 mb-3">
          <span
            className="w-2 h-5 rounded-sm"
            style={{ backgroundColor: accent }}
            aria-hidden
          />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-white">{title}</h2>
        </header>
        <div className="flex-1 flex items-center justify-center text-xs text-slate-500">
          No {title} products at this point
        </div>
      </div>
    );
  }

  const activeProduct = products.find((p) => p.product_id === activeProductId) ?? products[0];

  return (
    <div className="flex flex-col h-full">
      <header className="flex items-center gap-2 px-4 py-3 border-b border-border-dark bg-bg-dark/40">
        <span
          className="w-2 h-5 rounded-sm flex-shrink-0"
          style={{ backgroundColor: accent }}
          aria-hidden
        />
        <h2 className="text-sm font-semibold uppercase tracking-wider text-white">
          {title}
        </h2>
        <span className="text-[10px] font-mono text-slate-500 ml-auto">
          {products.length} {products.length === 1 ? "product" : "products"}
        </span>
      </header>

      {/* Variant toggle */}
      {variants && variants.length > 1 && activeVariant && onVariantChange && (
        <div className="flex items-center gap-1 px-4 py-2 border-b border-border-dark/60">
          {variants.map((v) => (
            <button
              key={v.value}
              type="button"
              disabled={v.disabled}
              onClick={() => onVariantChange(v.value)}
              className={`px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-colors ${
                activeVariant === v.value
                  ? "bg-primary/20 text-primary"
                  : v.disabled
                  ? "text-slate-700 cursor-not-allowed"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
      )}

      {/* Product picker (only when 2+) */}
      {products.length > 1 && (
        <div className="px-4 py-2 border-b border-border-dark/60">
          <label className="text-[9px] uppercase tracking-wider text-slate-500 font-bold mb-1 block">
            Product
          </label>
          <select
            value={activeProduct?.product_id ?? ""}
            onChange={(e) => onSelectProduct(e.target.value)}
            className="w-full text-[11px] font-mono bg-bg-dark border border-border-dark rounded px-2 py-1 text-white focus:outline-none focus:border-primary"
          >
            {products.map((p) => (
              <option key={p.product_id} value={p.product_id}>
                {p.product_id}
                {p.distance_km != null ? ` — ${p.distance_km.toFixed(1)} km` : ""}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Body */}
      <div className="flex-1 overflow-y-auto">{children}</div>

      {/* Footer with active product metadata */}
      {activeProduct && (
        <footer className="px-4 py-2 border-t border-border-dark/60 bg-bg-dark/40 text-[10px] text-slate-500 font-mono">
          {activeProduct.lat != null && activeProduct.lon != null && (
            <span>
              {activeProduct.lat.toFixed(3)}°, {activeProduct.lon.toFixed(3)}°
            </span>
          )}
          {activeProduct.distance_km != null && (
            <span className="ml-2 text-slate-600">
              · {activeProduct.distance_km.toFixed(1)} km from cursor
            </span>
          )}
        </footer>
      )}
    </div>
  );
}
