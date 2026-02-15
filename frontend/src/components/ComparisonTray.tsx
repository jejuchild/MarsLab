import { useState, useCallback, useEffect, memo } from "react";

/* ==================================================
 * Types
 * ==================================================*/
export interface ComparisonTrayProduct {
  productId: string;
  instrument: string;
  label: string;
  thumbnailUrl?: string;
}

interface ComparisonTrayProps {
  /** Available products that can be added to comparison */
  products: ComparisonTrayProduct[];
  /** Callback when user initiates comparison with selected product IDs */
  onCompare: (selected: string[]) => void;
  /** Maximum number of products that can be compared */
  maxProducts?: number;
}

/* ==================================================
 * Instrument badge color mapping
 * ==================================================*/
const INSTRUMENT_COLORS: Record<string, string> = {
  CRISM: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  HIRISE: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  SHARAD: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  SHARAD_HIGHRES: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  CTX: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  HIRISE_DTM: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  CRISM_TRR3: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  CUSTOM: "bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/30",
};

function getInstrumentColor(instrument: string): string {
  return INSTRUMENT_COLORS[instrument] || "bg-slate-500/20 text-slate-400 border-slate-500/30";
}

/* ==================================================
 * ComparisonTray Component
 * ==================================================*/
function ComparisonTrayInner({
  products,
  onCompare,
  maxProducts = 4,
}: ComparisonTrayProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Prune selectedIds when products list changes (e.g., overlay removed)
  const productIdSet = new Set(products.map((p) => p.productId));
  useEffect(() => {
    setSelectedIds((prev) => {
      const pruned = new Set<string>();
      for (const id of prev) {
        if (productIdSet.has(id)) pruned.add(id);
      }
      return pruned.size !== prev.size ? pruned : prev;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [products]);

  const toggleProduct = useCallback(
    (productId: string) => {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        if (next.has(productId)) {
          next.delete(productId);
        } else if (next.size < maxProducts) {
          next.add(productId);
        }
        return next;
      });
    },
    [maxProducts]
  );

  const handleCompare = useCallback(() => {
    if (selectedIds.size >= 2) {
      onCompare(Array.from(selectedIds));
    }
  }, [selectedIds, onCompare]);

  const handleSelectAll = useCallback(() => {
    const allIds = products.slice(0, maxProducts).map((p) => p.productId);
    setSelectedIds(new Set(allIds));
  }, [products, maxProducts]);

  const handleClearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  if (products.length < 2) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 border-t border-border-dark bg-[#0d1520]/95 backdrop-blur-md shadow-2xl md:left-[300px]">
      {/* Header row */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#1e293b]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-sky-400 text-lg">compare</span>
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Compare Products
          </span>
          <span className="text-[10px] text-slate-500">
            ({selectedIds.size}/{maxProducts} selected)
          </span>
        </div>
        <div className="flex items-center gap-2">
          {selectedIds.size > 0 && (
            <button
              onClick={handleClearSelection}
              className="px-2 py-1 text-[10px] font-medium text-slate-400 hover:text-slate-200 transition-colors"
            >
              Clear
            </button>
          )}
          {products.length > 2 && selectedIds.size < maxProducts && (
            <button
              onClick={handleSelectAll}
              className="px-2 py-1 text-[10px] font-medium text-slate-400 hover:text-slate-200 transition-colors"
            >
              Select All
            </button>
          )}
          <button
            onClick={handleCompare}
            disabled={selectedIds.size < 2}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all
              ${
                selectedIds.size >= 2
                  ? "bg-sky-500/20 border border-sky-500/50 text-sky-400 hover:bg-sky-500/30 cursor-pointer"
                  : "bg-slate-700/30 border border-slate-600/30 text-slate-500 cursor-not-allowed"
              }
            `}
          >
            <span className="material-symbols-outlined text-sm">grid_view</span>
            Compare {selectedIds.size >= 2 ? `(${selectedIds.size})` : ""}
          </button>
        </div>
      </div>

      {/* Product list */}
      <div className="flex gap-2 px-4 py-3 overflow-x-auto scrollbar-dark">
        {products.map((product) => {
          const isSelected = selectedIds.has(product.productId);
          const isDisabled = !isSelected && selectedIds.size >= maxProducts;

          return (
            <button
              key={product.productId}
              onClick={() => !isDisabled && toggleProduct(product.productId)}
              disabled={isDisabled}
              className={`
                relative flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-lg border transition-all
                ${
                  isSelected
                    ? "bg-sky-500/10 border-sky-500/40 ring-1 ring-sky-500/20"
                    : isDisabled
                    ? "bg-[#0d1520] border-[#1e293b] opacity-40 cursor-not-allowed"
                    : "bg-[#0d1520] border-[#1e293b] hover:border-slate-500/50 hover:bg-[#131c2d] cursor-pointer"
                }
              `}
            >
              {/* Selection indicator */}
              <div
                className={`
                  w-4 h-4 rounded-sm border flex items-center justify-center flex-shrink-0 transition-colors
                  ${
                    isSelected
                      ? "bg-sky-500 border-sky-500"
                      : "border-slate-500/50 bg-transparent"
                  }
                `}
              >
                {isSelected && (
                  <span className="material-symbols-outlined text-white text-xs" style={{ fontSize: 12 }}>
                    check
                  </span>
                )}
              </div>

              {/* Thumbnail placeholder */}
              {product.thumbnailUrl ? (
                <img
                  src={product.thumbnailUrl}
                  alt={product.label}
                  className="w-8 h-8 rounded object-cover border border-[#1e293b]"
                />
              ) : (
                <div className="w-8 h-8 rounded bg-[#1a2333] border border-[#1e293b] flex items-center justify-center">
                  <span className="material-symbols-outlined text-slate-600 text-xs" style={{ fontSize: 14 }}>
                    image
                  </span>
                </div>
              )}

              {/* Product info */}
              <div className="text-left min-w-0">
                <div className="text-[10px] font-mono text-slate-300 truncate max-w-[120px]">
                  {product.label}
                </div>
                <span
                  className={`inline-block px-1.5 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider border ${getInstrumentColor(
                    product.instrument
                  )}`}
                >
                  {product.instrument}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

const ComparisonTray = memo(ComparisonTrayInner);
export default ComparisonTray;
