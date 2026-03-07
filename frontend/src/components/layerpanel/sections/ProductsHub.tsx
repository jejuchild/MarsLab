import { useState, useMemo } from "react";
import type { ProductsHubProps } from "../types";
import type { VisibleProduct, OverlapFilter } from "../types";
import type { OverlapStats } from "../types";
import type { ProductOverlay } from "../../../pages/MainPage";
import CollapsibleSection from "../shared/CollapsibleSection";
import { OVERLAY_LABELS, INSTRUMENT_COLORS, type InstrumentType } from "../tokens";

// ── Overlap Filter (always visible, above tabs) ────────────────────────────

function OverlapFilterBar({
  filter,
  onChange,
  stats,
}: {
  filter?: OverlapFilter;
  onChange?: (filter: OverlapFilter) => void;
  stats?: OverlapStats | null;
}) {
  const currentFilter = filter ?? { enabled: false, instruments: [] };

  const handleToggle = () => {
    onChange?.({ ...currentFilter, enabled: !currentFilter.enabled });
  };

  return (
    <div className="mb-2 p-2 rounded bg-[#0d1219] border border-[#232f48]">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-bold uppercase text-[#6b7c9c] flex items-center gap-1">
          <span className="material-symbols-outlined text-[10px]">filter_alt</span>
          Overlap Filter
        </span>
        <div className="flex items-center gap-2">
          {currentFilter.enabled && stats && stats.totalChecked > 0 && (
            <span className="text-sky-400 text-[10px] font-mono">
              {stats.totalPassing}/{stats.totalChecked}
            </span>
          )}
          <button
            onClick={handleToggle}
            className={`text-[8px] px-1.5 py-0.5 rounded font-bold uppercase transition-colors ${
              currentFilter.enabled
                ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48] hover:border-[#3a4a68]"
            }`}
          >
            {currentFilter.enabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      {currentFilter.enabled && (
        <div className="mt-1.5 space-y-1">
          <p className="text-[8px] text-[#6b7c9c] leading-relaxed">
            Showing only products that overlap with at least one product from another instrument.
          </p>

          {/* Per-instrument stats */}
          {stats && stats.totalChecked > 0 && (
            <div className="flex flex-wrap gap-1">
              {Array.from(stats.perInstrument.entries()).map(([inst, s]) =>
                s.checked > 0 ? (
                  <span
                    key={inst}
                    className="text-[9px] text-[#6b7c9c] font-mono bg-[#1a2333] px-1.5 py-0.5 rounded border border-[#232f48]"
                  >
                    {inst}: {s.passing}/{s.checked}
                  </span>
                ) : null,
              )}
            </div>
          )}

          {/* No overlap */}
          {stats && stats.totalPassing === 0 && stats.totalChecked > 0 && (
            <p className="text-[9px] text-orange-400/80 flex items-center gap-1">
              <span className="material-symbols-outlined text-xs">warning</span>
              No overlapping regions found
            </p>
          )}

          {/* No footprints loaded */}
          {(!stats || stats.totalChecked === 0) && (
            <p className="text-[9px] text-[#6b7c9c] flex items-center gap-1">
              <span className="material-symbols-outlined text-xs">info</span>
              Load footprints for at least 2 instruments
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── "On Map" tab ────────────────────────────────────────────────────────────

function OnMapTab({
  visibleProducts,
  activeOverlays,
  onSetOverlay,
  onSelectProduct,
  onFlyToProduct,
  customDatasets,
  onCustomDatasetToggle,
}: Pick<
  ProductsHubProps,
  | "visibleProducts"
  | "activeOverlays"
  | "onSetOverlay"
  | "onSelectProduct"
  | "onFlyToProduct"
  | "customDatasets"
  | "onCustomDatasetToggle"
>) {
  // Group products by instrument
  const groupedProducts = useMemo(() => {
    const groups: Record<InstrumentType, VisibleProduct[]> = {
      CRISM: [],
      HIRISE: [],
      SHARAD: [],
      SHARAD_HIGHRES: [],
      CTX: [],
      CUSTOM: [],
      HIRISE_DTM: [],
      CRISM_TRR3: [],
    };
    for (const product of visibleProducts) {
      if (product.instrument in groups) {
        groups[product.instrument as InstrumentType].push(product);
      }
    }
    return groups;
  }, [visibleProducts]);

  if (visibleProducts.length === 0) {
    return (
      <div className="text-center py-4">
        <span className="material-symbols-outlined text-2xl text-[#3a4a68] mb-1">
          layers_clear
        </span>
        <p className="text-[10px] text-[#6b7c9c]">No products in view</p>
        <p className="text-[9px] text-[#4a5a7c] mt-1">Enable footprints and zoom in</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 max-h-64 overflow-y-auto scrollbar-dark">
      {Object.entries(groupedProducts).map(([instrument, products]) => {
        if (products.length === 0) return null;

        const instColors = INSTRUMENT_COLORS[instrument as InstrumentType];

        return (
          <div key={instrument}>
            {/* Instrument group header */}
            <div
              className={`text-[9px] font-bold uppercase tracking-wider mb-1.5 ${instColors.text}`}
            >
              {instrument} ({products.length})
            </div>

            {/* Products list */}
            <div className="space-y-1">
              {products.map((product) => {
                const overlay = activeOverlays.get(product.productId);
                const hasOverlay = !!overlay;
                const isCustom = product.instrument === "CUSTOM";
                const customDs = isCustom
                  ? customDatasets?.find((d) => d.id === product.productId)
                  : null;
                const customVisible = customDs?.visible ?? true;

                return (
                  <div
                    key={product.productId}
                    className="flex items-center gap-2 p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50 hover:bg-[#1a2333] transition-colors group"
                  >
                    {/* Instrument badge */}
                    <span
                      className={`text-[8px] px-1 py-0.5 rounded font-bold ${instColors.bg} ${instColors.text} ${instColors.border} border`}
                    >
                      {instrument.slice(0, 3)}
                    </span>

                    {/* Product ID - clickable to open Inspector */}
                    <span
                      className="flex-1 text-[10px] font-mono text-[#92a4c9] truncate cursor-pointer hover:text-white"
                      onClick={() => onSelectProduct?.(product)}
                      title={`Click to inspect ${product.title || product.productId}`}
                    >
                      {product.title || product.productId}
                    </span>

                    {/* Fly-to button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onFlyToProduct?.(product.productId);
                      }}
                      className="p-0.5 rounded transition-colors text-slate-600 hover:text-sky-400 hover:bg-sky-500/20"
                      title={`Fly to ${product.productId}`}
                    >
                      <span className="material-symbols-outlined text-xs">my_location</span>
                    </button>

                    {/* Visibility toggle for CUSTOM datasets */}
                    {isCustom && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onCustomDatasetToggle?.(product.productId, !customVisible);
                        }}
                        className={`p-0.5 rounded transition-colors ${
                          customVisible
                            ? "text-fuchsia-400 bg-fuchsia-500/20"
                            : "text-slate-600 hover:text-slate-400"
                        }`}
                        title={customVisible ? "Hide overlay" : "Show overlay"}
                      >
                        <span className="material-symbols-outlined text-xs">
                          {customVisible ? "visibility" : "visibility_off"}
                        </span>
                      </button>
                    )}

                    {/* Simple quickview toggle (not for CUSTOM) */}
                    {!isCustom && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (hasOverlay && overlay.type === "quickview") {
                            onSetOverlay?.(product.productId, null);
                          } else {
                            onSetOverlay?.(product.productId, "quickview");
                          }
                        }}
                        className={`p-0.5 rounded transition-colors ${
                          hasOverlay && overlay.type === "quickview"
                            ? "text-emerald-400 bg-emerald-500/20"
                            : "text-slate-600 hover:text-slate-400"
                        }`}
                        title={
                          hasOverlay && overlay.type === "quickview"
                            ? "Turn off Quickview"
                            : "Show Quickview"
                        }
                      >
                        <span className="material-symbols-outlined text-xs">visibility</span>
                      </button>
                    )}

                    {/* Active overlay indicator */}
                    {hasOverlay && overlay.type !== "quickview" && (
                      <span
                        className={`text-[8px] px-1 py-0.5 rounded font-bold bg-${OVERLAY_LABELS[overlay.type].color}-500/20 text-${OVERLAY_LABELS[overlay.type].color}-400 border border-${OVERLAY_LABELS[overlay.type].color}-500/30`}
                      >
                        {OVERLAY_LABELS[overlay.type].short}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── "Active Overlays" tab ───────────────────────────────────────────────────

function ActiveOverlaysTab({
  activeOverlays,
  visibleProducts,
  onSetOverlay,
  onSetOpacity,
  onFlyToProduct,
  onDeactivateAll,
}: Pick<
  ProductsHubProps,
  | "activeOverlays"
  | "visibleProducts"
  | "onSetOverlay"
  | "onSetOpacity"
  | "onFlyToProduct"
  | "onDeactivateAll"
>) {
  const [expandedOpacity, setExpandedOpacity] = useState<string | null>(null);

  // Build active products list with overlay info
  const activeProducts = useMemo(() => {
    const result: Array<{
      productId: string;
      instrument: InstrumentType;
      overlay: ProductOverlay;
    }> = [];

    const productMap = new Map(visibleProducts.map((p) => [p.productId, p]));

    for (const [productId, overlay] of activeOverlays) {
      const product = productMap.get(productId);
      const instrument: InstrumentType =
        (product?.instrument as InstrumentType) ??
        (productId.startsWith("ESP_") || productId.startsWith("PSP_") ? "HIRISE" : "CRISM");

      result.push({ productId, instrument, overlay });
    }

    return result;
  }, [activeOverlays, visibleProducts]);

  // Group by instrument
  const groupedActive = useMemo(() => {
    const groups: Record<InstrumentType, typeof activeProducts> = {
      CRISM: [],
      HIRISE: [],
      SHARAD: [],
      SHARAD_HIGHRES: [],
      CTX: [],
      CUSTOM: [],
      HIRISE_DTM: [],
      CRISM_TRR3: [],
    };
    for (const product of activeProducts) {
      if (product.instrument in groups) {
        groups[product.instrument].push(product);
      }
    }
    return groups;
  }, [activeProducts]);

  const activeCount = activeProducts.length;

  return (
    <div>
      {/* Clear All header */}
      {activeCount > 0 && (
        <div className="flex justify-end mb-1.5">
          <button
            onClick={() => onDeactivateAll?.()}
            className="text-[8px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors border border-red-500/30 font-bold uppercase"
          >
            Clear All
          </button>
        </div>
      )}

      <div className="space-y-3 max-h-64 overflow-y-auto scrollbar-dark">
        {activeCount === 0 ? (
          <div className="text-center py-4">
            <span className="material-symbols-outlined text-2xl text-[#3a4a68] mb-1">
              check_circle
            </span>
            <p className="text-[10px] text-[#6b7c9c]">No active overlays</p>
            <p className="text-[9px] text-[#4a5a7c] mt-1">Click a product to activate</p>
          </div>
        ) : (
          Object.entries(groupedActive).map(([instrument, products]) => {
            if (products.length === 0) return null;

            const instColors = INSTRUMENT_COLORS[instrument as InstrumentType];

            return (
              <div key={instrument}>
                {/* Instrument group header */}
                <div
                  className={`text-[9px] font-bold uppercase tracking-wider mb-1.5 ${instColors.text}`}
                >
                  {instrument} ({products.length})
                </div>

                {/* Products list */}
                <div className="space-y-1">
                  {products.map(({ productId, overlay }) => {
                    const labelInfo = OVERLAY_LABELS[overlay.type];
                    const isOpacityExpanded = expandedOpacity === productId;

                    return (
                      <div key={productId} className="space-y-1">
                        <div className="flex items-center gap-2 p-1.5 rounded bg-[#1a2333]/40 border border-[#232f48]/50">
                          {/* Instrument badge */}
                          <span
                            className={`text-[8px] px-1 py-0.5 rounded font-bold ${instColors.bg} ${instColors.text} ${instColors.border} border`}
                          >
                            {instrument.slice(0, 3)}
                          </span>

                          {/* Product ID - clickable to fly to */}
                          <span
                            className="flex-1 text-[10px] font-mono text-[#92a4c9] truncate cursor-pointer hover:text-white"
                            onClick={() => onFlyToProduct?.(productId)}
                            title={`Click to fly to ${productId}`}
                          >
                            {productId}
                          </span>

                          {/* Overlay type badge with close button */}
                          <button
                            onClick={() => onSetOverlay?.(productId, null)}
                            className={`text-[8px] px-1 py-0.5 rounded font-bold flex items-center gap-0.5 transition-colors
                              ${labelInfo.color === "emerald" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30" : ""}
                              ${labelInfo.color === "purple" ? "bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30" : ""}
                              ${labelInfo.color === "fuchsia" ? "bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/30 hover:bg-fuchsia-500/30" : ""}
                              ${labelInfo.color === "blue" ? "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30" : ""}
                              ${labelInfo.color === "cyan" ? "bg-cyan-400/20 text-cyan-400 border-cyan-400/30 hover:bg-cyan-400/30" : ""}
                              ${labelInfo.color === "sky" ? "bg-sky-500/20 text-sky-400 border-sky-500/30 hover:bg-sky-500/30" : ""}
                              ${labelInfo.color === "rose" ? "bg-rose-500/20 text-rose-400 border-rose-500/30 hover:bg-rose-500/30" : ""}
                              ${labelInfo.color === "amber" ? "bg-amber-500/20 text-amber-400 border-amber-500/30 hover:bg-amber-500/30" : ""}
                              border`}
                            title={`Turn off ${labelInfo.full}`}
                          >
                            {labelInfo.short}
                            <span className="material-symbols-outlined text-[10px]">close</span>
                          </button>

                          {/* Opacity toggle */}
                          <button
                            onClick={() =>
                              setExpandedOpacity(isOpacityExpanded ? null : productId)
                            }
                            className={`p-0.5 rounded transition-colors ${
                              isOpacityExpanded
                                ? "text-primary bg-primary/20"
                                : "text-slate-600 hover:text-slate-400"
                            }`}
                            title="Adjust opacity"
                          >
                            <span className="material-symbols-outlined text-xs">opacity</span>
                          </button>
                        </div>

                        {/* Opacity slider (expanded) */}
                        {isOpacityExpanded && (
                          <div className="flex items-center gap-2 px-2 py-1.5 bg-[#0a0f18] rounded border border-[#232f48]">
                            <span className="text-[9px] text-[#6b7c9c] uppercase">Opacity</span>
                            <input
                              type="range"
                              min="0"
                              max="100"
                              value={overlay.opacity}
                              onChange={(e) =>
                                onSetOpacity?.(productId, Number(e.target.value))
                              }
                              className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                                [&::-webkit-slider-thumb]:appearance-none
                                [&::-webkit-slider-thumb]:h-3
                                [&::-webkit-slider-thumb]:w-3
                                [&::-webkit-slider-thumb]:rounded-full
                                [&::-webkit-slider-thumb]:bg-primary
                                [&::-webkit-slider-thumb]:cursor-pointer"
                            />
                            <span className="text-[10px] text-white font-mono w-8 text-right">
                              {overlay.opacity}%
                            </span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ── Main ProductsHub ────────────────────────────────────────────────────────

export default function ProductsHub({
  visibleProducts,
  activeOverlays,
  onSetOverlay,
  onSetOpacity,
  onSelectProduct,
  onFlyToProduct,
  onDeactivateAll,
  customDatasets,
  onCustomDatasetToggle,
  overlapFilter,
  onOverlapFilterChange,
  overlapStats,
}: ProductsHubProps) {
  const [activeTab, setActiveTab] = useState<"onmap" | "active">("onmap");
  const activeCount = activeOverlays.size;
  const totalCount = visibleProducts.length;

  return (
    <CollapsibleSection
      title="Products"
      icon="layers"
      defaultOpen={false}
      storageKey="products"
      trailing={
        <span className="text-primary text-[10px] font-mono">{totalCount}</span>
      }
    >
      {/* Overlap Filter — always visible above tabs */}
      <OverlapFilterBar
        filter={overlapFilter}
        onChange={onOverlapFilterChange}
        stats={overlapStats}
      />

      {/* Tab bar */}
      <div className="flex border-b border-[#232f48] mb-2">
        <button
          onClick={() => setActiveTab("onmap")}
          className={`flex-1 text-[10px] font-medium py-1.5 text-center transition-colors ${
            activeTab === "onmap"
              ? "bg-primary/20 text-primary border-b-2 border-primary"
              : "text-[#6b7c9c] hover:text-[#92a4c9]"
          }`}
        >
          On Map
        </button>
        <button
          onClick={() => setActiveTab("active")}
          className={`flex-1 text-[10px] font-medium py-1.5 text-center transition-colors ${
            activeTab === "active"
              ? "bg-primary/20 text-primary border-b-2 border-primary"
              : "text-[#6b7c9c] hover:text-[#92a4c9]"
          }`}
        >
          Active Overlays
          {activeCount > 0 && (
            <span className="ml-1 text-[9px] text-green-400 font-mono">({activeCount})</span>
          )}
        </button>
      </div>

      {/* Tab content */}
      {activeTab === "onmap" ? (
        <OnMapTab
          visibleProducts={visibleProducts}
          activeOverlays={activeOverlays}
          onSetOverlay={onSetOverlay}
          onSelectProduct={onSelectProduct}
          onFlyToProduct={onFlyToProduct}
          customDatasets={customDatasets}
          onCustomDatasetToggle={onCustomDatasetToggle}
        />
      ) : (
        <ActiveOverlaysTab
          activeOverlays={activeOverlays}
          visibleProducts={visibleProducts}
          onSetOverlay={onSetOverlay}
          onSetOpacity={onSetOpacity}
          onFlyToProduct={onFlyToProduct}
          onDeactivateAll={onDeactivateAll}
        />
      )}
    </CollapsibleSection>
  );
}
