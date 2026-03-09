import type { InspectorContext, HiRISETabKey, CRISMTabKey, RecentProduct } from "./types";

type InspectorHeaderProps = {
  selected: InspectorContext;
  // Tab state
  hiriseTab: HiRISETabKey;
  onHiriseTabChange: (tab: HiRISETabKey) => void;
  crismTab: CRISMTabKey;
  onCrismTabChange: (tab: CRISMTabKey) => void;
  // Panel controls
  onClose: () => void;
  onCollapse?: () => void;
  // Recent products
  recentProducts: RecentProduct[];
  onSelectRecent?: (product: RecentProduct) => void;
  onRemoveRecent?: (productId: string) => void;
};

const HIRISE_TABS: HiRISETabKey[] = ["Metadata", "Pixel"];
const CRISM_TABS: CRISMTabKey[] = ["Metadata", "Spectrum", "Bands"];

export default function InspectorHeader({
  selected,
  hiriseTab,
  onHiriseTabChange,
  crismTab,
  onCrismTabChange,
  onClose,
  onCollapse,
  recentProducts,
  onSelectRecent,
  onRemoveRecent,
}: InspectorHeaderProps) {
  const isHiRISE = selected.instrument === "HIRISE";
  const isCRISM = selected.instrument === "CRISM" || selected.instrument === "CRISM_TRR3";
  const isCustom = selected.instrument === "CUSTOM";
  const isDTM = selected.instrument === "HIRISE_DTM";

  return (
    <div className="flex-shrink-0">
      {/* Tab Bar */}
      <div className="flex border-b border-border-dark">
        {isHiRISE &&
          HIRISE_TABS.map((t) => (
            <TabButton
              key={t}
              label={t}
              active={hiriseTab === t}
              onClick={() => onHiriseTabChange(t)}
            />
          ))}

        {isCRISM &&
          CRISM_TABS.map((t) => (
            <TabButton
              key={t}
              label={t}
              active={crismTab === t}
              onClick={() => onCrismTabChange(t)}
            />
          ))}

        {isCustom && (
          <TabButton label="Metadata" active onClick={() => {}} />
        )}

        {isDTM && (
          <TabButton label="HiRISE DTM" active onClick={() => {}} accentColor="amber" />
        )}

        {/* Generic instruments (SHARAD, SHARAD_HIGHRES, CTX) */}
        {!isHiRISE && !isCRISM && !isCustom && !isDTM && (
          <TabButton label="Metadata" active onClick={() => {}} />
        )}

        {/* Panel controls */}
        <div className="flex items-center shrink-0 ml-auto">
          {onCollapse && (
            <button
              onClick={onCollapse}
              className="flex items-center justify-center px-1.5 text-slate-500 hover:text-white transition-colors"
              title="Collapse panel"
              aria-label="Collapse inspector panel"
            >
              <span className="material-symbols-outlined text-base">chevron_right</span>
            </button>
          )}
          <button
            onClick={onClose}
            className="flex items-center justify-center px-3 text-slate-500 hover:text-red-400 transition-colors"
            aria-label="Close inspector"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* Recent Products Chip Bar */}
      {recentProducts.length > 1 && onSelectRecent && (
        <div className="flex gap-1.5 px-3 py-2 border-b border-border-dark overflow-x-auto scrollbar-dark">
          {recentProducts.map((p) => (
            <div
              key={p.productId}
              className={`flex-shrink-0 flex items-center rounded-full text-[10px] font-medium transition-all ${
                p.productId === selected.productId
                  ? "bg-primary/20 border border-primary/50 text-primary"
                  : "bg-surface-dark border border-border-dark text-slate-400 hover:text-white hover:border-slate-500"
              }`}
            >
              <button
                onClick={() => onSelectRecent(p)}
                className="pl-2.5 pr-1 py-1"
                title={`${p.instrument} — ${p.productId}`}
              >
                {p.productId.length > 16
                  ? p.productId.slice(0, 14) + "…"
                  : p.productId}
              </button>
              {onRemoveRecent && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveRecent(p.productId);
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
      )}
    </div>
  );
}

/* ── Tab Button ── */
function TabButton({
  label,
  active,
  onClick,
  accentColor,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  accentColor?: "amber" | "primary";
}) {
  const accent = accentColor === "amber" ? "border-amber-600 bg-amber-600/5" : "border-primary bg-primary/5";

  return (
    <button
      onClick={onClick}
      aria-label={`${label} tab`}
      className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors ${
        active
          ? `border-b-2 ${accent} text-white`
          : "border-b-2 border-transparent text-slate-500 hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}
