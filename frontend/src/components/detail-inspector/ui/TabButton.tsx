type Props = {
  productId: string;
  instrument?: string;
  active: boolean;
  onSelect: () => void;
  onRemove: () => void;
};

export default function TabButton({
  productId,
  instrument,
  active,
  onSelect,
  onRemove,
}: Props) {
  return (
    <div
      className={`flex items-center overflow-hidden rounded-lg border transition-colors ${
        active
          ? "border-primary/30 bg-primary/10"
          : "border-border-dark bg-surface-dark/50 hover:bg-surface-dark"
      }`}
    >
      <button
        onClick={onSelect}
        className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium transition-colors ${
          active ? "text-white" : "text-slate-400 hover:text-white"
        }`}
      >
        {instrument && (
          <span
            className={`text-[10px] uppercase ${
              instrument === "CRISM" ? "text-purple-400" : "text-blue-400"
            }`}
          >
            {instrument}
          </span>
        )}
        <span className="font-mono">{productId}</span>
      </button>

      <button
        onClick={onRemove}
        className="flex h-full items-center border-l border-border-dark/50 px-2 text-slate-500 transition-colors hover:bg-red-500/20 hover:text-red-400"
      >
        <span className="material-symbols-outlined text-sm">close</span>
      </button>
    </div>
  );
}
