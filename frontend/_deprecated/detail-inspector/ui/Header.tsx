type Props = {
  expanded: boolean;
  onToggle: () => void;
};

export default function Header({ expanded, onToggle }: Props) {
  return (
    <div className="flex items-center justify-between border-b border-border-dark px-4 py-2">
      <div className="flex items-center gap-3">
        <span className="material-symbols-outlined text-primary">analytics</span>
        <span className="text-sm font-medium">Detail Inspector</span>
      </div>

      <button
        onClick={onToggle}
        className="flex items-center gap-2 rounded-lg bg-surface-dark px-3 py-1.5 text-xs font-medium text-slate-400 transition-colors hover:bg-primary/20 hover:text-white"
      >
        <span className="material-symbols-outlined text-sm">
          {expanded ? "expand_more" : "expand_less"}
        </span>
        {expanded ? "Collapse" : "Expand"}
      </button>
    </div>
  );
}
