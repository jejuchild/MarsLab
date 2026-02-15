interface EmptyStateProps {
  icon: string;           // Material Symbols icon name
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
}

export default function EmptyState({ icon, title, description, actionLabel, onAction }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 px-8 py-12 text-center max-w-xs mx-auto">
      {/* Icon */}
      <span className="material-symbols-outlined text-[48px] text-[#3a4a68]">
        {icon}
      </span>

      {/* Title */}
      <h3 className="text-white text-sm font-semibold leading-snug">
        {title}
      </h3>

      {/* Description */}
      <p className="text-[#92a4c9] text-xs leading-relaxed">
        {description}
      </p>

      {/* Optional action button */}
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-2 flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-primary/20 border border-primary/50 text-primary hover:bg-primary/30 transition-colors"
        >
          <span className="material-symbols-outlined text-sm">smart_toy</span>
          {actionLabel}
        </button>
      )}
    </div>
  );
}
