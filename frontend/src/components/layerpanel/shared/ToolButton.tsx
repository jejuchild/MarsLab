interface ToolButtonProps {
  active: boolean;
  onClick: () => void;
  icon: string;
  title: string;
  description: string;
  /** Tailwind color name, e.g. "primary", "emerald", "fuchsia" */
  color: string;
  /** Badge text, e.g. "BETA", "NEW" */
  badge?: string;
}

export default function ToolButton({
  active,
  onClick,
  icon,
  title,
  description,
  color,
  badge,
}: ToolButtonProps) {
  const activeClass = `bg-${color}-500/20 border border-${color}-500/50 text-${color}-400`;
  const inactiveClass = `bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-${color}-500/30`;

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 w-full p-2 rounded transition-colors text-left ${
        active ? activeClass : inactiveClass
      }`}
      aria-pressed={active}
    >
      <span className="material-symbols-outlined text-sm">{icon}</span>
      <div className="flex-1">
        <span className="text-[11px] font-medium">
          {title}
          {badge && (
            <span
              className={`text-[8px] px-1 py-0.5 rounded bg-${color}-500/20 text-${color}-400 border border-${color}-500/30 font-bold ml-1`}
            >
              {badge}
            </span>
          )}
        </span>
        <p className="text-[9px] text-[#6b7c9c]">{description}</p>
      </div>
      {active && (
        <span
          className={`text-[8px] px-1.5 py-0.5 rounded bg-${color}-500/20 text-${color}-400 border border-${color}-500/30 font-bold uppercase`}
        >
          ON
        </span>
      )}
    </button>
  );
}
