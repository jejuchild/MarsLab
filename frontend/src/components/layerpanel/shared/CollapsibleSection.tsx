import { useState, type ReactNode } from "react";
import { lp } from "../tokens";

interface CollapsibleSectionProps {
  title: string;
  icon?: string;
  /** Extra node rendered right of the title (count badge, status, etc.) */
  trailing?: ReactNode;
  defaultOpen?: boolean;
  /** Extra wrapper className (applied to the outer div) */
  className?: string;
  children: ReactNode;
}

export default function CollapsibleSection({
  title,
  icon,
  trailing,
  defaultOpen = true,
  className = "",
  children,
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`${lp.section} ${className}`}>
      <div
        className="flex items-center justify-between mb-2 cursor-pointer select-none"
        onClick={() => setIsOpen(!isOpen)}
        role="button"
        tabIndex={0}
        aria-expanded={isOpen}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setIsOpen(!isOpen);
          }
        }}
      >
        <h3 className={`${lp.h3} flex items-center gap-1`}>
          <span className="material-symbols-outlined text-xs">
            {isOpen ? "expand_less" : "expand_more"}
          </span>
          {icon && <span className="material-symbols-outlined text-xs">{icon}</span>}
          {title}
        </h3>
        {trailing}
      </div>
      {isOpen && children}
    </div>
  );
}
