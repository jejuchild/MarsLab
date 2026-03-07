import { useState, useEffect, type ReactNode } from "react";
import { lp } from "../tokens";

interface CollapsibleSectionProps {
  title: string;
  icon?: string;
  /** Extra node rendered right of the title (count badge, status, etc.) */
  trailing?: ReactNode;
  defaultOpen?: boolean;
  /** localStorage key for persisting collapse state. Omit to skip persistence. */
  storageKey?: string;
  /** Extra wrapper className (applied to the outer div) */
  className?: string;
  children: ReactNode;
}

const LS_PREFIX = "marslab-section-";

function readStorage(key: string | undefined, fallback: boolean): boolean {
  if (!key) return fallback;
  try {
    const stored = localStorage.getItem(LS_PREFIX + key);
    if (stored === "true") return true;
    if (stored === "false") return false;
    return fallback;
  } catch {
    return fallback;
  }
}

export default function CollapsibleSection({
  title,
  icon,
  trailing,
  defaultOpen = true,
  storageKey,
  className = "",
  children,
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(() => readStorage(storageKey, defaultOpen));

  // Persist to localStorage when changed
  useEffect(() => {
    if (!storageKey) return;
    try {
      localStorage.setItem(LS_PREFIX + storageKey, String(isOpen));
    } catch {
      // Ignore — localStorage unavailable (incognito, quota)
    }
  }, [isOpen, storageKey]);

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
