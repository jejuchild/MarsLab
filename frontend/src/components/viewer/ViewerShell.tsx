import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { useEscapeLayer } from "./escapeStack";

export interface ViewerShellProps {
  title: string;
  /** Mono line under the title: product id, coordinates… */
  subtitle?: ReactNode;
  icon?: string;
  /** Small status chip next to the title (e.g. "JP2 High-Res"). */
  badge?: ReactNode;
  /** Header buttons left of Close (download, open elsewhere…). */
  actions?: ReactNode;
  footer?: ReactNode;
  onClose: () => void;
  /** fullscreen: edge to edge; modal: centred card, backdrop click closes. */
  variant?: "fullscreen" | "modal";
  /** Tailwind max-width for the modal card. */
  maxWidthClass?: string;
  children: ReactNode;
}

/**
 * Common frame for every image/data viewer: portaled above the app (map
 * controls can never cover it), one header layout, one labelled close
 * button, and Escape closing only this viewer.
 */
export default function ViewerShell({
  title,
  subtitle,
  icon,
  badge,
  actions,
  footer,
  onClose,
  variant = "modal",
  maxWidthClass = "max-w-4xl",
  children,
}: ViewerShellProps) {
  useEscapeLayer(onClose);
  const full = variant === "fullscreen";

  const card = (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onMouseDown={(e) => e.stopPropagation()}
      className={
        full
          ? "flex h-full w-full flex-col bg-[#05080d]"
          : `flex max-h-[90vh] w-full ${maxWidthClass} flex-col overflow-hidden rounded-xl border border-border-dark bg-surface-dark shadow-2xl`
      }
    >
      <header className="flex shrink-0 items-center gap-3 border-b border-border-dark bg-black/40 px-4 py-2.5">
        {icon && <span className="material-symbols-outlined text-[20px] text-primary">{icon}</span>}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="truncate text-sm font-bold text-white">{title}</h2>
            {badge}
          </div>
          {subtitle && <div className="truncate font-mono text-[11px] text-slate-400">{subtitle}</div>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close viewer"
          title="Close (Esc)"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-300 transition-colors hover:bg-white/10 hover:text-white"
        >
          <span className="material-symbols-outlined text-[20px]">close</span>
        </button>
      </header>
      <div className="relative min-h-0 flex-1">{children}</div>
      {footer && <footer className="shrink-0 border-t border-border-dark bg-black/30 px-4 py-2">{footer}</footer>}
    </div>
  );

  return createPortal(
    <div
      className={`fixed inset-0 z-[1000] flex items-center justify-center ${full ? "" : "bg-black/70 p-4 backdrop-blur-sm"}`}
      onMouseDown={full ? undefined : onClose}
    >
      {card}
    </div>,
    document.body,
  );
}

/** Header action button with a consistent 32px hit target. */
export function ViewerAction({
  icon,
  label,
  onClick,
  href,
  download,
}: {
  icon: string;
  label: string;
  onClick?: () => void;
  href?: string;
  download?: string;
}) {
  const cls =
    "flex h-8 items-center gap-1.5 rounded-lg border border-border-dark px-2.5 text-[11px] font-semibold text-slate-200 transition-colors hover:bg-white/10";
  const inner = (
    <>
      <span className="material-symbols-outlined text-[16px]">{icon}</span>
      <span className="hidden sm:inline">{label}</span>
    </>
  );
  return href ? (
    <a href={href} download={download} className={cls} title={label} aria-label={label}>
      {inner}
    </a>
  ) : (
    <button type="button" onClick={onClick} className={cls} title={label} aria-label={label}>
      {inner}
    </button>
  );
}
