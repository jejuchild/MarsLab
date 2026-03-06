import { useEffect } from "react";
import { createPortal } from "react-dom";

/* =========================================================
 * Shortcut data
 * =======================================================*/
interface ShortcutEntry {
  keys: string[];
  description: string;
  category: "navigation" | "instruments" | "general" | "map";
}

const SHORTCUTS: ShortcutEntry[] = [
  // General
  { keys: ["?"], description: "Show / hide keyboard shortcuts", category: "general" },
  { keys: ["Esc"], description: "Close current panel or modal", category: "general" },
  { keys: ["Ctrl", "K"], description: "Open command palette", category: "general" },
  { keys: ["G"], description: "Toggle coordinate grid", category: "general" },
  { keys: ["L"], description: "Toggle layer panel", category: "general" },
  { keys: ["M"], description: "Toggle measurement mode", category: "general" },

  // Navigation
  { keys: ["N"], description: "Select next product", category: "navigation" },
  { keys: ["P"], description: "Select previous product", category: "navigation" },
  { keys: ["+", "="], description: "Zoom in", category: "navigation" },
  { keys: ["-"], description: "Zoom out", category: "navigation" },
  { keys: ["Arrow Keys"], description: "Pan map", category: "navigation" },
  { keys: ["F"], description: "Fly to current selection", category: "navigation" },
  { keys: ["B"], description: "Add bookmark at current position", category: "navigation" },

  // Map Controls
  { keys: ["+", "="], description: "Zoom in", category: "map" },
  { keys: ["-"], description: "Zoom out", category: "map" },
  { keys: ["↑", "↓", "←", "→"], description: "Pan map", category: "map" },
  { keys: ["L"], description: "Toggle layer panel", category: "map" },
  { keys: ["M"], description: "Toggle measurement mode", category: "map" },
  { keys: ["F"], description: "Fly to current selection", category: "map" },
  { keys: ["B"], description: "Add bookmark at current position", category: "map" },

  // Instruments (1-7)
  { keys: ["1"], description: "Toggle CRISM footprints", category: "instruments" },
  { keys: ["2"], description: "Toggle HiRISE footprints", category: "instruments" },
  { keys: ["3"], description: "Toggle SHARAD footprints", category: "instruments" },
  { keys: ["4"], description: "Toggle SHARAD Hi-Res footprints", category: "instruments" },
  { keys: ["5"], description: "Toggle CTX footprints", category: "instruments" },
  { keys: ["6"], description: "Toggle HiRISE DTM footprints", category: "instruments" },
  { keys: ["7"], description: "Toggle CRISM TRR3 footprints", category: "instruments" },
];

const CATEGORY_LABELS: Record<string, { label: string; icon: string }> = {
  general: { label: "General", icon: "settings" },
  navigation: { label: "Navigation", icon: "explore" },
  instruments: { label: "Instruments", icon: "satellite_alt" },
  map: { label: "Map Controls", icon: "map" },
};

const CATEGORY_ORDER = ["general", "navigation", "map", "instruments"];

/* =========================================================
 * Props
 * =======================================================*/
interface KeyboardShortcutsProps {
  isOpen: boolean;
  onClose: () => void;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function KeyboardShortcuts({ isOpen, onClose }: KeyboardShortcutsProps) {
  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown, { capture: true });
    return () => document.removeEventListener("keydown", handleKeyDown, { capture: true });
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Group shortcuts by category
  const grouped = CATEGORY_ORDER.map((cat) => ({
    category: cat,
    items: SHORTCUTS.filter((s) => s.category === cat),
  })).filter((g) => g.items.length > 0);

  const modal = (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center"
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm animate-fadeIn"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal container */}
      <div className="relative w-full max-w-lg mx-4 animate-slideDown">
        <div className="rounded-xl border border-[#232f48] bg-[#0d1520] shadow-2xl shadow-black/50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#232f48]">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-xl text-primary">keyboard</span>
              <h2 className="text-white text-sm font-bold">Keyboard Shortcuts</h2>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-[#232f48] transition-colors text-[#92a4c9] hover:text-white"
            >
              <span className="material-symbols-outlined text-xl">close</span>
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-4 max-h-[60vh] overflow-y-auto scrollbar-dark space-y-5">
            {grouped.map(({ category, items }) => {
              const meta = CATEGORY_LABELS[category]!;
              return (
                <div key={category}>
                  {/* Category header */}
                  <div className="flex items-center gap-2 mb-3">
                    <span className="material-symbols-outlined text-sm text-[#6b7c9c]">
                      {meta.icon}
                    </span>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-[#6b7c9c]">
                      {meta.label}
                    </span>
                  </div>

                  {/* Shortcut cards */}
                  <div className="grid grid-cols-1 gap-1.5">
                    {items.map((shortcut, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between px-3 py-2 rounded-lg bg-[#101622] border border-[#232f48]/60"
                      >
                        <span className="text-xs text-[#92a4c9]">
                          {shortcut.description}
                        </span>
                        <div className="flex items-center gap-1 shrink-0 ml-4">
                          {shortcut.keys.map((key, ki) => (
                            <span key={ki} className="flex items-center">
                              {ki > 0 && (
                                <span className="text-[#4a5a7c] text-[10px] mx-0.5">+</span>
                              )}
                              <kbd className="inline-flex items-center justify-center min-w-[24px] h-6 px-1.5 rounded border border-[#3a4a68] bg-[#1a2333] text-[11px] font-mono text-white font-medium">
                                {key}
                              </kbd>
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Footer */}
          <div className="px-6 py-3 border-t border-[#232f48] flex items-center justify-between">
            <span className="text-[10px] text-[#4a5a7c]">
              Press <kbd className="px-1 py-0.5 rounded border border-[#3a4a68] bg-[#1a2333] text-[9px] font-mono text-[#92a4c9]">?</kbd> to toggle this help
            </span>
            <span className="text-[10px] text-[#4a5a7c]">
              {SHORTCUTS.length} shortcuts
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(modal, document.body);
}
