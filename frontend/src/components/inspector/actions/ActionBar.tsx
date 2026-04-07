import { useMemo } from "react";
import type { FieldNote } from "../../../api/fieldnotes";
import type { InstrumentType } from "../types";

type ActionBarProps = {
  productId: string;
  instrument: InstrumentType;
  lat: number;
  lon: number;
  // DTM
  isDTM: boolean;
  onShow3DView?: (productId: string, lat: number, lon: number) => void;
  // Field Notes
  fieldNotes: FieldNote[];
  onOpenFieldNote?: (productId: string, instrument: string, lat: number, lon: number) => void;
};

export default function ActionBar({
  productId,
  instrument,
  lat,
  lon,
  isDTM,
  onShow3DView,
  fieldNotes,
  onOpenFieldNote,
}: ActionBarProps) {
  const hasNote = useMemo(
    () => fieldNotes.some((n) => n.product_id === productId),
    [fieldNotes, productId],
  );
  const noteCount = useMemo(
    () => fieldNotes.filter((n) => n.product_id === productId).length,
    [fieldNotes, productId],
  );

  return (
    <div className="mt-4 space-y-2">
      {/* ── Tier 1: Primary Actions ── */}

      {/* 3D View (DTM only) */}
      {isDTM && onShow3DView && (
        <ActionButton
          onClick={() => onShow3DView(productId, lat, lon)}
          icon="terrain"
          label="Show 3D View"
          colorScheme="amber"
          size="lg"
        />
      )}

      {/* ── Tier 2: Field Note ── */}
      {onOpenFieldNote && (
        <ActionButton
          onClick={() => onOpenFieldNote(productId, instrument, lat, lon)}
          icon={hasNote ? "description" : "note_add"}
          label={hasNote ? `Field Notes (${noteCount})` : "Add Field Note"}
          colorScheme={hasNote ? "amber" : "slate"}
        />
      )}
    </div>
  );
}

/* ── Reusable Action Button ── */
type ColorScheme = "amber" | "violet" | "emerald" | "purple" | "slate" | "primary";

const COLOR_CLASSES: Record<ColorScheme, { bg: string; border: string; text: string; hover: string }> = {
  amber: { bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400", hover: "hover:bg-amber-500/30" },
  violet: { bg: "bg-violet-500/20", border: "border-violet-500/50", text: "text-violet-400", hover: "hover:bg-violet-500/30" },
  emerald: { bg: "bg-emerald-500/20", border: "border-emerald-500/50", text: "text-emerald-400", hover: "hover:bg-emerald-500/30" },
  purple: { bg: "bg-purple-500/20", border: "border-purple-500/50", text: "text-purple-400", hover: "hover:bg-purple-500/30" },
  slate: { bg: "bg-[#1a2333]", border: "border-[#232f48]", text: "text-[#92a4c9]", hover: "hover:text-amber-400 hover:border-amber-500/30" },
  primary: { bg: "bg-primary/20", border: "border-primary/50", text: "text-primary", hover: "hover:bg-primary/30" },
};

function ActionButton({
  onClick,
  icon,
  label,
  colorScheme,
  size = "md",
  active,
}: {
  onClick: () => void;
  icon: string;
  label: string;
  colorScheme: ColorScheme;
  size?: "md" | "lg";
  active?: boolean;
}) {
  const c = COLOR_CLASSES[colorScheme];
  const py = size === "lg" ? "py-3" : "py-2.5";

  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center justify-center gap-2 rounded-lg ${py} text-[10px] font-bold uppercase tracking-widest border transition-all active:scale-[0.98] ${c.bg} ${c.border} ${c.text} ${c.hover} ${active ? "ring-1 ring-current" : ""}`}
    >
      <span className="material-symbols-outlined text-sm">{icon}</span>
      {label}
    </button>
  );
}
