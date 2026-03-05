import { useState, useMemo, useCallback } from "react";
import type { FieldNotesSectionProps } from "../types";
import CollapsibleSection from "../shared/CollapsibleSection";
import { INSTRUMENT_COLORS, type InstrumentType } from "../tokens";

export default function FieldNotesSection({
  fieldNotes,
  showFieldNotesOnMap,
  onToggleFieldNotesOnMap,
  onFieldNoteClick,
  onActiveTagChange,
}: FieldNotesSectionProps) {
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  const handleTagSelect = useCallback(
    (tag: string) => {
      const next = selectedTag === tag ? null : tag;
      setSelectedTag(next);
      onActiveTagChange?.(next);
      // Auto-enable "Show on Map" when a tag is selected
      if (next && !showFieldNotesOnMap) {
        onToggleFieldNotesOnMap?.(true);
      }
    },
    [selectedTag, onActiveTagChange, showFieldNotesOnMap, onToggleFieldNotesOnMap],
  );

  // All unique tags sorted
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    fieldNotes.forEach((n) => n.tags.forEach((t) => tagSet.add(t)));
    return Array.from(tagSet).sort();
  }, [fieldNotes]);

  // Group notes by tag, or flat list when no tag selected
  const grouped = useMemo(() => {
    if (selectedTag) {
      // Single-tag filter: one group
      return [
        { tag: selectedTag, notes: fieldNotes.filter((n) => n.tags.includes(selectedTag)) },
      ];
    }
    // Group by every tag; notes without tags go into "(Untagged)"
    const map = new Map<string, typeof fieldNotes>();
    const untagged: typeof fieldNotes = [];
    for (const n of fieldNotes) {
      if (n.tags.length === 0) {
        untagged.push(n);
      } else {
        for (const t of n.tags) {
          if (!map.has(t)) map.set(t, []);
          map.get(t)!.push(n);
        }
      }
    }
    const groups = Array.from(map.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([tag, notes]) => ({ tag, notes }));
    if (untagged.length > 0) groups.push({ tag: "", notes: untagged });
    return groups;
  }, [fieldNotes, selectedTag]);

  if (fieldNotes.length === 0) return null;

  return (
    <CollapsibleSection
      title="Field Notes"
      icon="sticky_note_2"
      defaultOpen={false}
      trailing={
        <span className="text-amber-400 text-[10px] font-mono">{fieldNotes.length}</span>
      }
    >
      <div className="space-y-3">
        {/* Show on Map toggle */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showFieldNotesOnMap}
            onChange={(e) => onToggleFieldNotesOnMap?.(e.target.checked)}
            className="accent-amber-400 w-3 h-3"
          />
          <span className="text-[10px] text-[#92a4c9]">Show on Map</span>
        </label>

        {/* Tag filter pills */}
        {allTags.length > 0 && (
          <div className="flex flex-wrap gap-1 max-h-16 overflow-y-auto scrollbar-dark">
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => handleTagSelect(tag)}
                aria-label={`Filter by tag: ${tag}`}
                className={`px-2 py-0.5 rounded-full text-[9px] border transition-colors ${
                  selectedTag === tag
                    ? "bg-amber-500/20 text-amber-400 border-amber-500/50"
                    : "bg-[#1a2333] text-[#92a4c9] border-[#232f48] hover:border-amber-500/30 hover:text-amber-400"
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        )}

        {/* Grouped notes list */}
        <div className="max-h-64 overflow-y-auto scrollbar-dark space-y-2">
          {grouped.map(({ tag, notes }) => (
            <div key={tag || "__untagged"}>
              {/* Group header */}
              <div className="flex items-center gap-1.5 mb-1 px-1">
                <span className="material-symbols-outlined text-amber-400 text-[10px]">
                  label
                </span>
                <span className="text-[10px] font-bold text-amber-400">
                  {tag || "Untagged"}
                </span>
                <span className="text-[9px] text-[#6b7c9c]">({notes.length})</span>
              </div>
              {/* Notes under this group */}
              <div className="space-y-1 pl-1">
                {notes.map((note) => {
                  const instColors =
                    INSTRUMENT_COLORS[note.instrument as InstrumentType] ??
                    INSTRUMENT_COLORS.CUSTOM;
                  // In grouped view, exclude the group tag from displayed tags
                  const otherTags = tag ? note.tags.filter((t) => t !== tag) : note.tags;
                  return (
                    <button
                      key={note.id}
                      onClick={() => onFieldNoteClick?.(note)}
                      aria-label={`Field note: ${note.product_id}`}
                      className="w-full text-left p-2 rounded bg-[#0a0f18] border border-[#232f48] hover:border-amber-500/30 transition-colors space-y-1"
                    >
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[8px] font-bold ${instColors.bg} ${instColors.text} border ${instColors.border}`}
                        >
                          {note.instrument}
                        </span>
                        <span className="text-[10px] text-white font-mono truncate flex-1">
                          {note.product_id}
                        </span>
                      </div>
                      {note.memo && (
                        <p className="text-[9px] text-[#6b7c9c] truncate">{note.memo}</p>
                      )}
                      {otherTags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {otherTags.map((t) => (
                            <span
                              key={t}
                              className="px-1.5 py-0.5 rounded-full text-[8px] bg-amber-500/10 text-amber-400/70 border border-amber-500/20"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </CollapsibleSection>
  );
}
