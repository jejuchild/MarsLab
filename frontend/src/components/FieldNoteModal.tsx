import { useEffect, useState, useCallback } from "react";
import {
  getNotesForProduct,
  getAllTags,
  createFieldNote,
  updateFieldNote,
  deleteFieldNote,
} from "../api/fieldnotes";
import type { FieldNote } from "../api/fieldnotes";

interface Props {
  productId: string;
  instrument: string;
  lat: number;
  lon: number;
  onClose: () => void;
  onNoteSaved: () => void;
}

export default function FieldNoteModal({
  productId,
  instrument,
  lat,
  lon,
  onClose,
  onNoteSaved,
}: Props) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [existing, setExisting] = useState<FieldNote | null>(null);
  const [memo, setMemo] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");
  const [allTags, setAllTags] = useState<string[]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Load existing note + all tags
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const [notes, tagList] = await Promise.all([
          getNotesForProduct(productId),
          getAllTags(),
        ]);
        if (cancelled) return;
        if (notes.length > 0) {
          const note = notes[0];
          setExisting(note);
          setMemo(note.memo);
          setTags(note.tags);
        }
        setAllTags(tagList);
      } catch (e) {
        console.error("Failed to load field notes:", e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [productId]);

  const addTag = useCallback((tag: string) => {
    const t = tag.trim();
    if (t && !tags.includes(t)) {
      setTags(prev => [...prev, t]);
    }
    setTagInput("");
  }, [tags]);

  const removeTag = useCallback((tag: string) => {
    setTags(prev => prev.filter(t => t !== tag));
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTag(tagInput);
    }
  }, [tagInput, addTag]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      if (existing) {
        await updateFieldNote(existing.id, { tags, memo: memo.trim() });
      } else {
        await createFieldNote({
          product_id: productId,
          instrument,
          lat,
          lon,
          tags,
          memo: memo.trim(),
        });
      }
      onNoteSaved();
      onClose();
    } catch (e) {
      console.error("Failed to save field note:", e);
    } finally {
      setSaving(false);
    }
  }, [existing, tags, memo, productId, instrument, lat, lon, onNoteSaved, onClose]);

  const handleDelete = useCallback(async () => {
    if (!existing) return;
    setSaving(true);
    try {
      await deleteFieldNote(existing.id);
      onNoteSaved();
      onClose();
    } catch (e) {
      console.error("Failed to delete field note:", e);
    } finally {
      setSaving(false);
    }
  }, [existing, onNoteSaved, onClose]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  // Suggested tags (exclude already-added)
  const suggestedTags = allTags.filter(t => !tags.includes(t));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative w-[420px] max-h-[80vh] bg-[#101622] rounded-lg border border-[#232f48] shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48] bg-[#0a0f18]">
          <div className="flex items-center gap-2 min-w-0">
            <span className="material-symbols-outlined text-amber-400 text-base">description</span>
            <div className="min-w-0">
              <h3 className="text-white font-bold text-sm truncate">{productId}</h3>
              <p className="text-[#92a4c9] text-[10px]">
                {instrument} Field Note
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-[#232f48] transition-colors text-[#92a4c9] hover:text-white"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-dark">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <span className="material-symbols-outlined animate-spin text-2xl text-amber-400">
                progress_activity
              </span>
            </div>
          ) : (
            <>
              {/* Tags */}
              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-widest text-[#92a4c9]">
                  Tags
                </label>

                {/* Current tags */}
                {tags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {tags.map(tag => (
                      <span
                        key={tag}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30"
                      >
                        {tag}
                        <button
                          onClick={() => removeTag(tag)}
                          className="hover:text-red-400 transition-colors"
                        >
                          <span className="material-symbols-outlined text-[11px]">close</span>
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                {/* Tag input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Add tag..."
                    className="flex-1 bg-[#0a0f18] border border-[#232f48] rounded px-2.5 py-1.5 text-[11px] text-white placeholder-[#6b7c9c] focus:outline-none focus:border-amber-500/40"
                  />
                  <button
                    onClick={() => addTag(tagInput)}
                    disabled={!tagInput.trim()}
                    className="px-2.5 py-1.5 text-[10px] font-bold rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    Add
                  </button>
                </div>

                {/* Tag suggestions */}
                {suggestedTags.length > 0 && (
                  <div className="space-y-1">
                    <span className="text-[9px] text-[#6b7c9c] uppercase">Existing tags</span>
                    <div className="flex flex-wrap gap-1">
                      {suggestedTags.map(tag => (
                        <button
                          key={tag}
                          onClick={() => addTag(tag)}
                          className="px-2 py-0.5 rounded-full text-[9px] bg-[#1a2333] text-[#92a4c9] border border-[#232f48] hover:border-amber-500/30 hover:text-amber-400 transition-colors"
                        >
                          + {tag}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Memo */}
              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-widest text-[#92a4c9]">
                  Memo
                </label>
                <textarea
                  value={memo}
                  onChange={(e) => setMemo(e.target.value)}
                  placeholder="Write your notes here..."
                  rows={5}
                  className="w-full bg-[#0a0f18] border border-[#232f48] rounded px-3 py-2 text-[11px] text-white placeholder-[#6b7c9c] resize-y focus:outline-none focus:border-amber-500/40 scrollbar-dark"
                />
              </div>

              {/* Existing note info */}
              {existing && (
                <div className="text-[9px] text-[#6b7c9c] space-y-0.5">
                  <div>Created: {new Date(existing.created_at).toLocaleString()}</div>
                  {existing.updated_at !== existing.created_at && (
                    <div>Updated: {new Date(existing.updated_at).toLocaleString()}</div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        {!loading && (
          <div className="px-4 py-3 border-t border-[#232f48] bg-[#0a0f18] flex items-center gap-2">
            {/* Delete (left side) */}
            {existing && (
              confirmDelete ? (
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-red-400">Delete?</span>
                  <button
                    onClick={handleDelete}
                    disabled={saving}
                    className="px-2 py-1 text-[10px] font-bold rounded bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => setConfirmDelete(false)}
                    className="px-2 py-1 text-[10px] rounded border border-[#232f48] text-[#92a4c9] hover:text-white transition-colors"
                  >
                    No
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setConfirmDelete(true)}
                  className="px-2.5 py-1.5 text-[10px] font-medium rounded border border-red-500/30 text-red-400/70 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  Delete
                </button>
              )
            )}

            <div className="flex-1" />

            {/* Cancel / Save (right side) */}
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-[11px] font-medium rounded border border-[#232f48] text-[#92a4c9] hover:text-white hover:bg-[#1a2333] transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving || (!memo.trim() && tags.length === 0)}
              className="px-4 py-1.5 text-[11px] font-bold rounded bg-amber-500/20 border border-amber-500/50 text-amber-400 hover:bg-amber-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {saving ? "Saving..." : existing ? "Update" : "Save"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
