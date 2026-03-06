import { useState } from "react";
import type { MapBookmark } from "../../hooks/useBookmarks";

interface BookmarkPanelProps {
  bookmarks: MapBookmark[];
  onSelect: (bookmark: MapBookmark) => void;
  onRemove: (id: string) => void;
  onRename: (id: string, name: string) => void;
  isOpen: boolean;
  onClose: () => void;
}

export default function BookmarkPanel({
  bookmarks,
  onSelect,
  onRemove,
  onRename,
  isOpen,
  onClose,
}: BookmarkPanelProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");

  const handleDoubleClick = (id: string, currentName: string) => {
    setEditingId(id);
    setEditingName(currentName);
  };

  const handleSaveRename = (id: string) => {
    if (editingName.trim()) {
      onRename(id, editingName.trim());
    }
    setEditingId(null);
    setEditingName("");
  };

  const handleKeyDown = (e: React.KeyboardEvent, id: string) => {
    if (e.key === "Enter") {
      handleSaveRename(id);
    } else if (e.key === "Escape") {
      setEditingId(null);
      setEditingName("");
    }
  };

  if (!isOpen) return null;

  return (
    <div className="absolute bottom-20 left-6 z-30 animate-fade-in">
      <div className="rounded-xl border border-border-dark bg-bg-dark/95 backdrop-blur-md shadow-2xl w-72 overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border-dark/50">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-amber-400 text-sm">star</span>
            <h3 className="text-sm font-semibold text-slate-200">Bookmarks</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-700/30 transition-colors text-slate-400 hover:text-slate-200"
            aria-label="Close bookmarks panel"
          >
            <span className="material-symbols-outlined text-sm">close</span>
          </button>
        </div>

        {/* Bookmarks List */}
        <div className="max-h-80 overflow-y-auto">
          {bookmarks.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <p className="text-xs text-slate-500">
                No bookmarks yet. Press B to add one.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-border-dark/30">
              {bookmarks.map((bookmark) => (
                <div
                  key={bookmark.id}
                  className="group px-3 py-2.5 hover:bg-slate-700/20 transition-colors cursor-pointer"
                  onClick={() => onSelect(bookmark)}
                >
                  <div className="flex items-start gap-2">
                    {/* Star icon */}
                    <span className="material-symbols-outlined text-amber-400 text-[14px] flex-shrink-0 mt-0.5">
                      star
                    </span>

                    {/* Name and coordinates */}
                    <div className="flex-1 min-w-0">
                      {editingId === bookmark.id ? (
                        <input
                          autoFocus
                          type="text"
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          onBlur={() => handleSaveRename(bookmark.id)}
                          onKeyDown={(e) => handleKeyDown(e, bookmark.id)}
                          className="w-full bg-slate-700/40 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-primary/50"
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        <div
                          onDoubleClick={() =>
                            handleDoubleClick(bookmark.id, bookmark.name)
                          }
                        >
                          <p className="text-xs text-slate-200 truncate font-medium">
                            {bookmark.name}
                          </p>
                          <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                            {bookmark.lat.toFixed(4)}°, {bookmark.lon.toFixed(4)}°
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Delete button (visible on hover) */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemove(bookmark.id);
                      }}
                      className="p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/20"
                      aria-label="Delete bookmark"
                    >
                      <span className="material-symbols-outlined text-red-400/70 text-[14px]">
                        delete
                      </span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        {bookmarks.length > 0 && (
          <div className="px-4 py-2 border-t border-border-dark/50 bg-slate-900/30">
            <p className="text-[10px] text-slate-500">
              {bookmarks.length} {bookmarks.length === 1 ? "bookmark" : "bookmarks"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
