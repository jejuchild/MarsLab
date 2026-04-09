import { useEffect, useRef, useState } from "react";
import useCatalogSearch, { type SearchResult } from "../../hooks/useCatalogSearch";

export interface SearchBarProps {
  /** Called when the user picks/submits a result. */
  onSelect: (result: SearchResult) => void;
  /** Compact mode for mobile/topbar. */
  compact?: boolean;
}

/**
 * Catalog-first search bar (Phase 3).
 *
 * Priority: easter egg → catalog → coordinate → product_id → none.
 * Live suggestions dropdown shows top matches as the user types.
 */
export default function SearchBar({ onSelect, compact = false }: SearchBarProps) {
  const search = useCatalogSearch();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const suggestions = query.trim() ? search.suggestions(query, 6) : [];

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const submit = (result?: SearchResult) => {
    const final = result ?? search.parse(query);
    if (final.type === "none") return;
    onSelect(final);
    setQuery("");
    setOpen(false);
    setHighlightedIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (highlightedIndex >= 0 && suggestions[highlightedIndex]) {
        submit(suggestions[highlightedIndex]);
      } else {
        submit();
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((i) => Math.max(i - 1, -1));
    } else if (e.key === "Escape") {
      setOpen(false);
      setHighlightedIndex(-1);
    }
  };

  return (
    <div ref={wrapperRef} className={`relative ${compact ? "w-48" : "w-80"}`}>
      <div className="relative">
        <span className="material-symbols-outlined absolute left-2 top-1/2 -translate-y-1/2 text-sm text-slate-500 pointer-events-none">
          search
        </span>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
            setHighlightedIndex(-1);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder="Search 'jezero', '18.4, 77.7', or product ID"
          className="w-full pl-7 pr-2 py-1.5 rounded-lg bg-white/5 border border-border-dark text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30"
          aria-label="Search Mars locations, coordinates, or product IDs"
        />
      </div>

      {/* Suggestions dropdown */}
      {open && suggestions.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 rounded-lg border border-border-dark bg-[#0a0f18] shadow-xl overflow-hidden">
          {suggestions.map((s, i) => (
            <SuggestionRow
              key={`${s.type}-${i}`}
              result={s}
              highlighted={i === highlightedIndex}
              onMouseEnter={() => setHighlightedIndex(i)}
              onClick={() => submit(s)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function SuggestionRow({
  result,
  highlighted,
  onMouseEnter,
  onClick,
}: {
  result: SearchResult;
  highlighted: boolean;
  onMouseEnter: () => void;
  onClick: () => void;
}) {
  const baseClass =
    "flex items-center gap-2 w-full px-3 py-2 text-left text-xs transition-colors";
  const stateClass = highlighted ? "bg-primary/15 text-white" : "text-slate-300 hover:bg-white/5";

  if (result.type === "catalog") {
    return (
      <button type="button" className={`${baseClass} ${stateClass}`} onMouseEnter={onMouseEnter} onClick={onClick}>
        <span className="material-symbols-outlined text-sm text-primary flex-shrink-0">place</span>
        <div className="flex-1 min-w-0">
          <div className="font-semibold truncate">{result.entry.name}</div>
          <div className="text-[10px] text-slate-500 truncate">{result.entry.description}</div>
        </div>
        <span className="text-[10px] font-mono text-slate-500 flex-shrink-0">
          {result.entry.lat.toFixed(2)}°, {result.entry.lon.toFixed(2)}°
        </span>
      </button>
    );
  }

  if (result.type === "coordinate") {
    return (
      <button type="button" className={`${baseClass} ${stateClass}`} onMouseEnter={onMouseEnter} onClick={onClick}>
        <span className="material-symbols-outlined text-sm text-emerald-400 flex-shrink-0">my_location</span>
        <div className="flex-1">
          <div className="font-semibold">Coordinate</div>
          <div className="text-[10px] text-slate-500">
            {result.lat.toFixed(3)}°, {result.lon.toFixed(3)}°
          </div>
        </div>
      </button>
    );
  }

  if (result.type === "product") {
    return (
      <button type="button" className={`${baseClass} ${stateClass}`} onMouseEnter={onMouseEnter} onClick={onClick}>
        <span className="material-symbols-outlined text-sm text-amber-400 flex-shrink-0">database</span>
        <div className="flex-1 min-w-0">
          <div className="font-semibold font-mono truncate">{result.productId}</div>
          <div className="text-[10px] text-slate-500">{result.instrument} product</div>
        </div>
      </button>
    );
  }

  return null;
}
