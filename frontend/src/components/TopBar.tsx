import { useState, useRef, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";

/* =========================================================
 * Types
 * =======================================================*/
type SearchResultItem = {
  product_id: string;
  instrument: string;
  title?: string;
  lat?: number | null;
  lon?: number | null;
};

type SearchResponse = {
  query: string;
  results: SearchResultItem[];
  count: number;
  total_products: number;
};

export interface SearchableItem {
  productId: string;
  title?: string;
}

interface TopBarProps {
  onSearch?: (query: string) => void;
  onSelectResult?: (productId: string, instrument?: string, lat?: number | null, lon?: number | null) => void;
  searchableItems?: SearchableItem[];
}

/* =========================================================
 * Instrument badge colors
 * =======================================================*/
const INSTRUMENT_COLORS: Record<string, string> = {
  CRISM:          "text-cyan-400 bg-cyan-500/20 border-cyan-500/30",
  HIRISE:         "text-yellow-400 bg-yellow-500/20 border-yellow-500/30",
  SHARAD:         "text-orange-400 bg-orange-500/20 border-orange-500/30",
  SHARAD_HIGHRES: "text-amber-400 bg-amber-500/20 border-amber-500/30",
  CTX:            "text-pink-400 bg-pink-500/20 border-pink-500/30",
  REGION:         "text-emerald-400 bg-emerald-500/20 border-emerald-500/30",
};

/* =========================================================
 * TopBar Component
 * =======================================================*/
export default function TopBar({
  onSearch,
  onSelectResult,
}: TopBarProps) {
  const [query, setQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [totalProducts, setTotalProducts] = useState<number | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Reset selected index when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [results.length]);

  // Debounced backend search
  const doSearch = useCallback((q: string) => {
    // Cancel previous request
    abortRef.current?.abort();

    if (!q.trim() || q.trim().length < 2) {
      setResults([]);
      setSearching(false);
      return;
    }

    setSearching(true);
    const controller = new AbortController();
    abortRef.current = controller;

    fetch(`/api/search/local?q=${encodeURIComponent(q.trim())}&limit=20`, {
      signal: controller.signal,
    })
      .then((res) => res.json())
      .then((data: SearchResponse) => {
        setResults(data.results);
        setTotalProducts(data.total_products);
        setSearching(false);
      })
      .catch((e) => {
        if (e.name !== "AbortError") {
          setResults([]);
          setSearching(false);
        }
      });
  }, []);

  const handleInputChange = (value: string) => {
    setQuery(value);
    setShowDropdown(value.trim().length > 0);

    // Debounce the search
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(value), 200);
  };

  const handleSelect = (item: SearchResultItem) => {
    setQuery(item.product_id);
    setShowDropdown(false);
    onSelectResult?.(item.product_id, item.instrument, item.lat, item.lon);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown || results.length === 0) {
      if (e.key === "Enter" && onSearch) {
        onSearch(query);
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
        break;
      case "Enter":
        e.preventDefault();
        if (results[selectedIndex]) {
          handleSelect(results[selectedIndex]);
        }
        break;
      case "Escape":
        setShowDropdown(false);
        break;
    }
  };

  // Highlight matching substring
  const highlightMatch = (text: string, q: string) => {
    if (!q.trim()) return <span>{text}</span>;
    const idx = text.toLowerCase().indexOf(q.toLowerCase());
    if (idx === -1) return <span>{text}</span>;
    return (
      <>
        <span>{text.slice(0, idx)}</span>
        <mark className="bg-primary/30 text-primary rounded px-0.5">
          {text.slice(idx, idx + q.length)}
        </mark>
        <span>{text.slice(idx + q.length)}</span>
      </>
    );
  };

  return (
    <header className="flex h-14 items-center justify-between border-b border-border-dark bg-bg-dark px-6">
      {/* Brand + Navigation */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="flex size-6 items-center justify-center text-primary">
            <span className="material-symbols-outlined text-2xl">rocket_launch</span>
          </div>
          <h1 className="text-xl font-bold tracking-tight">MarsLab</h1>
        </div>

        <div className="h-6 w-px bg-border-dark" />

        <nav className="flex items-center gap-6">
          <Link
            to="/"
            className="text-sm font-medium text-white hover:text-primary transition-colors border-b-2 border-primary pb-1"
          >
            Workbench
          </Link>
          <Link
            to="/download"
            className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
          >
            Data Download
          </Link>
          <Link
            to="/upload"
            className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
          >
            Data Upload
          </Link>
        </nav>
      </div>

      {/* Search */}
      <div className="flex flex-1 items-center justify-center px-10">
        <div className="relative w-full max-w-xl">
          <div className="flex h-9 w-full items-stretch rounded-lg bg-surface-dark">
            <div className="flex items-center justify-center pl-3 text-slate-400">
              {searching ? (
                <span className="material-symbols-outlined text-[20px] animate-spin">progress_activity</span>
              ) : (
                <span className="material-symbols-outlined text-[20px]">search</span>
              )}
            </div>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => query.trim() && setShowDropdown(true)}
              className="w-full border-none bg-transparent px-2 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-0"
              placeholder="Search product IDs across all instruments..."
            />
            {query && (
              <button
                onClick={() => {
                  setQuery("");
                  setResults([]);
                  setShowDropdown(false);
                  inputRef.current?.focus();
                }}
                className="flex items-center justify-center pr-3 text-slate-400 hover:text-white"
              >
                <span className="material-symbols-outlined text-[18px]">close</span>
              </button>
            )}
          </div>

          {/* Search Results Dropdown */}
          {showDropdown && results.length > 0 && (
            <div
              ref={dropdownRef}
              className="absolute top-full left-0 right-0 z-50 mt-1 max-h-80 overflow-y-auto rounded-lg border border-border-dark bg-surface-dark shadow-xl"
            >
              {/* Result count header */}
              <div className="px-3 py-1.5 border-b border-border-dark/50 flex justify-between">
                <span className="text-[10px] text-slate-500">
                  {results.length} result{results.length !== 1 ? "s" : ""}
                </span>
                {totalProducts && (
                  <span className="text-[10px] text-slate-600">
                    {totalProducts.toLocaleString()} products in database
                  </span>
                )}
              </div>

              <div className="p-1">
                {results.map((item, index) => {
                  const colorClass = INSTRUMENT_COLORS[item.instrument] || "text-slate-400 bg-slate-500/20 border-slate-500/30";
                  return (
                    <button
                      key={`${item.instrument}-${item.product_id}`}
                      onClick={() => handleSelect(item)}
                      onMouseEnter={() => setSelectedIndex(index)}
                      className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                        index === selectedIndex
                          ? "bg-primary/20 text-white"
                          : "text-slate-300 hover:bg-white/5"
                      }`}
                    >
                      {/* Instrument badge */}
                      <span className={`text-[8px] font-bold uppercase px-1.5 py-0.5 rounded border shrink-0 ${colorClass}`}>
                        {item.instrument}
                      </span>

                      <div className="flex flex-col flex-1 min-w-0">
                        <span className="font-mono text-sm truncate">
                          {highlightMatch(item.product_id, query)}
                        </span>
                        {item.title && (
                          <span className="text-[11px] text-slate-400 truncate mt-0.5">
                            {item.title}
                          </span>
                        )}
                      </div>

                      {/* Coordinates */}
                      {item.lat != null && item.lon != null && (
                        <span className="text-[9px] text-slate-600 font-mono shrink-0">
                          {item.lat.toFixed(1)}°, {item.lon.toFixed(1)}°
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* No results message */}
          {showDropdown && query.trim().length >= 2 && !searching && results.length === 0 && (
            <div
              ref={dropdownRef}
              className="absolute top-full left-0 right-0 z-50 mt-1 rounded-lg border border-border-dark bg-surface-dark p-4 text-center shadow-xl"
            >
              <span className="text-sm text-slate-400">No products found for "{query}"</span>
            </div>
          )}
        </div>
      </div>

      {/* Suggest Feature */}
      <Link
        to="/suggestions"
        className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium text-slate-400 hover:text-white border border-border-dark rounded-md hover:bg-white/5 transition-colors shrink-0"
      >
        <span className="material-symbols-outlined text-sm">lightbulb</span>
        Suggest Feature
      </Link>
    </header>
  );
}
