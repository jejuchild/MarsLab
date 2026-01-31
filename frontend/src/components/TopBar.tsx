import { useState, useRef, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { searchWithHighlight } from "../utils/search";

export interface SearchableItem {
  productId: string;
  title?: string;
}

interface TopBarProps {
  onSearch?: (query: string) => void;
  onSelectResult?: (productId: string) => void;
  searchableItems?: SearchableItem[];
}

export default function TopBar({
  onSearch,
  onSelectResult,
  searchableItems = [],
}: TopBarProps) {
  const [query, setQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Create searchable strings that include both productId and title
  // Format: "productId|title" so we can search both but display them separately
  const searchStrings = searchableItems.map(item =>
    item.title ? `${item.productId}|${item.title}` : item.productId
  );
  const results = searchWithHighlight(searchStrings, query, 10);

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
  }, [results.length, query]);

  const handleInputChange = (value: string) => {
    setQuery(value);
    setShowDropdown(value.trim().length > 0);
  };

  const handleSelect = (item: string) => {
    // Extract productId from "productId|title" format
    const productId = item.split("|")[0];
    setQuery(productId);
    setShowDropdown(false);
    onSelectResult?.(productId);
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
          handleSelect(results[selectedIndex].item);
        }
        break;
      case "Escape":
        setShowDropdown(false);
        break;
    }
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
        </nav>
      </div>

      {/* Search */}
      <div className="flex flex-1 items-center justify-center px-10">
        <div className="relative w-full max-w-xl">
          <div className="flex h-9 w-full items-stretch rounded-lg bg-surface-dark">
            <div className="flex items-center justify-center pl-3 text-slate-400">
              <span className="material-symbols-outlined text-[20px]">search</span>
            </div>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => query.trim() && setShowDropdown(true)}
              className="w-full border-none bg-transparent px-2 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-0"
              placeholder="Search coordinates, feature names, or product IDs..."
            />
            {query && (
              <button
                onClick={() => {
                  setQuery("");
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
              <div className="p-1">
                {results.map((result, index) => {
                  // Parse "productId|title" format
                  const [productId, title] = result.item.split("|");
                  return (
                    <button
                      key={result.item}
                      onClick={() => handleSelect(result.item)}
                      onMouseEnter={() => setSelectedIndex(index)}
                      className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                        index === selectedIndex
                          ? "bg-primary/20 text-white"
                          : "text-slate-300 hover:bg-white/5"
                      }`}
                    >
                      <span className="material-symbols-outlined text-sm text-slate-400">
                        {result.isPrefix ? "star" : "search"}
                      </span>
                      <div className="flex flex-col flex-1 min-w-0">
                        <span className="font-mono text-sm truncate">
                          {result.segments.map((seg, i) =>
                            seg.highlight ? (
                              <mark
                                key={i}
                                className="bg-primary/30 text-primary rounded px-0.5"
                              >
                                {seg.text}
                              </mark>
                            ) : (
                              <span key={i}>{seg.text}</span>
                            )
                          )}
                        </span>
                        {title && (
                          <span className="text-[11px] text-slate-400 truncate mt-0.5">
                            {title}
                          </span>
                        )}
                      </div>
                      {result.isPrefix && (
                        <span className="ml-auto text-[10px] uppercase tracking-wider text-slate-500">
                          prefix
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* No results message */}
          {showDropdown && query.trim() && results.length === 0 && (
            <div
              ref={dropdownRef}
              className="absolute top-full left-0 right-0 z-50 mt-1 rounded-lg border border-border-dark bg-surface-dark p-4 text-center shadow-xl"
            >
              <span className="text-sm text-slate-400">No results found for "{query}"</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
