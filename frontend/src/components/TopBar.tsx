import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { Link } from "react-router-dom";
import toast from "react-hot-toast";

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

type SearchMode = "product" | "coordinates" | "region" | "text";

interface TopBarProps {
  onSearch?: (query: string) => void;
  onSelectResult?: (productId: string, instrument?: string, lat?: number | null, lon?: number | null) => void;
  searchableItems?: SearchableItem[];
  isMobile?: boolean;
  onEasterEgg?: () => void;
  onTerraform?: () => void;
  onCuriositySelfie?: () => void;
  onMarkWatney?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  lastActionDescription?: string;
  onUndo?: () => void;
  onRedo?: () => void;
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
 * Search mode detection
 * =======================================================*/
function detectSearchMode(query: string): SearchMode {
  const trimmed = query.trim();

  // Coordinates: "lat, lon" pattern (e.g., "-18.4, 77.5" or "18.4 77.5")
  if (/^-?\d+\.?\d*\s*[,\s]\s*-?\d+\.?\d*$/.test(trimmed)) {
    return "coordinates";
  }

  // Product ID patterns
  const productPatterns = [
    /^(frt|hrl|hrs|frs|att|atr)/i,           // CRISM
    /^(esp|psp|tra)_\d/i,                       // HiRISE
    /^(dteec|dteed)/i,                          // HiRISE DTM
    /^s_\d/i,                                    // SHARAD
    /^(b|p|d|g|j|k|n|t)\d{4}_\d/i,             // CTX
  ];
  if (productPatterns.some(p => p.test(trimmed))) {
    return "product";
  }

  // Known region names (simple check for common Mars locations)
  const regions = [
    "jezero", "gale", "olympus", "valles", "hellas", "argyre", "syrtis",
    "utopia", "isidis", "arcadia", "amazonis", "elysium", "tharsis",
    "chryse", "acidalia", "arabia", "terra", "planitia", "mons"
  ];
  if (regions.some(r => trimmed.toLowerCase().includes(r))) {
    return "region";
  }

  return "text";
}

/* =========================================================
 * Coordinate parsing
 * =======================================================*/
function parseCoordinates(query: string): { lat: number; lon: number } | null {
  const match = query.trim().match(/^(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)$/);
  if (match) {
    const lat = parseFloat(match[1]!);
    const lon = parseFloat(match[2]!);
    if (lat >= -90 && lat <= 90 && lon >= -360 && lon <= 360) {
      return { lat, lon };
    }
  }
  return null;
}

/* =========================================================
 * Search mode badge config
 * =======================================================*/
const MODE_BADGE: Record<SearchMode, { label: string; color: string }> = {
  product:     { label: "ID",     color: "text-cyan-400 bg-cyan-500/20 border-cyan-500/30" },
  coordinates: { label: "XY",     color: "text-emerald-400 bg-emerald-500/20 border-emerald-500/30" },
  region:      { label: "Region", color: "text-amber-400 bg-amber-500/20 border-amber-500/30" },
  text:        { label: "Search", color: "text-slate-400 bg-slate-500/20 border-slate-500/30" },
};

/* =========================================================
 * TopBar Component
 * =======================================================*/
export default function TopBar({
  onSearch,
  onSelectResult,
  isMobile = false,
  onEasterEgg,
  onTerraform,
  onCuriositySelfie,
  onMarkWatney,
  canUndo = false,
  canRedo = false,
  lastActionDescription,
  onUndo,
  onRedo,
}: TopBarProps) {
  const [query, setQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [totalProducts, setTotalProducts] = useState<number | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Easter egg #3: Curiosity selfie — click logo 7 times rapidly
  const logoClickCountRef = useRef(0);
  const logoClickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleLogoClick = useCallback((e: React.MouseEvent) => {
    logoClickCountRef.current += 1;
    if (logoClickTimerRef.current) clearTimeout(logoClickTimerRef.current);
    logoClickTimerRef.current = setTimeout(() => {
      logoClickCountRef.current = 0;
    }, 2000);

    if (logoClickCountRef.current >= 7) {
      e.preventDefault();
      logoClickCountRef.current = 0;
      if (logoClickTimerRef.current) clearTimeout(logoClickTimerRef.current);
      onCuriositySelfie?.();
    }
  }, [onCuriositySelfie]);

  // Detect search mode from query
  const searchMode = useMemo<SearchMode>(() => {
    if (!query.trim()) return "text";
    return detectSearchMode(query);
  }, [query]);

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
    // Easter egg: typing "game" triggers the space shooter
    if (value.trim().toLowerCase() === "game" && onEasterEgg) {
      setQuery("");
      setResults([]);
      setShowDropdown(false);
      onEasterEgg();
      return;
    }

    // Easter egg #2: terraform mode
    if (value.trim().toLowerCase() === "terraform" && onTerraform) {
      setQuery("");
      setResults([]);
      setShowDropdown(false);
      onTerraform();
      return;
    }

    // Easter egg #5: Mark Watney / The Martian
    if (value.trim().toLowerCase() === "markwatney" && onMarkWatney) {
      setQuery("");
      setResults([]);
      setShowDropdown(false);
      onMarkWatney();
      return;
    }

    // Easter egg #6: metric enforcement
    const lower = value.trim().toLowerCase();
    if (lower === "feet" || lower === "miles" || lower === "inches" || lower === "yards") {
      toast("This is a science tool. We use metric here.", {
        icon: "\u{1F4CF}",
        duration: 3000,
        style: {
          background: "#101622",
          color: "#92a4c9",
          border: "1px solid #232f48",
        },
      });
    }

    setQuery(value);
    setShowDropdown(value.trim().length > 0);

    // Only do backend search for non-coordinate modes
    const mode = detectSearchMode(value);
    if (mode !== "coordinates") {
      // Debounce the search
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => doSearch(value), 200);
    } else {
      // Clear results for coordinate mode
      setResults([]);
      setSearching(false);
    }
  };

  const handleSelect = (item: SearchResultItem) => {
    setQuery(item.product_id);
    setShowDropdown(false);
    onSelectResult?.(item.product_id, item.instrument, item.lat, item.lon);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Handle coordinates mode: Enter flies to coordinates
    if (searchMode === "coordinates" && e.key === "Enter") {
      e.preventDefault();
      const coords = parseCoordinates(query);
      if (coords) {
        setShowDropdown(false);
        onSelectResult?.(
          `${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)}`,
          "REGION",
          coords.lat,
          coords.lon,
        );
      }
      return;
    }

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

  // Mode badge component
  const badge = query.trim() ? MODE_BADGE[searchMode] : null;

  /* -- Search bar (shared between desktop and mobile menu) -- */
  const searchBar = (
    <div className={`relative ${isMobile ? "w-full" : "w-full max-w-xl"}`}>
      <div className="flex h-9 w-full items-stretch rounded-lg bg-surface-dark">
        <div className="flex items-center justify-center pl-3 text-slate-400">
          {searching ? (
            <span className="material-symbols-outlined text-[20px] animate-spin">progress_activity</span>
          ) : (
            <span className="material-symbols-outlined text-[20px]">search</span>
          )}
        </div>
        {/* Mode badge */}
        {badge && (
          <div className="flex items-center pl-2">
            <span className={`text-[8px] font-bold uppercase px-1.5 py-0.5 rounded border shrink-0 ${badge.color}`}>
              {badge.label}
            </span>
          </div>
        )}
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => handleInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => query.trim() && setShowDropdown(true)}
          className="w-full border-none bg-transparent px-2 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-0"
          placeholder="Search product IDs, coordinates, regions..."
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

      {/* Coordinates hint */}
      {showDropdown && searchMode === "coordinates" && (
        <div
          ref={dropdownRef}
          className="absolute top-full left-0 right-0 z-50 mt-1 rounded-lg border border-border-dark bg-surface-dark p-3 shadow-xl"
        >
          <div className="flex items-center gap-2 text-emerald-400">
            <span className="material-symbols-outlined text-base">my_location</span>
            <span className="text-sm">
              Press <kbd className="px-1.5 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-[10px] font-bold">Enter</kbd> to fly to these coordinates
            </span>
          </div>
          {(() => {
            const coords = parseCoordinates(query);
            if (coords) {
              return (
                <p className="mt-1.5 text-[11px] text-slate-400 font-mono pl-6">
                  {coords.lat.toFixed(4)}° lat, {coords.lon.toFixed(4)}° lon
                </p>
              );
            }
            return null;
          })()}
        </div>
      )}

      {/* Search Results Dropdown */}
      {showDropdown && searchMode !== "coordinates" && results.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute top-full left-0 right-0 z-50 mt-1 max-h-80 overflow-y-auto rounded-lg border border-border-dark bg-surface-dark shadow-xl"
        >
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
                  onClick={() => { handleSelect(item); setMobileMenuOpen(false); }}
                  onMouseEnter={() => setSelectedIndex(index)}
                  className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                    index === selectedIndex
                      ? "bg-primary/20 text-white"
                      : "text-slate-300 hover:bg-white/5"
                  }`}
                >
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

      {showDropdown && searchMode !== "coordinates" && query.trim().length >= 2 && !searching && results.length === 0 && (
        <div
          ref={dropdownRef}
          className="absolute top-full left-0 right-0 z-50 mt-1 rounded-lg border border-border-dark bg-surface-dark p-4 text-center shadow-xl"
        >
          <span className="text-sm text-slate-400">No products found for &ldquo;{query}&rdquo;</span>
        </div>
      )}
    </div>
  );

  /* -- Undo/Redo buttons -- */
  const undoRedoButtons = (
    <div className="flex items-center gap-0.5 ml-2">
      <button
        onClick={onUndo}
        disabled={!canUndo}
        className={`p-1.5 rounded transition-colors ${canUndo ? 'text-[#92a4c9] hover:text-white hover:bg-[#232f48]' : 'text-[#3a4a68] cursor-not-allowed'}`}
        title={lastActionDescription ? `Undo: ${lastActionDescription}` : 'Nothing to undo'}
      >
        <span className="material-symbols-outlined text-lg">undo</span>
      </button>
      <button
        onClick={onRedo}
        disabled={!canRedo}
        className={`p-1.5 rounded transition-colors ${canRedo ? 'text-[#92a4c9] hover:text-white hover:bg-[#232f48]' : 'text-[#3a4a68] cursor-not-allowed'}`}
        title="Redo"
      >
        <span className="material-symbols-outlined text-lg">redo</span>
      </button>
    </div>
  );

  /* -- Mobile header -- */
  if (isMobile) {
    return (
      <>
        <header className="flex h-12 items-center justify-between border-b border-border-dark bg-bg-dark px-4">
          {/* Brand */}
          <Link to="/" className="flex items-center gap-2" onClick={handleLogoClick}>
            <div className="flex size-5 items-center justify-center text-primary">
              <span className="material-symbols-outlined text-xl">rocket_launch</span>
            </div>
            <h1 className="text-lg font-bold tracking-tight">MarsLab</h1>
          </Link>

          {/* Hamburger */}
          <button
            onClick={() => setMobileMenuOpen((p) => !p)}
            className="flex items-center justify-center w-9 h-9 rounded-lg hover:bg-white/10 text-slate-300"
          >
            <span className="material-symbols-outlined text-2xl">
              {mobileMenuOpen ? "close" : "menu"}
            </span>
          </button>
        </header>

        {/* Mobile slide-down menu */}
        {mobileMenuOpen && (
          <div className="absolute top-12 left-0 right-0 z-50 border-b border-border-dark bg-bg-dark p-4 flex flex-col gap-4 shadow-xl">
            {/* Search */}
            {searchBar}

            {/* Nav links */}
            <nav className="flex flex-col gap-2">
              <Link
                to="/"
                onClick={() => setMobileMenuOpen(false)}
                className="text-sm font-medium text-white hover:text-primary px-2 py-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                Workbench
              </Link>
              <Link
                to="/download"
                onClick={() => setMobileMenuOpen(false)}
                className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                Data Download
              </Link>
              <Link
                to="/upload"
                onClick={() => setMobileMenuOpen(false)}
                className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                Data Upload
              </Link>
              <Link
                to="/suggestions"
                onClick={() => setMobileMenuOpen(false)}
                className="text-sm font-medium text-slate-400 hover:text-white px-2 py-2 rounded-lg hover:bg-white/5 transition-colors flex items-center gap-2"
              >
                <span className="material-symbols-outlined text-sm">lightbulb</span>
                Suggest Feature
              </Link>
            </nav>
          </div>
        )}
      </>
    );
  }

  /* -- Desktop header -- */
  return (
    <header className="flex h-14 items-center justify-between border-b border-border-dark bg-bg-dark px-6">
      {/* Brand + Navigation */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3 cursor-pointer" onClick={handleLogoClick}>
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

      {/* Search + Undo/Redo */}
      <div className="flex flex-1 items-center justify-center px-10">
        {searchBar}
        {undoRedoButtons}
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
