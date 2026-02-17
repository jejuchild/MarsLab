import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
  memo,
} from "react";
import { createPortal } from "react-dom";

/* =========================================================
 * Types
 * =======================================================*/
export type CommandCategory =
  | "navigate"
  | "instrument"
  | "analysis"
  | "view"
  | "page";

export interface CommandAction {
  id: string;
  label: string;
  category: CommandCategory;
  /** Material Symbols icon name */
  icon: string;
  /** Extra searchable keywords */
  keywords: string[];
  /** Payload dispatched on selection */
  action:
    | { type: "fly_to"; lat: number; lon: number }
    | { type: "toggle_instrument"; instrumentId: string }
    | { type: "set_analysis"; mode: string }
    | { type: "set_map_mode"; mode: "2D" | "3D" }
    | { type: "toggle_grid" }
    | { type: "navigate_page"; path: string }
    | { type: "show_keyboard_shortcuts" }
    | { type: "show_tour" };
}

/* =========================================================
 * Category metadata
 * =======================================================*/
const CATEGORY_META: Record<
  CommandCategory,
  { label: string; icon: string }
> = {
  navigate: { label: "Navigate", icon: "explore" },
  instrument: { label: "Instruments", icon: "satellite_alt" },
  analysis: { label: "Analysis", icon: "science" },
  view: { label: "View", icon: "visibility" },
  page: { label: "Pages", icon: "web" },
};

/** Canonical order for rendering categories */
const CATEGORY_ORDER: CommandCategory[] = [
  "navigate",
  "instrument",
  "analysis",
  "view",
  "page",
];

/* =========================================================
 * Built-in action registry
 * =======================================================*/
const ACTIONS: CommandAction[] = [
  // ── Navigate (popular Mars regions) ──────────────────────
  {
    id: "fly-jezero",
    label: "Fly to Jezero Crater",
    category: "navigate",
    icon: "flight",
    keywords: ["mars 2020", "perseverance", "landing site", "delta", "lake"],
    action: { type: "fly_to", lat: 18.41, lon: 77.69 },
  },
  {
    id: "fly-gale",
    label: "Fly to Gale Crater",
    category: "navigate",
    icon: "flight",
    keywords: ["curiosity", "mount sharp", "aeolis mons"],
    action: { type: "fly_to", lat: -5.34, lon: 137.65 },
  },
  {
    id: "fly-valles",
    label: "Fly to Valles Marineris",
    category: "navigate",
    icon: "flight",
    keywords: ["canyon", "rift", "tectonic"],
    action: { type: "fly_to", lat: -11.03, lon: -60.80 },
  },
  {
    id: "fly-olympus",
    label: "Fly to Olympus Mons",
    category: "navigate",
    icon: "flight",
    keywords: ["volcano", "tallest", "shield"],
    action: { type: "fly_to", lat: 18.58, lon: -133.52 },
  },
  {
    id: "fly-hellas",
    label: "Fly to Hellas Basin",
    category: "navigate",
    icon: "flight",
    keywords: ["impact", "basin", "deep", "planitia"],
    action: { type: "fly_to", lat: -41.64, lon: 70.85 },
  },
  {
    id: "fly-syrtis",
    label: "Fly to Syrtis Major",
    category: "navigate",
    icon: "flight",
    keywords: ["dark", "albedo", "volcanic", "planum"],
    action: { type: "fly_to", lat: 8.90, lon: 68.00 },
  },
  {
    id: "fly-nili",
    label: "Fly to Nili Fossae",
    category: "navigate",
    icon: "flight",
    keywords: ["trough", "carbonate", "olivine", "clay"],
    action: { type: "fly_to", lat: 22.07, lon: 76.90 },
  },
  {
    id: "fly-mawrth",
    label: "Fly to Mawrth Vallis",
    category: "navigate",
    icon: "flight",
    keywords: ["valley", "clay", "phyllosilicate", "outflow"],
    action: { type: "fly_to", lat: 22.46, lon: -16.60 },
  },
  {
    id: "fly-south-pole",
    label: "Fly to South Polar Region",
    category: "navigate",
    icon: "flight",
    keywords: ["ice", "cap", "polar", "co2", "layered"],
    action: { type: "fly_to", lat: -87.0, lon: 0.0 },
  },
  {
    id: "fly-north-pole",
    label: "Fly to North Polar Region",
    category: "navigate",
    icon: "flight",
    keywords: ["ice", "cap", "polar", "water", "layered", "planum boreum"],
    action: { type: "fly_to", lat: 87.0, lon: 0.0 },
  },
  {
    id: "fly-eberswalde",
    label: "Fly to Eberswalde Crater",
    category: "navigate",
    icon: "flight",
    keywords: ["delta", "sediment", "ancient lake"],
    action: { type: "fly_to", lat: -23.98, lon: -33.30 },
  },
  {
    id: "fly-columbia-hills",
    label: "Fly to Columbia Hills",
    category: "navigate",
    icon: "flight",
    keywords: ["spirit", "rover", "gusev", "silica"],
    action: { type: "fly_to", lat: -14.57, lon: 175.47 },
  },
  {
    id: "fly-meridiani",
    label: "Fly to Meridiani Planum",
    category: "navigate",
    icon: "flight",
    keywords: ["opportunity", "hematite", "rover", "blueberries"],
    action: { type: "fly_to", lat: 2.39, lon: -1.95 },
  },
  {
    id: "fly-elysium",
    label: "Fly to Elysium Planitia",
    category: "navigate",
    icon: "flight",
    keywords: ["insight", "lander", "seismology", "volcanic"],
    action: { type: "fly_to", lat: 1.82, lon: 153.69 },
  },
  {
    id: "fly-tharsis",
    label: "Fly to Tharsis Rise",
    category: "navigate",
    icon: "flight",
    keywords: ["volcano", "bulge", "ascraeus", "pavonis", "arsia"],
    action: { type: "fly_to", lat: 0.0, lon: -100.0 },
  },

  // ── Instruments ──────────────────────────────────────────
  {
    id: "toggle-crism",
    label: "Toggle CRISM Footprints",
    category: "instrument",
    icon: "science",
    keywords: ["spectrometer", "mineral", "spectral", "mtrdr"],
    action: { type: "toggle_instrument", instrumentId: "crism" },
  },
  {
    id: "toggle-hirise",
    label: "Toggle HiRISE Footprints",
    category: "instrument",
    icon: "photo_camera",
    keywords: ["high resolution", "imaging", "camera"],
    action: { type: "toggle_instrument", instrumentId: "hirise" },
  },
  {
    id: "toggle-sharad",
    label: "Toggle SHARAD Footprints",
    category: "instrument",
    icon: "radar",
    keywords: ["radar", "subsurface", "ice", "shallow"],
    action: { type: "toggle_instrument", instrumentId: "sharad" },
  },
  {
    id: "toggle-sharad-highres",
    label: "Toggle SHARAD High-Res Footprints",
    category: "instrument",
    icon: "radar",
    keywords: ["radar", "subsurface", "rdr", "high resolution"],
    action: { type: "toggle_instrument", instrumentId: "sharad_highres" },
  },
  {
    id: "toggle-ctx",
    label: "Toggle CTX Footprints",
    category: "instrument",
    icon: "photo_camera",
    keywords: ["context", "camera", "mro"],
    action: { type: "toggle_instrument", instrumentId: "ctx" },
  },
  {
    id: "toggle-hirise-dtm",
    label: "Toggle HiRISE DTM Footprints",
    category: "instrument",
    icon: "terrain",
    keywords: ["digital terrain", "elevation", "topography", "dtm"],
    action: { type: "toggle_instrument", instrumentId: "hirise_dtm" },
  },
  {
    id: "toggle-crism-trr3",
    label: "Toggle CRISM TRR3 Footprints",
    category: "instrument",
    icon: "science",
    keywords: ["trdr", "targeted", "spectral", "trr3"],
    action: { type: "toggle_instrument", instrumentId: "crism_trr3" },
  },

  // ── Analysis ─────────────────────────────────────────────
  {
    id: "analysis-slope",
    label: "Run Slope Analysis",
    category: "analysis",
    icon: "landscape",
    keywords: ["slope", "gradient", "terrain", "elevation", "safety"],
    action: { type: "set_analysis", mode: "slope" },
  },
  {
    id: "analysis-agentic",
    label: "Open Agentic AI",
    category: "analysis",
    icon: "smart_toy",
    keywords: ["agent", "ai", "llm", "autonomous", "investigate"],
    action: { type: "set_analysis", mode: "agentic" },
  },
  {
    id: "analysis-assistant",
    label: "Open MARVIS Agent",
    category: "analysis",
    icon: "smart_toy",
    keywords: ["assistant", "marvis", "agent", "workflow", "research", "plan", "sharad", "ice", "terrace"],
    action: { type: "set_analysis", mode: "agentic" },
  },
  {
    id: "analysis-report",
    label: "Generate Landing Site Report",
    category: "analysis",
    icon: "summarize",
    keywords: ["report", "landing", "compare", "regions", "multi"],
    action: { type: "set_analysis", mode: "report" },
  },
  {
    id: "analysis-ai",
    label: "Open AI Analysis",
    category: "analysis",
    icon: "psychology",
    keywords: ["ai", "analysis", "gemini", "context", "science"],
    action: { type: "set_analysis", mode: "ai_analysis" },
  },
  {
    id: "analysis-line",
    label: "Draw Line Profile",
    category: "analysis",
    icon: "show_chart",
    keywords: ["line", "profile", "elevation", "cross section", "transect"],
    action: { type: "set_analysis", mode: "line" },
  },

  // ── View ─────────────────────────────────────────────────
  {
    id: "view-2d",
    label: "Switch to 2D Map",
    category: "view",
    icon: "map",
    keywords: ["flat", "mercator", "2d", "projection"],
    action: { type: "set_map_mode", mode: "2D" },
  },
  {
    id: "view-3d",
    label: "Switch to 3D Globe",
    category: "view",
    icon: "public",
    keywords: ["globe", "sphere", "3d", "cesium"],
    action: { type: "set_map_mode", mode: "3D" },
  },
  {
    id: "view-grid",
    label: "Toggle Coordinate Grid",
    category: "view",
    icon: "grid_on",
    keywords: ["grid", "coordinates", "latitude", "longitude", "graticule"],
    action: { type: "toggle_grid" },
  },
  {
    id: "view-shortcuts",
    label: "Show Keyboard Shortcuts",
    category: "view",
    icon: "keyboard",
    keywords: ["keyboard", "shortcuts", "help", "keys", "hotkeys"],
    action: { type: "show_keyboard_shortcuts" },
  },
  {
    id: "view-tour",
    label: "Re-take Onboarding Tour",
    category: "view",
    icon: "school",
    keywords: ["tour", "onboarding", "tutorial", "guide", "welcome", "help"],
    action: { type: "show_tour" },
  },

  // ── Pages ────────────────────────────────────────────────
  {
    id: "page-workbench",
    label: "Go to Workbench",
    category: "page",
    icon: "home",
    keywords: ["main", "map", "home", "workspace"],
    action: { type: "navigate_page", path: "/" },
  },
  {
    id: "page-download",
    label: "Go to Data Download",
    category: "page",
    icon: "download",
    keywords: ["download", "data", "export", "ode"],
    action: { type: "navigate_page", path: "/download" },
  },
  {
    id: "page-upload",
    label: "Go to Data Upload",
    category: "page",
    icon: "upload",
    keywords: ["upload", "import", "geotiff", "custom"],
    action: { type: "navigate_page", path: "/upload" },
  },
];

/* =========================================================
 * Props
 * =======================================================*/
interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onAction: (action: CommandAction) => void;
}

/* =========================================================
 * Fuzzy search — simple substring match on label + keywords
 * =======================================================*/
function matchesQuery(action: CommandAction, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  const haystack = [
    action.label,
    action.category,
    ...action.keywords,
  ]
    .join(" ")
    .toLowerCase();
  // Split query into tokens for multi-word matching
  const tokens = q.split(/\s+/).filter(Boolean);
  return tokens.every((token) => haystack.includes(token));
}

/* =========================================================
 * Highlight matching substring in text
 * =======================================================*/
function highlightMatch(text: string, query: string): JSX.Element {
  if (!query.trim()) return <>{text}</>;
  const q = query.trim().toLowerCase();
  const idx = text.toLowerCase().indexOf(q);
  if (idx === -1) return <>{text}</>;
  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-primary/30 text-sky-300 rounded px-0.5">
        {text.slice(idx, idx + q.length)}
      </mark>
      {text.slice(idx + q.length)}
    </>
  );
}

/* =========================================================
 * Component
 * =======================================================*/
function CommandPalette({ isOpen, onClose, onAction }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<Map<number, HTMLButtonElement>>(new Map());

  // Filtered results
  const filtered = useMemo(
    () => ACTIONS.filter((a) => matchesQuery(a, query)),
    [query]
  );

  // Group filtered results by category (preserving category order)
  const grouped = useMemo(() => {
    const groups: { category: CommandCategory; items: CommandAction[] }[] = [];
    for (const cat of CATEGORY_ORDER) {
      const items = filtered.filter((a) => a.category === cat);
      if (items.length > 0) {
        groups.push({ category: cat, items });
      }
    }
    return groups;
  }, [filtered]);

  // Flat list for keyboard navigation
  const flatItems = useMemo(
    () => grouped.flatMap((g) => g.items),
    [grouped]
  );

  // Reset state when opening/closing
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      // Focus input after a tick to allow the portal to mount
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  }, [isOpen]);

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Scroll selected item into view
  useEffect(() => {
    const el = itemRefs.current.get(selectedIndex);
    if (el) {
      el.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  const handleSelect = useCallback(
    (action: CommandAction) => {
      onAction(action);
      onClose();
    },
    [onAction, onClose]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            Math.min(prev + 1, flatItems.length - 1)
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => Math.max(prev - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (flatItems[selectedIndex]) {
            handleSelect(flatItems[selectedIndex]);
          }
          break;
        case "Tab":
          // Prevent focus escaping the palette
          e.preventDefault();
          break;
      }
    },
    [flatItems, selectedIndex, handleSelect]
  );

  // Store refs for scroll-into-view
  const setItemRef = useCallback(
    (index: number, el: HTMLButtonElement | null) => {
      if (el) {
        itemRefs.current.set(index, el);
      } else {
        itemRefs.current.delete(index);
      }
    },
    []
  );

  if (!isOpen) return null;

  // Build flat index counter for keyboard navigation
  let flatIndex = 0;

  const palette = (
    <div
      className="fixed inset-0 z-[9999] flex items-start justify-center"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fadeIn"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Palette container */}
      <div
        className="relative mt-[12vh] w-full max-w-xl px-4 sm:px-0 animate-slideDown"
        onKeyDown={handleKeyDown}
      >
        <div className="overflow-hidden rounded-xl border border-border-dark bg-[#0d1520] shadow-2xl shadow-black/50">
          {/* Search input */}
          <div className="flex items-center gap-3 border-b border-border-dark px-4 py-3">
            <span className="material-symbols-outlined text-xl text-slate-400">
              search
            </span>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 bg-transparent text-sm text-white placeholder:text-slate-500 focus:outline-none"
              placeholder="Type a command or search..."
              aria-label="Search commands"
              aria-activedescendant={
                flatItems[selectedIndex]
                  ? `cmd-item-${flatItems[selectedIndex].id}`
                  : undefined
              }
              role="combobox"
              aria-haspopup="listbox"
              aria-expanded="true"
              aria-controls="command-list"
              aria-autocomplete="list"
            />
            <kbd className="hidden sm:flex items-center gap-0.5 rounded border border-border-dark bg-surface-dark px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
              ESC
            </kbd>
          </div>

          {/* Results list */}
          <div
            ref={listRef}
            id="command-list"
            role="listbox"
            className="max-h-[60vh] overflow-y-auto scrollbar-dark p-2"
          >
            {flatItems.length === 0 ? (
              <div className="flex flex-col items-center gap-2 py-8 text-slate-500">
                <span className="material-symbols-outlined text-3xl">
                  search_off
                </span>
                <span className="text-sm">
                  No results for &ldquo;{query}&rdquo;
                </span>
              </div>
            ) : (
              grouped.map((group) => {
                const meta = CATEGORY_META[group.category];
                return (
                  <div key={group.category} className="mb-1">
                    {/* Category header */}
                    <div className="flex items-center gap-2 px-3 pb-1 pt-2">
                      <span className="material-symbols-outlined text-[14px] text-slate-600">
                        {meta.icon}
                      </span>
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                        {meta.label}
                      </span>
                    </div>

                    {/* Items */}
                    {group.items.map((item) => {
                      const currentFlatIndex = flatIndex++;
                      const isSelected = currentFlatIndex === selectedIndex;
                      return (
                        <button
                          key={item.id}
                          id={`cmd-item-${item.id}`}
                          ref={(el) => setItemRef(currentFlatIndex, el)}
                          role="option"
                          aria-selected={isSelected}
                          onClick={() => handleSelect(item)}
                          onMouseEnter={() =>
                            setSelectedIndex(currentFlatIndex)
                          }
                          className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors ${
                            isSelected
                              ? "bg-primary/15 text-white"
                              : "text-slate-300 hover:bg-white/5"
                          }`}
                        >
                          <span
                            className={`material-symbols-outlined text-lg ${
                              isSelected ? "text-sky-400" : "text-slate-500"
                            }`}
                          >
                            {item.icon}
                          </span>
                          <span className="flex-1 text-sm">
                            {highlightMatch(item.label, query)}
                          </span>
                          {isSelected && (
                            <kbd className="hidden sm:flex items-center rounded border border-border-dark bg-surface-dark px-1.5 py-0.5 text-[10px] text-slate-500">
                              Enter
                            </kbd>
                          )}
                        </button>
                      );
                    })}
                  </div>
                );
              })
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between border-t border-border-dark px-4 py-2">
            <div className="flex items-center gap-3 text-[10px] text-slate-600">
              <span className="flex items-center gap-1">
                <kbd className="rounded border border-border-dark bg-surface-dark px-1 py-0.5 font-mono text-[9px]">
                  ↑↓
                </kbd>
                navigate
              </span>
              <span className="flex items-center gap-1">
                <kbd className="rounded border border-border-dark bg-surface-dark px-1 py-0.5 font-mono text-[9px]">
                  ↵
                </kbd>
                select
              </span>
              <span className="flex items-center gap-1">
                <kbd className="rounded border border-border-dark bg-surface-dark px-1 py-0.5 font-mono text-[9px]">
                  esc
                </kbd>
                close
              </span>
            </div>
            <span className="text-[10px] text-slate-600">
              {flatItems.length} command{flatItems.length !== 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(palette, document.body);
}

export default memo(CommandPalette);
