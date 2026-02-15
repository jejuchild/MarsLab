import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  scoreColor,
  EMPTY_SCORES,
  type RegionScoresWithComposite,
  type RegionScoreEntry,
  type RegionScoresResponse,
} from "../utils/regionScoring";

/* =========================================================
 * Types
 * =======================================================*/

interface RegionRaw {
  region_id: string;
  display_name: string;
  description: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  tags: string[];
}

interface RegionCard {
  regionId: string;
  displayName: string;
  description: string;
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
  tags: string[];
  scores: RegionScoresWithComposite;
  productCounts: Record<string, number>;
}

export interface RegionDashboardProps {
  isOpen: boolean;
  onClose: () => void;
  onFlyTo: (lat: number, lon: number) => void;
  onRunReport: (regionId: string) => void;
}

/* =========================================================
 * Sort options
 * =======================================================*/

type SortKey = "composite" | "name" | "latitude" | "ice" | "slope";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "composite", label: "Composite Score" },
  { key: "name", label: "Name (A-Z)" },
  { key: "latitude", label: "Latitude" },
  { key: "ice", label: "Ice Score" },
  { key: "slope", label: "Slope Safety" },
];

type ViewMode = "grid" | "list";

/* =========================================================
 * Score bar config
 * =======================================================*/

const SCORE_BARS: { key: keyof RegionScoresWithComposite; label: string; color: string }[] = [
  { key: "ice", label: "Ice", color: "#22d3ee" },
  { key: "slope", label: "Slope", color: "#4ade80" },
  { key: "subsurface", label: "SHARAD", color: "#a78bfa" },
  { key: "coverage", label: "Cov", color: "#fbbf24" },
];

/* =========================================================
 * Helpers
 * =======================================================*/

function centerLat(r: RegionCard): number {
  return (r.latMin + r.latMax) / 2;
}

function centerLon(r: RegionCard): number {
  if (r.lonMin <= r.lonMax) return (r.lonMin + r.lonMax) / 2;
  // Wraps around antimeridian
  const c = (r.lonMin + r.lonMax + 360) / 2;
  return c <= 180 ? c : c - 360;
}

/** Collect all unique tags from a list of regions. */
function collectTags(regions: RegionCard[]): string[] {
  const tagSet = new Set<string>();
  for (const r of regions) {
    for (const t of r.tags) tagSet.add(t);
  }
  return Array.from(tagSet).sort();
}

/** Null-safe sort comparator: nulls go to bottom (desc) or top (asc). */
function nullSafeCompare(a: number | null, b: number | null): number {
  if (a === null && b === null) return 0;
  if (a === null) return -1; // null sorts low
  if (b === null) return 1;
  return a - b;
}

/* =========================================================
 * Sub-components
 * =======================================================*/

function ScoreBar({ value, color }: { value: number | null; color: string }) {
  if (value === null) {
    return (
      <div className="flex items-center gap-1.5" title="Insufficient data">
        <div className="h-[5px] flex-1 rounded-full bg-[#1a2333] overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{ width: "100%", backgroundColor: "#1e2a3a" }}
          />
        </div>
        <span className="text-[10px] font-mono text-[#3d4f6f] w-6 text-right">N/A</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-1.5" title={`${value}/100`}>
      <div className="h-[5px] flex-1 rounded-full bg-[#1a2333] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${value}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[10px] font-mono text-[#6b7c9c] w-6 text-right tabular-nums">{value}</span>
    </div>
  );
}

function TagChip({
  tag,
  active,
  onClick,
}: {
  tag: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 rounded-full text-[10px] font-medium transition-colors whitespace-nowrap
        ${active
          ? "bg-sky-500/25 text-sky-300 border border-sky-500/40"
          : "bg-[#1a2333] text-[#6b7c9c] border border-[#232f48] hover:border-[#3d4f6f] hover:text-[#92a4c9]"
        }`}
    >
      {tag}
    </button>
  );
}

/* =========================================================
 * Grid card
 * =======================================================*/

function GridCard({
  region,
  onExplore,
  onReport,
}: {
  region: RegionCard;
  onExplore: () => void;
  onReport: () => void;
}) {
  const compositeTextClass = scoreColor(region.scores.composite);

  return (
    <article
      className="group rounded-xl border border-[#232f48] bg-[#111827] p-4 flex flex-col gap-3
                 transition-all duration-200 hover:border-sky-500/50 hover:shadow-lg hover:shadow-sky-500/5
                 focus-within:border-sky-500/50 focus-within:ring-1 focus-within:ring-sky-500/30"
      tabIndex={0}
      role="article"
      aria-label={`${region.displayName} — composite score ${region.scores.composite ?? "N/A"}`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-white truncate">{region.displayName}</h3>
          <p className="text-[11px] text-[#6b7c9c] mt-0.5 line-clamp-2 leading-snug">
            {region.description || "No description available."}
          </p>
        </div>
        {/* Composite score badge */}
        <div
          className="shrink-0 flex flex-col items-center justify-center w-12 h-12 rounded-lg bg-[#0d1117] border border-[#232f48]"
          title={region.scores.composite !== null ? `Composite score: ${region.scores.composite}/100` : "Composite score: N/A"}
        >
          <span className={`text-lg font-bold tabular-nums leading-none ${compositeTextClass}`}>
            {region.scores.composite !== null ? region.scores.composite : "---"}
          </span>
          <span className="text-[8px] text-[#4a5a78] uppercase tracking-wider mt-0.5">score</span>
        </div>
      </div>

      {/* Score bars */}
      <div className="flex flex-col gap-1.5">
        {SCORE_BARS.map((bar) => (
          <div key={bar.key} className="flex items-center gap-1">
            <span className="text-[9px] text-[#6b7c9c] w-10 shrink-0 uppercase tracking-wider font-medium">
              {bar.label}
            </span>
            <div className="flex-1">
              <ScoreBar value={region.scores[bar.key]} color={bar.color} />
            </div>
          </div>
        ))}
      </div>

      {/* Product counts */}
      {Object.keys(region.productCounts).length > 0 && (
        <div className="flex flex-wrap gap-x-2 gap-y-0.5 px-0.5">
          {Object.entries(region.productCounts).map(([type, count]) => (
            <span key={type} className="text-[9px] text-[#5a6a8a] font-mono">
              {type.replace("_", " ")}: <span className="text-[#8a9abc]">{count}</span>
            </span>
          ))}
        </div>
      )}

      {/* Tags */}
      {region.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {region.tags.slice(0, 5).map((tag) => (
            <span
              key={tag}
              className="px-1.5 py-0.5 rounded text-[9px] text-[#7b8daf] bg-[#1a2333] border border-[#232f48]"
            >
              {tag}
            </span>
          ))}
          {region.tags.length > 5 && (
            <span className="px-1.5 py-0.5 rounded text-[9px] text-[#4a5a78]">
              +{region.tags.length - 5}
            </span>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 mt-auto pt-1">
        <button
          onClick={onExplore}
          className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold
                     bg-sky-500/15 text-sky-400 border border-sky-500/30 hover:bg-sky-500/25 transition-colors"
          aria-label={`Explore ${region.displayName} on map`}
        >
          <span className="material-symbols-outlined text-sm">explore</span>
          Explore
        </button>
        <button
          onClick={onReport}
          className="flex items-center justify-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold
                     text-[#92a4c9] border border-[#2d3a54] hover:bg-[#1a2333] hover:border-[#3d4f6f] transition-colors"
          aria-label={`Run report for ${region.displayName}`}
        >
          <span className="material-symbols-outlined text-sm">assessment</span>
          Report
        </button>
      </div>
    </article>
  );
}

/* =========================================================
 * List row
 * =======================================================*/

function ListRow({
  region,
  onExplore,
  onReport,
}: {
  region: RegionCard;
  onExplore: () => void;
  onReport: () => void;
}) {
  const compositeTextClass = scoreColor(region.scores.composite);

  return (
    <tr
      className="group border-b border-[#1a2333] hover:bg-[#111827]/70 transition-colors"
      tabIndex={0}
      role="row"
      aria-label={`${region.displayName} — composite score ${region.scores.composite ?? "N/A"}`}
    >
      <td className="py-2.5 px-3">
        <div className="font-medium text-sm text-white truncate max-w-[180px]">{region.displayName}</div>
        <div className="text-[10px] text-[#4a5a78] mt-0.5">
          {centerLat(region).toFixed(1)}, {centerLon(region).toFixed(1)}
        </div>
      </td>
      <td className="py-2.5 px-2">
        <span className={`text-base font-bold tabular-nums ${compositeTextClass}`}>
          {region.scores.composite !== null ? region.scores.composite : "---"}
        </span>
      </td>
      {SCORE_BARS.map((bar) => (
        <td key={bar.key} className="py-2.5 px-2">
          <div className="w-20">
            <ScoreBar value={region.scores[bar.key]} color={bar.color} />
          </div>
        </td>
      ))}
      <td className="py-2.5 px-2">
        <div className="flex flex-wrap gap-1 max-w-[150px]">
          {region.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="px-1 py-0.5 rounded text-[9px] text-[#7b8daf] bg-[#1a2333]">{tag}</span>
          ))}
        </div>
      </td>
      <td className="py-2.5 px-2">
        <div className="flex gap-1">
          <button
            onClick={onExplore}
            className="p-1.5 rounded-md text-sky-400 hover:bg-sky-500/15 transition-colors"
            aria-label={`Explore ${region.displayName}`}
            title="Explore on map"
          >
            <span className="material-symbols-outlined text-base">explore</span>
          </button>
          <button
            onClick={onReport}
            className="p-1.5 rounded-md text-[#92a4c9] hover:bg-[#1a2333] transition-colors"
            aria-label={`Run report for ${region.displayName}`}
            title="Run report"
          >
            <span className="material-symbols-outlined text-base">assessment</span>
          </button>
        </div>
      </td>
    </tr>
  );
}

/* =========================================================
 * Main Component
 * =======================================================*/

export default function RegionDashboard({
  isOpen,
  onClose,
  onFlyTo,
  onRunReport,
}: RegionDashboardProps) {
  /* --- State ---*/
  const [regionData, setRegionData] = useState<RegionRaw[]>([]);
  const [scoresMap, setScoresMap] = useState<Record<string, RegionScoresWithComposite> | null>(null);
  const [scoresLoading, setScoresLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("composite");
  const [sortAsc, setSortAsc] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [activeTags, setActiveTags] = useState<Set<string>>(new Set());
  const [animateIn, setAnimateIn] = useState(false);

  const overlayRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  /* --- Fetch regions on mount ---*/
  useEffect(() => {
    if (!isOpen) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch("/api/proximity/regions")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<RegionRaw[]>;
      })
      .then((data) => {
        if (cancelled) return;
        setRegionData(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  /* --- Fetch real scores ---*/
  useEffect(() => {
    if (!isOpen) return;

    let cancelled = false;
    setScoresLoading(true);

    fetch("/api/regions/scores")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<RegionScoresResponse>;
      })
      .then((data) => {
        if (cancelled) return;
        setScoresMap(data.scores);
        setScoresLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setScoresLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  /* --- Build region cards from raw data + scores ---*/
  const regions: RegionCard[] = useMemo(() => {
    return regionData.map((r) => {
      const entry = scoresMap?.[r.region_id] as RegionScoreEntry | undefined;
      return {
        regionId: r.region_id,
        displayName: r.display_name,
        description: r.description,
        latMin: r.lat_min,
        latMax: r.lat_max,
        lonMin: r.lon_min,
        lonMax: r.lon_max,
        tags: r.tags,
        scores: entry ?? EMPTY_SCORES,
        productCounts: entry?.details?.product_counts ?? {},
      };
    });
  }, [regionData, scoresMap]);

  /* --- Animate in ---*/
  useEffect(() => {
    if (isOpen) {
      // Trigger animation after a paint
      const frame = requestAnimationFrame(() => setAnimateIn(true));
      return () => cancelAnimationFrame(frame);
    } else {
      setAnimateIn(false);
    }
  }, [isOpen]);

  /* --- Focus search on open ---*/
  useEffect(() => {
    if (isOpen && !loading) {
      const timer = setTimeout(() => searchRef.current?.focus(), 150);
      return () => clearTimeout(timer);
    }
  }, [isOpen, loading]);

  /* --- Keyboard: Escape to close ---*/
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  /* --- All tags ---*/
  const allTags = useMemo(() => collectTags(regions), [regions]);

  /* --- Filtered and sorted ---*/
  const filteredRegions = useMemo(() => {
    let result = regions;

    // Text search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (r) =>
          r.displayName.toLowerCase().includes(q) ||
          r.description.toLowerCase().includes(q) ||
          r.regionId.includes(q),
      );
    }

    // Tag filter
    if (activeTags.size > 0) {
      result = result.filter((r) => r.tags.some((t) => activeTags.has(t)));
    }

    // Sort (null-safe)
    const sorted = [...result].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "composite":
          cmp = nullSafeCompare(a.scores.composite, b.scores.composite);
          break;
        case "name":
          cmp = a.displayName.localeCompare(b.displayName);
          break;
        case "latitude":
          cmp = centerLat(a) - centerLat(b);
          break;
        case "ice":
          cmp = nullSafeCompare(a.scores.ice, b.scores.ice);
          break;
        case "slope":
          cmp = nullSafeCompare(a.scores.slope, b.scores.slope);
          break;
      }
      return sortAsc ? cmp : -cmp;
    });

    return sorted;
  }, [regions, searchQuery, activeTags, sortKey, sortAsc]);

  /* --- Tag toggle ---*/
  const toggleTag = useCallback((tag: string) => {
    setActiveTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) {
        next.delete(tag);
      } else {
        next.add(tag);
      }
      return next;
    });
  }, []);

  /* --- Sort toggle (click same key to reverse) ---*/
  const handleSort = useCallback(
    (key: SortKey) => {
      if (key === sortKey) {
        setSortAsc((prev) => !prev);
      } else {
        setSortKey(key);
        // Default descending for scores, ascending for name/latitude
        setSortAsc(key === "name" || key === "latitude");
      }
    },
    [sortKey],
  );

  /* --- Composite avg for footer (null-safe) ---*/
  const avgComposite = useMemo(() => {
    const withScore = filteredRegions.filter((r) => r.scores.composite !== null);
    if (withScore.length === 0) return null;
    return Math.round(
      withScore.reduce((acc, r) => acc + (r.scores.composite as number), 0) / withScore.length,
    );
  }, [filteredRegions]);

  /* --- Early return if closed ---*/
  if (!isOpen) return null;

  return (
    <div
      ref={overlayRef}
      className={`fixed inset-0 z-[100] flex flex-col bg-[#0a0e17]/95 backdrop-blur-sm
        transition-all duration-300 ${animateIn ? "opacity-100" : "opacity-0"}`}
      role="dialog"
      aria-modal="true"
      aria-label="Mars Region Explorer"
    >
      {/* ===== Header ===== */}
      <header className="shrink-0 border-b border-[#1e293b] bg-[#0d1117]/80 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto px-4 md:px-6 py-4">
          {/* Title row */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-sky-400 text-2xl">public</span>
              <div>
                <h1 className="text-lg font-bold text-white leading-tight">Mars Region Explorer</h1>
                <p className="text-xs text-[#6b7c9c]">
                  {filteredRegions.length} of {regions.length} regions
                  {activeTags.size > 0 && ` \u00B7 ${activeTags.size} tag filter${activeTags.size > 1 ? "s" : ""}`}
                  {scoresLoading && " \u00B7 Computing scores..."}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-[#1a2333] transition-colors text-[#6b7c9c] hover:text-white"
              aria-label="Close dashboard"
            >
              <span className="material-symbols-outlined text-xl">close</span>
            </button>
          </div>

          {/* Scores computing banner */}
          {scoresLoading && (
            <div className="flex items-center gap-2 px-3 py-2 mb-3 rounded-lg bg-sky-500/10 border border-sky-500/20">
              <span className="material-symbols-outlined text-sm text-sky-400 animate-spin">progress_activity</span>
              <span className="text-xs text-sky-300">Computing real scores from DEM + instrument data... (first time only)</span>
            </div>
          )}

          {/* Controls row */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Search */}
            <div className="relative flex-1 min-w-[200px] max-w-md">
              <span className="material-symbols-outlined absolute left-2.5 top-1/2 -translate-y-1/2 text-[#4a5a78] text-base">
                search
              </span>
              <input
                ref={searchRef}
                type="text"
                placeholder="Search regions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-[#111827] border border-[#232f48] text-sm text-white
                           placeholder:text-[#4a5a78] focus:border-sky-500/50 focus:outline-none focus:ring-1 focus:ring-sky-500/20
                           transition-colors"
                aria-label="Search regions by name"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-[#4a5a78] hover:text-white transition-colors"
                  aria-label="Clear search"
                >
                  <span className="material-symbols-outlined text-sm">close</span>
                </button>
              )}
            </div>

            {/* Sort select */}
            <div className="flex items-center gap-1">
              <select
                value={sortKey}
                onChange={(e) => handleSort(e.target.value as SortKey)}
                className="px-2.5 py-1.5 rounded-lg bg-[#111827] border border-[#232f48] text-xs text-[#92a4c9]
                           focus:border-sky-500/50 focus:outline-none cursor-pointer appearance-none pr-7"
                aria-label="Sort regions by"
              >
                {SORT_OPTIONS.map((opt) => (
                  <option key={opt.key} value={opt.key}>
                    Sort: {opt.label}
                  </option>
                ))}
              </select>
              <button
                onClick={() => setSortAsc((prev) => !prev)}
                className="p-1.5 rounded-lg bg-[#111827] border border-[#232f48] text-[#6b7c9c] hover:text-white hover:border-[#3d4f6f] transition-colors"
                title={sortAsc ? "Ascending" : "Descending"}
                aria-label={`Sort direction: ${sortAsc ? "ascending" : "descending"}`}
              >
                <span className="material-symbols-outlined text-base">
                  {sortAsc ? "arrow_upward" : "arrow_downward"}
                </span>
              </button>
            </div>

            {/* View toggle */}
            <div className="flex items-center bg-[#111827] border border-[#232f48] rounded-lg p-0.5">
              <button
                onClick={() => setViewMode("grid")}
                className={`p-1.5 rounded-md transition-colors ${
                  viewMode === "grid"
                    ? "bg-sky-500/20 text-sky-400"
                    : "text-[#6b7c9c] hover:text-white"
                }`}
                title="Grid view"
                aria-label="Grid view"
                aria-pressed={viewMode === "grid"}
              >
                <span className="material-symbols-outlined text-base">grid_view</span>
              </button>
              <button
                onClick={() => setViewMode("list")}
                className={`p-1.5 rounded-md transition-colors ${
                  viewMode === "list"
                    ? "bg-sky-500/20 text-sky-400"
                    : "text-[#6b7c9c] hover:text-white"
                }`}
                title="List view"
                aria-label="List view"
                aria-pressed={viewMode === "list"}
              >
                <span className="material-symbols-outlined text-base">view_list</span>
              </button>
            </div>
          </div>

          {/* Tag filters */}
          {allTags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3" role="group" aria-label="Filter by tags">
              {activeTags.size > 0 && (
                <button
                  onClick={() => setActiveTags(new Set())}
                  className="px-2 py-0.5 rounded-full text-[10px] font-medium text-red-400 bg-red-500/10 border border-red-500/30 hover:bg-red-500/20 transition-colors"
                >
                  Clear filters
                </button>
              )}
              {allTags.map((tag) => (
                <TagChip
                  key={tag}
                  tag={tag}
                  active={activeTags.has(tag)}
                  onClick={() => toggleTag(tag)}
                />
              ))}
            </div>
          )}
        </div>
      </header>

      {/* ===== Content ===== */}
      <main className="flex-1 overflow-y-auto scrollbar-dark">
        <div className="max-w-[1600px] mx-auto px-4 md:px-6 py-4">
          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-24 text-[#6b7c9c]">
              <span className="material-symbols-outlined text-3xl animate-spin mb-3">progress_activity</span>
              <p className="text-sm">Loading regions...</p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex flex-col items-center justify-center py-24 text-red-400">
              <span className="material-symbols-outlined text-3xl mb-3">error</span>
              <p className="text-sm">Failed to load regions: {error}</p>
              <button
                onClick={() => window.location.reload()}
                className="mt-3 px-4 py-1.5 rounded-lg text-xs bg-red-500/15 border border-red-500/30 hover:bg-red-500/25 transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {/* Empty state */}
          {!loading && !error && filteredRegions.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-[#6b7c9c]">
              <span className="material-symbols-outlined text-3xl mb-3">search_off</span>
              <p className="text-sm">No regions match your filters.</p>
              <button
                onClick={() => {
                  setSearchQuery("");
                  setActiveTags(new Set());
                }}
                className="mt-3 px-4 py-1.5 rounded-lg text-xs text-sky-400 bg-sky-500/15 border border-sky-500/30 hover:bg-sky-500/25 transition-colors"
              >
                Clear all filters
              </button>
            </div>
          )}

          {/* Grid view */}
          {!loading && !error && filteredRegions.length > 0 && viewMode === "grid" && (
            <div
              className="grid gap-4"
              style={{ gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))" }}
              role="list"
            >
              {filteredRegions.map((region) => (
                <div key={region.regionId} role="listitem">
                  <GridCard
                    region={region}
                    onExplore={() => {
                      onFlyTo(centerLat(region), centerLon(region));
                      onClose();
                    }}
                    onReport={() => onRunReport(region.regionId)}
                  />
                </div>
              ))}
            </div>
          )}

          {/* List view */}
          {!loading && !error && filteredRegions.length > 0 && viewMode === "list" && (
            <div className="overflow-x-auto rounded-xl border border-[#232f48]">
              <table className="w-full text-left" role="table">
                <thead>
                  <tr className="border-b border-[#232f48] bg-[#111827]/50">
                    <th className="py-2 px-3 text-[10px] font-bold uppercase tracking-wider text-[#6b7c9c]">Region</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-[#6b7c9c]">Score</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-cyan-500/70">Ice</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-green-500/70">Slope</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-violet-500/70">SHARAD</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-amber-500/70">Cov</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-[#6b7c9c]">Tags</th>
                    <th className="py-2 px-2 text-[10px] font-bold uppercase tracking-wider text-[#6b7c9c]">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRegions.map((region) => (
                    <ListRow
                      key={region.regionId}
                      region={region}
                      onExplore={() => {
                        onFlyTo(centerLat(region), centerLon(region));
                        onClose();
                      }}
                      onReport={() => onRunReport(region.regionId)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>

      {/* ===== Footer stats ===== */}
      {!loading && !error && regions.length > 0 && (
        <footer className="shrink-0 border-t border-[#1e293b] bg-[#0d1117]/80 backdrop-blur-md">
          <div className="max-w-[1600px] mx-auto px-4 md:px-6 py-2 flex items-center justify-between text-[10px] text-[#4a5a78]">
            <span>
              Showing {filteredRegions.length} of {regions.length} regions
            </span>
            <span>
              Avg composite: {avgComposite !== null ? `${avgComposite} / 100` : "N/A"}
            </span>
          </div>
        </footer>
      )}
    </div>
  );
}
