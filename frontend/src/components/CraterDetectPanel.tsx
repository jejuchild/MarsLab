import { useState, useCallback, useEffect, type MutableRefObject } from "react";

/* =========================================================
 * Types
 * =======================================================*/
export type DetectedFeature = {
  id: string;
  type: string;
  lat: number;
  lon: number;
  diameter_km?: number;
  depth_m?: number;
  depth_diameter_ratio?: number;
  circularity?: number;
  rim_elevation_m?: number;
  floor_elevation_m?: number;
  n_terraces?: number;
  morphology?: string;
  confidence: number;
  description: string;
  length_km?: number;
  width_km?: number;
  area_km2?: number;
  sinuosity?: number;
  path?: [number, number][];
  boundary?: [number, number][];
  terrace_depth_m?: number;
  terrace_ring_radii_km?: number[];
};

const FEATURE_COLORS: Record<string, string> = {
  crater: "#fb923c",
  terraced_crater: "#f43f5e",
  volcanic: "#ef4444",
  graben: "#a855f7",
  channel: "#3b82f6",
  wrinkle_ridge: "#eab308",
  lda: "#22d3ee",
};

const FEATURE_ICONS: Record<string, string> = {
  crater: "circle",
  terraced_crater: "stacks",
  volcanic: "volcano",
  graben: "horizontal_rule",
  channel: "waves",
  wrinkle_ridge: "show_chart",
  lda: "landscape",
};

const FEATURE_LABELS: Record<string, string> = {
  crater: "Crater",
  terraced_crater: "Terraced Crater",
  volcanic: "Volcanic Construct",
  graben: "Graben / Fossa",
  channel: "Channel / Valley",
  wrinkle_ridge: "Wrinkle Ridge",
  lda: "Lobate Debris Apron",
};

/* =========================================================
 * Props
 * =======================================================*/
interface CraterDetectPanelProps {
  cameraViewportRef?: MutableRefObject<{ minLat: number; maxLat: number; westLon: number; eastLon: number } | null>;
  onClose: () => void;
  onFlyTo?: (lat: number, lon: number) => void;
  onSearchHiRISE?: (lat: number, lon: number) => void;
  onSearchSHARAD?: (lat: number, lon: number) => void;
  onFeaturesChanged?: (features: DetectedFeature[]) => void;
  onRunEpsilonInversion?: (feature: DetectedFeature) => void;
  onOpenStratColumn?: (feature: DetectedFeature) => void;
  // Legacy props kept for compatibility
  scanCenter?: { lat: number; lon: number } | null;
  viewBounds?: unknown;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function CraterDetectPanel({
  cameraViewportRef,
  onClose,
  onFlyTo,
  onSearchHiRISE,
  onSearchSHARAD,
  onFeaturesChanged,
  onRunEpsilonInversion,
  onOpenStratColumn,
}: CraterDetectPanelProps) {
  // Precomputed features state — loaded on demand via "Load" button
  const [features, setFeatures] = useState<DetectedFeature[]>([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [totalMatched, setTotalMatched] = useState(0);
  const [truncated, setTruncated] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Type selection for loading
  const [loadTypes, setLoadTypes] = useState<Record<string, boolean>>({
    crater: true,
    terraced_crater: true,
    volcanic: true,
    graben: true,
    channel: true,
    wrinkle_ridge: true,
    lda: true,
  });

  // Display filters (post-load)
  const [filterType, setFilterType] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"confidence" | "size" | "type">("confidence");

  const MAX_FEATURES = 1000;

  // Check if precomputed cache is available on mount
  useEffect(() => {
    fetch("/api/mola-detect/status")
      .then((r) => r.json())
      .then((d) => {
        setAvailable(!!d.precomputed);
        setTotalCount(d.precomputed_count ?? 0);
      })
      .catch(() => {
        setAvailable(false);
      });
  }, []);

  // Notify parent of feature changes
  useEffect(() => {
    onFeaturesChanged?.(features);
  }, [features, onFeaturesChanged]);

  // Load precomputed features for current camera viewport
  const loadFeatures = useCallback(() => {
    const vp = cameraViewportRef?.current;
    if (!vp) {
      setError("Camera viewport not available. Try panning the map first.");
      return;
    }

    // Build selected types
    const selectedTypes = Object.entries(loadTypes)
      .filter(([, on]) => on)
      .map(([t]) => t);
    if (selectedTypes.length === 0) {
      setError("Select at least one landform type to load.");
      return;
    }

    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      west: String(vp.westLon),
      south: String(vp.minLat),
      east: String(vp.eastLon),
      north: String(vp.maxLat),
      types: selectedTypes.join(","),
      limit: String(MAX_FEATURES),
    });
    fetch(`/api/mola-detect/features?${params}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.features) {
          setFeatures(d.features as DetectedFeature[]);
          setTotalMatched(d.total_matched ?? d.count ?? 0);
          setTruncated(!!d.truncated);
          setLoaded(true);
        } else if (d.error) {
          setError(d.error);
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [cameraViewportRef, loadTypes]);

  // Unload — clear all features from map
  const unloadFeatures = useCallback(() => {
    setFeatures([]);
    setLoaded(false);
    setTruncated(false);
    setTotalMatched(0);
    setError(null);
  }, []);

  const toggleLoadType = useCallback((type: string) => {
    setLoadTypes((prev) => ({ ...prev, [type]: !prev[type] }));
  }, []);

  // Computed
  const filteredFeatures = features.filter(
    (f) => filterType === "all" || f.type === filterType
  );

  const sortedFeatures = [...filteredFeatures].sort((a, b) => {
    if (sortBy === "confidence") return b.confidence - a.confidence;
    if (sortBy === "size") return (b.diameter_km ?? b.area_km2 ?? b.length_km ?? 0) - (a.diameter_km ?? a.area_km2 ?? a.length_km ?? 0);
    return a.type.localeCompare(b.type);
  });

  const typeCounts = features.reduce((acc, f) => {
    acc[f.type] = (acc[f.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  /* =========================================================
   * Render
   * =======================================================*/
  return (
    <aside className="flex flex-col h-full bg-[#101622] text-[#c8d4e8] overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-[#232f48] bg-[#0d1219] shrink-0">
        <span
          className="material-symbols-outlined text-base"
          style={{ color: "#f43f5e" }}
        >
          target
        </span>
        <h2 className="text-xs font-semibold flex-1 text-[#c8d4e8]">
          Landform Detection
        </h2>
        <button
          onClick={onClose}
          className="text-[#6b7c9c] hover:text-white p-0.5 rounded transition-colors"
          title="Close"
        >
          <span className="material-symbols-outlined text-sm">close</span>
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3 text-[11px]">
        {/* Unavailable warning */}
        {available === false && (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded p-2 text-[10px] text-amber-400">
            <span className="material-symbols-outlined text-xs mr-1 align-middle">warning</span>
            No pre-computed landform cache available. Run the precompute script first.
          </div>
        )}

        {/* Type selection checkboxes */}
        {available !== false && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold text-[#6b7c9c] uppercase tracking-wider">
              Landform Types
            </div>
            <div className="grid grid-cols-2 gap-1">
              {Object.entries(FEATURE_LABELS).map(([type, label]) => (
                <label
                  key={type}
                  className="flex items-center gap-1.5 cursor-pointer select-none"
                >
                  <input
                    type="checkbox"
                    checked={loadTypes[type] ?? false}
                    onChange={() => toggleLoadType(type)}
                    disabled={loading}
                    className="rounded border-[#232f48] bg-[#1a2333] text-rose-500 focus:ring-rose-500/20 w-3 h-3"
                  />
                  <span className="text-[10px]" style={{ color: FEATURE_COLORS[type] }}>
                    {label}
                  </span>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Load / Unload button */}
        {available !== false && !loaded && (
          <button
            onClick={loadFeatures}
            disabled={loading || available === null}
            className={`w-full py-2 rounded text-[11px] font-medium transition-colors flex items-center justify-center gap-1.5 ${
              loading || available === null
                ? "bg-[#1a2333] text-[#4a5568] border border-[#232f48] cursor-wait"
                : "bg-rose-500/20 text-rose-400 border border-rose-500/40 hover:bg-rose-500/30"
            }`}
          >
            {loading ? (
              <>
                <div className="animate-spin w-3 h-3 border-2 border-rose-500/30 border-t-rose-500 rounded-full" />
                Loading...
              </>
            ) : available === null ? (
              <>
                <div className="animate-spin w-3 h-3 border-2 border-[#4a5568]/30 border-t-[#4a5568] rounded-full" />
                Checking cache...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-sm">download</span>
                Load Viewport Landforms
              </>
            )}
          </button>
        )}

        {/* Loaded state */}
        {loaded && (
          <div className="bg-[#1a2333] rounded p-2 border border-emerald-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="material-symbols-outlined text-xs text-emerald-400">check_circle</span>
                <span className="text-[10px] text-[#c8d4e8] font-medium">
                  {features.length.toLocaleString()} landforms loaded
                  {truncated && (
                    <span className="text-amber-400 ml-1">
                      (max {MAX_FEATURES}, {totalMatched.toLocaleString()} in viewport)
                    </span>
                  )}
                </span>
              </div>
              <button
                onClick={unloadFeatures}
                className="text-[9px] text-[#6b7c9c] hover:text-red-400 transition-colors px-1.5 py-0.5 rounded hover:bg-red-500/10"
                title="Unload landforms from map"
              >
                <span className="material-symbols-outlined text-xs align-middle mr-0.5">close</span>
                Unload
              </button>
            </div>
            <div className="flex items-center gap-3 mt-1.5">
              <button
                onClick={loadFeatures}
                disabled={loading}
                className="text-[9px] text-[#6b7c9c] hover:text-rose-400 transition-colors flex items-center gap-0.5"
                title="Reload for current viewport"
              >
                {loading ? (
                  <div className="animate-spin w-2.5 h-2.5 border border-rose-500/30 border-t-rose-500 rounded-full" />
                ) : (
                  <span className="material-symbols-outlined text-xs">refresh</span>
                )}
                Reload viewport
              </button>
            </div>
          </div>
        )}

        {/* Cache info */}
        {totalCount > 0 && !loaded && (
          <p className="text-[9px] text-[#4a5568]">
            {totalCount.toLocaleString()} pre-computed landforms in cache.
            Select types, zoom to an area, then click Load (max {MAX_FEATURES}).
          </p>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded p-2 text-[10px] text-red-400">
            <span className="material-symbols-outlined text-xs mr-1 align-middle">error</span>
            {error}
          </div>
        )}

        {/* Results Summary */}
        {features.length > 0 && (
          <>
            {/* Type breakdown badges */}
            <div className="space-y-1.5">
              <div className="text-[10px] font-semibold text-[#6b7c9c] uppercase tracking-wider">
                Results
              </div>
              <div className="flex flex-wrap gap-1">
                {Object.entries(typeCounts).map(([type, count]) => (
                  <button
                    key={type}
                    onClick={() => setFilterType(filterType === type ? "all" : type)}
                    className={`flex items-center gap-1 px-1.5 py-0.5 rounded border cursor-pointer transition-colors ${
                      filterType === type ? "ring-1 ring-white/20" : ""
                    }`}
                    style={{
                      borderColor: (FEATURE_COLORS[type] || "#6b7c9c") + "40",
                      backgroundColor: (FEATURE_COLORS[type] || "#6b7c9c") + (filterType === type ? "25" : "10"),
                    }}
                  >
                    <span
                      className="material-symbols-outlined"
                      style={{ fontSize: "10px", color: FEATURE_COLORS[type] }}
                    >
                      {FEATURE_ICONS[type] || "circle"}
                    </span>
                    <span
                      className="text-[9px] font-medium"
                      style={{ color: FEATURE_COLORS[type] }}
                    >
                      {count} {FEATURE_LABELS[type] || type}
                      {count > 1 ? "s" : ""}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Sort */}
            <div className="flex items-center gap-2">
              <span className="text-[9px] text-[#4a5568]">Sort:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                className="bg-[#1a2333] border border-[#232f48] rounded px-2 py-1 text-[10px] text-[#c8d4e8]"
              >
                <option value="confidence">Confidence</option>
                <option value="size">Size</option>
                <option value="type">Type</option>
              </select>
              {filterType !== "all" && (
                <button
                  onClick={() => setFilterType("all")}
                  className="text-[9px] text-[#6b7c9c] hover:text-white transition-colors ml-auto"
                >
                  Show all
                </button>
              )}
            </div>

            {/* Feature list */}
            <div className="space-y-1">
              {sortedFeatures.map((f) => (
                <FeatureRow
                  key={f.id}
                  feature={f}
                  onFlyTo={onFlyTo}
                  onSearchHiRISE={onSearchHiRISE}
                  onSearchSHARAD={onSearchSHARAD}
                  onRunEpsilonInversion={onRunEpsilonInversion}
                  onOpenStratColumn={onOpenStratColumn}
                />
              ))}
            </div>
          </>
        )}

        {/* Empty state */}
        {features.length === 0 && !loading && !loaded && !error && available !== false && (
          <div className="text-center py-6 text-[#4a5568] space-y-2">
            <span className="material-symbols-outlined text-3xl">radar</span>
            <p className="text-[11px]">
              Navigate to an area of interest, then click "Load Viewport Landforms" to display detected features.
            </p>
            <p className="text-[9px]">
              Includes craters, terraced craters, volcanic constructs, graben,
              channels, wrinkle ridges, and lobate debris aprons.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}

/* =========================================================
 * Feature Row
 * =======================================================*/
function FeatureRow({
  feature: f,
  onFlyTo,
  onSearchHiRISE,
  onSearchSHARAD,
  onRunEpsilonInversion,
  onOpenStratColumn,
}: {
  feature: DetectedFeature;
  onFlyTo?: (lat: number, lon: number) => void;
  onSearchHiRISE?: (lat: number, lon: number) => void;
  onSearchSHARAD?: (lat: number, lon: number) => void;
  onRunEpsilonInversion?: (feature: DetectedFeature) => void;
  onOpenStratColumn?: (feature: DetectedFeature) => void;
}) {
  const color = FEATURE_COLORS[f.type] || "#6b7c9c";
  const [expanded, setExpanded] = useState(false);

  const sizeLabel =
    f.diameter_km
      ? `${f.diameter_km.toFixed(1)} km`
      : f.area_km2
        ? `${f.area_km2.toFixed(0)} km\u00b2`
        : f.length_km
          ? `${f.length_km.toFixed(1)} km long`
          : "";

  return (
    <div
      className="bg-[#1a2333] rounded border border-[#232f48] hover:border-[#3a4a68] transition-colors"
      style={{ borderLeftColor: color, borderLeftWidth: "2px" }}
    >
      {/* Main row */}
      <div
        className="flex items-center gap-1.5 px-2 py-1.5 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <span
          className="material-symbols-outlined"
          style={{ fontSize: "12px", color }}
        >
          {FEATURE_ICONS[f.type] || "circle"}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1">
            <span className="text-[10px] font-medium text-[#c8d4e8] truncate">
              {FEATURE_LABELS[f.type] || f.type}
            </span>
            {sizeLabel && (
              <span className="text-[9px] text-[#6b7c9c] font-mono shrink-0">
                {sizeLabel}
              </span>
            )}
          </div>
          <div className="text-[9px] text-[#4a5568] font-mono">
            {f.lat.toFixed(3)}\u00b0, {f.lon.toFixed(3)}\u00b0
          </div>
        </div>
        {/* Confidence badge */}
        <div
          className="text-[8px] font-bold px-1 py-0.5 rounded shrink-0"
          style={{
            backgroundColor: color + "20",
            color,
            border: `1px solid ${color}40`,
          }}
        >
          {(f.confidence * 100).toFixed(0)}%
        </div>
        <span className="material-symbols-outlined text-[10px] text-[#4a5568]">
          {expanded ? "expand_less" : "expand_more"}
        </span>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="px-2 pb-2 space-y-1.5 border-t border-[#232f48]">
          <p className="text-[9px] text-[#6b7c9c] pt-1">{f.description}</p>

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-1 text-[9px]">
            {f.depth_m != null && f.depth_m > 0 && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">Depth</div>
                <div className="font-mono text-[#c8d4e8]">{f.depth_m.toFixed(0)} m</div>
              </div>
            )}
            {f.depth_diameter_ratio != null && f.depth_diameter_ratio > 0 && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">d/D ratio</div>
                <div className="font-mono text-[#c8d4e8]">{f.depth_diameter_ratio.toFixed(3)}</div>
              </div>
            )}
            {f.n_terraces != null && f.n_terraces > 0 && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">Terraces</div>
                <div className="font-mono text-rose-400">{f.n_terraces}</div>
              </div>
            )}
            {f.terrace_depth_m != null && f.terrace_depth_m > 0 && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">Terrace Depth</div>
                <div className="font-mono text-rose-400">{f.terrace_depth_m.toFixed(0)} m</div>
              </div>
            )}
            {f.sinuosity != null && f.sinuosity > 0 && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">Sinuosity</div>
                <div className="font-mono text-[#c8d4e8]">{f.sinuosity.toFixed(2)}</div>
              </div>
            )}
            {f.morphology && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">Morphology</div>
                <div className="font-mono text-[#c8d4e8] capitalize">{f.morphology}</div>
              </div>
            )}
            {f.circularity != null && f.circularity > 0 && (
              <div className="bg-[#0d1219] rounded px-1.5 py-1">
                <div className="text-[8px] text-[#4a5568]">Circularity</div>
                <div className="font-mono text-[#c8d4e8]">{f.circularity.toFixed(2)}</div>
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-1 pt-0.5">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onFlyTo?.(f.lat, f.lon);
              }}
              className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#0d1219] border border-[#232f48] text-[9px] text-[#92a4c9] hover:text-white hover:border-[#3a4a68] transition-colors"
              title="Fly to location"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "10px" }}>
                flight
              </span>
              Fly To
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSearchHiRISE?.(f.lat, f.lon);
              }}
              className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#0d1219] border border-[#232f48] text-[9px] text-[#92a4c9] hover:text-cyan-400 hover:border-cyan-500/30 transition-colors"
              title="Search HiRISE data"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "10px" }}>
                photo_camera
              </span>
              HiRISE
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSearchSHARAD?.(f.lat, f.lon);
              }}
              className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#0d1219] border border-[#232f48] text-[9px] text-[#92a4c9] hover:text-purple-400 hover:border-purple-500/30 transition-colors"
              title="Search SHARAD data"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "10px" }}>
                radar
              </span>
              SHARAD
            </button>
            {f.type === "terraced_crater" && f.terrace_depth_m != null && f.terrace_depth_m > 0 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRunEpsilonInversion?.(f);
                }}
                className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-rose-500/10 border border-rose-500/30 text-[9px] text-rose-400 hover:text-rose-300 hover:border-rose-500/50 hover:bg-rose-500/20 transition-colors"
                title="Run dielectric inversion using terrace depth + SHARAD"
              >
                <span className="material-symbols-outlined" style={{ fontSize: "10px" }}>
                  science
                </span>
                Run \u03b5 Inversion
              </button>
            )}
            {(f.type === "crater" || f.type === "terraced_crater") && f.diameter_km != null && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onOpenStratColumn?.(f);
                }}
                className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-[9px] text-emerald-400 hover:text-emerald-300 hover:border-emerald-500/50 hover:bg-emerald-500/20 transition-colors"
                title="Build composite stratigraphic column from HiRISE + CRISM + SHARAD"
              >
                <span className="material-symbols-outlined" style={{ fontSize: "10px" }}>
                  view_column
                </span>
                Strat Column
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
