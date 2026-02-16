import { useState, useCallback, useRef, useEffect } from "react";

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

type ScanPhase =
  | "idle"
  | "starting"
  | "detecting_craters"
  | "craters_done"
  | "detecting_channels"
  | "channels_done"
  | "detecting_ridges"
  | "ridges_done"
  | "detecting_ldas"
  | "ldas_done"
  | "complete"
  | "error";

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
  scanCenter?: { lat: number; lon: number } | null;
  onClose: () => void;
  onFlyTo?: (lat: number, lon: number) => void;
  onSearchHiRISE?: (lat: number, lon: number) => void;
  onSearchSHARAD?: (lat: number, lon: number) => void;
  onFeaturesChanged?: (features: DetectedFeature[]) => void;
  onRunEpsilonInversion?: (feature: DetectedFeature) => void;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function CraterDetectPanel({
  scanCenter,
  onClose,
  onFlyTo,
  onSearchHiRISE,
  onSearchSHARAD,
  onFeaturesChanged,
  onRunEpsilonInversion,
}: CraterDetectPanelProps) {
  // Scan config
  const [radiusKm, setRadiusKm] = useState(100);
  const [detectCraters, setDetectCraters] = useState(true);
  const [detectLdas, setDetectLdas] = useState(true);
  const [detectChannels, setDetectChannels] = useState(true);
  const [detectRidges, setDetectRidges] = useState(true);

  // Scan state
  const [features, setFeatures] = useState<DetectedFeature[]>([]);
  const [phase, setPhase] = useState<ScanPhase>("idle");
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  // Filters
  const [filterType, setFilterType] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"confidence" | "size" | "type">("confidence");

  // Notify parent of feature changes
  useEffect(() => {
    onFeaturesChanged?.(features);
  }, [features, onFeaturesChanged]);

  // Scan execution
  const startScan = useCallback(async () => {
    if (!scanCenter) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("starting");
    setFeatures([]);
    setError(null);
    setProgress(0);
    setStatusMessage("Connecting...");
    setElapsed(0);

    try {
      const res = await fetch("/api/mola-detect/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: scanCenter.lat,
          lon: scanCenter.lon,
          radius_km: radiusKm,
          detect_craters: detectCraters,
          detect_ldas: detectLdas,
          detect_channels: detectChannels,
          detect_ridges: detectRidges,
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              handleSSE(currentEvent, data);
            } catch {
              /* skip malformed */
            }
          }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== "AbortError") {
        setPhase("error");
        setError(e.message);
      }
    }
  }, [scanCenter, radiusKm, detectCraters, detectLdas, detectChannels, detectRidges]);

  const handleSSE = useCallback((event: string, data: Record<string, unknown>) => {
    if (event === "status") {
      setPhase(data.phase as ScanPhase);
      if (typeof data.progress === "number") setProgress(data.progress);
      if (typeof data.message === "string") setStatusMessage(data.message);
    } else if (event === "feature") {
      setFeatures((prev) => [...prev, data as unknown as DetectedFeature]);
    } else if (event === "complete") {
      setPhase("complete");
      setProgress(1);
      if (typeof data.elapsed_s === "number") setElapsed(data.elapsed_s);
      if (typeof data.message === "string") setStatusMessage(data.message);
    } else if (event === "error") {
      setPhase("error");
      setError(data.message as string);
    }
  }, []);

  const stopScan = useCallback(() => {
    abortRef.current?.abort();
    setPhase("idle");
    setStatusMessage("Scan cancelled");
  }, []);

  const clearResults = useCallback(() => {
    setFeatures([]);
    setPhase("idle");
    setProgress(0);
    setStatusMessage("");
    setError(null);
  }, []);

  // Computed
  const isScanning = phase !== "idle" && phase !== "complete" && phase !== "error";
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
        {/* Scan Config */}
        <div className="space-y-2">
          <div className="text-[10px] font-semibold text-[#6b7c9c] uppercase tracking-wider">
            Scan Region
          </div>

          {/* Center coordinates */}
          <div className="flex gap-2">
            <div className="flex-1 bg-[#1a2333] rounded px-2 py-1.5 border border-[#232f48]">
              <span className="text-[9px] text-[#6b7c9c]">Lat</span>
              <div className="text-[11px] font-mono">
                {scanCenter ? scanCenter.lat.toFixed(3) + "°" : "Click map..."}
              </div>
            </div>
            <div className="flex-1 bg-[#1a2333] rounded px-2 py-1.5 border border-[#232f48]">
              <span className="text-[9px] text-[#6b7c9c]">Lon</span>
              <div className="text-[11px] font-mono">
                {scanCenter ? scanCenter.lon.toFixed(3) + "°" : "—"}
              </div>
            </div>
          </div>

          {/* Radius slider */}
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-[#6b7c9c]">
              <span>Scan Radius</span>
              <span className="text-[#c8d4e8] font-mono">{radiusKm} km</span>
            </div>
            <input
              type="range"
              min={10}
              max={500}
              step={10}
              value={radiusKm}
              onChange={(e) => setRadiusKm(Number(e.target.value))}
              className="w-full h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer accent-rose-500"
              disabled={isScanning}
            />
            <div className="flex justify-between text-[9px] text-[#4a5568]">
              <span>10 km</span>
              <span>500 km</span>
            </div>
          </div>

          {/* Feature type toggles */}
          <div className="space-y-1">
            <div className="text-[10px] text-[#6b7c9c]">Detect</div>
            <div className="grid grid-cols-2 gap-1">
              {[
                { label: "Craters", state: detectCraters, set: setDetectCraters, color: "#fb923c" },
                { label: "LDAs", state: detectLdas, set: setDetectLdas, color: "#22d3ee" },
                { label: "Channels", state: detectChannels, set: setDetectChannels, color: "#3b82f6" },
                { label: "Ridges", state: detectRidges, set: setDetectRidges, color: "#eab308" },
              ].map((t) => (
                <label
                  key={t.label}
                  className="flex items-center gap-1.5 cursor-pointer select-none"
                >
                  <input
                    type="checkbox"
                    checked={t.state}
                    onChange={() => t.set(!t.state)}
                    disabled={isScanning}
                    className="rounded border-[#232f48] bg-[#1a2333] text-rose-500 focus:ring-rose-500/20 w-3 h-3"
                  />
                  <span className="text-[10px]" style={{ color: t.color }}>
                    {t.label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Scan button */}
          {isScanning ? (
            <button
              onClick={stopScan}
              className="w-full py-1.5 rounded text-[11px] font-medium bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
            >
              <span className="material-symbols-outlined text-xs mr-1 align-middle">stop</span>
              Stop Scan
            </button>
          ) : (
            <button
              onClick={startScan}
              disabled={!scanCenter}
              className={`w-full py-1.5 rounded text-[11px] font-medium transition-colors ${
                scanCenter
                  ? "bg-rose-500/20 text-rose-400 border border-rose-500/40 hover:bg-rose-500/30"
                  : "bg-[#1a2333] text-[#4a5568] border border-[#232f48] cursor-not-allowed"
              }`}
            >
              <span className="material-symbols-outlined text-xs mr-1 align-middle">radar</span>
              {scanCenter ? "Scan Region" : "Click map to set center"}
            </button>
          )}
        </div>

        {/* Progress */}
        {isScanning && (
          <div className="space-y-1.5 bg-[#1a2333] rounded p-2 border border-[#232f48]">
            <div className="flex items-center gap-2">
              <div className="animate-spin w-3 h-3 border-2 border-rose-500/30 border-t-rose-500 rounded-full" />
              <span className="text-[10px] text-rose-400">{statusMessage}</span>
            </div>
            <div className="h-1 bg-[#232f48] rounded overflow-hidden">
              <div
                className="h-full bg-rose-500 transition-all duration-500"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            {features.length > 0 && (
              <div className="text-[9px] text-[#6b7c9c]">
                {features.length} features found so far
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {phase === "error" && error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded p-2 text-[10px] text-red-400">
            <span className="material-symbols-outlined text-xs mr-1 align-middle">error</span>
            {error}
          </div>
        )}

        {/* Results Summary */}
        {features.length > 0 && (
          <>
            {/* Summary cards */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="text-[10px] font-semibold text-[#6b7c9c] uppercase tracking-wider">
                  Results
                  {phase === "complete" && (
                    <span className="ml-1 text-[#4a5568]">({elapsed}s)</span>
                  )}
                </div>
                <button
                  onClick={clearResults}
                  className="text-[9px] text-[#6b7c9c] hover:text-red-400 transition-colors"
                >
                  Clear
                </button>
              </div>

              <div className="flex flex-wrap gap-1">
                {Object.entries(typeCounts).map(([type, count]) => (
                  <div
                    key={type}
                    className="flex items-center gap-1 px-1.5 py-0.5 rounded border"
                    style={{
                      borderColor: (FEATURE_COLORS[type] || "#6b7c9c") + "40",
                      backgroundColor: (FEATURE_COLORS[type] || "#6b7c9c") + "10",
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
                  </div>
                ))}
              </div>
            </div>

            {/* Filters */}
            <div className="flex items-center gap-2">
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="flex-1 bg-[#1a2333] border border-[#232f48] rounded px-2 py-1 text-[10px] text-[#c8d4e8]"
              >
                <option value="all">All types ({features.length})</option>
                {Object.entries(typeCounts).map(([type, count]) => (
                  <option key={type} value={type}>
                    {FEATURE_LABELS[type] || type} ({count})
                  </option>
                ))}
              </select>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                className="bg-[#1a2333] border border-[#232f48] rounded px-2 py-1 text-[10px] text-[#c8d4e8]"
              >
                <option value="confidence">Confidence</option>
                <option value="size">Size</option>
                <option value="type">Type</option>
              </select>
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
                />
              ))}
            </div>
          </>
        )}

        {/* Empty state */}
        {phase === "idle" && features.length === 0 && (
          <div className="text-center py-6 text-[#4a5568] space-y-2">
            <span className="material-symbols-outlined text-3xl">radar</span>
            <p className="text-[11px]">
              Click on the map to set scan center, then click "Scan Region" to
              detect landforms.
            </p>
            <p className="text-[9px]">
              Detects craters, terraced craters, volcanic constructs, graben,
              channels, wrinkle ridges, and LDAs from MOLA DEM (200m/px).
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
}: {
  feature: DetectedFeature;
  onFlyTo?: (lat: number, lon: number) => void;
  onSearchHiRISE?: (lat: number, lon: number) => void;
  onSearchSHARAD?: (lat: number, lon: number) => void;
  onRunEpsilonInversion?: (feature: DetectedFeature) => void;
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
            {f.lat.toFixed(3)}°, {f.lon.toFixed(3)}°
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
          <div className="flex gap-1 pt-0.5">
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
                Run ε Inversion
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
