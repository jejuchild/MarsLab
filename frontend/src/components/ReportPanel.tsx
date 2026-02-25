import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import DOMPurify from 'dompurify';

/* =========================================================
 * Types
 * =======================================================*/
type RegionInfo = {
  region_id: string;
  display_name: string;
  center_lat: number;
  center_lon: number;
  tags: string[];
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
};


type StepData = {
  id: string;
  type: string;
  description: string;
  instrument: string | null;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  result_summary: string | null;
  error: string | null;
};

type RegionProgress = {
  region_id: string;
  region_name: string;
  status: "pending" | "running" | "completed" | "failed";
  steps: StepData[];
  score: number | null;
  recommendation: string | null;
  passes_ground_rules: boolean;
  ground_rule_violations: string[];
};

type RankingEntry = {
  rank: number;
  region_id: string;
  region_name: string;
  overall_score: number;
  scores: { engineering: number; subsurface: number; ice: number; coverage: number };
  recommendation: string;
  highlights: string[];
  passes_ground_rules: boolean;
  ground_rule_violations: string[];
  engineering_safety: string;
  subsurface_coverage: string;
  subsurface_detections: number;
  ice_count: number;
  total_products: number;
};

type ReportEvent = {
  event: string;
  data: Record<string, unknown>;
};

type ReportSessionSummary = {
  session_id: string;
  status: string;
  region_count: number;
  region_names: string[];
  created_at: string;
  recommended: { region_name: string; score: number } | null;
};

type ViewState = "config" | "progress" | "report";
type SessionState = "idle" | "filtering" | "analyzing" | "comparing" | "generating" | "done" | "error";

const STATUS_COLORS: Record<string, string> = {
  pending: "text-[#6b7c9c]",
  running: "text-amber-400",
  completed: "text-emerald-400",
  failed: "text-red-400",
  skipped: "text-[#6b7c9c]",
};

const SCORE_MAX = { engineering: 30, subsurface: 25, ice: 25, coverage: 20 };

const ALL_TAGS = ["crater", "volcanic", "polar", "plains", "canyon", "channel", "basin", "ice", "landing_site", "geological"];

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function ReportPanel({
  onClose,
  isMobile = false,
}: {
  onClose: () => void;
  isMobile?: boolean;
}) {
  // View management
  const [view, setView] = useState<ViewState>("config");
  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Panel width (resizable like AgenticPanel)
  const [panelWidth, setPanelWidth] = useState(620);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.7);
      setPanelWidth(Math.max(400, Math.min(maxW, startW + delta)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [panelWidth]);

  // ── Region data ──
  const [allRegions, setAllRegions] = useState<RegionInfo[]>([]);
  const [regionsLoading, setRegionsLoading] = useState(true);

  useEffect(() => {
    fetch("/api/report/regions")
      .then((r) => r.json())
      .then((data) => setAllRegions(data))
      .catch(() => {})
      .finally(() => setRegionsLoading(false));
  }, []);

  // ── Ground rules state ──
  const [latMin, setLatMin] = useState<number>(-50);
  const [latMax, setLatMax] = useState<number>(50);
  const [lonEnabled, setLonEnabled] = useState(false);
  const [lonMin, setLonMin] = useState<number>(-180);
  const [lonMax, setLonMax] = useState<number>(180);
  const [includeRegions, setIncludeRegions] = useState<Set<string>>(new Set());
  const [excludeTags, setExcludeTags] = useState<Set<string>>(new Set(["polar"]));
  const [includeTags, setIncludeTags] = useState<Set<string>>(new Set());
  const [minSlopeSafety, setMinSlopeSafety] = useState<string | null>(null);
  const [customNotes, setCustomNotes] = useState("");
  const [maxRegions, setMaxRegions] = useState(5);

  // Analysis options
  const [analysisSlope, setAnalysisSlope] = useState(true);
  const [analysisSubsurface, setAnalysisSubsurface] = useState(true);
  const [analysisMineral, setAnalysisMineral] = useState(true);
  const [autoDownload, setAutoDownload] = useState(true);

  // Compute matching regions
  const matchingRegions = useMemo(() => {
    return allRegions.filter((r) => {
      // When specific regions are selected, bypass lat/lon/tag filters
      if (includeRegions.size > 0) return includeRegions.has(r.region_id);
      if (r.center_lat < latMin || r.center_lat > latMax) return false;
      if (lonEnabled && (r.center_lon < lonMin || r.center_lon > lonMax)) return false;
      if (includeTags.size > 0) {
        const rTags = r.tags.map((t) => t.toLowerCase());
        if (!Array.from(includeTags).some((t) => rTags.includes(t.toLowerCase()))) return false;
      }
      if (excludeTags.size > 0) {
        const rTags = r.tags.map((t) => t.toLowerCase());
        if (Array.from(excludeTags).some((t) => rTags.includes(t.toLowerCase()))) return false;
      }
      return true;
    });
  }, [allRegions, latMin, latMax, lonEnabled, lonMin, lonMax, includeRegions, includeTags, excludeTags]);

  // ── Progress state ──
  const [regionProgress, setRegionProgress] = useState<RegionProgress[]>([]);
  const [expandedRegion, setExpandedRegion] = useState<number | null>(null);

  // ── Report state ──
  const [rankings, setRankings] = useState<RankingEntry[]>([]);
  const [categoryWinners, setCategoryWinners] = useState<Record<string, { region_name: string; value: string }>>({});
  const [recommended, setRecommended] = useState<{ region_name: string; score: number; recommendation: string; highlights: string[] } | null>(null);
  const [executiveSummary, setExecutiveSummary] = useState("");
  const [reasoningText, setReasoningText] = useState("");
  const reasoningTextRef = useRef("");
  const [reasoningPhase, setReasoningPhase] = useState<string | null>(null);

  // History panel state
  const [showHistory, setShowHistory] = useState(false);
  const [pastReports, setPastReports] = useState<ReportSessionSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Time tracking
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!startTime) return;
    const isActive = sessionState !== "idle" && sessionState !== "done" && sessionState !== "error";
    if (!isActive) return;
    const interval = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000);
    return () => clearInterval(interval);
  }, [startTime, sessionState]);

  // Auto-scroll
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [regionProgress, rankings, executiveSummary, reasoningText]);

  // ── SSE event processor ──
  const processEvent = useCallback((event: ReportEvent) => {
    switch (event.event) {
      case "session_start":
        setSessionId(event.data.session_id as string);
        break;

      case "filter_result":
        setSessionState("analyzing");
        break;

      case "region_start": {
        const idx = event.data.region_index as number;
        const name = event.data.region_name as string;
        const rid = event.data.region_id as string;
        setRegionProgress((prev) => {
          const next = [...prev];
          if (!next[idx]) {
            next[idx] = {
              region_id: rid,
              region_name: name,
              status: "running",
              steps: [],
              score: null,
              recommendation: null,
              passes_ground_rules: true,
              ground_rule_violations: [],
            };
          } else {
            next[idx] = { ...next[idx], status: "running" };
          }
          return next;
        });
        setExpandedRegion(idx);
        break;
      }

      case "region_step_start":
      case "region_step_complete":
      case "region_step_failed": {
        const idx = event.data.region_index as number;
        const step = event.data.step as StepData;
        setRegionProgress((prev) => {
          const next = [...prev];
          if (next[idx]) {
            const steps = [...next[idx].steps];
            const si = steps.findIndex((s) => s.id === step.id);
            if (si >= 0) {
              steps[si] = step;
            } else {
              steps.push(step);
            }
            next[idx] = { ...next[idx], steps };
          }
          return next;
        });
        break;
      }

      case "region_complete": {
        const idx = event.data.region_index as number;
        setRegionProgress((prev) => {
          const next = [...prev];
          if (next[idx]) {
            next[idx] = {
              ...next[idx],
              status: "completed",
              score: event.data.score as number,
              recommendation: event.data.recommendation as string,
              passes_ground_rules: (event.data.passes_ground_rules as boolean) ?? true,
              ground_rule_violations: (event.data.ground_rule_violations as string[]) || [],
            };
          }
          return next;
        });
        break;
      }

      case "comparison_complete":
        setSessionState("generating");
        setRankings((event.data.rankings as RankingEntry[]) || []);
        setCategoryWinners((event.data.category_winners as Record<string, { region_name: string; value: string }>) || {});
        setRecommended((event.data.recommended as typeof recommended) || null);
        break;

      case "reasoning_start":
        setReasoningPhase(event.data.phase as string);
        setReasoningText("");
        reasoningTextRef.current = "";
        break;

      case "reasoning_chunk": {
        const chunk = event.data.text as string;
        reasoningTextRef.current += chunk;
        setReasoningText((prev) => prev + chunk);
        break;
      }

      case "reasoning_end":
        setReasoningPhase(null);
        setExecutiveSummary(reasoningTextRef.current);
        break;

      case "report_ready":
        break;

      case "done":
        setSessionState("done");
        setView("report");
        break;

      case "error":
        setError((event.data.error as string) || "Unknown error");
        setSessionState("error");
        break;
    }
  }, []);

  // ── SSE consumer ──
  const consumeSSE = useCallback(async (res: Response) => {
    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            processEvent(JSON.parse(line.slice(6)));
          } catch { /* skip malformed */ }
        }
      }
    }

    if (buffer.startsWith("data: ")) {
      try { processEvent(JSON.parse(buffer.slice(6))); } catch { /* skip */ }
    }
  }, [processEvent]);

  // ── Generate handler ──
  const handleGenerate = useCallback(async () => {
    // Reset state
    setSessionState("filtering");
    setView("progress");
    setError(null);
    setRegionProgress([]);
    setRankings([]);
    setCategoryWinners({});
    setRecommended(null);
    setExecutiveSummary("");
    setReasoningText("");
    setStartTime(Date.now());
    setElapsed(0);

    const analyses: string[] = [];
    if (analysisSlope) analyses.push("slope");
    if (analysisSubsurface) analyses.push("subsurface");
    if (analysisMineral) analyses.push("mineral");

    const body = {
      ground_rules: {
        lat_min: latMin,
        lat_max: latMax,
        lon_min: lonEnabled ? lonMin : null,
        lon_max: lonEnabled ? lonMax : null,
        include_regions: Array.from(includeRegions),
        exclude_regions: [],
        include_tags: Array.from(includeTags),
        exclude_tags: Array.from(excludeTags),
        min_slope_safety: minSlopeSafety,
        custom_notes: customNotes,
      },
      analyses,
      auto_download: autoDownload,
      max_regions: maxRegions,
    };

    try {
      const res = await fetch("/api/report/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to start report");
      }
      await consumeSSE(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setSessionState("error");
    }
  }, [latMin, latMax, lonEnabled, lonMin, lonMax, includeRegions, includeTags, excludeTags, minSlopeSafety, customNotes, maxRegions, analysisSlope, analysisSubsurface, analysisMineral, autoDownload, consumeSSE]);

  const handleOpenHistory = useCallback(async () => {
    setShowHistory(true);
    setHistoryLoading(true);
    try {
      const res = await fetch("/api/report/sessions");
      if (res.ok) setPastReports(await res.json());
    } catch { /* empty list is fine */ }
    finally { setHistoryLoading(false); }
  }, []);

  const handleLoadSession = useCallback(async (sid: string) => {
    setShowHistory(false);
    setSessionState("filtering");
    setView("progress");
    setError(null);
    setRegionProgress([]);
    setRankings([]);
    setCategoryWinners({});
    setRecommended(null);
    setExecutiveSummary("");
    setReasoningText("");
    setStartTime(null);
    setElapsed(0);
    setSessionId(sid);

    try {
      // Use JSON polling endpoint for instant load
      const res = await fetch(`/api/report/session/${sid}`);
      if (!res.ok) throw new Error("Failed to load report session");
      const data = await res.json();

      // Populate region progress from loaded data
      const loadedRegions: RegionProgress[] = (data.regions || []).map((ra: Record<string, unknown>) => ({
        region_id: ra.region_id as string,
        region_name: ra.region_name as string,
        status: (ra.status as string) || "completed",
        steps: ((ra.steps as Record<string, unknown>[]) || []).map((s: Record<string, unknown>) => ({
          id: s.id as string,
          type: s.type as string,
          description: s.description as string,
          instrument: (s.instrument as string) || null,
          status: (s.status as string) || "completed",
          result_summary: (s.result_summary as string) || null,
          error: (s.error as string) || null,
        })),
        score: (ra.synthesis as Record<string, unknown>)?.overall_score as number ?? null,
        recommendation: (ra.synthesis as Record<string, unknown>)?.recommendation as string ?? null,
        passes_ground_rules: (ra.passes_ground_rules as boolean) ?? true,
        ground_rule_violations: (ra.ground_rule_violations as string[]) || [],
      }));
      setRegionProgress(loadedRegions);

      // Populate comparison data
      const comparison = data.comparison as Record<string, unknown> | null;
      if (comparison) {
        setRankings((comparison.rankings as RankingEntry[]) || []);
        setCategoryWinners((comparison.category_winners as Record<string, { region_name: string; value: string }>) || {});
        setRecommended((comparison.recommended as { region_name: string; score: number; recommendation: string; highlights: string[] } | null) || null);
      }

      setExecutiveSummary((data.executive_summary as string) || "");

      if (data.error) {
        setError(data.error as string);
        setSessionState("error");
      } else {
        setSessionState("done");
        setView("report");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load session");
      setSessionState("error");
    }
  }, []);

  const handleStop = useCallback(async () => {
    if (!sessionId) return;
    try {
      await fetch(`/api/report/stop/${sessionId}`, { method: "POST" });
    } catch { /* SSE stream will deliver the error event */ }
  }, [sessionId]);

  const isRunning = sessionState !== "idle" && sessionState !== "done" && sessionState !== "error";
  const completedRegions = regionProgress.filter((r) => r.status === "completed").length;
  const totalRegionsCount = regionProgress.length || 1;
  const progressPct = Math.round((completedRegions / totalRegionsCount) * 100);

  return (
    <div
      className={`relative flex flex-col h-full bg-[#0d1520] text-[#c8d6e5] ${isMobile ? "w-full" : "border-l border-[#232f48]"}`}
      style={isMobile ? undefined : { width: panelWidth, minWidth: panelWidth }}
    >
      {/* Resize handle */}
      {!isMobile && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-amber-500/30 active:bg-amber-500/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-lg text-amber-400">assignment</span>
          <h2 className="text-sm font-bold text-white tracking-wide">Landing Site Report</h2>
          {sessionState !== "idle" && (
            <span className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
              sessionState === "done" ? "bg-emerald-500/20 text-emerald-400"
              : sessionState === "error" ? "bg-red-500/20 text-red-400"
              : "bg-amber-500/20 text-amber-400"
            }`}>
              {sessionState.toUpperCase()}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {(view === "progress" || view === "report") && (
            <button
              onClick={() => { setView("config"); setSessionState("idle"); }}
              className="text-[#6b7c9c] hover:text-white transition-colors text-xs mr-2"
              title="New report"
            >
              <span className="material-symbols-outlined text-lg">add_circle</span>
            </button>
          )}
          <button
            onClick={handleOpenHistory}
            disabled={isRunning}
            className="text-[#6b7c9c] hover:text-white transition-colors disabled:opacity-30"
            title="Past reports"
          >
            <span className="material-symbols-outlined text-lg">history</span>
          </button>
          <button onClick={onClose} className="text-[#6b7c9c] hover:text-white transition-colors">
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* Time bar (when running or done) */}
      {(isRunning || sessionState === "done") && (
        <div className="px-4 py-2 border-b border-[#232f48]">
          <div className="flex items-center justify-between text-[9px] mb-1">
            <span className="text-[#92a4c9] font-mono">
              <span className="material-symbols-outlined text-[10px] align-middle mr-0.5">timer</span>
              {formatTime(elapsed)}
            </span>
            <span className="flex items-center gap-2">
              <span className="text-[#6b7c9c] font-mono">
                {sessionState === "done" ? "Complete" : `Region ${completedRegions + 1} of ${totalRegionsCount}`}
              </span>
              {isRunning && (
                <button
                  onClick={handleStop}
                  className="px-2 py-0.5 rounded text-[8px] font-bold uppercase bg-red-600 hover:bg-red-500 text-white transition-colors"
                >
                  Stop
                </button>
              )}
            </span>
          </div>
          <div className="h-1 bg-[#232f48] rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${sessionState === "done" ? "bg-emerald-500" : "bg-amber-500"}`}
              style={{ width: `${sessionState === "done" ? 100 : Math.max(progressPct, isRunning ? 3 : 0)}%` }}
            />
          </div>
        </div>
      )}

      {/* History Overlay */}
      {showHistory && (
        <div className="absolute inset-0 top-[49px] z-10 bg-[#0d1520] flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
            <span className="text-xs font-bold text-[#92a4c9] uppercase tracking-widest">Past Reports</span>
            <button
              onClick={() => setShowHistory(false)}
              className="text-[#6b7c9c] hover:text-white transition-colors"
            >
              <span className="material-symbols-outlined text-sm">close</span>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {historyLoading && (
              <div className="flex items-center justify-center py-8">
                <span className="material-symbols-outlined text-sm text-amber-400 animate-spin">progress_activity</span>
              </div>
            )}
            {!historyLoading && pastReports.length === 0 && (
              <div className="text-center py-8 text-xs text-[#6b7c9c]">No past reports found.</div>
            )}
            {!historyLoading && pastReports.map((s) => (
              <button
                key={s.session_id}
                onClick={() => handleLoadSession(s.session_id)}
                className="w-full text-left bg-[#1a2333] hover:bg-[#232f48] border border-[#232f48] rounded-lg p-3 transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-mono text-[#6b7c9c]">
                    {new Date(s.created_at).toLocaleString()}
                  </span>
                  <span className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
                    s.status === "done"
                      ? "bg-emerald-500/20 text-emerald-400"
                      : s.status === "error"
                      ? "bg-red-500/20 text-red-400"
                      : "bg-amber-500/20 text-amber-400"
                  }`}>
                    {s.status.toUpperCase()}
                  </span>
                </div>
                <div className="text-[10px] text-[#92a4c9] mb-1">
                  {s.region_count} region{s.region_count !== 1 ? "s" : ""} compared
                </div>
                <div className="flex flex-wrap gap-1 mb-1.5">
                  {s.region_names.map((name) => (
                    <span key={name} className="text-[8px] px-1.5 py-0.5 bg-[#232f48] rounded font-mono text-sky-400">
                      {name}
                    </span>
                  ))}
                </div>
                {s.recommended && (
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[10px] text-amber-400">flag</span>
                    <span className="text-[9px] text-white font-medium">{s.recommended.region_name}</span>
                    <span className={`text-[9px] font-bold ${
                      s.recommended.score >= 70 ? "text-emerald-400" : s.recommended.score >= 45 ? "text-amber-400" : "text-red-400"
                    }`}>
                      {s.recommended.score}/100
                    </span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main content area */}
      <div ref={logRef} className="flex-1 overflow-y-auto p-4 space-y-4">

        {/* ═══ VIEW 1: CONFIG ═══ */}
        {view === "config" && (
          <>
            {/* Ground Rules Section */}
            <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
              <div className="flex items-center gap-2 mb-3">
                <span className="material-symbols-outlined text-sm text-amber-400">rule</span>
                <span className="text-[10px] font-bold uppercase tracking-widest text-amber-400">Ground Rules</span>
              </div>

              {/* Latitude Range */}
              <div className="mb-3">
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Latitude Range ({latMin}° to {latMax}°)
                </label>
                <div className="flex items-center gap-3 mt-1">
                  <span className="text-[9px] text-[#6b7c9c] w-8">-90°</span>
                  <div className="flex-1 flex items-center gap-2">
                    <input
                      type="range" min={-90} max={90} value={latMin}
                      onChange={(e) => setLatMin(Math.min(Number(e.target.value), latMax - 1))}
                      className="flex-1 accent-amber-500 h-1"
                    />
                    <input
                      type="range" min={-90} max={90} value={latMax}
                      onChange={(e) => setLatMax(Math.max(Number(e.target.value), latMin + 1))}
                      className="flex-1 accent-amber-500 h-1"
                    />
                  </div>
                  <span className="text-[9px] text-[#6b7c9c] w-8 text-right">90°</span>
                </div>
              </div>

              {/* Longitude (optional) */}
              <div className="mb-3">
                <label className="flex items-center gap-1.5 text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  <input
                    type="checkbox" checked={lonEnabled}
                    onChange={(e) => setLonEnabled(e.target.checked)}
                    className="rounded border-[#232f48] accent-amber-500"
                  />
                  Longitude Filter
                </label>
                {lonEnabled && (
                  <div className="flex items-center gap-2 mt-1">
                    <input
                      type="number" value={lonMin} min={-180} max={180}
                      onChange={(e) => setLonMin(Number(e.target.value))}
                      className="w-16 bg-[#0d1520] border border-[#232f48] rounded px-2 py-1 text-[10px] text-white"
                    />
                    <span className="text-[9px] text-[#6b7c9c]">to</span>
                    <input
                      type="number" value={lonMax} min={-180} max={180}
                      onChange={(e) => setLonMax(Number(e.target.value))}
                      className="w-16 bg-[#0d1520] border border-[#232f48] rounded px-2 py-1 text-[10px] text-white"
                    />
                  </div>
                )}
              </div>

              {/* Tag Exclusions */}
              <div className="mb-3">
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Exclude Tags
                </label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {ALL_TAGS.map((tag) => (
                    <button
                      key={tag}
                      onClick={() => {
                        setExcludeTags((prev) => {
                          const next = new Set(prev);
                          if (next.has(tag)) next.delete(tag); else next.add(tag);
                          return next;
                        });
                      }}
                      className={`px-2 py-0.5 rounded text-[8px] font-mono transition-colors ${
                        excludeTags.has(tag)
                          ? "bg-red-500/20 text-red-400 border border-red-500/30"
                          : "bg-[#0d1520] text-[#6b7c9c] border border-[#232f48] hover:border-[#4a5a7a]"
                      }`}
                    >
                      {tag}
                    </button>
                  ))}
                </div>
              </div>

              {/* Include Tags */}
              <div className="mb-3">
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Required Tags (empty = any)
                </label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {ALL_TAGS.map((tag) => (
                    <button
                      key={tag}
                      onClick={() => {
                        setIncludeTags((prev) => {
                          const next = new Set(prev);
                          if (next.has(tag)) next.delete(tag); else next.add(tag);
                          return next;
                        });
                      }}
                      className={`px-2 py-0.5 rounded text-[8px] font-mono transition-colors ${
                        includeTags.has(tag)
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                          : "bg-[#0d1520] text-[#6b7c9c] border border-[#232f48] hover:border-[#4a5a7a]"
                      }`}
                    >
                      {tag}
                    </button>
                  ))}
                </div>
              </div>

              {/* Specific Regions */}
              <div className="mb-3">
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Include Only Specific Regions (empty = all matching)
                </label>
                <div className="max-h-32 overflow-y-auto mt-1 bg-[#0d1520] rounded border border-[#232f48] p-1.5">
                  {regionsLoading ? (
                    <div className="text-[9px] text-[#6b7c9c] text-center py-2">Loading regions...</div>
                  ) : (
                    allRegions.map((r) => (
                      <label
                        key={r.region_id}
                        className="flex items-center gap-1.5 px-1 py-0.5 hover:bg-[#1a2333] rounded text-[9px] cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          checked={includeRegions.has(r.region_id)}
                          onChange={() => {
                            setIncludeRegions((prev) => {
                              const next = new Set(prev);
                              if (next.has(r.region_id)) next.delete(r.region_id); else next.add(r.region_id);
                              return next;
                            });
                          }}
                          className="rounded border-[#232f48] accent-amber-500"
                        />
                        <span className="text-white">{r.display_name}</span>
                        <span className="text-[#6b7c9c] ml-auto">
                          {r.center_lat.toFixed(1)}°, {r.center_lon.toFixed(1)}°
                        </span>
                      </label>
                    ))
                  )}
                </div>
              </div>

              {/* Slope Safety */}
              <div className="mb-3">
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Min Slope Safety
                </label>
                <select
                  value={minSlopeSafety || ""}
                  onChange={(e) => setMinSlopeSafety(e.target.value || null)}
                  className="mt-1 w-full bg-[#0d1520] border border-[#232f48] rounded px-2 py-1 text-[10px] text-white"
                >
                  <option value="">Any</option>
                  <option value="MARGINAL">MARGINAL or better</option>
                  <option value="FAVORABLE">FAVORABLE only</option>
                </select>
              </div>

              {/* Max Regions */}
              <div className="mb-3">
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Max Regions to Analyze
                </label>
                <input
                  type="number" min={1} max={10} value={maxRegions}
                  onChange={(e) => setMaxRegions(Math.max(1, Math.min(10, Number(e.target.value))))}
                  className="mt-1 w-20 bg-[#0d1520] border border-[#232f48] rounded px-2 py-1 text-[10px] text-white"
                />
              </div>

              {/* Custom Notes */}
              <div>
                <label className="text-[9px] text-[#92a4c9] uppercase tracking-wider font-bold">
                  Custom Notes (optional)
                </label>
                <textarea
                  value={customNotes}
                  onChange={(e) => setCustomNotes(e.target.value)}
                  placeholder="e.g., Focus on ice-rich terrains, prioritize flat landing zones..."
                  className="mt-1 w-full h-12 bg-[#0d1520] border border-[#232f48] rounded px-2 py-1 text-[9px] text-[#c8d6e5] placeholder-[#4a5a7a] resize-none focus:outline-none focus:border-amber-500/50"
                />
              </div>
            </div>

            {/* Analysis Options */}
            <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="material-symbols-outlined text-sm text-amber-400">tune</span>
                <span className="text-[10px] font-bold uppercase tracking-widest text-amber-400">Analysis Options</span>
              </div>
              <div className="space-y-1.5">
                {[
                  { label: "Slope Analysis", desc: "Terrain engineering feasibility", checked: analysisSlope, set: setAnalysisSlope },
                  { label: "Subsurface Scan", desc: "SHARAD radar ice detection", checked: analysisSubsurface, set: setAnalysisSubsurface },
                  { label: "Mineral Analysis", desc: "CRISM ice/hydration signatures", checked: analysisMineral, set: setAnalysisMineral },
                  { label: "Auto-Download", desc: "Download missing products", checked: autoDownload, set: setAutoDownload },
                ].map((opt) => (
                  <label key={opt.label} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox" checked={opt.checked}
                      onChange={(e) => opt.set(e.target.checked)}
                      className="rounded border-[#232f48] accent-amber-500"
                    />
                    <div>
                      <span className="text-[10px] text-white">{opt.label}</span>
                      <span className="text-[8px] text-[#6b7c9c] ml-1.5">{opt.desc}</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Preview + Generate */}
            <div className="bg-[#1a2333] border border-amber-500/20 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] text-[#92a4c9]">
                  <span className="font-bold text-white">{matchingRegions.length}</span> regions match your ground rules
                  {matchingRegions.length > maxRegions && (
                    <span className="text-amber-400"> (top {maxRegions} will be analyzed)</span>
                  )}
                </span>
              </div>
              {matchingRegions.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {matchingRegions.slice(0, 8).map((r) => (
                    <span key={r.region_id} className="px-1.5 py-0.5 bg-[#0d1520] rounded text-[8px] text-sky-400 font-mono">
                      {r.display_name}
                    </span>
                  ))}
                  {matchingRegions.length > 8 && (
                    <span className="px-1.5 py-0.5 text-[8px] text-[#6b7c9c]">
                      +{matchingRegions.length - 8} more
                    </span>
                  )}
                </div>
              )}
              <button
                onClick={handleGenerate}
                disabled={matchingRegions.length === 0}
                className={`w-full py-2 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
                  matchingRegions.length === 0
                    ? "bg-[#232f48] text-[#4a5a7a] cursor-not-allowed"
                    : "bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-600/20"
                }`}
              >
                Generate Comparison Report ({Math.min(matchingRegions.length, maxRegions)} regions)
              </button>
            </div>
          </>
        )}

        {/* ═══ VIEW 2: PROGRESS ═══ */}
        {view === "progress" && (
          <>
            {regionProgress.map((rp, i) => (
              <div key={rp.region_id} className={`bg-[#1a2333] border rounded-lg overflow-hidden ${
                rp.status === "running" ? "border-amber-500/30" :
                rp.status === "completed" ? "border-emerald-500/20" :
                rp.status === "failed" ? "border-red-500/20" : "border-[#232f48]"
              }`}>
                {/* Region header */}
                <button
                  onClick={() => setExpandedRegion(expandedRegion === i ? null : i)}
                  className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-[#232f48]/50 transition-colors"
                >
                  <span className={`material-symbols-outlined text-sm ${
                    rp.status === "running" ? "text-amber-400 animate-spin" :
                    rp.status === "completed" ? "text-emerald-400" :
                    rp.status === "failed" ? "text-red-400" : "text-[#6b7c9c]"
                  }`}>
                    {rp.status === "running" ? "progress_activity" :
                     rp.status === "completed" ? "check_circle" :
                     rp.status === "failed" ? "error" : "circle"}
                  </span>
                  <span className="text-[10px] font-medium text-white flex-1">{rp.region_name}</span>
                  {rp.score !== null && (
                    <span className={`text-[10px] font-bold ${
                      rp.score >= 70 ? "text-emerald-400" : rp.score >= 45 ? "text-amber-400" : "text-red-400"
                    }`}>
                      {rp.score}/100
                    </span>
                  )}
                  {!rp.passes_ground_rules && (
                    <span className="text-[8px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 font-mono">FAIL</span>
                  )}
                  <span className="material-symbols-outlined text-sm text-[#6b7c9c]">
                    {expandedRegion === i ? "expand_less" : "expand_more"}
                  </span>
                </button>

                {/* Expanded steps */}
                {expandedRegion === i && (
                  <div className="px-3 pb-2 space-y-1">
                    {rp.steps.map((step) => (
                      <div key={step.id} className="flex items-center gap-2 py-0.5">
                        <span className={`material-symbols-outlined text-[11px] ${STATUS_COLORS[step.status]}`}>
                          {step.status === "running" ? "progress_activity" :
                           step.status === "completed" ? "check" :
                           step.status === "failed" ? "close" : "circle"}
                        </span>
                        <span className={`text-[9px] ${STATUS_COLORS[step.status]}`}>{step.description}</span>
                        {step.instrument && (
                          <span className="text-[7px] px-1 bg-[#232f48] rounded font-mono text-sky-400">{step.instrument}</span>
                        )}
                        {step.result_summary && (
                          <span className="text-[8px] text-[#6b7c9c] ml-auto truncate max-w-[200px]">{step.result_summary}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Reasoning panel */}
            {reasoningText && (
              <div className="bg-[#1a2333] border border-amber-500/20 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`material-symbols-outlined text-sm text-amber-400 ${reasoningPhase ? "animate-pulse" : ""}`}>psychology</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-amber-400">
                    {reasoningPhase ? "Writing Executive Summary..." : "Executive Summary"}
                  </span>
                </div>
                <pre className="text-[9px] text-[#92a4c9] font-mono whitespace-pre-wrap leading-relaxed max-h-32 overflow-y-auto">
                  {reasoningText}
                  {reasoningPhase && <span className="animate-pulse text-amber-400">|</span>}
                </pre>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                <span className="material-symbols-outlined text-sm text-red-400 mr-1">error</span>
                <span className="text-xs text-red-400">{error}</span>
              </div>
            )}
          </>
        )}

        {/* ═══ VIEW 3: REPORT ═══ */}
        {view === "report" && (
          <>
            {/* Executive Summary */}
            {(executiveSummary || reasoningText) && (
              <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-sm text-amber-400">summarize</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-amber-400">Executive Summary</span>
                </div>
                <div
                  className="text-[10px] text-[#c8d6e5] leading-relaxed [&_strong]:text-white [&_p]:mb-1.5"
                  dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(simpleMarkdownToHtml(executiveSummary || reasoningText)) }}
                />
              </div>
            )}

            {/* Rankings Table */}
            {rankings.length > 0 && (
              <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
                <div className="flex items-center gap-2 mb-3">
                  <span className="material-symbols-outlined text-sm text-emerald-400">leaderboard</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">Rankings</span>
                </div>
                <div className="space-y-2">
                  {rankings.map((r) => (
                    <div key={r.region_id} className={`bg-[#0d1520] rounded-lg p-2.5 ${
                      r.rank === 1 ? "border border-amber-500/30" : "border border-transparent"
                    }`}>
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className={`text-lg font-bold ${
                          r.rank === 1 ? "text-amber-400" : r.rank === 2 ? "text-[#c0c0c0]" : "text-[#6b7c9c]"
                        }`}>#{r.rank}</span>
                        <span className="text-[11px] font-medium text-white flex-1">{r.region_name}</span>
                        {!r.passes_ground_rules && (
                          <span className="text-[7px] px-1 py-0.5 rounded bg-red-500/20 text-red-400 font-mono">CONSTRAINT FAIL</span>
                        )}
                        <span className={`text-sm font-bold ${
                          r.overall_score >= 70 ? "text-emerald-400" : r.overall_score >= 45 ? "text-amber-400" : "text-red-400"
                        }`}>{r.overall_score}/100</span>
                      </div>

                      {/* Score breakdown bars */}
                      <div className="grid grid-cols-4 gap-1.5">
                        {(["engineering", "subsurface", "ice", "coverage"] as const).map((cat) => {
                          const val = r.scores[cat];
                          const max = SCORE_MAX[cat];
                          const pct = Math.round((val / max) * 100);
                          return (
                            <div key={cat}>
                              <div className="text-[7px] text-[#6b7c9c] uppercase mb-0.5">{cat}</div>
                              <div className="h-1.5 bg-[#232f48] rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    pct >= 80 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500"
                                  }`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              <div className="text-[7px] text-[#6b7c9c] mt-0.5">{val}/{max}</div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Highlights */}
                      {r.highlights.length > 0 && (
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {r.highlights.map((h, hi) => (
                            <span key={`${h}-${hi}`} className="text-[7px] px-1.5 py-0.5 bg-[#1a2333] rounded text-[#92a4c9]">{h}</span>
                          ))}
                        </div>
                      )}

                      {/* Recommendation badge */}
                      <div className="mt-1.5">
                        <span className={`text-[8px] px-1.5 py-0.5 rounded font-bold uppercase ${
                          r.recommendation === "HIGHLY_RECOMMENDED" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                          : r.recommendation === "PROMISING" ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                          : r.recommendation === "MODERATE" ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                          : "bg-red-500/20 text-red-400 border border-red-500/30"
                        }`}>
                          {r.recommendation.replace(/_/g, " ")}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Category Winners */}
            {Object.keys(categoryWinners).length > 0 && (
              <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-sm text-sky-400">emoji_events</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-sky-400">Category Winners</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(categoryWinners).map(([cat, info]) => (
                    <div key={cat} className="bg-[#0d1520] rounded p-2">
                      <div className="text-[7px] uppercase text-[#6b7c9c] mb-0.5">
                        {cat.replace("best_", "Best ").replace("_", " ")}
                      </div>
                      <div className="text-[9px] text-white font-medium">{info.region_name}</div>
                      <div className="text-[8px] text-[#6b7c9c]">{info.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended Site */}
            {recommended && (
              <div className="bg-[#1a2333] border border-amber-500/30 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-sm text-amber-400">flag</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-amber-400">Recommended Landing Site</span>
                </div>
                <div className="text-[11px] text-white font-medium">{recommended.region_name}</div>
                <div className="text-[10px] text-[#92a4c9] mt-1">
                  Score: <span className="text-white font-bold">{recommended.score}/100</span>
                  {" — "}
                  <span className="text-amber-400">{recommended.recommendation.replace(/_/g, " ")}</span>
                </div>
                {recommended.highlights.length > 0 && (
                  <div className="mt-1.5 space-y-0.5">
                    {recommended.highlights.map((h, i) => (
                      <div key={`${h}-${i}`} className="text-[9px] text-[#92a4c9]">• {h}</div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Download Buttons */}
            {sessionState === "done" && sessionId && (
              <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-sm text-sky-400">download</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-sky-400">Download Report</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => window.open(`/api/report/download/${sessionId}?format=md`, "_blank")}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-[10px] font-bold uppercase tracking-wider transition-colors"
                  >
                    <span className="material-symbols-outlined text-sm">description</span>
                    Markdown
                  </button>
                  <button
                    onClick={() => window.open(`/api/report/download/${sessionId}?format=pdf`, "_blank")}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-[10px] font-bold uppercase tracking-wider transition-colors"
                  >
                    <span className="material-symbols-outlined text-sm">picture_as_pdf</span>
                    PDF
                  </button>
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                <span className="material-symbols-outlined text-sm text-red-400 mr-1">error</span>
                <span className="text-xs text-red-400">{error}</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}


/* =========================================================
 * Simple markdown -> HTML
 * =======================================================*/
function simpleMarkdownToHtml(md: string): string {
  return md
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br/>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}
