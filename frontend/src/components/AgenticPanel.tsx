import { useState, useRef, useEffect, useCallback } from "react";

/* =========================================================
 * Types
 * =======================================================*/
type StepData = {
  id: string;
  type: string;
  description: string;
  instrument: string | null;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  result_summary: string | null;
  error: string | null;
};

type AgentEvent = {
  event: string;
  data: Record<string, unknown>;
};

type SessionState = "idle" | "connecting" | "planning" | "executing" | "synthesizing" | "done" | "error";

type SessionSummary = {
  session_id: string;
  objective: string;
  status: string;
  region_name: string | null;
  created_at: string;
  overall_score: number | null;
};

const STEP_ICONS: Record<string, string> = {
  search: "travel_explore",
  check_data: "inventory_2",
  download: "cloud_download",
  slope: "terrain",
  subsurface: "radar",
  mineral: "science",
  synthesize: "analytics",
  recommend: "pin_drop",
};

const STATUS_COLORS: Record<string, string> = {
  pending: "text-[#6b7c9c]",
  running: "text-amber-400",
  completed: "text-emerald-400",
  failed: "text-red-400",
  skipped: "text-[#6b7c9c]",
};

type ChatEntry =
  | { type: "thought"; text: string; iteration: number; streaming: boolean }
  | { type: "action"; tool: string; params: Record<string, unknown>; iteration: number }
  | { type: "observation"; tool: string; text: string; success: boolean; iteration: number }
  | { type: "finish"; summary: string; recommendation: string };

const TOOL_ICONS: Record<string, string> = {
  resolve_region: "location_on",
  search_products: "travel_explore",
  check_local_data: "inventory_2",
  download_products: "cloud_download",
  analyze_slope: "terrain",
  analyze_subsurface: "radar",
  analyze_minerals: "science",
  classify_minerals_cnn: "neurology",
  estimate_dielectric: "electric_bolt",
  recommend_site: "pin_drop",
  finish: "task_alt",
};

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/* =========================================================
 * Component
 * =======================================================*/
export default function AgenticPanel({
  onClose,
}: {
  onClose: () => void;
}) {
  const [objective, setObjective] = useState("");
  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [steps, setSteps] = useState<StepData[]>([]);
  const [narrative, setNarrative] = useState("");
  const [synthesis, setSynthesis] = useState<Record<string, unknown> | null>(null);
  const [planInfo, setPlanInfo] = useState<{ region: string; instruments: string[]; planned_by: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [autoDownload, setAutoDownload] = useState(true);
  const [ollamaStatus, setOllamaStatus] = useState<boolean | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const reasoningRef = useRef<HTMLDivElement>(null);

  // Resizable panel width (wider default than Inspector's 384px)
  const [panelWidth, setPanelWidth] = useState(560);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.65);
      setPanelWidth(Math.max(360, Math.min(maxW, startW + delta)));
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

  // Reasoning state
  const [reasoningText, setReasoningText] = useState("");
  const [reasoningPhase, setReasoningPhase] = useState<string | null>(null);
  const [reasoningCollapsed, setReasoningCollapsed] = useState(false);

  // Download progress state
  const [downloadProgress, setDownloadProgress] = useState<{ completed: number; failed: number; skipped: number; total: number } | null>(null);
  const downloadStartRef = useRef<number | null>(null);

  // History panel state
  const [showHistory, setShowHistory] = useState(false);
  const [pastSessions, setPastSessions] = useState<SessionSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Time tracking state
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [completedSteps, setCompletedSteps] = useState(0);

  // ReAct mode state
  const [chatLog, setChatLog] = useState<ChatEntry[]>([]);
  const [agentMode, setAgentMode] = useState<"react" | "rules" | null>(null);
  const [iterationCount, setIterationCount] = useState(0);
  const narrativePhaseRef = useRef(false);

  // Check Ollama status on mount
  useEffect(() => {
    fetch("/api/agent/status")
      .then((r) => r.json())
      .then((d) => setOllamaStatus(d.ollama_available))
      .catch(() => setOllamaStatus(false));
  }, []);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [steps, narrative, reasoningText, chatLog]);

  // Auto-scroll reasoning panel
  useEffect(() => {
    if (reasoningRef.current && reasoningPhase) {
      reasoningRef.current.scrollTop = reasoningRef.current.scrollHeight;
    }
  }, [reasoningText, reasoningPhase]);

  // Elapsed time ticker
  useEffect(() => {
    if (!startTime) return;
    const isActive = sessionState !== "idle" && sessionState !== "done" && sessionState !== "error";
    if (!isActive) return;

    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime, sessionState]);

  // ── Shared SSE event processor ──────────────────────────
  const processEvent = useCallback((event: AgentEvent) => {
    switch (event.event) {
      case "session_start":
        setSessionId((event.data.session_id as string) || null);
        if (event.data.mode === "react") {
          setAgentMode("react");
          setSessionState("executing");
        }
        break;
      case "reasoning_start":
        setReasoningPhase(event.data.phase as string);
        setReasoningText("");
        setReasoningCollapsed(false);
        break;
      case "reasoning_chunk":
        setReasoningText((prev) => prev + (event.data.text as string));
        break;
      case "reasoning_end":
        setReasoningPhase(null);
        setReasoningCollapsed(true);
        break;
      case "plan":
        setPlanInfo({
          region: (event.data.region as string) || "Unknown",
          instruments: (event.data.instruments as string[]) || [],
          planned_by: (event.data.planned_by as string) || "rules",
        });
        setSteps((event.data.steps as StepData[]) || []);
        setTotalSteps(((event.data.steps as StepData[]) || []).length);
        setSessionState("executing");
        break;
      case "step_start":
        setSteps((prev) =>
          prev.map((s) =>
            s.id === (event.data as StepData).id ? { ...s, status: "running" } : s
          )
        );
        if ((event.data as StepData).type === "download") {
          setDownloadProgress(null);
          downloadStartRef.current = Date.now();
        }
        break;
      case "download_progress":
        setDownloadProgress(event.data as { completed: number; failed: number; skipped: number; total: number });
        break;
      case "step_complete":
      case "step_failed": {
        const updated = event.data as StepData;
        setSteps((prev) =>
          prev.map((s) => (s.id === updated.id ? updated : s))
        );
        setCompletedSteps((prev) => prev + 1);
        if (updated.type === "download") {
          setDownloadProgress(null);
          downloadStartRef.current = null;
        }
        break;
      }
      case "narrative":
        setNarrative((event.data.narrative as string) || "");
        setSessionState("synthesizing");
        break;
      case "done":
        setSynthesis((event.data.synthesis as Record<string, unknown>) || null);
        setSessionState("done");
        break;
      case "error":
        setError((event.data.error as string) || "Unknown error");
        setSessionState("error");
        break;

      // ── ReAct events ──
      case "thought_start": {
        const iter = event.data.iteration as number;
        if (event.data.phase === "narrative") {
          narrativePhaseRef.current = true;
          setReasoningPhase("narrative");
          setReasoningText("");
          setReasoningCollapsed(false);
        } else {
          narrativePhaseRef.current = false;
          setIterationCount(iter);
          setChatLog(prev => [...prev, {
            type: "thought" as const,
            text: "",
            iteration: iter,
            streaming: true,
          }]);
        }
        break;
      }
      case "thought_chunk":
        if (narrativePhaseRef.current) {
          setReasoningText(prev => prev + (event.data.text as string));
        } else {
          setChatLog(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.type === "thought") {
              updated[updated.length - 1] = {
                ...last,
                text: last.text + (event.data.text as string),
              };
            }
            return updated;
          });
        }
        break;
      case "thought_end":
        if (event.data.phase === "narrative") {
          narrativePhaseRef.current = false;
          setReasoningPhase(null);
          setReasoningCollapsed(true);
        } else {
          setChatLog(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.type === "thought") {
              updated[updated.length - 1] = { ...last, streaming: false };
            }
            return updated;
          });
        }
        break;
      case "action_start":
        setChatLog(prev => [...prev, {
          type: "action" as const,
          tool: event.data.tool as string,
          params: (event.data.params as Record<string, unknown>) || {},
          iteration: event.data.iteration as number,
        }]);
        if ((event.data.tool as string) === "download_products") {
          setDownloadProgress(null);
          downloadStartRef.current = Date.now();
        }
        break;
      case "action_complete":
        setChatLog(prev => [...prev, {
          type: "observation" as const,
          tool: event.data.tool as string,
          text: event.data.observation as string,
          success: event.data.success as boolean,
          iteration: event.data.iteration as number,
        }]);
        setCompletedSteps(prev => prev + 1);
        if ((event.data.tool as string) === "download_products") {
          setDownloadProgress(null);
          downloadStartRef.current = null;
        }
        break;
      case "action": {
        const actionTool = event.data.tool as string;
        if (actionTool === "finish") {
          const finishParams = (event.data.params as Record<string, unknown>) || {};
          setChatLog(prev => [...prev, {
            type: "finish" as const,
            summary: (finishParams.summary as string) || "",
            recommendation: (finishParams.recommendation as string) || "",
          }]);
        }
        break;
      }
    }
  }, []);

  // ── Shared SSE stream consumer ─────────────────────────
  const consumeSSEStream = useCallback(async (res: Response) => {
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
            const event: AgentEvent = JSON.parse(line.slice(6));
            processEvent(event);
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }

    if (buffer.startsWith("data: ")) {
      try {
        const event: AgentEvent = JSON.parse(buffer.slice(6));
        processEvent(event);
      } catch {
        // Skip
      }
    }
  }, [processEvent]);

  // ── Handlers ───────────────────────────────────────────
  const handleRun = useCallback(async () => {
    if (!objective.trim()) return;

    setSessionState("connecting");
    setSteps([]);
    setNarrative("");
    setSynthesis(null);
    setPlanInfo(null);
    setError(null);
    setReasoningText("");
    setReasoningPhase(null);
    setReasoningCollapsed(false);
    setChatLog([]);
    setAgentMode(null);
    setIterationCount(0);
    narrativePhaseRef.current = false;
    setStartTime(Date.now());
    setElapsed(0);
    setTotalSteps(0);
    setCompletedSteps(0);

    try {
      // Start the agent via POST — runs in background on server
      const res = await fetch("/api/agent/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective, auto_download: autoDownload }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to start agent");
      }

      setSessionState("planning");
      await consumeSSEStream(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setSessionState("error");
    }
  }, [objective, autoDownload, consumeSSEStream]);

  const handleOpenHistory = useCallback(async () => {
    setShowHistory(true);
    setHistoryLoading(true);
    try {
      const res = await fetch("/api/agent/sessions");
      if (res.ok) {
        setPastSessions(await res.json());
      }
    } catch {
      // empty list is fine
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const loadSessionFromJson = useCallback((data: Record<string, unknown>) => {
    const loadedSteps: StepData[] = (data.steps as Record<string, unknown>[] || []).map((s: Record<string, unknown>) => ({
      id: s.id as string,
      type: s.type as string,
      description: s.description as string,
      instrument: (s.instrument as string) || null,
      status: (s.status as string) || "completed",
      result_summary: (s.result_summary as string) || null,
      error: (s.error as string) || null,
    }));

    setSteps(loadedSteps);
    setTotalSteps(loadedSteps.length);
    setCompletedSteps(loadedSteps.filter(s => s.status === "completed" || s.status === "failed").length);
    setObjective((data.objective as string) || "");
    setNarrative((data.narrative as string) || "");
    setSynthesis((data.synthesis as Record<string, unknown>) || null);
    setPlanInfo({
      region: (data.region_name as string) || "Unknown",
      instruments: ((data.synthesis as Record<string, unknown>)?.instruments_searched as string[]) || [],
      planned_by: "restored",
    });

    if (data.status === "error" || data.error) {
      setError((data.error as string) || "Session ended with error");
      setSessionState("error");
    } else {
      setSessionState("done");
    }
  }, []);

  const handleLoadSession = useCallback(async (sid: string, status?: string, sessionObjective?: string) => {
    setShowHistory(false);
    setSessionState("connecting");
    setSteps([]);
    setNarrative("");
    setSynthesis(null);
    setPlanInfo(null);
    setError(null);
    setReasoningText("");
    setReasoningPhase(null);
    setChatLog([]);
    setAgentMode(null);
    setIterationCount(0);
    narrativePhaseRef.current = false;
    setStartTime(null);
    setElapsed(0);
    setTotalSteps(0);
    setCompletedSteps(0);
    setSessionId(sid);
    // Set objective immediately from session list data
    if (sessionObjective) setObjective(sessionObjective);

    try {
      // Always try JSON endpoint first — works for completed + stale sessions
      const jsonRes = await fetch(`/api/agent/session/${sid}`);
      if (!jsonRes.ok) throw new Error("Failed to load session");
      const data = await jsonRes.json();

      const effectiveStatus = data.status as string;
      if (effectiveStatus === "done" || effectiveStatus === "error") {
        // Session is finished — populate from JSON snapshot
        loadSessionFromJson(data);
      } else {
        // Truly in-progress — use SSE resume to stream live events
        setObjective((data.objective as string) || sessionObjective || "");
        const res = await fetch(`/api/agent/resume/${sid}`);
        if (!res.ok) throw new Error("Failed to resume session");

        setSessionState("planning");
        setStartTime(Date.now());
        await consumeSSEStream(res);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load session");
      setSessionState("error");
    }
  }, [consumeSSEStream, loadSessionFromJson]);

  const handleStop = useCallback(async () => {
    if (!sessionId) return;
    try {
      await fetch(`/api/agent/stop/${sessionId}`, { method: "POST" });
    } catch { /* SSE stream will deliver the error event */ }
  }, [sessionId]);

  const isRunning = sessionState !== "idle" && sessionState !== "done" && sessionState !== "error";

  // Estimated time remaining
  const estRemaining = agentMode === "react"
    ? (iterationCount > 2 && elapsed > 0
      ? Math.round((elapsed / iterationCount) * (15 - iterationCount))
      : null)
    : (completedSteps > 0 && totalSteps > 0
      ? Math.round((elapsed / completedSteps) * (totalSteps - completedSteps))
      : null);
  const progressPct = agentMode === "react"
    ? Math.round((iterationCount / 15) * 100)
    : (totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0);

  return (
    <div className="relative flex flex-col h-full bg-[#0d1520] text-[#c8d6e5] border-l border-[#232f48]" style={{ width: panelWidth, minWidth: panelWidth }}>
      {/* Resize handle (left edge — drag to widen/narrow) */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/50 transition-colors"
        onMouseDown={handleResizeStart}
      />
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-lg text-violet-400">smart_toy</span>
          <h2 className="text-sm font-bold text-white tracking-wide">Agentic AI</h2>
          {ollamaStatus !== null && (
            <span className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
              ollamaStatus ? "bg-emerald-500/20 text-emerald-400" : "bg-amber-500/20 text-amber-400"
            }`}>
              {ollamaStatus ? "LLAMA" : "RULES"}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleOpenHistory}
            disabled={isRunning}
            className="text-[#6b7c9c] hover:text-white transition-colors disabled:opacity-30"
            title="Past sessions"
          >
            <span className="material-symbols-outlined text-lg">history</span>
          </button>
          <button
            onClick={onClose}
            className="text-[#6b7c9c] hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* History Overlay */}
      {showHistory && (
        <div className="absolute inset-0 top-[49px] z-10 bg-[#0d1520] flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
            <span className="text-xs font-bold text-[#92a4c9] uppercase tracking-widest">Past Sessions</span>
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
                <span className="material-symbols-outlined text-sm text-violet-400 animate-spin">progress_activity</span>
              </div>
            )}
            {!historyLoading && pastSessions.length === 0 && (
              <div className="text-center py-8 text-xs text-[#6b7c9c]">No past sessions found.</div>
            )}
            {!historyLoading && pastSessions.map((s) => (
              <button
                key={s.session_id}
                onClick={() => handleLoadSession(s.session_id, s.status, s.objective)}
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
                <div className="text-[11px] text-white font-medium leading-snug line-clamp-2">
                  {s.objective}
                </div>
                <div className="flex items-center gap-2 mt-1.5">
                  {s.region_name && (
                    <span className="text-[9px] text-[#6b7c9c]">
                      <span className="material-symbols-outlined text-[10px] align-middle mr-0.5">location_on</span>
                      {s.region_name}
                    </span>
                  )}
                  {s.overall_score !== null && (
                    <span className={`text-[9px] font-bold ${
                      s.overall_score >= 75 ? "text-emerald-400" : s.overall_score >= 55 ? "text-amber-400" : "text-red-400"
                    }`}>
                      {s.overall_score}/100
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 border-b border-[#232f48]">
        <textarea
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
          placeholder="Describe your mission objective...&#10;e.g., Plan a rover traverse in Jezero Crater to find shallow ice. Evaluate slope feasibility and subsurface ice indicators."
          className="w-full h-20 bg-[#1a2333] border border-[#232f48] rounded-lg px-3 py-2 text-xs text-[#c8d6e5] placeholder-[#4a5a7a] resize-none focus:outline-none focus:border-violet-500/50"
          disabled={isRunning}
        />
        <div className="flex items-center justify-between mt-2">
          <label className="flex items-center gap-1.5 text-[9px] text-[#6b7c9c]">
            <input
              type="checkbox"
              checked={autoDownload}
              onChange={(e) => setAutoDownload(e.target.checked)}
              disabled={isRunning}
              className="rounded border-[#232f48]"
            />
            Auto-download missing data
          </label>
          {isRunning ? (
            <button
              onClick={handleStop}
              className="px-4 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/20"
            >
              <span className="flex items-center gap-1.5">
                <span className="material-symbols-outlined text-sm">stop_circle</span>
                Stop
              </span>
            </button>
          ) : (
            <button
              onClick={handleRun}
              disabled={!objective.trim()}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
                !objective.trim()
                  ? "bg-[#1a2333] text-[#4a5a7a] cursor-not-allowed"
                  : "bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-600/20"
              }`}
            >
              Run Agent
            </button>
          )}
        </div>
      </div>

      {/* Time Bar (only for live sessions, not restored past ones) */}
      {(isRunning || sessionState === "done") && startTime !== null && (
        <div className="px-4 py-2 border-b border-[#232f48] bg-[#0d1520]">
          <div className="flex items-center justify-between text-[9px] mb-1">
            <span className="text-[#92a4c9] font-mono">
              <span className="material-symbols-outlined text-[10px] align-middle mr-0.5">timer</span>
              {formatTime(elapsed)}
            </span>
            <span className="text-[#6b7c9c] font-mono">
              {sessionState === "done"
                ? "Complete"
                : sessionState === "planning" || sessionState === "connecting"
                ? "Planning..."
                : sessionState === "synthesizing"
                ? "Synthesizing..."
                : agentMode === "react"
                ? `Iteration ${iterationCount}/15`
                : estRemaining !== null
                ? `~${formatTime(estRemaining)} remaining`
                : "Estimating..."}
            </span>
          </div>
          <div className="h-1 bg-[#232f48] rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                sessionState === "done" ? "bg-emerald-500" : "bg-violet-500"
              }`}
              style={{ width: `${sessionState === "done" ? 100 : Math.max(progressPct, isRunning ? 3 : 0)}%` }}
            />
          </div>
        </div>
      )}

      {/* Results Area */}
      <div ref={logRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Reasoning Panel (Planning phase) */}
        {reasoningText && sessionState !== "synthesizing" && sessionState !== "done" && !narrative && (
          <div className="bg-[#1a2333] border border-violet-500/20 rounded-lg overflow-hidden">
            <button
              onClick={() => setReasoningCollapsed((p) => !p)}
              className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-[#232f48]/50 transition-colors"
            >
              <span className={`material-symbols-outlined text-sm text-violet-400 ${reasoningPhase ? "animate-pulse" : ""}`}>
                psychology
              </span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-violet-400">
                Llama Reasoning
              </span>
              {reasoningPhase && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-300 font-mono animate-pulse">
                  {reasoningPhase === "planning" ? "Planning..." : "Synthesizing..."}
                </span>
              )}
              <span className="material-symbols-outlined text-sm text-[#6b7c9c] ml-auto">
                {reasoningCollapsed ? "expand_more" : "expand_less"}
              </span>
            </button>
            {!reasoningCollapsed && (
              <div
                ref={reasoningRef}
                className="px-3 pb-3 max-h-48 overflow-y-auto"
              >
                <pre className="text-[9px] text-[#92a4c9] font-mono whitespace-pre-wrap leading-relaxed">
                  {reasoningText}
                  {reasoningPhase && <span className="animate-pulse text-violet-400">|</span>}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Plan Info */}
        {planInfo && (
          <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-sm text-violet-400">map</span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-violet-400">Execution Plan</span>
              <span className="text-[8px] px-1.5 py-0.5 rounded bg-[#232f48] text-[#6b7c9c] font-mono ml-auto">
                {planInfo.planned_by}
              </span>
            </div>
            <div className="text-[10px] text-[#92a4c9] space-y-1">
              <div>Region: <span className="text-white font-medium">{planInfo.region}</span></div>
              <div className="flex flex-wrap gap-1">
                {planInfo.instruments.map((inst) => (
                  <span key={inst} className="px-1.5 py-0.5 bg-[#232f48] rounded text-[8px] font-mono text-sky-400">
                    {inst}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ReAct Chat Log */}
        {agentMode === "react" && chatLog.length > 0 && (
          <div className="space-y-2">
            {chatLog.map((entry, i) => {
              if (entry.type === "thought") {
                return (
                  <div key={i} className="bg-violet-500/5 border border-violet-500/20 rounded-lg p-3">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className={`material-symbols-outlined text-xs text-violet-400 ${entry.streaming ? "animate-pulse" : ""}`}>
                        psychology
                      </span>
                      <span className="text-[9px] font-bold uppercase tracking-widest text-violet-400">
                        Thought {entry.iteration}
                      </span>
                    </div>
                    <pre className="text-[10px] text-[#c8d6e5] font-mono whitespace-pre-wrap leading-relaxed">
                      {entry.text}
                      {entry.streaming && <span className="animate-pulse text-violet-400">|</span>}
                    </pre>
                  </div>
                );
              }
              if (entry.type === "action") {
                return (
                  <div key={i} className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-2.5">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="material-symbols-outlined text-xs text-amber-400">
                        {TOOL_ICONS[entry.tool] || "build"}
                      </span>
                      <span className="text-[10px] font-bold text-amber-400">{entry.tool}</span>
                      {Object.entries(entry.params).map(([k, v]) => (
                        <span key={k} className="text-[8px] px-1.5 py-0.5 bg-amber-500/10 rounded font-mono text-amber-300">
                          {k}: {String(v)}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              }
              if (entry.type === "observation") {
                return (
                  <div key={i} className={`border rounded-lg p-2.5 ${
                    entry.success
                      ? "bg-emerald-500/5 border-emerald-500/20"
                      : "bg-red-500/5 border-red-500/20"
                  }`}>
                    <div className="flex items-start gap-1.5">
                      <span className={`material-symbols-outlined text-xs mt-0.5 ${
                        entry.success ? "text-emerald-400" : "text-red-400"
                      }`}>
                        {entry.success ? "check_circle" : "error"}
                      </span>
                      <span className="text-[10px] text-[#c8d6e5] leading-relaxed">{entry.text}</span>
                    </div>
                  </div>
                );
              }
              if (entry.type === "finish") {
                return (
                  <div key={i} className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="material-symbols-outlined text-xs text-emerald-400">task_alt</span>
                      <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">Agent Finished</span>
                      {entry.recommendation && (
                        <span className="text-[8px] px-1.5 py-0.5 bg-emerald-500/20 rounded font-mono text-emerald-300 ml-auto">
                          {entry.recommendation.replace(/_/g, " ")}
                        </span>
                      )}
                    </div>
                    {entry.summary && (
                      <div className="text-[10px] text-[#c8d6e5] leading-relaxed">{entry.summary}</div>
                    )}
                  </div>
                );
              }
              return null;
            })}

            {/* Download progress (during download action) */}
            {downloadProgress && (() => {
              const { completed, failed: f, skipped: sk, total } = downloadProgress;
              const done = completed + f + sk;
              const realDownloads = total - sk;
              const dlElapsed = downloadStartRef.current ? (Date.now() - downloadStartRef.current) / 1000 : 0;
              const rate = completed > 0 && dlElapsed > 0 ? dlElapsed / completed : 0;
              const remaining = rate > 0 ? Math.round(rate * (realDownloads - completed - f)) : null;
              return (
                <div className="bg-sky-500/5 border border-sky-500/20 rounded-lg p-2.5">
                  <div className="flex items-center justify-between text-[9px] mb-1">
                    <span className="text-sky-400 font-mono">
                      <span className="material-symbols-outlined text-[10px] align-middle mr-0.5">cloud_download</span>
                      {completed}/{realDownloads} downloaded
                      {f > 0 && <span className="text-red-400 ml-1">({f} failed)</span>}
                    </span>
                    <span className="text-[#6b7c9c] font-mono">
                      {remaining !== null && remaining > 0 ? `~${formatTime(remaining)} left` : done === total ? "Finishing..." : "Estimating..."}
                    </span>
                  </div>
                  <div className="h-1 bg-[#232f48] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-sky-500 rounded-full transition-all duration-300"
                      style={{ width: `${realDownloads > 0 ? Math.round((done / total) * 100) : 0}%` }}
                    />
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* Steps (rules-based mode) */}
        {agentMode !== "react" && steps.length > 0 && (
          <div className="space-y-1.5">
            {steps.map((step) => (
              <div
                key={step.id}
                className={`flex items-start gap-2 px-3 py-2 rounded-lg transition-colors ${
                  step.status === "running"
                    ? "bg-amber-500/5 border border-amber-500/20"
                    : step.status === "completed"
                    ? "bg-emerald-500/5 border border-emerald-500/10"
                    : step.status === "failed"
                    ? "bg-red-500/5 border border-red-500/20"
                    : "bg-[#1a2333]/50 border border-transparent"
                }`}
              >
                {/* Icon */}
                <span className={`material-symbols-outlined text-sm mt-0.5 ${STATUS_COLORS[step.status]}`}>
                  {step.status === "running"
                    ? "progress_activity"
                    : step.status === "completed"
                    ? "check_circle"
                    : step.status === "failed"
                    ? "error"
                    : step.status === "skipped"
                    ? "skip_next"
                    : STEP_ICONS[step.type] || "task_alt"}
                </span>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] font-medium ${STATUS_COLORS[step.status]}`}>
                      {step.description}
                    </span>
                    {step.instrument && (
                      <span className="text-[8px] px-1 py-0.5 bg-[#232f48] rounded font-mono text-sky-400">
                        {step.instrument}
                      </span>
                    )}
                  </div>
                  {step.result_summary && (
                    <div className="text-[9px] text-[#6b7c9c] mt-0.5">{step.result_summary}</div>
                  )}
                  {step.error && (
                    <div className="text-[9px] text-red-400 mt-0.5">{step.error}</div>
                  )}
                  {/* Download progress sub-line */}
                  {step.type === "download" && step.status === "running" && downloadProgress && (
                    (() => {
                      const { completed, failed: f, skipped: sk, total } = downloadProgress;
                      const done = completed + f + sk;
                      const realDownloads = total - sk;
                      const dlElapsed = downloadStartRef.current ? (Date.now() - downloadStartRef.current) / 1000 : 0;
                      const rate = completed > 0 && dlElapsed > 0 ? dlElapsed / completed : 0;
                      const remaining = rate > 0 ? Math.round(rate * (realDownloads - completed - f)) : null;
                      return (
                        <div className="mt-1 space-y-1">
                          <div className="flex items-center justify-between text-[9px]">
                            <span className="text-sky-400 font-mono">
                              {completed}/{realDownloads} downloaded
                              {f > 0 && <span className="text-red-400 ml-1">({f} failed)</span>}
                            </span>
                            <span className="text-[#6b7c9c] font-mono">
                              {remaining !== null && remaining > 0 ? `~${formatTime(remaining)} left` : done === total ? "Finishing..." : "Estimating..."}
                            </span>
                          </div>
                          <div className="h-1 bg-[#232f48] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-sky-500 rounded-full transition-all duration-300"
                              style={{ width: `${realDownloads > 0 ? Math.round((done / total) * 100) : 0}%` }}
                            />
                          </div>
                        </div>
                      );
                    })()
                  )}
                </div>

                {/* Spinner for running */}
                {step.status === "running" && (
                  <span className="material-symbols-outlined text-sm text-amber-400 animate-spin">
                    progress_activity
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Reasoning Panel (Narrative phase) */}
        {reasoningText && (sessionState === "synthesizing" || (narrative && reasoningPhase === null)) && (
          <div className="bg-[#1a2333] border border-violet-500/20 rounded-lg overflow-hidden">
            <button
              onClick={() => setReasoningCollapsed((p) => !p)}
              className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-[#232f48]/50 transition-colors"
            >
              <span className={`material-symbols-outlined text-sm text-violet-400 ${reasoningPhase ? "animate-pulse" : ""}`}>
                psychology
              </span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-violet-400">
                Llama Reasoning
              </span>
              {reasoningPhase && (
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-300 font-mono animate-pulse">
                  Synthesizing...
                </span>
              )}
              <span className="material-symbols-outlined text-sm text-[#6b7c9c] ml-auto">
                {reasoningCollapsed ? "expand_more" : "expand_less"}
              </span>
            </button>
            {!reasoningCollapsed && (
              <div
                ref={reasoningRef}
                className="px-3 pb-3 max-h-48 overflow-y-auto"
              >
                <pre className="text-[9px] text-[#92a4c9] font-mono whitespace-pre-wrap leading-relaxed">
                  {reasoningText}
                  {reasoningPhase && <span className="animate-pulse text-violet-400">|</span>}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Synthesis Score */}
        {synthesis && (
          <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-3">
              <span className="material-symbols-outlined text-sm text-emerald-400">analytics</span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">Assessment</span>
            </div>

            {/* Score Bar */}
            <div className="mb-3">
              <div className="flex items-end justify-between mb-1">
                <span className="text-[10px] text-[#92a4c9]">Overall Score</span>
                <span className="text-lg font-bold text-white">
                  {(synthesis.score_range as any)?.low !== undefined
                    ? `${(synthesis.score_range as any).low}–${(synthesis.score_range as any).high}`
                    : synthesis.overall_score as number}/100
                </span>
              </div>
              <div className="h-2 bg-[#232f48] rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${
                    (synthesis.overall_score as number) >= 75
                      ? "bg-emerald-500"
                      : (synthesis.overall_score as number) >= 55
                      ? "bg-amber-500"
                      : "bg-red-500"
                  }`}
                  style={{ width: `${synthesis.overall_score as number}%` }}
                />
              </div>
            </div>

            {/* Recommendation badge */}
            <div className="flex items-center gap-2">
              <span className={`text-[10px] px-2 py-1 rounded font-bold uppercase ${
                synthesis.recommendation === "STRONG_CANDIDATE"
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : synthesis.recommendation === "PROMISING_WITH_CAVEATS"
                  ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                  : synthesis.recommendation === "REQUIRES_FURTHER_INVESTIGATION"
                  ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                  : "bg-red-500/20 text-red-400 border border-red-500/30"
              }`}>
                {(synthesis.recommendation as string || "").replace("_", " ")}
              </span>
              <span className="text-[9px] text-[#6b7c9c]">
                {(synthesis.total_products_found as number) || 0} products across {(synthesis.instruments_searched as string[])?.length || 0} instruments
              </span>
            </div>

            {/* Sub-scores */}
            <div className="mt-3 grid grid-cols-2 xl:grid-cols-4 gap-2">
              {/* Engineering */}
              {synthesis.engineering_feasibility && (
                <div className="bg-[#0d1520] rounded p-2">
                  <div className="text-[8px] uppercase text-[#6b7c9c] mb-1">Engineering</div>
                  <div className={`text-xs font-bold ${
                    (synthesis.engineering_feasibility as Record<string, unknown>).safety === "FAVORABLE"
                      ? "text-emerald-400"
                      : (synthesis.engineering_feasibility as Record<string, unknown>).safety === "MARGINAL"
                      ? "text-amber-400"
                      : "text-red-400"
                  }`}>
                    {(synthesis.engineering_feasibility as Record<string, unknown>).safety as string}
                  </div>
                  <div className="text-[8px] text-[#6b7c9c]">
                    Mean: {(synthesis.engineering_feasibility as Record<string, unknown>).mean_slope as number}deg
                  </div>
                </div>
              )}

              {/* Subsurface */}
              {synthesis.subsurface_coverage && (
                <div className="bg-[#0d1520] rounded p-2">
                  <div className="text-[8px] uppercase text-[#6b7c9c] mb-1">Subsurface</div>
                  <div className={`text-xs font-bold ${
                    (synthesis.subsurface_coverage as Record<string, unknown>).coverage === "EXCELLENT"
                      ? "text-emerald-400"
                      : (synthesis.subsurface_coverage as Record<string, unknown>).coverage === "GOOD"
                      ? "text-sky-400"
                      : "text-amber-400"
                  }`}>
                    {(synthesis.subsurface_coverage as Record<string, unknown>).coverage as string}
                  </div>
                  <div className="text-[8px] text-[#6b7c9c]">
                    {(synthesis.subsurface_coverage as Record<string, unknown>).total_tracks as number} tracks
                  </div>
                </div>
              )}

              {/* CRISM Ice */}
              {synthesis.mineral_signatures && (
                <div className="bg-[#0d1520] rounded p-2">
                  <div className="text-[8px] uppercase text-[#6b7c9c] mb-1">Ice Signatures</div>
                  <div className="text-xs font-bold text-cyan-400">
                    {(synthesis.mineral_signatures as Record<string, unknown>).high_ice_count as number} products
                  </div>
                  <div className="text-[8px] text-[#6b7c9c]">
                    of {(synthesis.mineral_signatures as Record<string, unknown>).crism_count as number} CRISM
                  </div>
                </div>
              )}

              {/* Data Coverage */}
              <div className="bg-[#0d1520] rounded p-2">
                <div className="text-[8px] uppercase text-[#6b7c9c] mb-1">Data Coverage</div>
                <div className="text-xs font-bold text-sky-400">
                  {(synthesis.total_products_found as number) || 0} found
                </div>
                <div className="text-[8px] text-[#6b7c9c]">
                  {(synthesis.total_available_locally as number) || 0} local
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Narrative Report */}
        {narrative && (
          <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-sm text-violet-400">description</span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-violet-400">Analysis Report</span>
            </div>
            <div
              className="text-[10px] text-[#c8d6e5] leading-relaxed prose prose-invert prose-xs max-w-none
                         [&_h2]:text-xs [&_h2]:font-bold [&_h2]:text-white [&_h2]:mt-3 [&_h2]:mb-1
                         [&_h3]:text-[11px] [&_h3]:font-bold [&_h3]:text-[#92a4c9] [&_h3]:mt-2 [&_h3]:mb-1
                         [&_ul]:pl-4 [&_li]:text-[10px] [&_strong]:text-white [&_p]:mb-1.5"
              dangerouslySetInnerHTML={{ __html: markdownToHtml(narrative) }}
            />
          </div>
        )}

        {/* Download Report */}
        {sessionState === "done" && sessionId && (
          <div className="bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-sm text-sky-400">download</span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-sky-400">Download Report</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => window.open(`/api/agent/report/${sessionId}?format=md`, "_blank")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-[10px] font-bold uppercase tracking-wider transition-colors"
              >
                <span className="material-symbols-outlined text-sm">description</span>
                Markdown
              </button>
              <button
                onClick={() => window.open(`/api/agent/report/${sessionId}?format=pdf`, "_blank")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-[10px] font-bold uppercase tracking-wider transition-colors"
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
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-sm text-red-400">error</span>
              <span className="text-xs text-red-400 font-medium">{error}</span>
            </div>
          </div>
        )}

        {/* Idle state */}
        {sessionState === "idle" && !error && steps.length === 0 && !narrative && !synthesis && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <span className="material-symbols-outlined text-4xl text-[#232f48] mb-3">smart_toy</span>
            <p className="text-xs text-[#6b7c9c] max-w-[340px] leading-relaxed">
              Describe a mission objective and the agent will autonomously search, download, and analyze multi-instrument data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}


/* =========================================================
 * Simple markdown -> HTML converter
 * =======================================================*/
function markdownToHtml(md: string): string {
  return md
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^\- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>")
    .replace(/<\/ul>\s*<ul>/g, "")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br/>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}
