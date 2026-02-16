/**
 * AssistantPanel — Adaptive Research Copilot (ARC) Chat UI.
 *
 * Conversational, turn-based research assistant with:
 * - Chat message bubbles (user right, copilot left)
 * - Suggestion chips below copilot messages
 * - Inline tool execution cards
 * - Action confirm/reject buttons
 * - Resizable panel with drag handle
 */
import { useState, useRef, useEffect, useCallback } from "react";
import type {
  CopilotSessionData,
  CopilotMessageData,
  CopilotAction,
  CopilotSSEEvent,
} from "../api/copilot";
import {
  createSession,
  sendMessage,
  confirmAction,
  rejectAction,
  getSession,
  listSessions,
  consumeCopilotSSE,
} from "../api/copilot";

/* =========================================================
 * Tool icons (reused from old AssistantPanel)
 * =======================================================*/
const TOOL_ICONS: Record<string, string> = {
  resolve_region: "location_on",
  search_products: "travel_explore",
  check_local_data: "inventory_2",
  download_products: "cloud_download",
  analyze_slope: "terrain",
  analyze_subsurface: "radar",
  analyze_minerals: "science",
  classify_minerals_cnn: "neurology",
  find_sharad_intersections: "compare_arrows",
  targeted_subsurface_at_ice: "ac_unit",
  run_sharad_inversion: "electric_bolt",
  detect_terraces: "stacked_line_chart",
  epsilon_inversion: "functions",
  evaluate_ice_evidence: "assessment",
  generate_report: "description",
};

/* =========================================================
 * Types for internal chat state
 * =======================================================*/
type ChatMessage = {
  id: string;
  role: "user" | "copilot" | "system";
  text: string;
  suggestions: string[];
  actions: CopilotAction[];
  flyTo: { lat: number; lon: number; height?: number } | null;
  isThinking?: boolean;
  isActionInline?: boolean;  // Tool execution status card
  isNew?: boolean;           // Attention highlight (fades after 2s)
};

/* =========================================================
 * Component
 * =======================================================*/
export default function AssistantPanel({
  onClose,
  onFlyTo,
  onHighlight,
  initialGoal,
  onPanelAttention,
}: {
  onClose: () => void;
  onFlyTo?: (lat: number, lon: number) => void;
  onHighlight?: (productIds: string[]) => void;
  initialGoal?: string | null;
  onPanelAttention?: () => void;
}) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingActions, setPendingActions] = useState<CopilotAction[]>([]);
  const [inputText, setInputText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<Record<string, unknown>[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [reportMarkdown, setReportMarkdown] = useState<string | null>(null);

  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const initialGoalFired = useRef(false);
  const msgIdCounter = useRef(0);

  // Resizable panel
  const [panelWidth, setPanelWidth] = useState(480);
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

  const nextId = () => `msg-${++msgIdCounter.current}`;

  // Auto-scroll
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [messages]);

  // ─── Initialize session (runs once) ───
  const sessionInitRef = useRef(false);
  useEffect(() => {
    if (!sessionId && !sessionInitRef.current) {
      sessionInitRef.current = true;
      createSession().then((session) => {
        setSessionId(session.session_id);
        // Load greeting message
        if (session.messages.length > 0) {
          const greeting = session.messages[0];
          setMessages([{
            id: nextId(),
            role: "copilot",
            text: greeting.text,
            suggestions: greeting.suggestions,
            actions: [],
            flyTo: null,
          }]);
        }
      }).catch((err) => {
        console.error("Failed to create copilot session:", err);
        sessionInitRef.current = false; // Allow retry on actual failure
      });
    }
  }, [sessionId]);

  // ─── Auto-send initialGoal (fires at most once per mount) ───
  useEffect(() => {
    if (initialGoal && sessionId && !initialGoalFired.current && messages.length > 0) {
      initialGoalFired.current = true;
      handleSend(initialGoal);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialGoal, sessionId, messages.length]);

  // ─── SSE event handler ───
  const handleSSEEvent = useCallback((event: CopilotSSEEvent) => {
    const d = event.data;
    switch (event.event) {
      case "thinking":
        setMessages(prev => [...prev, {
          id: nextId(),
          role: "copilot",
          text: "",
          suggestions: [],
          actions: [],
          flyTo: null,
          isThinking: true,
        }]);
        break;

      case "message":
        // Remove thinking indicator, add message
        setMessages(prev => {
          const filtered = prev.filter(m => !m.isThinking);
          return [...filtered, {
            id: nextId(),
            role: "copilot",
            text: (d.text as string) || "",
            suggestions: (d.suggestions as string[]) || [],
            actions: [],
            flyTo: (d.fly_to as { lat: number; lon: number; height?: number }) || null,
            isNew: true,
          }];
        });
        onPanelAttention?.();
        break;

      case "fly_to":
        if (onFlyTo && d.lat != null && d.lon != null) {
          onFlyTo(d.lat as number, d.lon as number);
        }
        break;

      case "action_started":
        setMessages(prev => [...prev, {
          id: nextId(),
          role: "system",
          text: `Running: ${d.description as string}`,
          suggestions: [],
          actions: [],
          flyTo: null,
          isActionInline: true,
        }]);
        break;

      case "action_completed": {
        const success = d.success as boolean;
        const summary = (d.summary as string) || "";
        const toolName = d.tool_name as string;
        // Update the last action_started message
        setMessages(prev => {
          const copy = [...prev];
          // Find the last action inline message
          for (let i = copy.length - 1; i >= 0; i--) {
            if (copy[i].isActionInline && copy[i].text.startsWith("Running:")) {
              copy[i] = {
                ...copy[i],
                text: success
                  ? `${toolName}: ${summary || "Done"}`
                  : `${toolName}: Failed`,
              };
              break;
            }
          }
          return copy;
        });
        break;
      }

      case "action_proposed": {
        const action: CopilotAction = {
          action_id: d.action_id as string,
          tool_name: d.tool_name as string,
          description: d.description as string,
          params: {},
          status: "proposed",
          requires_confirmation: true,
          result: null,
          error: null,
        };
        setPendingActions(prev => [...prev, action]);
        break;
      }

      case "state_updated":
        // Could update a context display, for now just log
        break;

      case "error":
        setMessages(prev => {
          const filtered = prev.filter(m => !m.isThinking);
          return [...filtered, {
            id: nextId(),
            role: "system",
            text: `Error: ${d.error as string}`,
            suggestions: [],
            actions: [],
            flyTo: null,
          }];
        });
        break;
    }
  }, [onFlyTo, onPanelAttention]);

  // ─── Send message ───
  const handleSend = useCallback(async (text?: string) => {
    const msg = text || inputText.trim();
    if (!msg || !sessionId || isStreaming) return;
    setInputText("");

    // Add user message
    setMessages(prev => [...prev, {
      id: nextId(),
      role: "user",
      text: msg,
      suggestions: [],
      actions: [],
      flyTo: null,
    }]);

    setIsStreaming(true);
    try {
      const response = await sendMessage(sessionId, msg);
      await consumeCopilotSSE(response, handleSSEEvent);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: nextId(),
        role: "system",
        text: `Error: ${e instanceof Error ? e.message : "Connection failed"}`,
        suggestions: [],
        actions: [],
        flyTo: null,
      }]);
    } finally {
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  }, [inputText, sessionId, isStreaming, handleSSEEvent]);

  // ─── Confirm/Reject actions ───
  const handleConfirm = useCallback(async (actionId: string) => {
    if (!sessionId) return;
    setPendingActions(prev => prev.filter(a => a.action_id !== actionId));
    setIsStreaming(true);
    try {
      const response = await confirmAction(sessionId, actionId);
      await consumeCopilotSSE(response, handleSSEEvent);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: nextId(),
        role: "system",
        text: `Error: ${e instanceof Error ? e.message : "Confirmation failed"}`,
        suggestions: [],
        actions: [],
        flyTo: null,
      }]);
    } finally {
      setIsStreaming(false);
    }
  }, [sessionId, handleSSEEvent]);

  const handleReject = useCallback(async (actionId: string) => {
    if (!sessionId) return;
    setPendingActions(prev => prev.filter(a => a.action_id !== actionId));
    try {
      await rejectAction(sessionId, actionId);
      setMessages(prev => [...prev, {
        id: nextId(),
        role: "copilot",
        text: "Understood, skipping that step.",
        suggestions: [],
        actions: [],
        flyTo: null,
      }]);
    } catch {
      // ignore
    }
  }, [sessionId]);

  // ─── Suggestion chip click ───
  const handleSuggestion = useCallback((text: string) => {
    if (!isStreaming) handleSend(text);
  }, [isStreaming, handleSend]);

  // ─── History ───
  const handleOpenHistory = useCallback(async () => {
    setShowHistory(true);
    setHistoryLoading(true);
    try {
      const sessions = await listSessions();
      setHistory(sessions);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const handleLoadSession = useCallback(async (sid: string) => {
    setShowHistory(false);
    try {
      const session = await getSession(sid);
      setSessionId(session.session_id);
      // Convert messages
      const chatMessages: ChatMessage[] = session.messages.map((m: CopilotMessageData) => ({
        id: nextId(),
        role: m.role,
        text: m.text,
        suggestions: m.suggestions,
        actions: m.actions,
        flyTo: m.fly_to,
      }));
      setMessages(chatMessages);
      setPendingActions(session.pending_actions || []);
      setReportMarkdown(session.report_markdown);
    } catch {
      setMessages(prev => [...prev, {
        id: nextId(),
        role: "system",
        text: "Failed to load session",
        suggestions: [],
        actions: [],
        flyTo: null,
      }]);
    }
  }, []);

  return (
    <div
      className="relative flex flex-col h-full bg-[#0d1520] text-[#c8d6e5] border-l border-[#232f48]"
      style={{ width: panelWidth, minWidth: panelWidth }}
    >
      {/* Resize handle */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-teal-500/30 active:bg-teal-500/50 transition-colors"
        onMouseDown={handleResizeStart}
      />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-lg text-teal-400">smart_toy</span>
          <h2 className="text-sm font-bold text-white tracking-wide">MARVIS</h2>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleOpenHistory}
            disabled={isStreaming}
            className="text-[#6b7c9c] hover:text-white transition-colors disabled:opacity-30"
            title="Past sessions"
          >
            <span className="material-symbols-outlined text-lg">history</span>
          </button>
          <button onClick={onClose} className="text-[#6b7c9c] hover:text-white transition-colors">
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* History overlay */}
      {showHistory && (
        <div className="absolute inset-0 top-[49px] z-10 bg-[#0d1520] flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48]">
            <span className="text-xs font-bold text-[#92a4c9] uppercase tracking-widest">Past Sessions</span>
            <button onClick={() => setShowHistory(false)} className="text-[#6b7c9c] hover:text-white transition-colors">
              <span className="material-symbols-outlined text-sm">close</span>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {historyLoading && (
              <div className="flex items-center justify-center py-8">
                <span className="material-symbols-outlined text-sm text-teal-400 animate-spin">progress_activity</span>
              </div>
            )}
            {!historyLoading && history.length === 0 && (
              <div className="text-center py-8 text-xs text-[#6b7c9c]">No past sessions found.</div>
            )}
            {!historyLoading && history.map((s: any) => (
              <button
                key={s.session_id}
                onClick={() => handleLoadSession(s.session_id)}
                className="w-full text-left bg-[#1a2333] hover:bg-[#232f48] border border-[#232f48] rounded-lg p-3 transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-mono text-[#6b7c9c]">{s.session_id?.slice(0, 8)}</span>
                  <span className="text-[8px] font-mono text-[#6b7c9c]">
                    {s.message_count || 0} messages
                  </span>
                </div>
                <div className="text-[11px] text-white font-medium leading-snug line-clamp-2">
                  {s.current_region || "New session"}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chat messages */}
      <div ref={logRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {messages.map(msg => (
          <ChatBubble
            key={msg.id}
            message={msg}
            onSuggestion={handleSuggestion}
            onFlyTo={onFlyTo}
            isStreaming={isStreaming}
          />
        ))}

        {/* Pending action cards */}
        {pendingActions.map(action => (
          <div key={action.action_id} className="mx-1 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-sm text-amber-400">
                {TOOL_ICONS[action.tool_name] || "build"}
              </span>
              <span className="text-[11px] text-amber-300 font-medium">{action.description}</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handleConfirm(action.action_id)}
                disabled={isStreaming}
                className="px-3 py-1.5 rounded text-[10px] font-bold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors disabled:opacity-50"
              >
                Confirm
              </button>
              <button
                onClick={() => handleReject(action.action_id)}
                disabled={isStreaming}
                className="px-3 py-1.5 rounded text-[10px] font-bold bg-[#232f48] hover:bg-[#2a3a58] text-[#92a4c9] transition-colors disabled:opacity-50"
              >
                Skip
              </button>
            </div>
          </div>
        ))}

        {/* Report */}
        {reportMarkdown && (
          <div className="mx-1 bg-[#1a2333] border border-[#232f48] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-sm text-teal-400">description</span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-teal-400">Report</span>
            </div>
            <div
              className="text-[10px] text-[#c8d6e5] leading-relaxed prose prose-invert prose-xs max-w-none
                [&_h1]:text-sm [&_h1]:font-bold [&_h1]:text-white [&_h1]:mt-3 [&_h1]:mb-1
                [&_h2]:text-xs [&_h2]:font-bold [&_h2]:text-white [&_h2]:mt-3 [&_h2]:mb-1
                [&_ul]:pl-4 [&_li]:text-[10px] [&_strong]:text-white [&_p]:mb-1.5"
              dangerouslySetInnerHTML={{ __html: simpleMarkdownToHtml(reportMarkdown) }}
            />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="px-3 py-2.5 border-t border-[#232f48]">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            placeholder="Ask MARVIS anything..."
            disabled={isStreaming}
            className="flex-1 h-9 bg-[#1a2333] border border-[#232f48] rounded-lg px-3 text-xs text-[#c8d6e5] placeholder-[#4a5a7a] focus:outline-none focus:border-teal-500/50 disabled:opacity-50"
          />
          <button
            onClick={() => handleSend()}
            disabled={isStreaming || !inputText.trim()}
            className={`h-9 w-9 rounded-lg flex items-center justify-center transition-all ${
              isStreaming || !inputText.trim()
                ? "bg-[#1a2333] text-[#4a5a7a] cursor-not-allowed"
                : "bg-teal-600 hover:bg-teal-500 text-white"
            }`}
          >
            {isStreaming ? (
              <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
            ) : (
              <span className="material-symbols-outlined text-sm">send</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
 * Chat bubble sub-component
 * =======================================================*/
function ChatBubble({
  message,
  onSuggestion,
  onFlyTo,
  isStreaming,
}: {
  message: ChatMessage;
  onSuggestion: (text: string) => void;
  onFlyTo?: (lat: number, lon: number) => void;
  isStreaming: boolean;
}) {
  if (message.isThinking) {
    return (
      <div className="flex items-center gap-2 px-1">
        <span className="material-symbols-outlined text-sm text-teal-400 animate-spin">progress_activity</span>
        <span className="text-[10px] text-[#6b7c9c] italic">Thinking...</span>
      </div>
    );
  }

  if (message.isActionInline) {
    const isDone = !message.text.startsWith("Running:");
    return (
      <div className="flex items-center gap-2 px-1 py-0.5">
        <span className={`material-symbols-outlined text-xs ${isDone ? "text-emerald-400" : "text-sky-400 animate-spin"}`}>
          {isDone ? "check_circle" : "progress_activity"}
        </span>
        <span className={`text-[10px] font-mono ${isDone ? "text-emerald-400/70" : "text-sky-400/70"}`}>
          {message.text}
        </span>
      </div>
    );
  }

  if (message.role === "system") {
    return (
      <div className="px-1 py-1">
        <span className="text-[10px] text-red-400/80 font-mono">{message.text}</span>
      </div>
    );
  }

  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}${!isUser && message.isNew ? " section-highlight-new" : ""}`}>
      <div className={`max-w-[85%] ${isUser ? "order-1" : ""}`}>
        {/* Bubble */}
        <div className={`rounded-2xl px-3.5 py-2 ${
          isUser
            ? "bg-teal-600 text-white rounded-br-md"
            : "bg-[#1a2333] border border-[#232f48] text-[#c8d6e5] rounded-bl-md"
        }`}>
          <p className="text-[11px] leading-relaxed whitespace-pre-wrap">{message.text}</p>

          {/* Fly-to button if location data */}
          {message.flyTo && onFlyTo && (
            <button
              onClick={() => onFlyTo(message.flyTo!.lat, message.flyTo!.lon)}
              className="mt-1.5 flex items-center gap-1 text-[9px] text-teal-300 hover:text-teal-200 transition-colors"
            >
              <span className="material-symbols-outlined text-xs">flight</span>
              Navigate to location
            </button>
          )}
        </div>

        {/* Suggestion chips */}
        {!isUser && message.suggestions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 px-1">
            {message.suggestions.map((s, i) => (
              <button
                key={i}
                onClick={() => onSuggestion(s)}
                disabled={isStreaming}
                className="px-2.5 py-1 rounded-full text-[10px] bg-teal-500/10 border border-teal-500/20 text-teal-400 hover:bg-teal-500/20 hover:border-teal-500/40 transition-colors disabled:opacity-40"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* =========================================================
 * Simple markdown -> HTML converter
 * =======================================================*/
function simpleMarkdownToHtml(md: string): string {
  return md
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^\| (.+)$/gm, (_, row) => {
      const cells = row.split("|").map((c: string) => c.trim()).filter(Boolean);
      return "<tr>" + cells.map((c: string) => `<td>${c}</td>`).join("") + "</tr>";
    })
    .replace(/(<tr>[\s\S]*?<\/tr>)/g, (match) => {
      if (match.includes("---")) return "";
      return match;
    })
    .replace(/^\- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>")
    .replace(/<\/ul>\s*<ul>/g, "")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br/>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}
