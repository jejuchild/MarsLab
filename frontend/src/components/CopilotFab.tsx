/**
 * CopilotFab — Floating MARVIS chat widget with grounded tool calling.
 *
 * Collapsed: small FAB button (bottom-right).
 * Expanded: floating chat panel that can navigate the map and load instruments.
 */
import { useState, useRef, useEffect, useCallback } from "react";
import { streamMarvisChat, type ChatMessage, type SessionContext } from "../api/marvis";

type MsgRole = "user" | "assistant" | "tool";

type Message = {
  id: number;
  role: MsgRole;
  content: string;
  toolCall?: { tool: string; params: Record<string, unknown> };
};

let _nextId = 0;

const TOOL_ICONS: Record<string, string> = {
  fly_to_location: "flight",
  load_instrument: "layers",
  find_intersections: "join",
  select_product: "info",
  search_minerals: "science",
};

export type IntersectionPair = {
  product_a: string;
  product_b: string;
  instrument_a: string;
  instrument_b: string;
  lat: number;
  lon: number;
};

export type MineralSearchResult = {
  product_id: string;
  obs_id: string;
  lat: number;
  lon: number;
  ice_percent: number;
  hyd_percent: number;
};

export default function CopilotFab({
  hidden = false,
  onFlyTo,
  onLoadInstrument,
  onShowIntersections,
  onSelectInstrumentProduct,
  onSearchMinerals,
  loadedInstruments = [],
  visibleProductCount = 0,
  currentLat,
  currentLon,
}: {
  hidden?: boolean;
  onFlyTo?: (lat: number, lon: number) => void;
  onLoadInstrument?: (instrument: string) => void;
  onShowIntersections?: (pairs: IntersectionPair[]) => void;
  onSelectInstrumentProduct?: (instrument: string) => void;
  onSearchMinerals?: (results: MineralSearchResult[]) => void;
  loadedInstruments?: string[];
  visibleProductCount?: number;
  currentLat?: number | null;
  currentLon?: number | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamText]);

  // Focus input when expanded
  useEffect(() => {
    if (expanded) setTimeout(() => inputRef.current?.focus(), 120);
  }, [expanded]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;

    setInput("");
    setError(null);
    const userMsg: Message = { id: _nextId++, role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setStreaming(true);
    setStreamText("");

    const history: ChatMessage[] = messages
      .filter(m => m.role === "user" || m.role === "assistant")
      .map(m => ({ role: m.role as "user" | "assistant", content: m.content }));

    const context: SessionContext = {
      loaded_instruments: loadedInstruments,
      visible_product_count: visibleProductCount,
      current_lat: currentLat,
      current_lon: currentLon,
    };

    try {
      let fullText = "";
      let loadDelay = 0; // Stagger multiple load_instrument calls to avoid React batching
      for await (const event of streamMarvisChat(text, history, context)) {
        if (event.event === "chunk") {
          fullText += event.data.text;
          setStreamText(fullText);
        } else if (event.event === "done") {
          fullText = event.data.full_text || fullText;
        } else if (event.event === "tool_call") {
          const { tool, params, message } = event.data;

          // Execute the tool
          if (tool === "fly_to_location" && onFlyTo) {
            const lat = params.lat as number;
            const lon = params.lon as number;
            if (typeof lat === "number" && typeof lon === "number") {
              onFlyTo(lat, lon);
            }
          } else if (tool === "load_instrument" && onLoadInstrument) {
            const inst = params.instrument as string;
            if (inst) {
              // Stagger calls: first one immediate, subsequent ones delayed
              // This prevents React batching from eating earlier triggers
              if (loadDelay === 0) {
                onLoadInstrument(inst);
              } else {
                const d = loadDelay;
                setTimeout(() => onLoadInstrument(inst), d);
              }
              loadDelay += 200;
            }
          } else if (tool === "find_intersections" && onShowIntersections) {
            const pairs = params.pairs as IntersectionPair[] | undefined;
            if (pairs && pairs.length > 0) {
              onShowIntersections(pairs);
            }
          } else if (tool === "select_product" && onSelectInstrumentProduct) {
            const inst = params.instrument as string;
            if (inst) {
              onSelectInstrumentProduct(inst);
            }
          } else if (tool === "search_minerals" && onSearchMinerals) {
            const results = params.results as MineralSearchResult[] | undefined;
            if (results && results.length > 0) {
              onSearchMinerals(results);
            }
          }

          // Add tool confirmation message
          setMessages(prev => [...prev, {
            id: _nextId++,
            role: "tool",
            content: message,
            toolCall: { tool, params },
          }]);
        } else if (event.event === "error") {
          setError(event.data.error);
        }
      }

      // Add text response (if any)
      if (fullText) {
        setMessages(prev => [...prev, { id: _nextId++, role: "assistant", content: fullText }]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
    } finally {
      setStreaming(false);
      setStreamText("");
    }
  }, [input, streaming, messages, loadedInstruments, visibleProductCount, currentLat, currentLon, onFlyTo, onLoadInstrument, onShowIntersections, onSelectInstrumentProduct, onSearchMinerals]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    if (e.key === "Escape") setExpanded(false);
  }, [handleSend]);

  if (hidden) return null;

  return (
    <>
      {/* Expanded chat panel */}
      {expanded && (
        <div
          className="fixed bottom-16 right-4 z-50 flex flex-col rounded-xl border border-[#232f48] bg-[#101622]/95 shadow-2xl backdrop-blur-sm"
          style={{ width: 380, maxHeight: 480 }}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-[#232f48] px-3 py-2.5 shrink-0">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-fuchsia-400 text-base">smart_toy</span>
              <span className="text-xs font-bold text-white tracking-wide">MARVIS</span>
              <span className="text-[8px] px-1.5 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30 font-bold">
                CHAT
              </span>
            </div>
            <div className="flex items-center gap-1">
              {messages.length > 0 && (
                <button
                  onClick={() => { setMessages([]); setError(null); }}
                  className="p-1 rounded hover:bg-[#232f48] text-[#6b7c9c] hover:text-white transition-colors"
                  title="Clear chat"
                >
                  <span className="material-symbols-outlined text-sm">delete_sweep</span>
                </button>
              )}
              <button
                onClick={() => setExpanded(false)}
                className="p-1 rounded hover:bg-[#232f48] text-[#6b7c9c] hover:text-white transition-colors"
                title="Minimize"
              >
                <span className="material-symbols-outlined text-sm">expand_more</span>
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-3 py-2.5 space-y-2.5 min-h-0" style={{ maxHeight: 340 }}>
            {messages.length === 0 && !streaming && (
              <div className="text-center py-8">
                <span className="material-symbols-outlined text-3xl text-fuchsia-500/30 mb-2 block">smart_toy</span>
                <p className="text-[11px] text-[#6b7c9c]">Ask me anything about Mars.</p>
                <p className="text-[10px] text-[#4a5a7a] mt-1">I can navigate the map and load data for you.</p>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                {msg.role === "tool" ? (
                  /* Tool confirmation bubble */
                  <div className="max-w-[90%] flex items-start gap-2 px-3 py-2 rounded-lg text-xs leading-relaxed bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                    <span className="material-symbols-outlined text-sm mt-0.5 shrink-0">
                      {TOOL_ICONS[msg.toolCall?.tool ?? ""] || "build"}
                    </span>
                    <span>{msg.content}</span>
                  </div>
                ) : (
                  <div className={`max-w-[85%] px-3 py-2 rounded-lg text-xs leading-relaxed ${
                    msg.role === "user"
                      ? "bg-fuchsia-500/20 text-fuchsia-100 border border-fuchsia-500/20"
                      : "bg-[#1a2333] text-[#c8d6e5] border border-[#232f48]"
                  }`}>
                    {msg.content}
                  </div>
                )}
              </div>
            ))}
            {streaming && streamText && (
              <div className="flex justify-start">
                <div className="max-w-[85%] px-3 py-2 rounded-lg text-xs leading-relaxed bg-[#1a2333] text-[#c8d6e5] border border-[#232f48]">
                  {streamText}
                  <span className="inline-block w-1.5 h-3.5 bg-fuchsia-400/60 ml-0.5 animate-pulse" />
                </div>
              </div>
            )}
            {streaming && !streamText && (
              <div className="flex justify-start">
                <div className="px-3 py-2 rounded-lg text-xs text-[#6b7c9c] bg-[#1a2333] border border-[#232f48]">
                  <span className="inline-flex gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </span>
                </div>
              </div>
            )}
            {error && (
              <div className="flex justify-start">
                <div className="px-3 py-2 rounded-lg text-xs text-red-400 bg-red-500/10 border border-red-500/20">
                  {error}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Context status bar */}
          <div className="px-3 py-1 border-t border-[#232f48]/50 flex items-center gap-2 text-[9px] text-[#4a5a7a]">
            <span className="material-symbols-outlined text-[10px]">layers</span>
            {loadedInstruments.length > 0
              ? loadedInstruments.join(", ")
              : "No instruments loaded"}
            <span className="mx-1">·</span>
            {visibleProductCount} products
          </div>

          {/* Input */}
          <div className="border-t border-[#232f48] p-2.5 shrink-0">
            <div className="flex items-center gap-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about Mars or give commands..."
                disabled={streaming}
                className="flex-1 bg-[#0d1520] border border-[#232f48] rounded-lg px-3 py-2 text-xs text-white placeholder-[#4a5a7a] outline-none focus:border-fuchsia-500/40 transition-colors disabled:opacity-50"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || streaming}
                className={`p-2 rounded-lg transition-colors ${
                  input.trim() && !streaming
                    ? "text-fuchsia-400 hover:bg-fuchsia-500/20"
                    : "text-[#3a4a68] cursor-not-allowed"
                }`}
              >
                <span className="material-symbols-outlined text-base">send</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* FAB button */}
      <button
        onClick={() => setExpanded(prev => !prev)}
        className={`fixed bottom-4 right-4 z-40 flex items-center gap-2 h-10 px-4 shadow-lg shadow-black/40 transition-all duration-200 ${
          expanded
            ? "bg-[#0d1520] border border-fuchsia-500/50 rounded-full"
            : "bg-[#0d1520]/90 border border-fuchsia-500/30 rounded-full hover:border-fuchsia-500/60 hover:shadow-fuchsia-500/10"
        }`}
      >
        <span className={`material-symbols-outlined text-base transition-colors ${
          expanded ? "text-fuchsia-400" : "text-fuchsia-500/70"
        }`}>
          {expanded ? "expand_more" : "smart_toy"}
        </span>
        <span className={`text-xs font-bold tracking-wide transition-colors ${
          expanded ? "text-fuchsia-400" : "text-fuchsia-500/70"
        }`}>
          MARVIS
        </span>
        {messages.length > 0 && !expanded && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-fuchsia-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center">
            {messages.filter(m => m.role === "assistant" || m.role === "tool").length}
          </span>
        )}
      </button>
    </>
  );
}
