import { useEffect, useState, useCallback, useRef } from "react";
import type { TerrainPoint } from "./SlopeAnalysis";

/* =========================================================
 * Types
 * =======================================================*/
type InstrumentEvidence = {
  count: number;
  products: { product_id: string; instrument: string; center?: { lat: number; lon: number } }[];
};

type Evidence = {
  pin: { lat: number; lon: number };
  radius_km: number;
  region: { name: string; description: string; distance_deg: number } | null;
  mola: {
    mean_slope: number;
    max_slope: number;
    elevation_m: number;
    count: number;
  } | null;
  instruments: Record<string, InstrumentEvidence>;
  field_notes: { product_id: string; instrument: string; tags: string[]; memo: string; distance_km: number }[];
  data_gaps: string[];
  total_products: number;
};

type AskResponse = {
  answer_markdown: string;
  highlights: {
    supports: string[];
    uncertainties: string[];
    actions: string[];
  };
  model_used: string;
};

const RADIUS_OPTIONS = [1, 5, 10, 20];

const INSTRUMENT_COLORS: Record<string, string> = {
  crism: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  hirise_dtm: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  sharad: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  sharad_highres: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
  ctx: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const INSTRUMENT_LABELS: Record<string, string> = {
  crism: "CRISM",
  hirise_dtm: "HiRISE DTM",
  sharad: "SHARAD",
  sharad_highres: "SHARAD Hi-Res",
  ctx: "CTX",
};

/* =========================================================
 * Component
 * =======================================================*/
export default function AiAnalysisPanel({
  pin,
  onClose,
}: {
  pin: TerrainPoint;
  onClose: () => void;
}) {
  const [radiusKm, setRadiusKm] = useState(10);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerError, setAnswerError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<"answer" | "evidence" | "products">("evidence");
  const [expandedInstruments, setExpandedInstruments] = useState<Set<string>>(new Set());

  // Panel width (resizable)
  const [panelWidth, setPanelWidth] = useState(420);
  const abortRef = useRef<AbortController | null>(null);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.6);
      setPanelWidth(Math.max(320, Math.min(maxW, startW + delta)));
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

  // Auto-gather evidence when pin or radius changes
  useEffect(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    async function fetchEvidence() {
      setEvidenceLoading(true);
      setEvidenceError(null);
      setAnswer(null);

      try {
        const res = await fetch("/api/ai_analysis/evidence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lat: pin.lat, lon: pin.lon, radius_km: radiusKm }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail || `HTTP ${res.status}`);
        }
        const data: Evidence = await res.json();
        setEvidence(data);
        setActiveTab("evidence");
      } catch (e: any) {
        if (e.name !== "AbortError") {
          setEvidenceError(e.message ?? "Failed to gather evidence");
        }
      } finally {
        setEvidenceLoading(false);
      }
    }

    fetchEvidence();
    return () => controller.abort();
  }, [pin.lat, pin.lon, radiusKm]);

  // Ask Gemini
  const handleAsk = useCallback(async () => {
    if (!question.trim() || !evidence) return;

    setAnswerLoading(true);
    setAnswerError(null);
    setAnswer(null);
    setActiveTab("answer");

    try {
      const res = await fetch("/api/ai_analysis/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), evidence }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `HTTP ${res.status}`);
      }
      const data: AskResponse = await res.json();
      setAnswer(data);
    } catch (e: any) {
      setAnswerError(e.message ?? "Analysis failed");
    } finally {
      setAnswerLoading(false);
    }
  }, [question, evidence]);

  const toggleInstrument = (inst: string) => {
    setExpandedInstruments((prev) => {
      const next = new Set(prev);
      if (next.has(inst)) next.delete(inst);
      else next.add(inst);
      return next;
    });
  };

  return (
    <aside
      className="relative flex h-full flex-col border-l border-border-dark bg-surface-dark/40"
      style={{ width: panelWidth }}
    >
      {/* Resize handle */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/50 transition-colors"
        onMouseDown={handleResizeStart}
      />

      {/* Header */}
      <div className="p-4 border-b border-border-dark">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-8 rounded-lg bg-violet-500/20 flex items-center justify-center text-violet-400">
              <span className="material-symbols-outlined text-lg">psychology</span>
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">AI Analysis</h2>
              <span className="text-[10px] text-slate-500">
                {pin.lat.toFixed(3)}°, {pin.lon.toFixed(3)}°
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex items-center justify-center p-1 text-slate-500 hover:text-red-400 transition-colors"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Radius selector */}
        <div className="mt-3 flex items-center gap-1">
          <span className="text-[10px] text-slate-500 mr-1">Radius</span>
          {RADIUS_OPTIONS.map((r) => (
            <button
              key={r}
              onClick={() => setRadiusKm(r)}
              className={`px-2 py-0.5 text-[10px] rounded-md border transition-colors ${
                radiusKm === r
                  ? "bg-violet-500/20 border-violet-500/50 text-violet-400"
                  : "bg-surface-dark border-border-dark text-slate-500 hover:text-slate-300"
              }`}
            >
              {r} km
            </button>
          ))}
        </div>
      </div>

      {/* Question input */}
      <div className="p-3 border-b border-border-dark">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about this area... e.g. Is there evidence of subsurface ice?"
          className="w-full h-16 px-2 py-1.5 text-xs bg-surface-dark border border-border-dark rounded-lg text-slate-300 placeholder:text-slate-600 resize-none focus:outline-none focus:border-violet-500/50"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleAsk();
          }}
        />
        <button
          onClick={handleAsk}
          disabled={!question.trim() || !evidence || answerLoading}
          className="mt-1.5 flex w-full items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-bold uppercase tracking-widest bg-violet-600/20 border border-violet-600/50 text-violet-400 hover:bg-violet-600/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {answerLoading ? (
            <>
              <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
              Analyzing...
            </>
          ) : (
            <>
              <span className="material-symbols-outlined text-sm">psychology</span>
              Run Analysis
              <span className="text-[9px] font-normal opacity-60 ml-1">⌘↵</span>
            </>
          )}
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border-dark">
        {(["answer", "evidence", "products"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
              activeTab === tab
                ? "text-violet-400 border-b-2 border-violet-500"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {tab === "answer" ? "Answer" : tab === "evidence" ? "Evidence" : "Products"}
            {tab === "products" && evidence && (
              <span className="ml-1 text-[9px] opacity-60">{evidence.total_products}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto scrollbar-dark p-3">
        {/* Loading state */}
        {evidenceLoading && (
          <div className="flex flex-col items-center justify-center py-12 gap-2">
            <span className="material-symbols-outlined text-2xl text-violet-400 animate-spin">progress_activity</span>
            <span className="text-xs text-slate-500">Gathering evidence...</span>
          </div>
        )}

        {/* Error state */}
        {evidenceError && !evidenceLoading && (
          <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-3">
            <p className="text-xs text-red-400">{evidenceError}</p>
          </div>
        )}

        {/* Answer tab */}
        {activeTab === "answer" && !evidenceLoading && (
          <div className="space-y-3">
            {answerError && (
              <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-3">
                <p className="text-xs text-red-400">{answerError}</p>
              </div>
            )}
            {answerLoading && (
              <div className="flex flex-col items-center justify-center py-12 gap-2">
                <span className="material-symbols-outlined text-2xl text-violet-400 animate-spin">progress_activity</span>
                <span className="text-xs text-slate-500">Gemini is analyzing...</span>
              </div>
            )}
            {answer && !answerLoading && (
              <>
                {/* Markdown answer */}
                <div className="prose prose-invert prose-xs max-w-none text-xs text-slate-300 leading-relaxed [&_h3]:text-violet-400 [&_h3]:text-xs [&_h3]:font-bold [&_h3]:mt-4 [&_h3]:mb-1 [&_ul]:space-y-0.5 [&_li]:text-slate-400">
                  <div dangerouslySetInnerHTML={{ __html: renderMarkdown(answer.answer_markdown) }} />
                </div>

                {/* Highlights chips */}
                {answer.highlights.supports.length > 0 && (
                  <div className="mt-3">
                    <span className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">Supports</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {answer.highlights.supports.map((s, i) => (
                        <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-green-500/10 border border-green-500/20 text-green-400">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
                {answer.highlights.uncertainties.length > 0 && (
                  <div className="mt-2">
                    <span className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">Uncertainties</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {answer.highlights.uncertainties.map((u, i) => (
                        <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400">{u}</span>
                      ))}
                    </div>
                  </div>
                )}
                {answer.highlights.actions.length > 0 && (
                  <div className="mt-2">
                    <span className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">Next Steps</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {answer.highlights.actions.map((a, i) => (
                        <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/20 text-violet-400">{a}</span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-3 text-[9px] text-slate-600">
                  Model: {answer.model_used}
                </div>
              </>
            )}
            {!answer && !answerLoading && !answerError && evidence && (
              <div className="flex flex-col items-center justify-center py-8 gap-2 text-center">
                <span className="material-symbols-outlined text-2xl text-slate-600">chat</span>
                <p className="text-xs text-slate-500">
                  Type a question above and press<br />
                  <span className="text-violet-400">Run Analysis</span> to get AI insights
                </p>
              </div>
            )}
          </div>
        )}

        {/* Evidence tab */}
        {activeTab === "evidence" && !evidenceLoading && evidence && (
          <div className="space-y-2">
            {/* Region */}
            {evidence.region && (
              <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-2.5">
                <div className="text-[10px] text-violet-400 font-semibold">{evidence.region.name}</div>
                <div className="text-[9px] text-slate-500 mt-0.5">{evidence.region.description}</div>
              </div>
            )}

            {/* MOLA */}
            {evidence.mola && (
              <div className="rounded-lg border border-border-dark bg-surface-dark/50 p-2.5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <span className="material-symbols-outlined text-xs text-orange-400">terrain</span>
                  <span className="text-[10px] font-semibold text-slate-300">MOLA Terrain</span>
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
                  <div><span className="text-slate-500">Elevation</span> <span className="text-slate-300">{evidence.mola.elevation_m?.toFixed(0)} m</span></div>
                  <div><span className="text-slate-500">Mean slope</span> <span className="text-slate-300">{evidence.mola.mean_slope?.toFixed(1)}°</span></div>
                  <div><span className="text-slate-500">Max slope</span> <span className="text-slate-300">{evidence.mola.max_slope?.toFixed(1)}°</span></div>
                  <div><span className="text-slate-500">Samples</span> <span className="text-slate-300">{evidence.mola.count}</span></div>
                </div>
              </div>
            )}

            {/* Instruments */}
            {Object.entries(evidence.instruments).map(([inst, ev]) => (
              <div key={inst} className="rounded-lg border border-border-dark bg-surface-dark/50">
                <button
                  onClick={() => toggleInstrument(inst)}
                  className="flex items-center justify-between w-full p-2.5 text-left"
                >
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[9px] px-1.5 py-0.5 rounded border ${INSTRUMENT_COLORS[inst] || "bg-slate-500/20 text-slate-400 border-slate-500/30"}`}>
                      {INSTRUMENT_LABELS[inst] || inst.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {ev.count} product{ev.count !== 1 ? "s" : ""}
                    </span>
                  </div>
                  <span className="material-symbols-outlined text-xs text-slate-500">
                    {expandedInstruments.has(inst) ? "expand_less" : "expand_more"}
                  </span>
                </button>
                {expandedInstruments.has(inst) && ev.products.length > 0 && (
                  <div className="px-2.5 pb-2.5 space-y-0.5">
                    {ev.products.slice(0, 20).map((p) => (
                      <div key={p.product_id} className="text-[9px] text-slate-500 font-mono truncate">
                        {p.product_id}
                      </div>
                    ))}
                    {ev.products.length > 20 && (
                      <div className="text-[9px] text-slate-600">+{ev.products.length - 20} more</div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Field notes */}
            {evidence.field_notes.length > 0 && (
              <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-2.5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <span className="material-symbols-outlined text-xs text-amber-400">sticky_note_2</span>
                  <span className="text-[10px] font-semibold text-slate-300">
                    Field Notes ({evidence.field_notes.length})
                  </span>
                </div>
                {evidence.field_notes.map((n, i) => (
                  <div key={i} className="text-[9px] text-slate-400 mt-1">
                    <span className="text-amber-400">{n.product_id}</span>
                    {n.memo && <span className="ml-1 text-slate-500">— {n.memo.slice(0, 80)}</span>}
                  </div>
                ))}
              </div>
            )}

            {/* Data gaps */}
            {evidence.data_gaps.length > 0 && (
              <div className="rounded-lg border border-red-500/15 bg-red-500/5 p-2.5">
                <div className="text-[10px] text-red-400 font-semibold mb-1">Data Gaps</div>
                <div className="flex flex-wrap gap-1">
                  {evidence.data_gaps.map((g) => (
                    <span key={g} className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 border border-red-500/20 text-red-400">
                      {INSTRUMENT_LABELS[g] || g.toUpperCase()}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Products tab */}
        {activeTab === "products" && !evidenceLoading && evidence && (
          <div className="space-y-1">
            {evidence.total_products === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 gap-2">
                <span className="material-symbols-outlined text-2xl text-slate-600">search_off</span>
                <p className="text-xs text-slate-500">No products found in this area</p>
              </div>
            ) : (
              Object.entries(evidence.instruments).flatMap(([inst, ev]) =>
                ev.products.map((p) => (
                  <div key={`${inst}-${p.product_id}`} className="flex items-center gap-2 p-1.5 rounded hover:bg-white/5 transition-colors">
                    <span className={`text-[8px] px-1 py-0.5 rounded border shrink-0 ${INSTRUMENT_COLORS[inst] || "bg-slate-500/20 text-slate-400 border-slate-500/30"}`}>
                      {INSTRUMENT_LABELS[inst] || inst.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-slate-300 font-mono truncate">{p.product_id}</span>
                  </div>
                ))
              )
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

/* =========================================================
 * Minimal markdown → HTML
 * =======================================================*/
function renderMarkdown(md: string): string {
  // Process line by line to properly group consecutive <li> items into <ul> blocks
  const lines = md.split("\n");
  const out: string[] = [];
  let inList = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ")) {
      if (!inList) { out.push("<ul>"); inList = true; }
      const content = trimmed.slice(2)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      out.push(`<li>${content}</li>`);
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      if (trimmed.startsWith("### ")) {
        out.push(`<h3>${trimmed.slice(4)}</h3>`);
      } else if (trimmed === "") {
        out.push("<br/>");
      } else {
        out.push(trimmed.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>"));
      }
    }
  }
  if (inList) out.push("</ul>");

  return out.join("\n");
}
