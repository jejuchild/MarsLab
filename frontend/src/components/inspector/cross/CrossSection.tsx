import { useMemo, useState } from "react";
import type { AtPointResponse } from "../../../api/inspector";

export interface CrossSectionProps {
  response: AtPointResponse;
}

type CrossTool = "stratigraphy" | "mineral_seq" | "temporal" | "spectral";

interface ToolDef {
  id: CrossTool;
  label: string;
  description: string;
  /** This tool is available when the response satisfies the predicate. */
  available: (r: AtPointResponse) => boolean;
}

const TOOLS: ToolDef[] = [
  {
    id: "stratigraphy",
    label: "Stratigraphy",
    description: "Layer counting and age estimation from a HiRISE DTM crater",
    available: (r) => (r.counts.HIRISE ?? 0) > 0 && r.lanes.HIRISE.some((p) => p.variant === "dtm"),
  },
  {
    id: "mineral_seq",
    label: "Mineral Sequence",
    description: "Cross-instrument mineral paragenesis (CRISM × HiRISE × SHARAD)",
    available: (r) => (r.counts.CRISM ?? 0) > 0 && (r.counts.HIRISE ?? 0) > 0,
  },
  {
    id: "temporal",
    label: "Temporal",
    description: "Before/after change detection over time",
    available: (r) => (r.counts.HIRISE ?? 0) >= 2 || (r.counts.CTX ?? 0) >= 2,
  },
  {
    id: "spectral",
    label: "Spectral Compare",
    description: "Pin and overlay multiple CRISM spectra",
    available: (r) => (r.counts.CRISM ?? 0) >= 2,
  },
];

/**
 * Cross-Analysis section.
 *
 * Per Q3=B: collapsed by default, manual expand. Header shows the count of
 * tools currently available for the active context.
 *
 * Tool implementations themselves (StratigraphyPanel etc.) are NOT yet
 * mounted from here — Phase 3 keeps the integration points as a stub
 * that points to the legacy panels via the analysisMode side-channel.
 * A follow-up phase will inline them once the cross-instrument data flow
 * is reorganized.
 */
export default function CrossSection({ response }: CrossSectionProps) {
  const [expanded, setExpanded] = useState(false);
  const [activeTool, setActiveTool] = useState<CrossTool>("stratigraphy");

  const availableTools = useMemo(
    () => TOOLS.filter((t) => t.available(response)),
    [response]
  );

  const tool = availableTools.find((t) => t.id === activeTool) ?? availableTools[0];

  return (
    <section className="border-t border-border-dark bg-bg-dark/40 flex-shrink-0">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-4 py-2 text-left hover:bg-white/5 transition-colors"
        aria-expanded={expanded}
      >
        <span className="material-symbols-outlined text-xs text-slate-500">
          {expanded ? "expand_more" : "chevron_right"}
        </span>
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
          Cross-Analysis
        </span>
        <span className="ml-auto text-[9px] font-mono text-slate-500">
          {availableTools.length} {availableTools.length === 1 ? "tool" : "tools"} available
        </span>
      </button>

      {expanded && availableTools.length === 0 && (
        <div className="px-4 py-3 text-[10px] text-slate-600 italic">
          No cross-instrument tools available at this point. Need at least two
          lanes with products.
        </div>
      )}

      {expanded && availableTools.length > 0 && (
        <div className="border-t border-border-dark/60">
          {/* Tool tabs */}
          <div className="flex items-center gap-1 px-2 pt-2">
            {availableTools.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setActiveTool(t.id)}
                className={`px-2 py-1 rounded-t text-[9px] font-bold uppercase tracking-wider transition-colors ${
                  tool?.id === t.id
                    ? "bg-bg-dark text-white"
                    : "text-slate-500 hover:text-white"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Tool body — placeholder description */}
          {tool && (
            <div className="px-4 py-3 border-t border-border-dark/60 bg-bg-dark/60">
              <p className="text-[11px] text-slate-300 mb-2">{tool.description}</p>
              <p className="text-[9px] text-slate-600 italic">
                Cross-analysis tool integration is being migrated. Use the
                legacy LayerPanel "Analysis Tools" for now.
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
