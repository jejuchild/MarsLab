import { useMemo, useState } from "react";
import type { AtPointResponse } from "../../../api/inspector";

export interface CrossActions {
  /** Open before/after comparison for this location. */
  onOpenTemporal?: (lat: number, lon: number, instrument: string) => void;
  /** Open the CNN mineral sequence for a CRISM TRR3 observation. */
  onOpenMineralSequence?: (obsId: string) => void;
  /** Open the crater detector (entry point for ε-inversion / strat column). */
  onOpenCraterDetect?: () => void;
}

export interface CrossSectionProps extends CrossActions {
  response: AtPointResponse;
}

type CrossTool = "stratigraphy" | "mineral_seq" | "temporal" | "spectral";

interface ToolDef {
  id: CrossTool;
  label: string;
  description: string;
  available: (r: AtPointResponse) => boolean;
}

const firstTrr3 = (r: AtPointResponse) => r.lanes.CRISM?.find((p) => p.variant === "trr3");
const count = (r: AtPointResponse, lane: keyof AtPointResponse["counts"]) => r.counts[lane] ?? 0;

const TOOLS: ToolDef[] = [
  {
    id: "stratigraphy",
    label: "Stratigraphy",
    description: "Detect craters in view, then run ε-inversion or build a stratigraphic column from a crater.",
    available: (r) => count(r, "HIRISE") > 0 || count(r, "SHARAD") > 0,
  },
  {
    id: "mineral_seq",
    label: "Mineral Sequence",
    description: "CNN mineral paragenesis for the nearest CRISM TRR3 observation.",
    available: (r) => firstTrr3(r) !== undefined,
  },
  {
    id: "temporal",
    label: "Temporal",
    description: "Find repeat observations of this location and compare before/after.",
    available: (r) => count(r, "HIRISE") >= 2 || count(r, "CTX") >= 2,
  },
  {
    id: "spectral",
    label: "Spectral Compare",
    description: "Pin spectra from the CRISM lane's Spectrum tab; pinned spectra open in an overlay chart.",
    available: (r) => count(r, "CRISM") >= 2,
  },
];

/**
 * Cross-Analysis section at the bottom of the Inspector (collapsed by default).
 * Each tool launches its panel as a drill-in; closing it returns to the Inspector.
 */
export default function CrossSection({
  response,
  onOpenTemporal,
  onOpenMineralSequence,
  onOpenCraterDetect,
}: CrossSectionProps) {
  const [expanded, setExpanded] = useState(false);
  const [activeTool, setActiveTool] = useState<CrossTool>("stratigraphy");

  const availableTools = useMemo(() => TOOLS.filter((t) => t.available(response)), [response]);
  const tool = availableTools.find((t) => t.id === activeTool) ?? availableTools[0];

  const action = ((): { label: string; run: () => void } | null => {
    if (!tool) return null;
    switch (tool.id) {
      case "stratigraphy":
        return onOpenCraterDetect ? { label: "Open crater detector", run: onOpenCraterDetect } : null;
      case "mineral_seq": {
        const p = firstTrr3(response);
        return p && onOpenMineralSequence
          ? { label: `Open sequence · ${p.product_id}`, run: () => onOpenMineralSequence(p.product_id.replace(/_\d{2}$/, "")) }
          : null;
      }
      case "temporal": {
        const instrument = count(response, "HIRISE") >= 2 ? "HIRISE" : "CTX";
        return onOpenTemporal
          ? { label: "Compare over time", run: () => onOpenTemporal(response.lat, response.lon, instrument) }
          : null;
      }
      case "spectral":
        return null;
    }
  })();

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
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Cross-Analysis</span>
        <span className="ml-auto text-[9px] font-mono text-slate-500">
          {availableTools.length} {availableTools.length === 1 ? "tool" : "tools"} available
        </span>
      </button>

      {expanded && availableTools.length === 0 && (
        <div className="px-4 py-3 text-[10px] text-slate-600 italic">
          No cross-instrument tools available at this point.
        </div>
      )}

      {expanded && tool && (
        <div className="border-t border-border-dark/60">
          <div className="flex items-center gap-1 px-2 pt-2" role="tablist">
            {availableTools.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={tool.id === t.id}
                onClick={() => setActiveTool(t.id)}
                className={`px-2 py-1 rounded-t text-[9px] font-bold uppercase tracking-wider transition-colors ${
                  tool.id === t.id ? "bg-bg-dark text-white" : "text-slate-500 hover:text-white"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="px-4 py-3 border-t border-border-dark/60 bg-bg-dark/60 space-y-2">
            <p className="text-[11px] text-slate-300">{tool.description}</p>
            {action && (
              <button
                type="button"
                onClick={action.run}
                className="w-full px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-primary/15 border border-primary/40 text-primary hover:bg-primary/25 transition-colors truncate"
              >
                {action.label} →
              </button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
