import { useState } from "react";
import type { AnalysisToolsProps, AnalysisMode } from "../types";
import ToolButton from "../shared/ToolButton";
import CollapsibleSection from "../shared/CollapsibleSection";

/** Category card with collapsible content */
function CategoryCard({
  icon,
  label,
  tint,
  count,
  defaultOpen = false,
  children,
}: {
  icon: string;
  label: string;
  tint: string;
  count: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`rounded-lg border ${tint} overflow-hidden`}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left"
      >
        <span className={`material-symbols-outlined text-sm`}>{icon}</span>
        <span className="text-[10px] font-bold uppercase tracking-wide text-[#92a4c9] flex-1">
          {label}
        </span>
        {count > 0 && (
          <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-primary/20 text-primary font-bold">
            {count}
          </span>
        )}
        <span className="material-symbols-outlined text-xs text-[#6b7c9c]">
          {open ? "expand_less" : "expand_more"}
        </span>
      </button>
      {open && <div className="px-3 pb-2 space-y-1">{children}</div>}
    </div>
  );
}

export default function AnalysisTools({
  analysisMode,
  onAnalysisModeChange,
  onShowRegionDashboard,
  showMeasurementTools,
  onToggleMeasurementTools,
}: AnalysisToolsProps) {
  /** Toggle an analysis mode on/off */
  const toggle = (mode: NonNullable<AnalysisMode>) => {
    onAnalysisModeChange?.(analysisMode === mode ? null : mode);
  };

  // Count active tools per category for badges
  const terrainActive = [
    analysisMode === "slope",
    analysisMode === "line",
    showMeasurementTools,
    analysisMode === "region_stats",
  ].filter(Boolean).length;

  const aiActive = [
    analysisMode === "agentic",
    analysisMode === "report",
    analysisMode === "guided",
    analysisMode === "pathfinder",
  ].filter(Boolean).length;

  const detectActive = [
    analysisMode === "crater_detect",
  ].filter(Boolean).length;

  return (
    <CollapsibleSection title="Analysis Tools" icon="build" defaultOpen={false} storageKey="analysis">
      <div className="space-y-2">
        {/* ── Manual Tools ── */}
        <CategoryCard
          icon="terrain"
          label="Manual Tools"
          tint="border-sky-500/20 bg-sky-500/[0.03]"
          count={terrainActive}
          defaultOpen={terrainActive > 0}
        >
          <ToolButton
            active={analysisMode === "slope"}
            onClick={() => toggle("slope")}
            icon="analytics"
            title="Slope Analysis"
            description="Click terrain to analyse"
            color="primary"
          />
          <ToolButton
            active={analysisMode === "line"}
            onClick={() => toggle("line")}
            icon="show_chart"
            title="Line Profile"
            description="Click two points for elevation"
            color="emerald"
          />
          <ToolButton
            active={showMeasurementTools ?? false}
            onClick={() => onToggleMeasurementTools?.(!showMeasurementTools)}
            icon="straighten"
            title="Measurement Tools"
            description="Distance, area, elevation, pins"
            color="cyan"
          />
          <ToolButton
            active={analysisMode === "region_stats"}
            onClick={() => toggle("region_stats")}
            icon="pentagon"
            title="Region Stats"
            description="Draw polygon for area statistics"
            color="indigo"
          />
          <ToolButton
            active={false}
            onClick={() => window.open("/mastcam-label", "_blank")}
            icon="panorama_photosphere"
            title="Mastcam Labeling"
            description="Label roughness on 360° panoramas"
            color="amber"
            badge="NEW"
          />
        </CategoryCard>

        {/* ── AI-Powered ── */}
        <CategoryCard
          icon="auto_awesome"
          label="AI-Powered"
          tint="border-fuchsia-500/20 bg-fuchsia-500/[0.03]"
          count={aiActive}
          defaultOpen={aiActive > 0}
        >
          <ToolButton
            active={analysisMode === "agentic"}
            onClick={() => toggle("agentic")}
            icon="smart_toy"
            title="Agentic AI"
            description="Autonomous multi-instrument analysis"
            color="fuchsia"
            badge="BETA"
          />
          <ToolButton
            active={analysisMode === "report"}
            onClick={() => toggle("report")}
            icon="assignment"
            title="AI Landing Site Report"
            description="Compare regions with ground rules"
            color="amber"
            badge="BETA"
          />
          <ToolButton
            active={analysisMode === "guided"}
            onClick={() => toggle("guided")}
            icon="explore"
            title="Guided Workflows"
            description="Step-by-step investigation guides"
            color="sky"
            badge="NEW"
          />
          <ToolButton
            active={analysisMode === "pathfinder"}
            onClick={() => toggle("pathfinder")}
            icon="route"
            title="Pathfinder"
            description="AI rover route planning (Field D*)"
            color="orange"
            badge="NEW"
          />
        </CategoryCard>

        {/* ── Detection & Dashboard ── */}
        <CategoryCard
          icon="search"
          label="Detection & Dashboard"
          tint="border-rose-500/20 bg-rose-500/[0.03]"
          count={detectActive}
          defaultOpen={detectActive > 0}
        >
          <ToolButton
            active={analysisMode === "crater_detect"}
            onClick={() => toggle("crater_detect")}
            icon="target"
            title="Landform Detect"
            description="Find craters, channels, LDAs from MOLA DEM"
            color="rose"
            badge="NEW"
          />
          <ToolButton
            active={false}
            onClick={() => onShowRegionDashboard?.()}
            icon="public"
            title="Region Dashboard"
            description="Browse all 55 regions at a glance"
            color="teal"
            badge="NEW"
          />
        </CategoryCard>
      </div>
    </CollapsibleSection>
  );
}
