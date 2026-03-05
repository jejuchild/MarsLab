import type { AnalysisToolsProps, AnalysisMode } from "../types";
import ToolButton from "../shared/ToolButton";
import CollapsibleSection from "../shared/CollapsibleSection";

/** Category group header */
function CategoryHeader({ icon, label }: { icon: string; label: string }) {
  return (
    <div className="text-[9px] font-bold uppercase text-[#6b7c9c] mb-1.5 flex items-center gap-1">
      <span className="material-symbols-outlined text-[10px]">{icon}</span>
      {label}
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

  return (
    <CollapsibleSection title="Analysis Tools" icon="build" defaultOpen={true}>
      <div className="space-y-2">
        {/* ── Terrain Analysis ── */}
        <CategoryHeader icon="terrain" label="Terrain Analysis" />

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

        {/* ── AI Analysis ── */}
        <div className="mt-3">
          <CategoryHeader icon="psychology" label="AI Analysis" />
        </div>

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

        {/* ── Detection & Dashboard ── */}
        <div className="mt-3">
          <CategoryHeader icon="search" label="Detection & Dashboard" />
        </div>

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
      </div>
    </CollapsibleSection>
  );
}
