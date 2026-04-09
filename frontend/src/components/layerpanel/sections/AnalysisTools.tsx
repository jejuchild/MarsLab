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
        </CategoryCard>

        {/* ── Detection ── */}
        <CategoryCard
          icon="search"
          label="Detection"
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
        </CategoryCard>
      </div>
    </CollapsibleSection>
  );
}
