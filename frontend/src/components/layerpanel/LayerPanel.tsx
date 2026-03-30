import { useState, useEffect, useCallback } from "react";
import type { LayerPanelProps } from "./types";
import { lp, PANEL_COLLAPSED_KEY } from "./tokens";

// Section components
import ViewModeSection from "./sections/ViewModeSection";
import NavigationSection from "./sections/NavigationSection";
import IceHub from "./sections/IceHub";
import FootprintSection from "./sections/FootprintSection";
import AnalysisTools from "./sections/AnalysisTools";
import ProductsHub from "./sections/ProductsHub";
import FieldNotesSection from "./sections/FieldNotesSection";

export default function LayerPanel({
  // View
  mapMode,
  onMapModeChange,
  baseLayer,
  onBaseLayerChange,
  // Navigation
  viewBounds,
  onViewBoundsChange,
  onFlyToCoords,
  viewBoundSelectionMode = false,
  onViewBoundSelectionModeChange,
  showGrid = false,
  onToggleGrid,
  showRegionLayer = false,
  onToggleRegionLayer,
  // Ice
  scienceLayerVisibility = {} as Record<string, boolean>,
  onScienceLayerToggle,
  scienceLayerDepth = "1-5m" as "0-1m" | "1-5m" | "5m-plus",
  onScienceLayerDepthChange,
  scienceLayerOpacities = {} as Record<string, number>,
  onScienceLayerOpacity,
  swimLayer = false,
  onSwimLayerChange,
  swimIceLat = null,
  swimIceLon = null,
  accessibilityVisible = false,
  onAccessibilityVisibleChange,
  accessibilityOpacity = 0.6,
  onAccessibilityOpacityChange,
  accessibilityExplainMode = false,
  onAccessibilityExplainModeChange,
  fusionVisible = false,
  onFusionVisibleChange,
  fusionOpacity = 0.6,
  onFusionOpacityChange,
  ctxMosaicOpacity = 1.0,
  onCtxMosaicOpacityChange,
  // Footprints
  instrumentVisibility,
  onToggleInstrument,
  onLoadFootprints,
  footprintsLoading = {},
  footprintCounts = {},
  highResOnly = false,
  onHighResOnlyChange,
  showCustomData,
  onToggleCustomData,
  onLoadCustomData,
  customDataLoading = false,
  customDatasets = [],
  onCustomDatasetToggle,
  // Analysis
  analysisMode = null,
  onAnalysisModeChange,
  onShowRegionDashboard,
  showMeasurementTools = false,
  onToggleMeasurementTools,
  // Products
  visibleProducts = [],
  activeOverlays = new Map(),
  onSetOverlay,
  onSetOpacity,
  onSelectProduct,
  onFlyToProduct,
  onDeactivateAll,
  overlapFilter,
  onOverlapFilterChange,
  overlapStats,
  // Field Notes
  fieldNotes = [],
  showFieldNotesOnMap = true,
  onToggleFieldNotesOnMap,
  onFieldNoteClick,
  onActiveTagChange,
  // Layout
  isMobile = false,
}: LayerPanelProps) {
  // ── Panel collapse state ──
  const [isCollapsed, setIsCollapsed] = useState(() => {
    try {
      return localStorage.getItem(PANEL_COLLAPSED_KEY) === "true";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(PANEL_COLLAPSED_KEY, String(isCollapsed));
    } catch {
      // Ignore localStorage errors
    }
  }, [isCollapsed]);

  // ── Resizable width ──
  const [panelWidth, setPanelWidth] = useState(320);

  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = panelWidth;
      const onMove = (ev: MouseEvent) => {
        const delta = ev.clientX - startX;
        const maxW = Math.floor(window.innerWidth * 0.5);
        setPanelWidth(Math.max(200, Math.min(maxW, startW + delta)));
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
    },
    [panelWidth],
  );

  const totalActiveOverlays = activeOverlays.size;

  // ── Collapsed sidebar (desktop) ──
  if (isCollapsed && !isMobile) {
    return (
      <div className="flex h-full w-12 flex-col border-r border-[#232f48] bg-[#101622] transition-all duration-300 ease-in-out">
        <button
          onClick={() => setIsCollapsed(false)}
          className="flex flex-col items-center justify-center gap-2 p-3 border-b border-[#232f48] hover:bg-[#1a2333] transition-colors"
          title="Expand Control Panel"
          aria-label="Expand Control Panel"
        >
          <span className="material-symbols-outlined text-[#92a4c9]">chevron_right</span>
        </button>
        <div className="flex flex-col items-center gap-3 py-4">
          <span className="material-symbols-outlined text-sm text-[#6b7c9c]" title="Layers">layers</span>
          <span className="material-symbols-outlined text-sm text-[#6b7c9c]" title="Footprints">hexagon</span>
        </div>
        {totalActiveOverlays > 0 && (
          <div className="mt-auto mb-4 flex flex-col items-center">
            <div className="w-6 h-6 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center">
              <span className="text-[9px] font-bold text-green-400">{totalActiveOverlays}</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── Expanded panel ──
  return (
    <div
      className={`relative flex flex-col bg-[#101622] ${isMobile ? "w-full" : "h-full border-r border-[#232f48]"}`}
      style={isMobile ? undefined : { width: panelWidth }}
    >
      {/* Resize handle */}
      {!isMobile && (
        <div
          className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}

      {/* Header */}
      <div className={lp.section}>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            {!isMobile && (
              <button
                onClick={() => setIsCollapsed(true)}
                className="p-1 rounded hover:bg-[#1a2333] transition-colors"
                title="Collapse Panel"
                aria-label="Collapse Panel"
              >
                <span className="material-symbols-outlined text-sm text-[#92a4c9]">chevron_left</span>
              </button>
            )}
            <h1 className="text-white text-xs font-bold uppercase tracking-wider">
              Control Center
            </h1>
          </div>
          <span className="text-[10px] text-[#92a4c9] font-mono">
            {visibleProducts.length} products
          </span>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto scrollbar-dark">
        <ViewModeSection
          mapMode={mapMode}
          onMapModeChange={onMapModeChange}
          baseLayer={baseLayer}
          onBaseLayerChange={onBaseLayerChange}
        />

        <NavigationSection
          onFlyToCoords={onFlyToCoords}
          viewBounds={viewBounds}
          onViewBoundsChange={onViewBoundsChange}
          viewBoundSelectionMode={viewBoundSelectionMode}
          onViewBoundSelectionModeChange={onViewBoundSelectionModeChange}
          showGrid={showGrid}
          onToggleGrid={onToggleGrid}
          showRegionLayer={showRegionLayer}
          onToggleRegionLayer={onToggleRegionLayer}
        />

        <IceHub
          scienceLayerVisibility={scienceLayerVisibility}
          onScienceLayerToggle={onScienceLayerToggle}
          scienceLayerDepth={scienceLayerDepth}
          onScienceLayerDepthChange={onScienceLayerDepthChange}
          scienceLayerOpacities={scienceLayerOpacities}
          onScienceLayerOpacity={onScienceLayerOpacity}
          swimLayer={swimLayer}
          onSwimLayerChange={onSwimLayerChange}
          swimIceLat={swimIceLat}
          swimIceLon={swimIceLon}
          accessibilityVisible={accessibilityVisible}
          onAccessibilityVisibleChange={onAccessibilityVisibleChange}
          accessibilityOpacity={accessibilityOpacity}
          onAccessibilityOpacityChange={onAccessibilityOpacityChange}
          accessibilityExplainMode={accessibilityExplainMode}
          onAccessibilityExplainModeChange={onAccessibilityExplainModeChange}
          fusionVisible={fusionVisible}
          onFusionVisibleChange={onFusionVisibleChange}
          fusionOpacity={fusionOpacity}
          onFusionOpacityChange={onFusionOpacityChange}
        />

        <FootprintSection
          instrumentVisibility={instrumentVisibility}
          onToggleInstrument={onToggleInstrument}
          onLoadFootprints={onLoadFootprints}
          footprintsLoading={footprintsLoading}
          footprintCounts={footprintCounts}
          highResOnly={highResOnly}
          onHighResOnlyChange={onHighResOnlyChange}
          showCustomData={showCustomData}
          onToggleCustomData={onToggleCustomData}
          onLoadCustomData={onLoadCustomData}
          customDataLoading={customDataLoading}
          customDatasets={customDatasets}
          onCustomDatasetToggle={onCustomDatasetToggle}
        />

        <AnalysisTools
          analysisMode={analysisMode}
          onAnalysisModeChange={onAnalysisModeChange}
          onShowRegionDashboard={onShowRegionDashboard}
          showMeasurementTools={showMeasurementTools}
          onToggleMeasurementTools={onToggleMeasurementTools}
        />

        <ProductsHub
          visibleProducts={visibleProducts}
          activeOverlays={activeOverlays}
          onSetOverlay={onSetOverlay}
          onSetOpacity={onSetOpacity}
          onSelectProduct={onSelectProduct}
          onFlyToProduct={onFlyToProduct}
          onDeactivateAll={onDeactivateAll}
          customDatasets={customDatasets}
          onCustomDatasetToggle={onCustomDatasetToggle}
          overlapFilter={overlapFilter}
          onOverlapFilterChange={onOverlapFilterChange}
          overlapStats={overlapStats}
        />

        <FieldNotesSection
          fieldNotes={fieldNotes}
          showFieldNotesOnMap={showFieldNotesOnMap}
          onToggleFieldNotesOnMap={onToggleFieldNotesOnMap}
          onFieldNoteClick={onFieldNoteClick}
          onActiveTagChange={onActiveTagChange}
        />
      </div>
    </div>
  );
}
