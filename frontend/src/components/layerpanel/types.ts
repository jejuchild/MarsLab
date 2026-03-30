/**
 * Type definitions for LayerPanel and its sub-sections.
 * Re-exports external types for convenient access.
 */
import type {
  VisibleProduct,
  BaseLayerType,
  BoundingBox,
  MapMode,
  ActiveOverlays,
  OverlayType,
  CustomDataset,
  OverlapFilter,
} from "../../pages/MainPage";
import type { FieldNote } from "../../api/fieldnotes";
import type { OverlapStats } from "../../utils/overlapFilter";
import type { SwimMethod, DepthRange } from "../../api/swim_ice";
import type { InstrumentId } from "../../config/instrumentRegistry";

// ── Re-exports ──
export type {
  VisibleProduct,
  BaseLayerType,
  BoundingBox,
  MapMode,
  ActiveOverlays,
  OverlayType,
  CustomDataset,
  OverlapFilter,
  FieldNote,
  OverlapStats,
  SwimMethod,
  DepthRange,
  InstrumentId,
};

// ── Local types ──
export type InstrumentVisibility = Record<InstrumentId, boolean>;
export type FootprintCount = { count: number; truncated: boolean; total: number } | null;
export type AnalysisMode =
  | "slope"
  | "hirise_dtm_3d"
  | "line"
  | "ai_analysis"
  | "agentic"
  | "report"
  | "guided"
  | "region_stats"
  | "crater_detect"
  | "regolith"
  | "stratigraphy"
  | "attenuation"
  | "mineral_sequence"
  | "strat_column"
  | "pathfinder"
  | null;

// ── Section Props ──

export interface ViewModeSectionProps {
  mapMode: MapMode;
  onMapModeChange: (mode: MapMode) => void;
  baseLayer: BaseLayerType;
  onBaseLayerChange: (layer: BaseLayerType) => void;
}

export interface NavigationSectionProps {
  onFlyToCoords?: (lat: number, lon: number) => void;
  viewBounds: BoundingBox;
  onViewBoundsChange: (bounds: BoundingBox) => void;
  viewBoundSelectionMode?: boolean;
  onViewBoundSelectionModeChange?: (active: boolean) => void;
  showGrid?: boolean;
  onToggleGrid?: (v: boolean) => void;
  showRegionLayer?: boolean;
  onToggleRegionLayer?: (v: boolean) => void;
}

export interface IceHubProps {
  scienceLayerVisibility?: Record<SwimMethod, boolean>;
  onScienceLayerToggle?: (method: SwimMethod, visible: boolean) => void;
  scienceLayerDepth?: DepthRange;
  onScienceLayerDepthChange?: (depth: DepthRange) => void;
  scienceLayerOpacities?: Record<SwimMethod, number>;
  onScienceLayerOpacity?: (method: SwimMethod, opacity: number) => void;
  swimLayer?: string | false;
  onSwimLayerChange?: (layer: string | false) => void;
  swimIceLat?: number | null;
  swimIceLon?: number | null;
  accessibilityVisible?: boolean;
  onAccessibilityVisibleChange?: (v: boolean) => void;
  accessibilityOpacity?: number;
  onAccessibilityOpacityChange?: (v: number) => void;
  accessibilityExplainMode?: boolean;
  onAccessibilityExplainModeChange?: (v: boolean) => void;
  fusionVisible?: boolean;
  onFusionVisibleChange?: (v: boolean) => void;
  fusionOpacity?: number;
  onFusionOpacityChange?: (v: number) => void;
}

export interface FootprintSectionProps {
  instrumentVisibility: InstrumentVisibility;
  onToggleInstrument: (id: InstrumentId, v: boolean) => void;
  onLoadFootprints?: (instrument: string) => void;
  footprintsLoading?: Record<string, boolean>;
  footprintCounts?: Record<string, FootprintCount>;
  highResOnly?: boolean;
  onHighResOnlyChange?: (v: boolean) => void;
  showCustomData: boolean;
  onToggleCustomData: (v: boolean) => void;
  onLoadCustomData?: () => void;
  customDataLoading?: boolean;
  customDatasets?: CustomDataset[];
  onCustomDatasetToggle?: (id: string, visible: boolean) => void;
}

export interface AnalysisToolsProps {
  analysisMode?: AnalysisMode;
  onAnalysisModeChange?: (mode: AnalysisMode) => void;
  onShowRegionDashboard?: () => void;
  showMeasurementTools?: boolean;
  onToggleMeasurementTools?: (v: boolean) => void;
}

export interface ProductsHubProps {
  visibleProducts: VisibleProduct[];
  activeOverlays: ActiveOverlays;
  onSetOverlay?: (productId: string, type: OverlayType | null) => void;
  onSetOpacity?: (productId: string, opacity: number) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
  onFlyToProduct?: (productId: string) => void;
  onDeactivateAll?: () => void;
  customDatasets?: CustomDataset[];
  onCustomDatasetToggle?: (id: string, visible: boolean) => void;
  overlapFilter?: OverlapFilter;
  onOverlapFilterChange?: (filter: OverlapFilter) => void;
  overlapStats?: OverlapStats | null;
}

export interface FieldNotesSectionProps {
  fieldNotes: FieldNote[];
  showFieldNotesOnMap?: boolean;
  onToggleFieldNotesOnMap?: (v: boolean) => void;
  onFieldNoteClick?: (note: FieldNote) => void;
  onActiveTagChange?: (tag: string | null) => void;
}

// ── Main Container Props (flat — mirrors current interface for MainPage compat) ──
export interface LayerPanelProps {
  mapMode: MapMode;
  onMapModeChange: (mode: MapMode) => void;
  baseLayer: BaseLayerType;
  onBaseLayerChange: (layer: BaseLayerType) => void;
  viewBounds: BoundingBox;
  onViewBoundsChange: (bounds: BoundingBox) => void;
  instrumentVisibility: InstrumentVisibility;
  onToggleInstrument: (id: InstrumentId, v: boolean) => void;
  onLoadFootprints?: (instrument: string) => void;
  footprintsLoading?: Record<string, boolean>;
  footprintCounts?: Record<string, FootprintCount>;
  highResOnly?: boolean;
  onHighResOnlyChange?: (v: boolean) => void;
  visibleProducts?: VisibleProduct[];
  activeOverlays?: ActiveOverlays;
  onSetOverlay?: (productId: string, type: OverlayType | null) => void;
  onSetOpacity?: (productId: string, opacity: number) => void;
  onSelectProduct?: (product: VisibleProduct) => void;
  onFlyToProduct?: (productId: string) => void;
  onDeactivateAll?: () => void;
  showCustomData: boolean;
  onToggleCustomData: (v: boolean) => void;
  onLoadCustomData?: () => void;
  customDataLoading?: boolean;
  customDatasets?: CustomDataset[];
  onCustomDatasetToggle?: (id: string, visible: boolean) => void;
  analysisMode?: AnalysisMode;
  onAnalysisModeChange?: (mode: AnalysisMode) => void;
  onFlyToCoords?: (lat: number, lon: number) => void;
  viewBoundSelectionMode?: boolean;
  onViewBoundSelectionModeChange?: (active: boolean) => void;
  onShowRegionDashboard?: () => void;
  showGrid?: boolean;
  onToggleGrid?: (v: boolean) => void;
  showRegionLayer?: boolean;
  onToggleRegionLayer?: (v: boolean) => void;
  swimLayer?: string | false;
  onSwimLayerChange?: (layer: string | false) => void;
  swimIceLat?: number | null;
  swimIceLon?: number | null;
  scienceLayerVisibility?: Record<SwimMethod, boolean>;
  onScienceLayerToggle?: (method: SwimMethod, visible: boolean) => void;
  scienceLayerDepth?: DepthRange;
  onScienceLayerDepthChange?: (depth: DepthRange) => void;
  scienceLayerOpacities?: Record<SwimMethod, number>;
  onScienceLayerOpacity?: (method: SwimMethod, opacity: number) => void;
  fieldNotes?: FieldNote[];
  showFieldNotesOnMap?: boolean;
  onToggleFieldNotesOnMap?: (v: boolean) => void;
  onFieldNoteClick?: (note: FieldNote) => void;
  onActiveTagChange?: (tag: string | null) => void;
  overlapFilter?: OverlapFilter;
  onOverlapFilterChange?: (filter: OverlapFilter) => void;
  overlapStats?: OverlapStats | null;
  accessibilityVisible?: boolean;
  onAccessibilityVisibleChange?: (v: boolean) => void;
  accessibilityOpacity?: number;
  onAccessibilityOpacityChange?: (v: number) => void;
  accessibilityExplainMode?: boolean;
  onAccessibilityExplainModeChange?: (v: boolean) => void;
  fusionVisible?: boolean;
  onFusionVisibleChange?: (v: boolean) => void;
  fusionOpacity?: number;
  onFusionOpacityChange?: (v: number) => void;
  ctxMosaicOpacity?: number;
  onCtxMosaicOpacityChange?: (v: number) => void;
  showMeasurementTools?: boolean;
  onToggleMeasurementTools?: (v: boolean) => void;
  isMobile?: boolean;
}
