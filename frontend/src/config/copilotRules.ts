/**
 * Copilot Rules Configuration
 *
 * Each rule defines a trigger condition, suggestion template, priority,
 * and cooldown. The ContextCopilot evaluates these rules against the
 * current application state and enqueues matching suggestions.
 */

export type SuggestionType =
  | "product_context"
  | "overlap_hint"
  | "region_info"
  | "analysis_tip"
  | "idle_tip";

export interface CopilotRuleDef {
  /** Unique rule identifier */
  id: string;
  /** Suggestion type category */
  type: SuggestionType;
  /** Material Symbols icon name */
  icon: string;
  /** Priority (higher = shown first) */
  priority: number;
  /** Auto-dismiss duration in ms (0 = no auto-dismiss) */
  autoDismissMs: number;
  /** Minimum cooldown between re-fires in ms */
  cooldownMs: number;
}

/* =========================================================
 * Product Context Rules
 * Fired when a product is selected/inspected
 * =======================================================*/

export const CRISM_SELECTED: CopilotRuleDef = {
  id: "crism_selected",
  type: "product_context",
  icon: "science",
  priority: 80,
  autoDismissMs: 12000,
  cooldownMs: 30000,
};

export const HIRISE_SELECTED: CopilotRuleDef = {
  id: "hirise_selected",
  type: "product_context",
  icon: "photo_camera",
  priority: 70,
  autoDismissMs: 12000,
  cooldownMs: 30000,
};

export const SHARAD_SELECTED: CopilotRuleDef = {
  id: "sharad_selected",
  type: "product_context",
  icon: "radar",
  priority: 75,
  autoDismissMs: 12000,
  cooldownMs: 30000,
};

export const CTX_SELECTED: CopilotRuleDef = {
  id: "ctx_selected",
  type: "product_context",
  icon: "photo_camera",
  priority: 60,
  autoDismissMs: 10000,
  cooldownMs: 30000,
};

export const DTM_SELECTED: CopilotRuleDef = {
  id: "dtm_selected",
  type: "product_context",
  icon: "terrain",
  priority: 75,
  autoDismissMs: 12000,
  cooldownMs: 30000,
};

/* =========================================================
 * Overlap / Multi-Product Rules
 * =======================================================*/

export const MANY_OVERLAYS: CopilotRuleDef = {
  id: "many_overlays",
  type: "overlap_hint",
  icon: "layers",
  priority: 65,
  autoDismissMs: 10000,
  cooldownMs: 60000,
};

export const OVERLAP_FILTER_ENABLED: CopilotRuleDef = {
  id: "overlap_filter_on",
  type: "overlap_hint",
  icon: "filter_alt",
  priority: 70,
  autoDismissMs: 12000,
  cooldownMs: 90000,
};

/* =========================================================
 * Analysis Mode Rules
 * =======================================================*/

export const SLOPE_ANALYSIS_ACTIVE: CopilotRuleDef = {
  id: "slope_active",
  type: "analysis_tip",
  icon: "landscape",
  priority: 60,
  autoDismissMs: 10000,
  cooldownMs: 120000,
};

export const AI_ANALYSIS_ACTIVE: CopilotRuleDef = {
  id: "ai_analysis_active",
  type: "analysis_tip",
  icon: "psychology",
  priority: 55,
  autoDismissMs: 10000,
  cooldownMs: 120000,
};

export const AGENTIC_MODE_ACTIVE: CopilotRuleDef = {
  id: "agentic_active",
  type: "analysis_tip",
  icon: "smart_toy",
  priority: 50,
  autoDismissMs: 10000,
  cooldownMs: 120000,
};

/* =========================================================
 * Idle Tips
 * =======================================================*/

export const IDLE_AGENTIC_TIP: CopilotRuleDef = {
  id: "idle_agentic",
  type: "idle_tip",
  icon: "smart_toy",
  priority: 30,
  autoDismissMs: 10000,
  cooldownMs: 300000, // 5 min between idle tips
};

export const IDLE_SLOPE_TIP: CopilotRuleDef = {
  id: "idle_slope",
  type: "idle_tip",
  icon: "landscape",
  priority: 25,
  autoDismissMs: 10000,
  cooldownMs: 300000,
};

export const IDLE_OVERLAP_TIP: CopilotRuleDef = {
  id: "idle_overlap",
  type: "idle_tip",
  icon: "filter_alt",
  priority: 20,
  autoDismissMs: 10000,
  cooldownMs: 300000,
};

/* =========================================================
 * All rules for iteration
 * =======================================================*/

export const ALL_RULES: CopilotRuleDef[] = [
  CRISM_SELECTED,
  HIRISE_SELECTED,
  SHARAD_SELECTED,
  CTX_SELECTED,
  DTM_SELECTED,
  MANY_OVERLAYS,
  OVERLAP_FILTER_ENABLED,
  SLOPE_ANALYSIS_ACTIVE,
  AI_ANALYSIS_ACTIVE,
  AGENTIC_MODE_ACTIVE,
  IDLE_AGENTIC_TIP,
  IDLE_SLOPE_TIP,
  IDLE_OVERLAP_TIP,
];

/**
 * Instrument-specific tip messages.
 * Used by ContextCopilot to build suggestions when a product is selected.
 */
export const INSTRUMENT_TIPS: Record<
  string,
  { title: string; detail: string; actionLabel?: string }
> = {
  CRISM: {
    title: "CRISM spectral observation selected",
    detail:
      "This is a mineral spectrometer footprint. Check the Spectrum tab for absorption features. SHARAD tracks may cross here -- enable radar footprints to investigate subsurface structure.",
    actionLabel: "Enable SHARAD",
  },
  CRISM_TRR3: {
    title: "CRISM TRR3 full-resolution data",
    detail:
      "Full hyperspectral cube available. Use the RGB compositor or run CNN mineral classification for automated mapping. Check if HiRISE DTMs are available for 3D context.",
    actionLabel: "Run mineral classification",
  },
  HIRISE: {
    title: "HiRISE high-resolution image",
    detail:
      "25 cm/pixel imagery. Look for surface textures: polygonal terrain, gullies, recurring slope lineae. If a DTM exists for this area, try the 3D viewer.",
    actionLabel: "Check for DTM",
  },
  SHARAD: {
    title: "SHARAD radar track selected",
    detail:
      "Subsurface radar sounding. Bright reflectors may indicate ice/rock interfaces. Enable SHARAD High-Res for improved clutter removal. Compare with CRISM to correlate surface minerals with subsurface structure.",
    actionLabel: "Enable SHARAD Hi-Res",
  },
  SHARAD_HIGHRES: {
    title: "SHARAD High-Resolution radargram",
    detail:
      "Enhanced clutter-removed radar profile. Look for layered deposits, basal reflectors, and dielectric contrasts. Cross-reference with nearby CRISM observations for mineral context.",
  },
  CTX: {
    title: "CTX context image",
    detail:
      "6 m/pixel regional context. Use this to identify geomorphic features (channels, craters, wrinkle ridges). Overlay with HiRISE for detail or CRISM for mineralogy.",
    actionLabel: "Enable CRISM",
  },
  HIRISE_DTM: {
    title: "HiRISE Digital Terrain Model",
    detail:
      "Stereo-derived topography at ~1 m/pixel. Launch the 3D viewer for slope analysis, or run the Agentic AI for a comprehensive landing site assessment.",
    actionLabel: "Open 3D viewer",
  },
};

/**
 * Idle tip rotation pool. One is picked at random when the user
 * has been inactive for 30+ seconds.
 */
export const IDLE_TIP_POOL = [
  {
    rule: IDLE_AGENTIC_TIP,
    title: "Try the Agentic AI",
    detail:
      "The Agentic AI runs a multi-step site assessment: slope safety, subsurface radar, mineral analysis, and data coverage -- all in one go.",
    actionLabel: "Launch Agentic AI",
    actionType: "run_analysis" as const,
    actionPayload: "agentic",
  },
  {
    rule: IDLE_SLOPE_TIP,
    title: "Analyze terrain slopes",
    detail:
      "Click any point on the map in Slope mode to get MOLA-derived slope statistics. Useful for evaluating landing site safety.",
    actionLabel: "Enable slope mode",
    actionType: "run_analysis" as const,
    actionPayload: "slope",
  },
  {
    rule: IDLE_OVERLAP_TIP,
    title: "Find multi-instrument coverage",
    detail:
      "The Overlap Filter finds areas where multiple instruments have data. Ideal for cross-validation of mineralogy with terrain and subsurface.",
    actionLabel: "Open overlap filter",
    actionType: "run_analysis" as const,
    actionPayload: "overlap",
  },
];
