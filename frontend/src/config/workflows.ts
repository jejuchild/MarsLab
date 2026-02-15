/**
 * Guided Investigation Workflow Definitions
 *
 * Each workflow is a sequence of steps that guide a user through a
 * common scientific investigation.  Steps carry an `actionType` that
 * the parent component maps to a concrete `WorkflowAction` dispatch.
 */

/* ---------------------------------------------------------------
 * Types
 * -------------------------------------------------------------*/
export type StepStatus = "pending" | "active" | "completed" | "skipped";

export interface WorkflowStepDef {
  id: string;
  title: string;
  description: string;
  /** Key the parent uses to build a WorkflowAction (may be null for manual steps). */
  actionType: string | null;
  /** If true the step may be skipped without blocking progress. */
  optional: boolean;
  /** Whether this step requires a location to be selected first. */
  requiresLocation: boolean;
}

export interface WorkflowDef {
  id: string;
  title: string;
  /** Material Symbols icon name */
  icon: string;
  description: string;
  /** Accent color class root (Tailwind), e.g. "sky" -> sky-400, sky-500/20, etc. */
  accent: string;
  steps: WorkflowStepDef[];
}

/* ---------------------------------------------------------------
 * Workflow Definitions
 * -------------------------------------------------------------*/

export const WORKFLOW_ICE: WorkflowDef = {
  id: "ice_detection",
  title: "Is there ice here?",
  icon: "ac_unit",
  description:
    "Detect water ice by combining CRISM spectral analysis with SHARAD subsurface radar.",
  accent: "sky",
  steps: [
    {
      id: "ice_pick_location",
      title: "Pick a location",
      description:
        "Click on the map to choose the area you want to investigate for ice signatures.",
      actionType: null,
      optional: false,
      requiresLocation: false,
    },
    {
      id: "ice_load_crism",
      title: "Load CRISM footprints",
      description:
        "Load CRISM mineral spectroscopy footprints to check for water-ice absorption bands.",
      actionType: "load_instrument",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "ice_check_score",
      title: "Check ice score",
      description:
        "Review the CRISM ice-score overlays to see how strong the spectral ice signature is.",
      actionType: "check_ice_score",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "ice_load_sharad",
      title: "Load SHARAD radar",
      description:
        "Load SHARAD subsurface radar tracks to look for dielectric reflections indicating buried ice.",
      actionType: "load_instrument",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "ice_check_subsurface",
      title: "Check subsurface",
      description:
        "Examine SHARAD radargrams for bright subsurface reflectors consistent with ice layers.",
      actionType: "check_subsurface",
      optional: true,
      requiresLocation: true,
    },
    {
      id: "ice_results",
      title: "View results summary",
      description:
        "Review the combined evidence from spectral and radar data for ice presence.",
      actionType: "show_results",
      optional: false,
      requiresLocation: false,
    },
  ],
};

export const WORKFLOW_LANDING: WorkflowDef = {
  id: "landing_site",
  title: "Is this a good landing site?",
  icon: "rocket_launch",
  description:
    "Evaluate a candidate landing site for slope safety, elevation, and multi-instrument coverage.",
  accent: "amber",
  steps: [
    {
      id: "land_pick_location",
      title: "Pick a location",
      description:
        "Click on the map to select the candidate landing site you want to evaluate.",
      actionType: null,
      optional: false,
      requiresLocation: false,
    },
    {
      id: "land_slope",
      title: "Run slope analysis",
      description:
        "Analyze terrain slope to determine if the surface is safe for landing (< 15 degrees).",
      actionType: "set_analysis_mode",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "land_elevation",
      title: "Check elevation",
      description:
        "Verify the site elevation is within acceptable limits for EDL (Entry, Descent, Landing).",
      actionType: "check_elevation",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "land_load_all",
      title: "Load all instruments",
      description:
        "Load footprints from CRISM, HiRISE, CTX, and SHARAD to assess data coverage.",
      actionType: "load_all_instruments",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "land_agentic",
      title: "Run Agentic AI analysis",
      description:
        "Launch the autonomous AI agent to perform a comprehensive multi-instrument site evaluation.",
      actionType: "run_agentic",
      optional: true,
      requiresLocation: true,
    },
    {
      id: "land_results",
      title: "View report",
      description:
        "Review the full site-evaluation report with composite scoring and recommendations.",
      actionType: "show_results",
      optional: false,
      requiresLocation: false,
    },
  ],
};

export const WORKFLOW_SUBSURFACE: WorkflowDef = {
  id: "subsurface_exploration",
  title: "What's the subsurface like?",
  icon: "radar",
  description:
    "Explore subsurface geology using SHARAD radar and HiRISE DTM terrain models.",
  accent: "orange",
  steps: [
    {
      id: "sub_pick_location",
      title: "Pick a location",
      description:
        "Click on the map to choose the area whose subsurface you want to explore.",
      actionType: null,
      optional: false,
      requiresLocation: false,
    },
    {
      id: "sub_load_sharad",
      title: "Load SHARAD + SHARAD High-Res",
      description:
        "Load both standard and high-resolution SHARAD radar tracks for the area.",
      actionType: "load_sharad",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "sub_load_dtm",
      title: "Load HiRISE DTM",
      description:
        "Load HiRISE Digital Terrain Models to correlate surface topography with subsurface features.",
      actionType: "load_instrument",
      optional: true,
      requiresLocation: true,
    },
    {
      id: "sub_epsilon_terrace",
      title: "Run subsurface analysis",
      description:
        "Analyze SHARAD radargrams and estimate dielectric properties of subsurface layers.",
      actionType: "run_subsurface_analysis",
      optional: false,
      requiresLocation: true,
    },
    {
      id: "sub_results",
      title: "View results",
      description:
        "Review the subsurface structure analysis with layer identification and dielectric estimates.",
      actionType: "show_results",
      optional: false,
      requiresLocation: false,
    },
  ],
};

/** All available workflows */
export const WORKFLOWS: WorkflowDef[] = [
  WORKFLOW_ICE,
  WORKFLOW_LANDING,
  WORKFLOW_SUBSURFACE,
];
