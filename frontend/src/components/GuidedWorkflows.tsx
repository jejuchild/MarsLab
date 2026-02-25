import { useReducer, useCallback, useMemo, useRef, useState } from "react";
import { WORKFLOWS, type WorkflowDef, type WorkflowStepDef, type StepStatus } from "../config/workflows";

/* =================================================================
 * Public types
 * ===============================================================*/

export type WorkflowAction =
  | { type: "fly_to"; lat: number; lon: number }
  | { type: "load_instrument"; instrument: string }
  | { type: "set_analysis_mode"; mode: string }
  | { type: "run_agentic"; query: string }
  | { type: "show_results"; data: Record<string, unknown> };

export interface GuidedWorkflowsProps {
  /** Whether the panel is visible (parent controls mounting, kept for API consistency) */
  isOpen: boolean;
  onClose: () => void;
  onAction: (action: WorkflowAction) => void;
  currentLocation?: { lat: number; lon: number } | null;
}

/* =================================================================
 * Internal state types
 * ===============================================================*/

interface StepState {
  status: StepStatus;
  resultSummary: string | null;
}

type View = "picker" | "stepper";

interface State {
  view: View;
  selectedWorkflowId: string | null;
  /** Per-step state, keyed by step id */
  steps: Record<string, StepState>;
  /** Index of the currently active step (0-based) */
  activeStepIndex: number;
}

const INITIAL_STATE: State = {
  view: "picker",
  selectedWorkflowId: null,
  steps: {},
  activeStepIndex: 0,
};

/* =================================================================
 * Reducer
 * ===============================================================*/

type Action =
  | { type: "SELECT_WORKFLOW"; workflow: WorkflowDef }
  | { type: "ADVANCE_STEP" }
  | { type: "SKIP_STEP" }
  | { type: "COMPLETE_STEP"; stepId: string; resultSummary?: string }
  | { type: "SET_RESULT"; stepId: string; resultSummary: string }
  | { type: "GO_BACK" }
  | { type: "RESET" };

function buildStepStates(workflow: WorkflowDef): Record<string, StepState> {
  const map: Record<string, StepState> = {};
  workflow.steps.forEach((step, i) => {
    map[step.id] = {
      status: i === 0 ? "active" : "pending",
      resultSummary: null,
    };
  });
  return map;
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SELECT_WORKFLOW": {
      return {
        view: "stepper",
        selectedWorkflowId: action.workflow.id,
        steps: buildStepStates(action.workflow),
        activeStepIndex: 0,
      };
    }

    case "ADVANCE_STEP": {
      const wf = WORKFLOWS.find((w) => w.id === state.selectedWorkflowId);
      if (!wf) return state;

      const currentStep = wf.steps[state.activeStepIndex];
      if (!currentStep) return state;

      const nextIndex = state.activeStepIndex + 1;
      const nextStep = wf.steps[nextIndex];

      const updatedSteps = { ...state.steps };

      // Mark current step completed if not already
      if (updatedSteps[currentStep.id]?.status !== "completed") {
        updatedSteps[currentStep.id] = {
          ...updatedSteps[currentStep.id]!,
          status: "completed",
        };
      }

      // Mark next step active if it exists
      if (nextStep) {
        updatedSteps[nextStep.id] = {
          ...updatedSteps[nextStep.id]!,
          status: "active",
        };
      }

      return {
        ...state,
        steps: updatedSteps,
        activeStepIndex: nextIndex,
      };
    }

    case "SKIP_STEP": {
      const wf = WORKFLOWS.find((w) => w.id === state.selectedWorkflowId);
      if (!wf) return state;

      const currentStep = wf.steps[state.activeStepIndex];
      if (!currentStep || !currentStep.optional) return state;

      const nextIndex = state.activeStepIndex + 1;
      const nextStep = wf.steps[nextIndex];

      const updatedSteps = { ...state.steps };
      updatedSteps[currentStep.id] = {
        ...updatedSteps[currentStep.id]!,
        status: "skipped",
      };

      if (nextStep) {
        updatedSteps[nextStep.id] = {
          ...updatedSteps[nextStep.id]!,
          status: "active",
        };
      }

      return {
        ...state,
        steps: updatedSteps,
        activeStepIndex: nextIndex,
      };
    }

    case "COMPLETE_STEP": {
      const existing = state.steps[action.stepId];
      if (!existing) return state;
      return {
        ...state,
        steps: {
          ...state.steps,
          [action.stepId]: {
            status: "completed",
            resultSummary: action.resultSummary ?? existing.resultSummary,
          },
        },
      };
    }

    case "SET_RESULT": {
      const existing = state.steps[action.stepId];
      if (!existing) return state;
      return {
        ...state,
        steps: {
          ...state.steps,
          [action.stepId]: {
            ...existing,
            resultSummary: action.resultSummary,
          },
        },
      };
    }

    case "GO_BACK":
      return { ...INITIAL_STATE };

    case "RESET":
      return { ...INITIAL_STATE };

    default:
      return state;
  }
}

/* =================================================================
 * Accent color helpers
 *
 * Tailwind cannot tree-shake dynamically-constructed class names, so
 * we map each accent token to its concrete classes at build time.
 * ===============================================================*/

const ACCENT_CLASSES: Record<
  string,
  {
    cardBorder: string;
    cardBg: string;
    text: string;
    textMuted: string;
    progressBg: string;
    progressBar: string;
    btnPrimary: string;
    btnBadge: string;
    stepActiveBorder: string;
    stepActiveRing: string;
    iconBg: string;
  }
> = {
  sky: {
    cardBorder: "border-sky-500/40",
    cardBg: "bg-sky-500/10",
    text: "text-sky-400",
    textMuted: "text-sky-400/70",
    progressBg: "bg-sky-900/30",
    progressBar: "bg-sky-500",
    btnPrimary:
      "bg-sky-500/20 text-sky-400 border border-sky-500/40 hover:bg-sky-500/30",
    btnBadge: "bg-sky-500/20 text-sky-400 border-sky-500/30",
    stepActiveBorder: "border-sky-500/50",
    stepActiveRing: "ring-sky-500/30",
    iconBg: "bg-sky-500/15",
  },
  amber: {
    cardBorder: "border-amber-500/40",
    cardBg: "bg-amber-500/10",
    text: "text-amber-400",
    textMuted: "text-amber-400/70",
    progressBg: "bg-amber-900/30",
    progressBar: "bg-amber-500",
    btnPrimary:
      "bg-amber-500/20 text-amber-400 border border-amber-500/40 hover:bg-amber-500/30",
    btnBadge: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    stepActiveBorder: "border-amber-500/50",
    stepActiveRing: "ring-amber-500/30",
    iconBg: "bg-amber-500/15",
  },
  orange: {
    cardBorder: "border-orange-500/40",
    cardBg: "bg-orange-500/10",
    text: "text-orange-400",
    textMuted: "text-orange-400/70",
    progressBg: "bg-orange-900/30",
    progressBar: "bg-orange-500",
    btnPrimary:
      "bg-orange-500/20 text-orange-400 border border-orange-500/40 hover:bg-orange-500/30",
    btnBadge: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    stepActiveBorder: "border-orange-500/50",
    stepActiveRing: "ring-orange-500/30",
    iconBg: "bg-orange-500/15",
  },
};

/* =================================================================
 * Sub-components
 * ===============================================================*/

/** Single step status indicator (the circle in the stepper line) */
function StepIndicator({ status, accent }: { status: StepStatus; accent: string }) {
  const a = (ACCENT_CLASSES[accent] ?? ACCENT_CLASSES.sky)!;

  const base =
    "w-6 h-6 rounded-full flex items-center justify-center shrink-0 transition-all duration-200";

  switch (status) {
    case "completed":
      return (
        <span className={`${base} bg-emerald-500/20 text-emerald-400 border border-emerald-500/40`}>
          <span className="material-symbols-outlined text-sm" style={{ fontSize: 14 }}>
            check
          </span>
        </span>
      );
    case "active":
      return (
        <span
          className={`${base} ${a.cardBg} ${a.text} border ${a.stepActiveBorder} ring-2 ${a.stepActiveRing}`}
        >
          <span
            className="material-symbols-outlined text-sm animate-pulse"
            style={{ fontSize: 14 }}
          >
            arrow_forward
          </span>
        </span>
      );
    case "skipped":
      return (
        <span className={`${base} bg-[#1a2333] text-[#6b7c9c] border border-[#232f48]`}>
          <span className="material-symbols-outlined text-sm" style={{ fontSize: 14 }}>
            remove
          </span>
        </span>
      );
    case "pending":
    default:
      return (
        <span className={`${base} bg-[#1a2333] text-[#3a4a68] border border-[#232f48]`}>
          <span className="text-[10px] font-mono font-bold">&bull;</span>
        </span>
      );
  }
}

/** Workflow picker card */
function WorkflowCard({
  workflow,
  onSelect,
}: {
  workflow: WorkflowDef;
  onSelect: (wf: WorkflowDef) => void;
}) {
  const a = (ACCENT_CLASSES[workflow.accent] ?? ACCENT_CLASSES.sky)!;

  return (
    <button
      onClick={() => onSelect(workflow)}
      className={`w-full text-left p-4 rounded-lg border ${a.cardBorder} ${a.cardBg}
        hover:brightness-110 transition-all duration-150 group cursor-pointer`}
    >
      <div className="flex items-start gap-3">
        <div
          className={`w-10 h-10 rounded-lg ${a.iconBg} flex items-center justify-center shrink-0`}
        >
          <span className={`material-symbols-outlined ${a.text}`} style={{ fontSize: 22 }}>
            {workflow.icon}
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <h3 className={`text-sm font-semibold ${a.text} leading-tight`}>
            {workflow.title}
          </h3>
          <p className="text-[11px] text-[#92a4c9] mt-1 leading-relaxed">
            {workflow.description}
          </p>
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-[9px] text-[#6b7c9c] font-mono">
              {workflow.steps.length} steps
            </span>
            <span className="text-[#3a4a68]">&middot;</span>
            <span className="text-[9px] text-[#6b7c9c] font-mono">
              ~{Math.ceil(workflow.steps.length * 1.5)} min
            </span>
          </div>
        </div>
        <span
          className={`material-symbols-outlined ${a.textMuted} text-lg
            group-hover:translate-x-0.5 transition-transform`}
        >
          chevron_right
        </span>
      </div>
    </button>
  );
}

/* =================================================================
 * Main component
 * ===============================================================*/

export default function GuidedWorkflows({
  onClose,
  onAction,
  currentLocation,
}: GuidedWorkflowsProps) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  /* Resizable panel (same pattern as AgenticPanel) */
  const [panelWidth, setPanelWidth] = useState(480);
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = panelWidth;
      const onMove = (ev: MouseEvent) => {
        const delta = startX - ev.clientX;
        const maxW = Math.floor(window.innerWidth * 0.6);
        setPanelWidth(Math.max(340, Math.min(maxW, startW + delta)));
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

  const scrollRef = useRef<HTMLDivElement>(null);

  /* Derived data */
  const selectedWorkflow = useMemo(
    () => WORKFLOWS.find((w) => w.id === state.selectedWorkflowId) ?? null,
    [state.selectedWorkflowId],
  );

  const accent = selectedWorkflow?.accent ?? "sky";
  const a = (ACCENT_CLASSES[accent] ?? ACCENT_CLASSES.sky)!;

  const totalSteps = selectedWorkflow?.steps.length ?? 0;
  const completedOrSkipped = useMemo(() => {
    if (!selectedWorkflow) return 0;
    return selectedWorkflow.steps.filter((s) => {
      const ss = state.steps[s.id];
      return ss?.status === "completed" || ss?.status === "skipped";
    }).length;
  }, [selectedWorkflow, state.steps]);

  const progressPct = totalSteps > 0 ? Math.round((completedOrSkipped / totalSteps) * 100) : 0;
  const allDone = totalSteps > 0 && completedOrSkipped === totalSteps;

  /* Check if current step requires location and none is set */
  const currentStepDef: WorkflowStepDef | null =
    selectedWorkflow?.steps[state.activeStepIndex] ?? null;
  const locationMissing =
    currentStepDef?.requiresLocation === true && !currentLocation;

  /* ---------------------------------------------------------------
   * Handlers
   * -------------------------------------------------------------*/

  const handleSelectWorkflow = useCallback(
    (wf: WorkflowDef) => {
      dispatch({ type: "SELECT_WORKFLOW", workflow: wf });
    },
    [],
  );

  const handleRunStep = useCallback(() => {
    if (!selectedWorkflow || !currentStepDef) return;
    if (locationMissing) return; // blocked

    const stepDef = currentStepDef;

    // Dispatch the appropriate action to the parent
    switch (stepDef.actionType) {
      case "load_instrument": {
        // Map step ids to instrument names
        const instrumentMap: Record<string, string> = {
          ice_load_crism: "CRISM",
          ice_load_sharad: "SHARAD",
          sub_load_dtm: "HIRISE_DTM",
        };
        const instrument = instrumentMap[stepDef.id] ?? "CRISM";
        onAction({ type: "load_instrument", instrument });
        break;
      }

      case "load_sharad":
        onAction({ type: "load_instrument", instrument: "SHARAD" });
        // Also trigger SHARAD_HIGHRES after a brief delay
        setTimeout(() => {
          onAction({ type: "load_instrument", instrument: "SHARAD_HIGHRES" });
        }, 200);
        break;

      case "load_all_instruments":
        for (const inst of ["CRISM", "HIRISE", "SHARAD", "CTX"]) {
          onAction({ type: "load_instrument", instrument: inst });
        }
        break;

      case "set_analysis_mode":
        onAction({ type: "set_analysis_mode", mode: "slope" });
        break;

      case "check_ice_score":
        // Show ice score overlays
        onAction({ type: "show_results", data: { overlay: "score_ice" } });
        break;

      case "check_elevation":
        onAction({ type: "set_analysis_mode", mode: "slope" });
        break;

      case "check_subsurface":
        onAction({
          type: "show_results",
          data: { panel: "sharad_inspector" },
        });
        break;

      case "run_agentic": {
        const query = currentLocation
          ? `Evaluate landing site at ${currentLocation.lat.toFixed(2)}N, ${currentLocation.lon.toFixed(2)}E`
          : "Evaluate this landing site";
        onAction({ type: "run_agentic", query });
        break;
      }

      case "run_subsurface_analysis":
        onAction({ type: "set_analysis_mode", mode: "slope" });
        break;

      case "show_results":
        onAction({
          type: "show_results",
          data: {
            workflowId: selectedWorkflow.id,
            completedSteps: completedOrSkipped,
            totalSteps,
          },
        });
        break;

      case null:
        // Manual step (e.g. "pick location") - just advance
        break;

      default:
        break;
    }

    // Mark the step completed and advance
    dispatch({
      type: "COMPLETE_STEP",
      stepId: stepDef.id,
      resultSummary: stepDef.actionType === null ? "Location selected" : "Done",
    });
    dispatch({ type: "ADVANCE_STEP" });
  }, [
    selectedWorkflow,
    currentStepDef,
    locationMissing,
    onAction,
    currentLocation,
    completedOrSkipped,
    totalSteps,
  ]);

  const handleSkipStep = useCallback(() => {
    dispatch({ type: "SKIP_STEP" });
  }, []);

  const handleGoBack = useCallback(() => {
    dispatch({ type: "GO_BACK" });
  }, []);

  const handleRestart = useCallback(() => {
    if (selectedWorkflow) {
      dispatch({ type: "SELECT_WORKFLOW", workflow: selectedWorkflow });
    }
  }, [selectedWorkflow]);

  /* ---------------------------------------------------------------
   * Render
   * -------------------------------------------------------------*/

  return (
    <div
      className="h-full flex flex-row bg-[#101622] border-l border-[#232f48]"
      style={{ width: panelWidth }}
    >
      {/* Resize handle */}
      <div
        className="w-1 cursor-col-resize hover:bg-sky-500/30 active:bg-sky-500/50 transition-colors shrink-0"
        onMouseDown={handleResizeStart}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize panel"
      />

      {/* Panel body */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#232f48] shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            {state.view === "stepper" && (
              <button
                onClick={handleGoBack}
                className="text-[#6b7c9c] hover:text-white transition-colors p-0.5 rounded"
                aria-label="Back to workflow list"
              >
                <span className="material-symbols-outlined text-lg">arrow_back</span>
              </button>
            )}
            <span className="material-symbols-outlined text-sky-400 text-lg">
              {state.view === "stepper" && selectedWorkflow
                ? selectedWorkflow.icon
                : "explore"}
            </span>
            <h2 className="text-sm font-semibold text-white truncate">
              {state.view === "stepper" && selectedWorkflow
                ? selectedWorkflow.title
                : "Guided Workflows"}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-[#6b7c9c] hover:text-white transition-colors p-0.5 rounded"
            aria-label="Close panel"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Progress bar (stepper view only) */}
        {state.view === "stepper" && selectedWorkflow && (
          <div className="px-4 py-2 border-b border-[#232f48] shrink-0">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] text-[#6b7c9c] font-mono uppercase tracking-wider">
                Progress
              </span>
              <span className={`text-[10px] font-mono font-bold ${a.text}`}>
                {completedOrSkipped}/{totalSteps} ({progressPct}%)
              </span>
            </div>
            <div
              className={`h-1.5 rounded-full ${a.progressBg} overflow-hidden`}
              role="progressbar"
              aria-valuenow={progressPct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`Workflow progress: ${progressPct}%`}
            >
              <div
                className={`h-full rounded-full ${a.progressBar} transition-all duration-500 ease-out`}
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Content area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto scrollbar-dark"
        >
          {state.view === "picker" ? (
            /* ---- Workflow Picker ---- */
            <div className="p-4 space-y-3">
              <p className="text-[11px] text-[#92a4c9] leading-relaxed mb-4">
                Select an investigation workflow to get step-by-step guidance
                through a common scientific analysis.
              </p>
              {WORKFLOWS.map((wf) => (
                <WorkflowCard
                  key={wf.id}
                  workflow={wf}
                  onSelect={handleSelectWorkflow}
                />
              ))}
            </div>
          ) : selectedWorkflow ? (
            /* ---- Stepper View ---- */
            <div className="p-4">
              {/* Current location indicator */}
              {currentLocation ? (
                <div className="flex items-center gap-2 mb-4 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                  <span className="material-symbols-outlined text-emerald-400 text-sm">
                    location_on
                  </span>
                  <span className="text-[11px] text-emerald-400 font-mono">
                    {currentLocation.lat.toFixed(3)}N, {currentLocation.lon.toFixed(3)}E
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-2 mb-4 px-3 py-2 rounded-lg bg-[#1a2333] border border-[#232f48]">
                  <span className="material-symbols-outlined text-[#6b7c9c] text-sm">
                    location_off
                  </span>
                  <span className="text-[11px] text-[#6b7c9c]">
                    No location selected. Click the map to pick one.
                  </span>
                </div>
              )}

              {/* Steps list */}
              <ol
                className="relative space-y-0"
                role="list"
                aria-label={`Steps for: ${selectedWorkflow.title}`}
              >
                {selectedWorkflow.steps.map((stepDef, idx) => {
                  const ss = state.steps[stepDef.id] ?? {
                    status: "pending" as StepStatus,
                    resultSummary: null,
                  };
                  const isActive = ss.status === "active";
                  const isLast = idx === selectedWorkflow.steps.length - 1;

                  return (
                    <li
                      key={stepDef.id}
                      role="listitem"
                      aria-current={isActive ? "step" : undefined}
                      className="relative flex gap-3"
                    >
                      {/* Vertical connector line + indicator */}
                      <div className="flex flex-col items-center shrink-0">
                        <StepIndicator status={ss.status} accent={accent} />
                        {!isLast && (
                          <div
                            className={`w-px flex-1 min-h-[16px] ${
                              ss.status === "completed" || ss.status === "skipped"
                                ? "bg-emerald-500/30"
                                : "bg-[#232f48]"
                            }`}
                          />
                        )}
                      </div>

                      {/* Step content card */}
                      <div
                        className={`flex-1 mb-3 p-3 rounded-lg border transition-all duration-200 ${
                          isActive
                            ? `${a.cardBg} ${a.stepActiveBorder} ring-1 ${a.stepActiveRing}`
                            : ss.status === "completed"
                              ? "bg-emerald-500/5 border-emerald-500/20"
                              : ss.status === "skipped"
                                ? "bg-[#0d1117] border-[#1a2333]"
                                : "bg-[#0d1117] border-[#1a2333]"
                        }`}
                      >
                        {/* Step header */}
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span
                              className={`text-[10px] font-mono font-bold ${
                                isActive
                                  ? a.text
                                  : ss.status === "completed"
                                    ? "text-emerald-400"
                                    : "text-[#6b7c9c]"
                              }`}
                            >
                              {idx + 1}.
                            </span>
                            <h4
                              className={`text-[12px] font-semibold truncate ${
                                isActive
                                  ? "text-white"
                                  : ss.status === "completed"
                                    ? "text-emerald-300/80"
                                    : ss.status === "skipped"
                                      ? "text-[#6b7c9c] line-through"
                                      : "text-[#92a4c9]"
                              }`}
                            >
                              {stepDef.title}
                            </h4>
                          </div>

                          {/* Status badge */}
                          {ss.status === "completed" && (
                            <span className="text-[8px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold uppercase shrink-0">
                              Done
                            </span>
                          )}
                          {ss.status === "skipped" && (
                            <span className="text-[8px] px-1.5 py-0.5 rounded bg-[#1a2333] text-[#6b7c9c] border border-[#232f48] font-bold uppercase shrink-0">
                              Skipped
                            </span>
                          )}
                        </div>

                        {/* Step description (shown for active and pending) */}
                        {(isActive || ss.status === "pending") && (
                          <p className="text-[10px] text-[#92a4c9] leading-relaxed mt-1.5">
                            {stepDef.description}
                          </p>
                        )}

                        {/* Result summary (shown when completed) */}
                        {ss.status === "completed" && ss.resultSummary && (
                          <p className="text-[10px] text-emerald-400/70 mt-1 flex items-center gap-1">
                            <span
                              className="material-symbols-outlined"
                              style={{ fontSize: 12 }}
                            >
                              check_circle
                            </span>
                            {ss.resultSummary}
                          </p>
                        )}

                        {/* Action buttons (active step only) */}
                        {isActive && (
                          <div className="flex items-center gap-2 mt-3">
                            {locationMissing ? (
                              <div className="flex items-center gap-1.5 text-[10px] text-amber-400/80">
                                <span
                                  className="material-symbols-outlined"
                                  style={{ fontSize: 14 }}
                                >
                                  warning
                                </span>
                                Click the map to select a location first
                              </div>
                            ) : (
                              <button
                                onClick={handleRunStep}
                                className={`text-[11px] px-3 py-1.5 rounded-md font-medium transition-colors ${a.btnPrimary}`}
                              >
                                <span className="flex items-center gap-1.5">
                                  <span
                                    className="material-symbols-outlined"
                                    style={{ fontSize: 14 }}
                                  >
                                    {stepDef.actionType === null
                                      ? "check"
                                      : "play_arrow"}
                                  </span>
                                  {stepDef.actionType === null
                                    ? "Mark complete"
                                    : "Run this step"}
                                </span>
                              </button>
                            )}

                            {stepDef.optional && (
                              <button
                                onClick={handleSkipStep}
                                className="text-[11px] px-3 py-1.5 rounded-md font-medium
                                  text-[#6b7c9c] bg-[#1a2333] border border-[#232f48]
                                  hover:border-[#3a4a68] hover:text-[#92a4c9] transition-colors"
                              >
                                Skip
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ol>

              {/* Summary card when all steps are done */}
              {allDone && (
                <div className="mt-4 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="material-symbols-outlined text-emerald-400">
                      task_alt
                    </span>
                    <h3 className="text-sm font-semibold text-emerald-400">
                      Workflow complete
                    </h3>
                  </div>
                  <p className="text-[11px] text-emerald-300/70 leading-relaxed mb-1">
                    You have completed all steps in the{" "}
                    <strong>{selectedWorkflow.title}</strong> workflow.
                  </p>
                  <div className="text-[10px] text-[#92a4c9] space-y-0.5 mt-2">
                    {selectedWorkflow.steps.map((sd) => {
                      const ss = state.steps[sd.id];
                      return (
                        <div key={sd.id} className="flex items-center gap-1.5">
                          <span
                            className={`material-symbols-outlined ${
                              ss?.status === "completed"
                                ? "text-emerald-400"
                                : "text-[#6b7c9c]"
                            }`}
                            style={{ fontSize: 12 }}
                          >
                            {ss?.status === "completed"
                              ? "check_circle"
                              : "remove_circle_outline"}
                          </span>
                          <span
                            className={
                              ss?.status === "skipped"
                                ? "line-through text-[#6b7c9c]"
                                : ""
                            }
                          >
                            {sd.title}
                          </span>
                          {ss?.resultSummary && (
                            <span className="text-[#6b7c9c] ml-1">
                              - {ss.resultSummary}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* Action buttons */}
                  <div className="flex items-center gap-2 mt-3">
                    <button
                      onClick={handleRestart}
                      className="text-[11px] px-3 py-1.5 rounded-md font-medium
                        bg-emerald-500/20 text-emerald-400 border border-emerald-500/40
                        hover:bg-emerald-500/30 transition-colors"
                    >
                      <span className="flex items-center gap-1.5">
                        <span
                          className="material-symbols-outlined"
                          style={{ fontSize: 14 }}
                        >
                          replay
                        </span>
                        Restart
                      </span>
                    </button>
                    <button
                      onClick={handleGoBack}
                      className="text-[11px] px-3 py-1.5 rounded-md font-medium
                        text-[#6b7c9c] bg-[#1a2333] border border-[#232f48]
                        hover:border-[#3a4a68] hover:text-[#92a4c9] transition-colors"
                    >
                      Other workflows
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
