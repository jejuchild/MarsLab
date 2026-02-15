import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import {
  type SuggestionType,
  type CopilotRuleDef,
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
  INSTRUMENT_TIPS,
  IDLE_TIP_POOL,
} from "../config/copilotRules";

/* =========================================================
 * Types
 * =======================================================*/

export interface CopilotAction {
  label: string;
  onClick: () => void;
}

export interface CopilotSuggestion {
  id: string;
  type: SuggestionType;
  icon: string;
  title: string;
  detail: string;
  actions: CopilotAction[];
  priority: number;
  autoDismissMs: number;
  /** Timestamp when the suggestion was created */
  createdAt: number;
}

export type CopilotDispatch =
  | { type: "load_instrument"; instrument: string }
  | { type: "run_analysis"; mode: string }
  | { type: "fly_to"; lat: number; lon: number }
  | { type: "show_product"; productId: string };

export interface ContextCopilotProps {
  selectedProduct: { productId: string; instrument: string } | null;
  visibleInstruments: string[];
  currentLocation: { lat: number; lon: number } | null;
  analysisMode: string | null;
  overlayCount: number;
  overlapFilterEnabled: boolean;
  onAction: (action: CopilotDispatch) => void;
}

/* =========================================================
 * Constants
 * =======================================================*/

const MAX_QUEUE_SIZE = 5;
const IDLE_THRESHOLD_MS = 30000;
const LOCALSTORAGE_KEY = "marslab_copilot_dismissed";
const LOCALSTORAGE_ENABLED_KEY = "marslab_copilot_enabled";

/* =========================================================
 * LocalStorage helpers
 * =======================================================*/

function getDismissedTypes(): Set<string> {
  try {
    const raw = localStorage.getItem(LOCALSTORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return new Set(parsed as string[]);
  } catch {
    // Corrupt or over-quota — reset
    try { localStorage.removeItem(LOCALSTORAGE_KEY); } catch { /* noop */ }
  }
  return new Set();
}

function addDismissedType(ruleId: string): void {
  try {
    const current = getDismissedTypes();
    current.add(ruleId);
    // Cap at 100 entries to avoid quota issues
    const entries = [...current].slice(-100);
    localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage full — silently fail
  }
}

function getCopilotEnabled(): boolean {
  try {
    const raw = localStorage.getItem(LOCALSTORAGE_ENABLED_KEY);
    return raw !== "false";
  } catch {
    return true;
  }
}

function setCopilotEnabled(enabled: boolean): void {
  try {
    localStorage.setItem(LOCALSTORAGE_ENABLED_KEY, String(enabled));
  } catch { /* noop */ }
}

/* =========================================================
 * Component
 * =======================================================*/

export default function ContextCopilot({
  selectedProduct,
  visibleInstruments,
  currentLocation: _currentLocation,
  analysisMode,
  overlayCount,
  overlapFilterEnabled,
  onAction,
}: ContextCopilotProps) {
  /* -------------------------------------------------------
   * State
   * -----------------------------------------------------*/
  const [queue, setQueue] = useState<CopilotSuggestion[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [visible, setVisible] = useState(false);  // controls enter/exit animation
  const [enabled, setEnabled] = useState(getCopilotEnabled);

  // Cooldown tracker: ruleId -> last fire timestamp
  const cooldownsRef = useRef<Map<string, number>>(new Map());

  // Dismissed types from localStorage
  const dismissedRef = useRef<Set<string>>(getDismissedTypes());

  // Idle timer ref
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-dismiss timer ref
  const autoDismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Dismiss animation timer ref (for the 200ms exit delay)
  const dismissAnimTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Progress bar animation ref
  const progressStartRef = useRef<number>(0);
  const progressDurationRef = useRef<number>(0);
  const animFrameRef = useRef<number>(0);
  const [progressPercent, setProgressPercent] = useState(100);

  /* -------------------------------------------------------
   * Derived: current suggestion = highest-priority in queue
   * -----------------------------------------------------*/
  const currentSuggestion = useMemo(() => {
    if (queue.length === 0) return null;
    // Already sorted by priority desc on enqueue
    return queue[0];
  }, [queue]);

  const pendingCount = queue.length;

  /* -------------------------------------------------------
   * Helper: check cooldown
   * -----------------------------------------------------*/
  const canFire = useCallback((rule: CopilotRuleDef): boolean => {
    if (dismissedRef.current.has(rule.id)) return false;
    const lastFired = cooldownsRef.current.get(rule.id) ?? 0;
    return Date.now() - lastFired >= rule.cooldownMs;
  }, []);

  /* -------------------------------------------------------
   * Helper: enqueue a suggestion
   * -----------------------------------------------------*/
  const enqueue = useCallback(
    (
      rule: CopilotRuleDef,
      title: string,
      detail: string,
      actions: CopilotAction[],
    ) => {
      if (!canFire(rule)) return;

      cooldownsRef.current.set(rule.id, Date.now());

      const suggestion: CopilotSuggestion = {
        id: `${rule.id}_${Date.now()}`,
        type: rule.type,
        icon: rule.icon,
        title,
        detail,
        actions,
        priority: rule.priority,
        autoDismissMs: rule.autoDismissMs,
        createdAt: Date.now(),
      };

      setQueue((prev) => {
        // Don't add if a suggestion with the same rule prefix is already queued
        const rulePrefix = rule.id;
        if (prev.some((s) => s.id.startsWith(rulePrefix))) return prev;

        const next = [...prev, suggestion]
          .sort((a, b) => b.priority - a.priority)
          .slice(0, MAX_QUEUE_SIZE);
        return next;
      });
    },
    [canFire],
  );

  /* -------------------------------------------------------
   * Helper: dismiss current suggestion
   * -----------------------------------------------------*/
  const dismissCurrent = useCallback(() => {
    setExpanded(false);
    setVisible(false);
    // Remove after exit animation completes
    if (dismissAnimTimerRef.current) clearTimeout(dismissAnimTimerRef.current);
    dismissAnimTimerRef.current = setTimeout(() => {
      setQueue((prev) => prev.slice(1));
      dismissAnimTimerRef.current = null;
    }, 200);
  }, []);

  /* -------------------------------------------------------
   * Helper: permanently dismiss a suggestion type
   * -----------------------------------------------------*/
  const dismissPermanently = useCallback(
    (ruleId: string) => {
      // Extract the base rule ID (without timestamp suffix)
      const baseId = ruleId.replace(/_\d+$/, "");
      addDismissedType(baseId);
      dismissedRef.current.add(baseId);
      dismissCurrent();
    },
    [dismissCurrent],
  );

  /* -------------------------------------------------------
   * Toggle copilot on/off
   * -----------------------------------------------------*/
  const toggleEnabled = useCallback(() => {
    setEnabled((prev) => {
      const next = !prev;
      setCopilotEnabled(next);
      if (!next) {
        setQueue([]);
        setExpanded(false);
        setVisible(false);
      }
      return next;
    });
  }, []);

  /* -------------------------------------------------------
   * Auto-dismiss timer with progress bar
   * -----------------------------------------------------*/
  useEffect(() => {
    // Cleanup previous timers
    if (autoDismissTimerRef.current) {
      clearTimeout(autoDismissTimerRef.current);
      autoDismissTimerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }

    if (!currentSuggestion || hovered || expanded) {
      setProgressPercent(100);
      return;
    }

    const duration = currentSuggestion.autoDismissMs;
    if (duration <= 0) return;

    progressStartRef.current = performance.now();
    progressDurationRef.current = duration;

    // Animate the progress bar
    function tick() {
      const elapsed = performance.now() - progressStartRef.current;
      const remaining = Math.max(0, 1 - elapsed / progressDurationRef.current);
      setProgressPercent(remaining * 100);
      if (remaining > 0) {
        animFrameRef.current = requestAnimationFrame(tick);
      }
    }
    animFrameRef.current = requestAnimationFrame(tick);

    // Auto-dismiss after duration
    autoDismissTimerRef.current = setTimeout(() => {
      dismissCurrent();
    }, duration);

    return () => {
      if (autoDismissTimerRef.current) {
        clearTimeout(autoDismissTimerRef.current);
        autoDismissTimerRef.current = null;
      }
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = 0;
      }
    };
  }, [currentSuggestion, hovered, expanded, dismissCurrent]);

  /* -------------------------------------------------------
   * Show animation when new suggestion appears
   * -----------------------------------------------------*/
  useEffect(() => {
    if (currentSuggestion && enabled) {
      // Small delay for enter animation
      const t = setTimeout(() => setVisible(true), 50);
      return () => clearTimeout(t);
    }
  }, [currentSuggestion?.id, enabled]);

  /* -------------------------------------------------------
   * Trigger: selectedProduct changes
   * -----------------------------------------------------*/
  useEffect(() => {
    if (!enabled || !selectedProduct) return;

    const inst = selectedProduct.instrument.toUpperCase();

    // Map instrument to rule
    let rule: CopilotRuleDef | null = null;
    if (inst === "CRISM" || inst === "CRISM_TRR3") rule = CRISM_SELECTED;
    else if (inst === "HIRISE") rule = HIRISE_SELECTED;
    else if (inst === "SHARAD" || inst === "SHARAD_HIGHRES") rule = SHARAD_SELECTED;
    else if (inst === "CTX") rule = CTX_SELECTED;
    else if (inst === "HIRISE_DTM") rule = DTM_SELECTED;

    if (!rule) return;

    const tip = INSTRUMENT_TIPS[inst];
    if (!tip) return;

    const actions: CopilotAction[] = [];

    // Build context-aware actions
    if (inst === "CRISM" && !visibleInstruments.includes("sharad")) {
      actions.push({
        label: "Enable SHARAD",
        onClick: () => onAction({ type: "load_instrument", instrument: "SHARAD" }),
      });
    }
    if (inst === "CRISM_TRR3") {
      actions.push({
        label: "Run mineral classification",
        onClick: () => onAction({ type: "show_product", productId: selectedProduct.productId }),
      });
    }
    if (inst === "HIRISE" && !visibleInstruments.includes("hirise_dtm")) {
      actions.push({
        label: "Check for DTM",
        onClick: () => onAction({ type: "load_instrument", instrument: "HIRISE_DTM" }),
      });
    }
    if ((inst === "SHARAD" || inst === "SHARAD_HIGHRES") && !visibleInstruments.includes("crism")) {
      actions.push({
        label: "Enable CRISM",
        onClick: () => onAction({ type: "load_instrument", instrument: "CRISM" }),
      });
    }
    if (inst === "CTX") {
      if (!visibleInstruments.includes("crism")) {
        actions.push({
          label: "Enable CRISM",
          onClick: () => onAction({ type: "load_instrument", instrument: "CRISM" }),
        });
      }
    }
    if (inst === "HIRISE_DTM") {
      actions.push({
        label: "Open 3D viewer",
        onClick: () => onAction({ type: "show_product", productId: selectedProduct.productId }),
      });
    }

    enqueue(rule, tip.title, tip.detail, actions);
  }, [selectedProduct?.productId, selectedProduct?.instrument, enabled, visibleInstruments, onAction, enqueue]);

  /* -------------------------------------------------------
   * Trigger: overlayCount > 3
   * -----------------------------------------------------*/
  useEffect(() => {
    if (!enabled || overlayCount <= 3) return;

    enqueue(
      MANY_OVERLAYS,
      `${overlayCount} overlays active`,
      "Multiple data layers are displayed. Consider using the Overlap Filter to find areas with multi-instrument coverage, or use the Agentic AI for a comprehensive cross-instrument analysis.",
      [
        {
          label: "Launch Agentic AI",
          onClick: () => onAction({ type: "run_analysis", mode: "agentic" }),
        },
      ],
    );
  }, [overlayCount, enabled, onAction, enqueue]);

  /* -------------------------------------------------------
   * Trigger: overlap filter enabled
   * -----------------------------------------------------*/
  useEffect(() => {
    if (!enabled || !overlapFilterEnabled) return;

    enqueue(
      OVERLAP_FILTER_ENABLED,
      "Overlap filter active",
      "Products are now filtered to multi-instrument overlap zones. Look for CRISM-HiRISE pairs with DTMs for the most comprehensive analysis. Enable the Agentic AI for automated site assessment.",
      [
        {
          label: "Launch Agentic AI",
          onClick: () => onAction({ type: "run_analysis", mode: "agentic" }),
        },
      ],
    );
  }, [overlapFilterEnabled, enabled, onAction, enqueue]);

  /* -------------------------------------------------------
   * Trigger: analysisMode changes
   * -----------------------------------------------------*/
  useEffect(() => {
    if (!enabled || !analysisMode) return;

    if (analysisMode === "slope" || analysisMode === "slope3d") {
      enqueue(
        SLOPE_ANALYSIS_ACTIVE,
        "Slope analysis mode",
        "Click on the map to measure terrain slopes. Safe landing requires less than 15 degrees. Check nearby CRISM observations for mineral context -- phyllosilicates may indicate aqueous alteration history.",
        [
          {
            label: "Enable CRISM",
            onClick: () => onAction({ type: "load_instrument", instrument: "CRISM" }),
          },
        ],
      );
    } else if (analysisMode === "ai_analysis") {
      enqueue(
        AI_ANALYSIS_ACTIVE,
        "AI analysis active",
        "Gemini is analyzing the selected area. After reviewing results, try the Agentic AI for a multi-step automated assessment including slope, subsurface, and mineral analysis.",
        [
          {
            label: "Try Agentic AI",
            onClick: () => onAction({ type: "run_analysis", mode: "agentic" }),
          },
        ],
      );
    } else if (analysisMode === "agentic") {
      enqueue(
        AGENTIC_MODE_ACTIVE,
        "Agentic AI running",
        "The agent is performing a multi-step site assessment. It will analyze slope safety, subsurface radar, mineral composition, and data coverage. Results will appear in the Agentic panel.",
        [],
      );
    }
  }, [analysisMode, enabled, onAction, enqueue]);

  /* -------------------------------------------------------
   * Trigger: idle detection (30s with no prop changes)
   * -----------------------------------------------------*/
  const resetIdleTimer = useCallback(() => {
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
    }

    if (!enabled) return;

    idleTimerRef.current = setTimeout(() => {
      // Pick a random idle tip that isn't on cooldown
      const available = IDLE_TIP_POOL.filter((tip) => canFire(tip.rule));
      if (available.length === 0) return;

      const tip = available[Math.floor(Math.random() * available.length)];

      const actions: CopilotAction[] = [];
      if (tip.actionLabel) {
        actions.push({
          label: tip.actionLabel,
          onClick: () => {
            if (tip.actionType === "run_analysis") {
              onAction({ type: "run_analysis", mode: tip.actionPayload });
            }
          },
        });
      }

      enqueue(tip.rule, tip.title, tip.detail, actions);
    }, IDLE_THRESHOLD_MS);
  }, [enabled, canFire, onAction, enqueue]);

  // Reset idle timer whenever relevant props change
  useEffect(() => {
    resetIdleTimer();
    return () => {
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    };
  }, [selectedProduct, analysisMode, overlayCount, overlapFilterEnabled, resetIdleTimer]);

  /* -------------------------------------------------------
   * Cleanup on unmount
   * -----------------------------------------------------*/
  useEffect(() => {
    return () => {
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      if (autoDismissTimerRef.current) clearTimeout(autoDismissTimerRef.current);
      if (dismissAnimTimerRef.current) clearTimeout(dismissAnimTimerRef.current);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  /* -------------------------------------------------------
   * Render: minimized badge (when no active suggestion)
   * -----------------------------------------------------*/
  if (!enabled) {
    return (
      <button
        onClick={toggleEnabled}
        className="fixed bottom-4 right-4 z-40 flex items-center gap-1.5 rounded-full
                   bg-surface-dark/80 border border-border-dark px-3 py-1.5
                   text-slate-500 hover:text-slate-300 hover:border-slate-500/50
                   backdrop-blur-sm transition-all duration-200"
        title="Enable AI Copilot"
      >
        <span className="material-symbols-outlined text-sm">assistant</span>
        <span className="text-[10px] font-medium">Copilot off</span>
      </button>
    );
  }

  if (!currentSuggestion) {
    return (
      <div className="fixed bottom-4 right-4 z-40 flex items-center gap-1">
        <button
          onClick={toggleEnabled}
          className="flex items-center gap-1.5 rounded-full
                     bg-surface-dark/80 border border-border-dark px-3 py-1.5
                     text-slate-500 hover:text-slate-300 hover:border-slate-500/50
                     backdrop-blur-sm transition-all duration-200"
          title="AI Copilot active -- click to disable"
        >
          <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>
            assistant
          </span>
          <span className="text-[10px] font-medium text-cyan-400/70">Copilot</span>
        </button>
      </div>
    );
  }

  /* -------------------------------------------------------
   * Render: active suggestion card
   * -----------------------------------------------------*/
  return (
    <div
      className="fixed bottom-4 right-4 z-40 pointer-events-none"
      style={{ maxWidth: "min(320px, calc(100vw - 32px))" }}
    >
      {/* Suggestion card */}
      <div
        className={`pointer-events-auto rounded-xl border border-border-dark
                     bg-surface-dark/95 backdrop-blur-md shadow-2xl shadow-black/40
                     transition-all duration-200 ease-out
                     ${visible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"}`}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Header row */}
        <div className="flex items-start gap-2.5 p-3 pb-0">
          {/* Icon */}
          <div className="flex-shrink-0 mt-0.5">
            <div className="size-7 rounded-lg bg-cyan-500/15 flex items-center justify-center">
              <span className="material-symbols-outlined text-sm text-cyan-400">
                {currentSuggestion.icon}
              </span>
            </div>
          </div>

          {/* Title + dismiss */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between gap-2">
              <h4 className="text-xs font-semibold text-slate-200 leading-tight truncate">
                {currentSuggestion.title}
              </h4>
              <div className="flex items-center gap-0.5 flex-shrink-0">
                {pendingCount > 1 && (
                  <span
                    className="text-[9px] font-bold text-cyan-400/60 bg-cyan-500/10
                               rounded-full px-1.5 py-0.5 leading-none"
                    title={`${pendingCount - 1} more suggestion${pendingCount > 2 ? "s" : ""}`}
                  >
                    +{pendingCount - 1}
                  </span>
                )}
                <button
                  onClick={dismissCurrent}
                  className="p-0.5 text-slate-600 hover:text-slate-400 transition-colors"
                  title="Dismiss"
                >
                  <span className="material-symbols-outlined text-sm">close</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Detail text */}
        <div className="px-3 pt-1.5 pb-2">
          <p
            className={`text-[11px] leading-relaxed text-slate-400 transition-all duration-200 ${
              expanded ? "" : "line-clamp-2"
            }`}
          >
            {currentSuggestion.detail}
          </p>

          {/* Expand toggle if text is long */}
          {currentSuggestion.detail.length > 120 && (
            <button
              onClick={() => setExpanded((prev) => !prev)}
              className="text-[10px] text-cyan-400/70 hover:text-cyan-400 mt-0.5 transition-colors"
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          )}
        </div>

        {/* Action buttons + "Don't show" */}
        <div className="flex items-center gap-1.5 px-3 pb-2.5">
          {currentSuggestion.actions.map((action, i) => (
            <button
              key={i}
              onClick={() => {
                action.onClick();
                dismissCurrent();
              }}
              className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[10px]
                         font-semibold text-cyan-400 bg-cyan-500/10 border border-cyan-500/20
                         hover:bg-cyan-500/20 hover:border-cyan-500/30 transition-all"
            >
              {action.label}
            </button>
          ))}

          {/* "Don't show again" — always visible */}
          <button
            onClick={() => dismissPermanently(currentSuggestion.id)}
            className="ml-auto text-[9px] text-slate-600 hover:text-slate-400 transition-colors"
            title="Don't show this type again"
          >
            Don't show
          </button>
        </div>

        {/* Auto-dismiss progress bar (uses scaleX transform for GPU compositing) */}
        {currentSuggestion.autoDismissMs > 0 && !hovered && !expanded && (
          <div className="h-[2px] bg-border-dark/50 rounded-b-xl overflow-hidden">
            <div
              className="h-full w-full bg-cyan-500/40 origin-left"
              style={{
                transform: `scaleX(${progressPercent / 100})`,
                willChange: "transform",
              }}
            />
          </div>
        )}
      </div>

      {/* Copilot badge (bottom-right, below card) */}
      <div className="pointer-events-auto flex justify-end mt-1.5">
        <button
          onClick={toggleEnabled}
          className="flex items-center gap-1 rounded-full bg-surface-dark/70 border border-border-dark/50
                     px-2 py-1 text-slate-600 hover:text-slate-400 hover:border-slate-500/30
                     backdrop-blur-sm transition-all duration-200"
          title="Disable AI Copilot"
        >
          <span className="material-symbols-outlined text-xs" style={{ fontVariationSettings: "'FILL' 1" }}>
            assistant
          </span>
          <span className="text-[9px]">Copilot</span>
        </button>
      </div>
    </div>
  );
}
