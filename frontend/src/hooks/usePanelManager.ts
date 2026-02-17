/**
 * usePanelManager — Intelligent panel auto-open, attention pulse, and persistence.
 *
 * Wraps the existing rightPanelCollapsed / mobilePanel states to add:
 * - ensurePanelVisible(reason) with debounce and manual-collapse cooldown
 * - Attention pulse coordination (CSS animation key)
 * - Autonomy toggle persisted to localStorage
 * - Section highlight tracking for new agent content
 */

import { useState, useCallback, useRef, useEffect } from "react";

// ── Types ──────────────────────────────────────────────────────────

export type PanelOpenReason =
  | "user_click"
  | "feature_select"
  | "search_result"
  | "agent_write"
  | "analysis_complete"
  | "planner_propose"
  | "keyboard_shortcut";

export interface PanelManagerConfig {
  rightPanelCollapsed: boolean;
  setRightPanelCollapsed: (v: boolean) => void;
  isMobile: boolean;
  setMobilePanel: (v: "none" | "layers" | "inspector") => void;
}

export interface PanelManagerAPI {
  /** Autonomy toggle — if false, only user_click opens panels */
  autoOpenEnabled: boolean;
  setAutoOpenEnabled: (v: boolean) => void;

  /** Core: ensure the right panel is visible */
  ensurePanelVisible: (reason?: PanelOpenReason) => void;

  /** Record that user manually collapsed (starts 5s cooldown) */
  recordManualCollapse: () => void;

  /** Attention pulse state for PanelAttentionWrapper */
  attentionPulse: boolean;
  attentionPulseKey: number;
  clearAttentionPulse: () => void;

  /** Whether the collapsed strip should pulse */
  stripPulse: boolean;
  stripPulseKey: number;
  clearStripPulse: () => void;

  /** Section highlight for new agent content */
  highlightedSection: string | null;
  setHighlightedSection: (id: string | null) => void;
}

// ── localStorage helpers ───────────────────────────────────────────

const STORAGE_KEY = "marslab_panel_prefs";
const COLLAPSE_COOLDOWN_MS = 5000;
const DEBOUNCE_MS = 300;

function loadPrefs(): { autoOpenEnabled: boolean } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (typeof parsed.autoOpenEnabled === "boolean") return parsed;
    }
  } catch {
    /* corrupt or unavailable */
  }
  return { autoOpenEnabled: true };
}

function savePrefs(prefs: { autoOpenEnabled: boolean }) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    /* storage full — silently fail */
  }
}

// ── Hook ───────────────────────────────────────────────────────────

export default function usePanelManager(config: PanelManagerConfig): PanelManagerAPI {
  const {
    rightPanelCollapsed,
    setRightPanelCollapsed,
    isMobile,
    setMobilePanel,
  } = config;

  // Autonomy toggle (persisted)
  const [autoOpenEnabled, setAutoOpenEnabledState] = useState(() => loadPrefs().autoOpenEnabled);

  const setAutoOpenEnabled = useCallback((v: boolean) => {
    setAutoOpenEnabledState(v);
    savePrefs({ autoOpenEnabled: v });
  }, []);

  // Manual collapse timestamp
  const lastManualCollapseRef = useRef(0);

  const recordManualCollapse = useCallback(() => {
    lastManualCollapseRef.current = Date.now();
  }, []);

  // Attention pulse (panel content)
  const [attentionPulse, setAttentionPulse] = useState(false);
  const [attentionPulseKey, setAttentionPulseKey] = useState(0);

  const clearAttentionPulse = useCallback(() => {
    setAttentionPulse(false);
  }, []);

  // Strip pulse (collapsed strip)
  const [stripPulse, setStripPulse] = useState(false);
  const [stripPulseKey, setStripPulseKey] = useState(0);

  const clearStripPulse = useCallback(() => {
    setStripPulse(false);
  }, []);

  // Section highlight
  const [highlightedSection, setHighlightedSection] = useState<string | null>(null);

  // Debounce timer
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Core: ensurePanelVisible
  const ensurePanelVisible = useCallback(
    (reason: PanelOpenReason = "user_click") => {
      // Non-user reasons respect autonomy toggle
      if (reason !== "user_click" && reason !== "feature_select" && reason !== "search_result" && reason !== "keyboard_shortcut") {
        if (!autoOpenEnabled) return;
      }

      // Debounce rapid calls
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        const withinCooldown =
          Date.now() - lastManualCollapseRef.current < COLLAPSE_COOLDOWN_MS;

        if (rightPanelCollapsed) {
          if (withinCooldown && reason !== "user_click") {
            // Don't reopen — pulse the collapsed strip instead
            setStripPulse(true);
            setStripPulseKey((k) => k + 1);
            return;
          }
          // Expand the panel
          setRightPanelCollapsed(false);
        }

        // Mobile: open inspector bottom sheet
        if (isMobile) {
          setMobilePanel("inspector");
        }

        // Trigger attention pulse on the panel content
        setAttentionPulse(true);
        setAttentionPulseKey((k) => k + 1);
      }, DEBOUNCE_MS);
    },
    [autoOpenEnabled, rightPanelCollapsed, isMobile, setRightPanelCollapsed, setMobilePanel],
  );

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return {
    autoOpenEnabled,
    setAutoOpenEnabled,
    ensurePanelVisible,
    recordManualCollapse,
    attentionPulse,
    attentionPulseKey,
    clearAttentionPulse,
    stripPulse,
    stripPulseKey,
    clearStripPulse,
    highlightedSection,
    setHighlightedSection,
  };
}
