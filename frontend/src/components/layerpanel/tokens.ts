/**
 * Design tokens for the LayerPanel component tree.
 * Single source of truth for colors, spacing, and reusable class strings.
 */
import type { OverlayType } from "../../pages/MainPage";
import type { InstrumentId } from "../../config/instrumentRegistry";

// ── Raw color values ──
export const LP_COLORS = {
  border: "#232f48",
  borderHover: "#3a4a68",
  textPrimary: "#92a4c9",
  textSecondary: "#6b7c9c",
  textMuted: "#4a5a7c",
  bgPanel: "#101622",
  bgSection: "#1a2333",
  bgInput: "#0a0f18",
  bgDeep: "#0d1219",
  placeholder: "#4a5568",
} as const;

// ── Reusable Tailwind class composites ──
export const lp = {
  // Section wrapper
  section: "p-3 border-b border-[#232f48]",
  // Typography
  h3: "text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest",
  h4: "text-[#92a4c9] text-[9px] font-bold uppercase tracking-widest",
  body: "text-[11px] font-medium",
  caption: "text-[9px] text-[#6b7c9c]",
  tiny: "text-[8px] text-[#6b7c9c]",
  // Form elements
  input:
    "w-full px-2 py-1 text-[11px] bg-[#0a0f18] border border-[#232f48] rounded text-white placeholder-[#4a5568] focus:border-primary focus:outline-none",
  select:
    "w-full bg-[#0a0f18] border border-[#232f48] rounded px-2 py-1 text-[10px] text-slate-300 focus:outline-none focus:border-primary/50",
  checkbox:
    "rounded bg-[#0a0f18] border-[#232f48] focus:ring-0 focus:ring-offset-0",
  // Buttons
  btnPrimary:
    "px-3 py-1.5 text-[10px] font-medium bg-primary/20 border border-primary/50 rounded text-primary hover:bg-primary/30 transition-colors",
  btnSecondary:
    "px-3 py-1.5 text-[10px] font-medium bg-[#1a2333] border border-[#232f48] rounded text-[#92a4c9] hover:border-[#3a4a68] transition-colors",
  btnSmall:
    "px-2 py-1 text-[9px] font-medium rounded transition-colors whitespace-nowrap",
  // Toggle buttons (active/inactive pair)
  toggleActive: "bg-primary/20 border border-primary/50 text-primary",
  toggleInactive:
    "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-primary/30",
} as const;

// ── Per-instrument styles (Tailwind static lookup) ──
export const INST_STYLES: Record<
  InstrumentId,
  { text: string; bgActive: string; checkbox: string; btn: string; btnLoading: string }
> = {
  crism: {
    text: "text-cyan-400",
    bgActive: "bg-cyan-500/15 border border-cyan-500/40",
    checkbox: "text-cyan-500",
    btn: "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30",
    btnLoading: "bg-cyan-500/10 text-cyan-400/50 border border-cyan-500/20 cursor-wait",
  },
  hirise: {
    text: "text-yellow-400",
    bgActive: "bg-yellow-500/15 border border-yellow-500/40",
    checkbox: "text-yellow-500",
    btn: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30",
    btnLoading: "bg-yellow-500/10 text-yellow-400/50 border border-yellow-500/20 cursor-wait",
  },
  sharad: {
    text: "text-orange-400",
    bgActive: "bg-orange-500/15 border border-orange-500/40",
    checkbox: "text-orange-500",
    btn: "bg-orange-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30",
    btnLoading: "bg-orange-500/10 text-orange-400/50 border border-orange-500/20 cursor-wait",
  },
  sharad_highres: {
    text: "text-amber-400",
    bgActive: "bg-amber-500/15 border border-amber-500/40",
    checkbox: "text-amber-500",
    btn: "bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30",
    btnLoading: "bg-amber-500/10 text-amber-400/50 border border-amber-500/20 cursor-wait",
  },
  ctx: {
    text: "text-pink-400",
    bgActive: "bg-pink-500/15 border border-pink-500/40",
    checkbox: "text-pink-500",
    btn: "bg-pink-500/20 text-pink-400 border border-pink-500/30 hover:bg-pink-500/30",
    btnLoading: "bg-pink-500/10 text-pink-400/50 border border-pink-500/20 cursor-wait",
  },
  hirise_dtm: {
    text: "text-amber-600",
    bgActive: "bg-amber-700/15 border border-amber-700/40",
    checkbox: "text-amber-600",
    btn: "bg-amber-700/20 text-amber-600 border border-amber-700/30 hover:bg-amber-700/30",
    btnLoading: "bg-amber-700/10 text-amber-600/50 border border-amber-700/20 cursor-wait",
  },
  crism_trr3: {
    text: "text-teal-400",
    bgActive: "bg-teal-500/15 border border-teal-500/40",
    checkbox: "text-teal-500",
    btn: "bg-teal-500/20 text-teal-400 border border-teal-500/30 hover:bg-teal-500/30",
    btnLoading: "bg-teal-500/10 text-teal-400/50 border border-teal-500/20 cursor-wait",
  },
};

// ── Overlay type display names ──
export const OVERLAY_LABELS: Record<OverlayType, { short: string; full: string; color: string }> = {
  quickview: { short: "QV", full: "Quickview", color: "emerald" },
  highres: { short: "HD", full: "High-Res", color: "purple" },
  browse_HYD: { short: "HYD", full: "Hydrated Minerals", color: "fuchsia" },
  browse_ICE: { short: "ICE", full: "Water Ice", color: "blue" },
  browse_IC2: { short: "IC2", full: "CO₂ Ice", color: "cyan" },
  score_ice: { short: "S-ICE", full: "Ice Score", color: "sky" },
  score_hyd: { short: "S-HYD", full: "Hydration Score", color: "rose" },
  mineral_cnn: { short: "MIN", full: "Mineral CNN", color: "amber" },
};

// ── Instrument badge colors (for product lists / field notes) ──
export type InstrumentType =
  | "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES"
  | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

export const INSTRUMENT_COLORS: Record<InstrumentType, { bg: string; text: string; border: string }> = {
  CRISM: { bg: "bg-cyan-500/20", text: "text-cyan-400", border: "border-cyan-500/30" },
  HIRISE: { bg: "bg-yellow-500/20", text: "text-yellow-400", border: "border-yellow-500/30" },
  SHARAD: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
  SHARAD_HIGHRES: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
  CTX: { bg: "bg-pink-500/20", text: "text-pink-400", border: "border-pink-500/30" },
  CUSTOM: { bg: "bg-fuchsia-500/20", text: "text-fuchsia-400", border: "border-fuchsia-500/30" },
  HIRISE_DTM: { bg: "bg-amber-700/20", text: "text-amber-600", border: "border-amber-700/30" },
  CRISM_TRR3: { bg: "bg-teal-500/20", text: "text-teal-400", border: "border-teal-500/30" },
};

// localStorage key for panel collapse state
export const PANEL_COLLAPSED_KEY = "marslab-layer-panel-collapsed";
