import { useState, useMemo } from "react";
import type { IceHubProps, SwimMethod, DepthRange } from "../types";
import OpacitySlider from "../shared/OpacitySlider";
import CollapsibleSection from "../shared/CollapsibleSection";
import SwimIcePanel from "../../SwimIcePanel";
import AccessibilityPanel from "../../AccessibilityPanel";
import { SWIM_METHODS } from "../../../api/swim_ice";
import { lp } from "../tokens";

// ── Per-method display metadata ──
const METHOD_META: Record<SwimMethod, { label: string; icon: string; activeClass: string }> = {
  neutron:          { label: "Neutron (WEH)",    icon: "radio_button_checked", activeClass: "bg-sky-500/10 border-sky-500/30" },
  thermal:          { label: "Thermal Inertia",  icon: "thermostat",           activeClass: "bg-orange-500/10 border-orange-500/30" },
  radar_surface:    { label: "Radar Surface",    icon: "radar",                activeClass: "bg-violet-500/10 border-violet-500/30" },
  radar_dielectric: { label: "Radar Dielectric", icon: "sensors",              activeClass: "bg-rose-500/10 border-rose-500/30" },
  geomorphic:       { label: "Geomorphology",    icon: "terrain",              activeClass: "bg-amber-500/10 border-amber-500/30" },
};

type IceTab = "methods" | "depth" | "access" | "fusion";

const TAB_DEFS: { id: IceTab; label: string; icon: string }[] = [
  { id: "methods", label: "Methods", icon: "science" },
  { id: "depth",   label: "Depth Map", icon: "water_drop" },
  { id: "access",  label: "Accessibility", icon: "explore" },
  { id: "fusion",  label: "Fusion", icon: "hub" },
];

export default function IceHub({
  scienceLayerVisibility = {} as Record<SwimMethod, boolean>,
  onScienceLayerToggle,
  scienceLayerDepth,
  onScienceLayerDepthChange,
  scienceLayerOpacities = {} as Record<SwimMethod, number>,
  onScienceLayerOpacity,
  swimLayer,
  onSwimLayerChange,
  swimIceLat,
  swimIceLon,
  accessibilityVisible,
  onAccessibilityVisibleChange,
  accessibilityOpacity = 0.7,
  onAccessibilityOpacityChange,
  accessibilityExplainMode,
  onAccessibilityExplainModeChange,
  fusionVisible,
  onFusionVisibleChange,
  fusionOpacity = 0.7,
  onFusionOpacityChange,
}: IceHubProps) {
  const [activeTab, setActiveTab] = useState<IceTab>("methods");

  // Badge counts per tab
  const methodCount = useMemo(
    () => SWIM_METHODS.filter((m) => scienceLayerVisibility[m]).length,
    [scienceLayerVisibility],
  );

  const tabBadge = (tab: IceTab): number | null => {
    if (tab === "methods" && methodCount > 0) return methodCount;
    if (tab === "depth" && swimLayer) return 1;
    if (tab === "access" && accessibilityVisible) return 1;
    if (tab === "fusion" && fusionVisible) return 1;
    return null;
  };

  // ── Trailing badge for collapse header ──
  const totalActive =
    methodCount + (swimLayer ? 1 : 0) + (accessibilityVisible ? 1 : 0) + (fusionVisible ? 1 : 0);

  const trailing = totalActive > 0 ? (
    <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-bold">
      {totalActive}
    </span>
  ) : undefined;

  return (
    <CollapsibleSection title="Ice Detection" icon="ac_unit" defaultOpen={false} storageKey="ice" trailing={trailing}>
      {/* ── Tab Bar ── */}
      <div className="flex gap-1 mb-3">
        {TAB_DEFS.map((tab) => {
          const isActive = activeTab === tab.id;
          const badge = tabBadge(tab.id);
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`${lp.btnSmall} flex items-center gap-1 flex-1 justify-center border ${
                isActive
                  ? "bg-primary/20 text-primary border-primary/50"
                  : "bg-[#1a2333] text-[#6b7c9c] border-[#232f48] hover:text-[#92a4c9]"
              }`}
            >
              <span className="material-symbols-outlined text-xs">{tab.icon}</span>
              <span className="hidden sm:inline">{tab.label}</span>
              {badge !== null && (
                <span className="text-[7px] px-1 py-px rounded-full bg-primary/30 text-primary font-bold leading-none">
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* ── Tab Contents ── */}
      <div className="space-y-2">
        {activeTab === "methods" && (
          <MethodsTab
            scienceLayerVisibility={scienceLayerVisibility}
            onScienceLayerToggle={onScienceLayerToggle}
            scienceLayerDepth={scienceLayerDepth}
            onScienceLayerDepthChange={onScienceLayerDepthChange}
            scienceLayerOpacities={scienceLayerOpacities}
            onScienceLayerOpacity={onScienceLayerOpacity}
          />
        )}
        {activeTab === "depth" && (
          <DepthMapTab
            swimLayer={swimLayer}
            onSwimLayerChange={onSwimLayerChange}
            swimIceLat={swimIceLat}
            swimIceLon={swimIceLon}
          />
        )}
        {activeTab === "access" && (
          <AccessibilityTab
            accessibilityVisible={accessibilityVisible}
            onAccessibilityVisibleChange={onAccessibilityVisibleChange}
            accessibilityOpacity={accessibilityOpacity}
            onAccessibilityOpacityChange={onAccessibilityOpacityChange}
            accessibilityExplainMode={accessibilityExplainMode}
            onAccessibilityExplainModeChange={onAccessibilityExplainModeChange}
            swimIceLat={swimIceLat}
            swimIceLon={swimIceLon}
          />
        )}
        {activeTab === "fusion" && (
          <FusionTab
            fusionVisible={fusionVisible}
            onFusionVisibleChange={onFusionVisibleChange}
            fusionOpacity={fusionOpacity}
            onFusionOpacityChange={onFusionOpacityChange}
          />
        )}
      </div>
    </CollapsibleSection>
  );
}

/* ================================================================
 *  Methods Tab — 5 SWIM detection methods with depth buttons
 * ================================================================*/

function MethodsTab({
  scienceLayerVisibility,
  onScienceLayerToggle,
  scienceLayerDepth,
  onScienceLayerDepthChange,
  scienceLayerOpacities,
  onScienceLayerOpacity,
}: Pick<
  IceHubProps,
  | "scienceLayerVisibility"
  | "onScienceLayerToggle"
  | "scienceLayerDepth"
  | "onScienceLayerDepthChange"
  | "scienceLayerOpacities"
  | "onScienceLayerOpacity"
>) {
  return (
    <>
      <p className={lp.tiny}>Individual detection methods — independent of SWIM map</p>

      {/* Depth range buttons */}
      <div className="flex gap-1 mb-2">
        {(["0-1m", "1-5m", "5m-plus"] as DepthRange[]).map((d) => (
          <button
            key={d}
            onClick={() => onScienceLayerDepthChange?.(d)}
            className={`flex-1 px-2 py-1 rounded text-[9px] font-medium transition-colors ${
              scienceLayerDepth === d
                ? "bg-emerald-500/20 border border-emerald-500/50 text-emerald-400"
                : "bg-[#1a2333] border border-[#232f48] text-[#6b7c9c] hover:border-emerald-500/30"
            }`}
          >
            {d === "0-1m" ? "0–1 m" : d === "1-5m" ? "1–5 m" : ">5 m"}
          </button>
        ))}
      </div>

      {/* SWIM method toggles */}
      <div className="space-y-1">
        {SWIM_METHODS.map((method) => {
          const visible = scienceLayerVisibility?.[method] ?? false;
          const opacity = scienceLayerOpacities?.[method] ?? 0.7;
          const m = METHOD_META[method];

          return (
            <div
              key={method}
              className={`rounded border px-2 py-1.5 transition-colors ${
                visible ? m.activeClass : "border-[#232f48] bg-[#111b2a]"
              }`}
            >
              <label className="flex cursor-pointer items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={visible}
                  onChange={(e) => onScienceLayerToggle?.(method, e.target.checked)}
                  className="h-3 w-3 rounded border-[#232f48] bg-[#0d1520] accent-emerald-500"
                />
                <span className="material-symbols-outlined text-[14px] text-[#92a4c9]">{m.icon}</span>
                <span className={`text-[10px] font-medium ${visible ? "text-slate-200" : "text-[#92a4c9]"}`}>
                  {m.label}
                </span>
              </label>
              {visible && (
                <div className="pl-5 mt-1">
                  <OpacitySlider
                    value={opacity}
                    onChange={(v) => onScienceLayerOpacity?.(method, v)}
                    color="emerald"
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}

/* ================================================================
 *  Depth Map Tab — SWIM layer dropdown + SwimIcePanel
 * ================================================================*/

function DepthMapTab({
  swimLayer,
  onSwimLayerChange,
  swimIceLat,
  swimIceLon,
}: Pick<IceHubProps, "swimLayer" | "onSwimLayerChange" | "swimIceLat" | "swimIceLon">) {
  return (
    <>
      <div
        className={`p-2 rounded transition-colors ${
          swimLayer
            ? "bg-blue-500/20 border border-blue-400/50"
            : "bg-[#1a2333] border border-[#232f48] hover:border-blue-400/30"
        }`}
      >
        <div className="flex items-center gap-2 mb-1.5">
          <span className="material-symbols-outlined text-xs text-blue-400">water_drop</span>
          <span className={`text-[11px] font-medium ${swimLayer ? "text-blue-300" : "text-[#92a4c9]"}`}>
            SWIM Ice Map
          </span>
          <span className="text-[9px] text-slate-500 ml-auto">3 km/px</span>
        </div>
        <select
          value={swimLayer || ""}
          onChange={(e) => onSwimLayerChange?.(e.target.value || false)}
          className="w-full bg-[#0a0f18] border border-[#232f48] rounded px-2 py-1 text-[10px] text-slate-300 focus:outline-none focus:border-blue-400/50"
        >
          <option value="">Off</option>
          <option value="0-1m">Shallow (0-1 m)</option>
          <option value="1-5m">Mid-depth (1-5 m)</option>
          <option value=">5m">Deep (&gt;5 m)</option>
        </select>
        {swimLayer && (
          <p className="text-[9px] text-slate-500 mt-1">
            Source: SWIM4MIM (Morgan &amp; Putzig et al. 2025)
          </p>
        )}
      </div>

      {swimLayer && <SwimIcePanel lat={swimIceLat ?? null} lon={swimIceLon ?? null} />}
    </>
  );
}

/* ================================================================
 *  Accessibility Tab — heatmap toggle + explain mode
 * ================================================================*/

function AccessibilityTab({
  accessibilityVisible,
  onAccessibilityVisibleChange,
  accessibilityOpacity,
  onAccessibilityOpacityChange,
  accessibilityExplainMode,
  onAccessibilityExplainModeChange,
  swimIceLat,
  swimIceLon,
}: Pick<
  IceHubProps,
  | "accessibilityVisible"
  | "onAccessibilityVisibleChange"
  | "accessibilityOpacity"
  | "onAccessibilityOpacityChange"
  | "accessibilityExplainMode"
  | "onAccessibilityExplainModeChange"
  | "swimIceLat"
  | "swimIceLon"
>) {
  return (
    <>
      <div
        className={`p-2 rounded transition-colors ${
          accessibilityVisible
            ? "bg-emerald-500/20 border border-emerald-400/50"
            : "bg-[#1a2333] border border-[#232f48] hover:border-emerald-400/30"
        }`}
      >
        <div className="flex items-center gap-2 mb-1.5">
          <span className="material-symbols-outlined text-xs text-emerald-400">explore</span>
          <label className="flex items-center gap-1.5 flex-1 cursor-pointer">
            <input
              type="checkbox"
              checked={accessibilityVisible}
              onChange={(e) => onAccessibilityVisibleChange?.(e.target.checked)}
              className="accent-emerald-500 w-3 h-3"
            />
            <span
              className={`text-[11px] font-medium ${accessibilityVisible ? "text-emerald-300" : "text-[#92a4c9]"}`}
            >
              Ice Accessibility
            </span>
          </label>
          <span className="text-[9px] text-slate-500">heatmap</span>
        </div>
        {accessibilityVisible && (
          <div className="space-y-1.5 mt-1.5">
            <OpacitySlider
              value={accessibilityOpacity ?? 0.7}
              onChange={(v) => onAccessibilityOpacityChange?.(v)}
              color="emerald"
            />
            <p className="text-[9px] text-slate-500">SWIM + TES + MOLA composite score</p>
            <label className="flex items-center gap-1.5 mt-1 cursor-pointer">
              <input
                type="checkbox"
                checked={accessibilityExplainMode}
                onChange={(e) => onAccessibilityExplainModeChange?.(e.target.checked)}
                className="accent-emerald-500 w-3 h-3"
              />
              <span className="text-[9px] text-slate-400">Explain on click</span>
            </label>
          </div>
        )}
      </div>

      {accessibilityVisible && <AccessibilityPanel lat={swimIceLat ?? null} lon={swimIceLon ?? null} />}
    </>
  );
}

/* ================================================================
 *  Fusion Tab — Landform × Accessibility overlay
 * ================================================================*/

function FusionTab({
  fusionVisible,
  onFusionVisibleChange,
  fusionOpacity,
  onFusionOpacityChange,
}: Pick<IceHubProps, "fusionVisible" | "onFusionVisibleChange" | "fusionOpacity" | "onFusionOpacityChange">) {
  return (
    <div
      className={`p-2 rounded transition-colors ${
        fusionVisible
          ? "bg-violet-500/20 border border-violet-400/50"
          : "bg-[#1a2333] border border-[#232f48] hover:border-violet-400/30"
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className="material-symbols-outlined text-xs text-violet-400">hub</span>
        <label className="flex items-center gap-1.5 flex-1 cursor-pointer">
          <input
            type="checkbox"
            checked={fusionVisible}
            onChange={(e) => onFusionVisibleChange?.(e.target.checked)}
            className="accent-violet-500 w-3 h-3"
          />
          <span className={`text-[11px] font-medium ${fusionVisible ? "text-violet-300" : "text-[#92a4c9]"}`}>
            Ice Prospecting (Fusion)
          </span>
        </label>
        <span className="text-[9px] text-slate-500">landform×access</span>
      </div>
      {fusionVisible && (
        <div className="space-y-1.5 mt-1.5">
          <OpacitySlider
            value={fusionOpacity ?? 0.7}
            onChange={(v) => onFusionOpacityChange?.(v)}
            color="violet"
          />
          <p className="text-[9px] text-slate-500">
            SWIM 2.0 depth-stratified + HiRISE landform overlay
          </p>
        </div>
      )}
    </div>
  );
}
