import { useEffect, useState, useCallback, useMemo } from "react";
import toast from "react-hot-toast";
import { fetchHiRISEWindow } from "../api/hirise";
import type { OverlayType, ProductOverlay, CustomDataset } from "../pages/MainPage";
import type { FieldNote } from "../api/fieldnotes";
import BandRatioCalculator from "./BandRatioCalculator";
import HiriseLandformPanel from "./HiriseLandformPanel";

/* =========================================================
 * Types
 * =======================================================*/
export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

export type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  // CRISM pixel coordinates for spectrum (optional)
  pixelLine?: number;
  pixelSample?: number;
  // HiRISE observation title (e.g., "Gullies in Arcadia Region")
  title?: string;
};

export type RGBWavelengths = {
  r: number;
  g: number;
  b: number;
};

type HiRISETabKey = "Metadata" | "Pixel";
type CRISMTabKey = "Metadata" | "Spectrum" | "Bands";

type SpectrumData = {
  wavelengths: number[];
  reflectance: (number | null)[];
  validBands: number;
};

type DustAssessment = {
  tau_estimated: number;
  risk_level: "LOW" | "MODERATE" | "HIGH";
  spectral_slope: number | null;
  band_depth_suppression_pct: number;
  warning_message: string | null;
};

type WindowStats = {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  sum: number;
  histogram: number[];
  binEdges: number[];
};

// Default CRISM wavelengths (in micrometers)
const DEFAULT_RGB: RGBWavelengths = {
  r: 2.53,  // Red channel
  g: 1.51,  // Green channel
  b: 1.08,  // Blue channel
};

// Overlay type display configuration
const OVERLAY_CONFIG: Record<OverlayType, { label: string; activeClass: string; icon: string }> = {
  quickview: { label: "Quickview", activeClass: "bg-emerald-500/20 border border-emerald-500/50 text-emerald-400", icon: "visibility" },
  highres: { label: "High-Res", activeClass: "bg-purple-500/20 border border-purple-500/50 text-purple-400", icon: "hd" },
  browse_HYD: { label: "HYD", activeClass: "bg-fuchsia-500/20 border border-fuchsia-500/50 text-fuchsia-400", icon: "water_drop" },
  browse_ICE: { label: "ICE", activeClass: "bg-blue-500/20 border border-blue-500/50 text-blue-400", icon: "ac_unit" },
  browse_IC2: { label: "IC2", activeClass: "bg-cyan-500/20 border border-cyan-500/50 text-cyan-400", icon: "ac_unit" },
  score_ice: { label: "S-ICE", activeClass: "bg-sky-500/20 border border-sky-500/50 text-sky-400", icon: "analytics" },
  score_hyd: { label: "S-HYD", activeClass: "bg-rose-500/20 border border-rose-500/50 text-rose-400", icon: "analytics" },
  mineral_cnn: { label: "Minerals", activeClass: "bg-amber-500/20 border border-amber-500/50 text-amber-400", icon: "science" },
};

/* =========================================================
 * CRISM Quickview Image Component
 * Tries multiple URL patterns to find the quickview image
 * =======================================================*/
function CRISMQuickviewImage({ productId, instrument }: { productId: string; instrument: string }) {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (instrument === "HIRISE") {
      setImgSrc(`/hirise/quickview/${productId}.jpg`);
      return;
    }

    if (instrument === "HIRISE_DTM") {
      setImgSrc(`/hirise_dtm/overlay/${productId}.png`);
      return;
    }

    if (instrument === "CRISM_TRR3") {
      const obsId = productId.replace(/_\d{2}$/, "");
      setImgSrc(`/api/mineral-cnn/quickview/${obsId}`);
      return;
    }

    // For CRISM, try multiple URL patterns
    const obsId = productId.split("_")[0]; // e.g., frt00008a1e
    const baseKey = productId.replace(/_if[0-9a-z]+_mtr3$/i, "").replace(/_br[a-z]+_mtr3$/i, ""); // e.g., frt00008a1e_07

    const patterns = [
      `/crism/quickview/${baseKey}.png`,                    // frt00008a1e_07.png
      `/crism/quickview/${obsId}_VNIR.png`,                 // frt00008a1e_VNIR.png
      `/crism/browse/${baseKey}_brvnaj_mtr3.png`,           // frt00008a1e_07_brvnaj_mtr3.png (browse dir)
      `/crism/quickview/${baseKey}_brvnaj_mtr3.png`,        // frt00008a1e_07_brvnaj_mtr3.png (quickview dir)
      `/crism/browse/${productId.replace(/_if[0-9a-z]+_mtr3$/i, "_brvnaj_mtr3.png")}`, // fallback browse
    ];

    // Try each pattern until one works
    const tryPatterns = async () => {
      for (const url of patterns) {
        try {
          const res = await fetch(url, { method: "HEAD" });
          if (res.ok) {
            setImgSrc(url);
            setError(false);
            return;
          }
        } catch {
          continue;
        }
      }
      // No pattern worked
      setError(true);
    };

    tryPatterns();
  }, [productId, instrument]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-32 bg-surface-dark text-slate-500 text-sm">
        No quickview available
      </div>
    );
  }

  if (!imgSrc) {
    return (
      <div className="flex items-center justify-center h-32 bg-surface-dark text-slate-500">
        <span className="material-symbols-outlined animate-spin">progress_activity</span>
      </div>
    );
  }

  return <img src={imgSrc} className="w-full" alt="Quickview" />;
}

/* =========================================================
 * Inspector Component
 * =======================================================*/
type RecentProduct = { productId: string; instrument: InstrumentType; lat: number; lon: number; title?: string };

export default function Inspector({
  selected,
  onClose,
  activeOverlay,
  onSetOverlay,
  onSetOpacity,
  rgbWavelengths,
  onRGBChange,
  hasHighResData = true,
  customDataset = null,
  onCustomDatasetOpacity,
  fieldNotes = [],
  onOpenFieldNote,
  onShow3DView,
  onFindRelated,
  recentProducts = [],
  onSelectRecent,
  onRemoveRecent,
  onDownloadProduct,
  onPinSpectrum,
  onFindTemporalPairs,
  onCollapse,
  onOpenMineralSequence,
  isMobile = false,
}: {
  selected: InspectorContext | null;
  onClose: () => void;
  onCollapse?: () => void;
  activeOverlay: ProductOverlay | null;
  onSetOverlay: (type: OverlayType | null) => void;
  onSetOpacity?: (opacity: number) => void;
  rgbWavelengths?: RGBWavelengths;
  onRGBChange?: (rgb: RGBWavelengths) => void;
  hasHighResData?: boolean;
  customDataset?: CustomDataset | null;
  onCustomDatasetOpacity?: (id: string, opacity: number) => void;
  fieldNotes?: FieldNote[];
  onOpenFieldNote?: (productId: string, instrument: string, lat: number, lon: number) => void;
  onShow3DView?: (productId: string, lat: number, lon: number) => void;
  onFindRelated?: (productId: string, instrument: string) => void;
  recentProducts?: RecentProduct[];
  onSelectRecent?: (product: RecentProduct) => void;
  onRemoveRecent?: (productId: string) => void;
  onDownloadProduct?: (productId: string, instrument: string) => void;
  onPinSpectrum?: (spectrum: { productId: string; lat: number; lon: number; wavelengths: number[]; reflectance: (number | null)[] }) => void;
  onFindTemporalPairs?: (lat: number, lon: number, instrument: string) => void;
  onOpenMineralSequence?: (obsId: string) => void;
  isMobile?: boolean;
}) {
  const hasNote = useMemo(
    () => selected ? fieldNotes.some(n => n.product_id === selected.productId) : false,
    [fieldNotes, selected]
  );
  const noteCount = useMemo(
    () => selected ? fieldNotes.filter(n => n.product_id === selected.productId).length : 0,
    [fieldNotes, selected]
  );

  // Keyboard: Escape to close panel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const [hiriseTab, setHiriseTab] = useState<HiRISETabKey>("Metadata");
  const [crismTab, setCrismTab] = useState<CRISMTabKey>("Metadata");
  const [windowSize, setWindowSize] = useState(5);
  const [stats, setStats] = useState<WindowStats | null>(null);
  const [loading, setLoading] = useState(false);

  // Panel width (resizable)
  const [panelWidth, setPanelWidth] = useState(384);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      const delta = startX - ev.clientX;
      const maxW = Math.floor(window.innerWidth * 0.6);
      setPanelWidth(Math.max(280, Math.min(maxW, startW + delta)));
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
  }, [panelWidth]);

  // CRISM spectrum state
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [dustAssessment, setDustAssessment] = useState<DustAssessment | null>(null);
  const [spectrumLoading, setSpectrumLoading] = useState(false);

  // Local RGB state for sliders
  const [localRGB, setLocalRGB] = useState<RGBWavelengths>(rgbWavelengths || DEFAULT_RGB);

  // Opacity slider expanded state
  const [showOpacity, setShowOpacity] = useState(false);
  const [showLandformPanel, setShowLandformPanel] = useState(false);
  const [showOverlaySection, setShowOverlaySection] = useState(false);

  // Sync local state with props
  useEffect(() => {
    if (rgbWavelengths) {
      setLocalRGB(rgbWavelengths);
    }
  }, [rgbWavelengths]);

  // Fetch HiRISE pixel stats when selection or window size changes
  useEffect(() => {
    if (!selected || selected.instrument !== "HIRISE") {
      setStats(null);
      return;
    }

    // Need pixel coordinates to fetch stats
    if (selected.pixelLine === undefined || selected.pixelSample === undefined) {
      setStats(null);
      return;
    }

    // Capture values for async function
    const productId = selected.productId;
    const pixelSample = selected.pixelSample;
    const pixelLine = selected.pixelLine;

    const controller = new AbortController();

    async function fetchStats() {
      setLoading(true);
      try {
        const halfSize = Math.floor(windowSize / 2);
        // Use pixel coordinates (sample = x, line = y)
        const data = await fetchHiRISEWindow(
          productId,
          pixelSample,  // x = sample (column)
          pixelLine,    // y = line (row)
          halfSize
        );

        const dn = data.dn.flat() as number[];
        if (dn.length === 0) {
          setStats(null);
          return;
        }

        const mean = dn.reduce((a, b) => a + b, 0) / dn.length;
        const sorted = [...dn].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)]!;
        const std = Math.sqrt(
          dn.reduce((acc, v) => acc + (v - mean) ** 2, 0) / dn.length
        );
        const min = Math.min(...dn);
        const max = Math.max(...dn);
        const sum = dn.reduce((a, b) => a + b, 0);

        const bins = 10;
        const binWidth = (max - min) / bins || 1;
        const histogram = new Array(bins).fill(0);
        const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + i * binWidth);

        dn.forEach((v) => {
          const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
          histogram[idx]++;
        });

        setStats({ mean, median, std, min, max, sum, histogram, binEdges });
      } catch (e) {
        console.error("Failed to fetch stats:", e);
        setStats(null);
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
    return () => controller.abort();
  }, [selected?.productId, selected?.pixelLine, selected?.pixelSample, windowSize]);

  // Fetch CRISM spectrum when pixel coordinates change
  useEffect(() => {
    if (!selected || (selected.instrument !== "CRISM" && selected.instrument !== "CRISM_TRR3")) {
      setSpectrumData(null);
      setDustAssessment(null);
      return;
    }

    if (selected.pixelLine === undefined || selected.pixelSample === undefined) {
      setSpectrumData(null);
      setDustAssessment(null);
      return;
    }

    // Capture values for async function
    const productId = selected.productId;
    const pixelLine = selected.pixelLine;
    const pixelSample = selected.pixelSample;
    const isTRR3 = selected.instrument === "CRISM_TRR3";
    const lat = selected.lat;
    const lon = selected.lon;

    async function fetchSpectrum() {
      setSpectrumLoading(true);
      try {
        // TRR3: use /api/crism-trr3/{obs_id}/spectrum
        // MTRDR: use /crism/{product_id}/spectrum
        let url: string;
        if (isTRR3) {
          const obsId = productId.replace(/_\d{2}$/, "");
          url = `/api/crism-trr3/${obsId}/spectrum`;
        } else {
          url = `/crism/${productId}/spectrum`;
        }

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            line: pixelLine,
            sample: pixelSample,
            lat,
            lon,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch spectrum: ${response.status}`);
        }

        const data = await response.json();
        setSpectrumData({
          wavelengths: data.wavelengths,
          reflectance: data.reflectance,
          validBands: data.valid_bands,
        });
        setDustAssessment(data.dust_assessment ?? null);
      } catch (e) {
        console.error("Failed to fetch spectrum:", e);
        toast.error("Failed to load spectrum data");
        setSpectrumData(null);
        setDustAssessment(null);
      } finally {
        setSpectrumLoading(false);
      }
    }

    fetchSpectrum();
  }, [selected?.productId, selected?.pixelLine, selected?.pixelSample, selected?.lat, selected?.lon]);

  if (!selected) return null;

  const isHiRISE = selected.instrument === "HIRISE";
  const isCRISM = selected.instrument === "CRISM" || selected.instrument === "CRISM_TRR3";
  const isTRR3 = selected.instrument === "CRISM_TRR3";
  const isCustom = selected.instrument === "CUSTOM";
  const isDTM = selected.instrument === "HIRISE_DTM";

  const handleRGBSliderChange = (channel: "r" | "g" | "b", value: number) => {
    const newRGB = { ...localRGB, [channel]: value };
    setLocalRGB(newRGB);
  };

  const handleApplyRGB = () => {
    // Auto-activate highres overlay if not already active
    if (activeOverlay?.type !== "highres") {
      onSetOverlay("highres");
    }
    onRGBChange?.(localRGB);
  };

  // Define tabs based on instrument
  const hiriseTabs: HiRISETabKey[] = ["Metadata", "Pixel"];
  const crismTabs: CRISMTabKey[] = ["Metadata", "Spectrum", "Bands"];

  // Available overlay types for this instrument
  const availableOverlays: OverlayType[] = isTRR3
    ? ["quickview", "mineral_cnn"]
    : isCRISM
      ? ["quickview", "highres", "browse_HYD", "browse_ICE", "browse_IC2", "score_ice", "score_hyd"]
      : ["quickview", "highres"];

  // Filter out highres if not available
  const displayOverlays = availableOverlays.filter(
    (type) => type !== "highres" || hasHighResData
  );

  return (
    <aside
      className={`relative flex flex-col bg-surface-dark/40 ${isMobile ? 'w-full' : 'h-full border-l border-border-dark'}`}
      style={isMobile ? undefined : { width: panelWidth }}
      role="complementary"
      aria-label="Product inspector"
    >
      {/* Resize handle (left edge) - desktop only */}
      {!isMobile && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}
      {/* Tabs */}
      <div className="flex border-b border-border-dark">
        {isHiRISE &&
          hiriseTabs.map((t) => (
            <button
              key={t}
              onClick={() => setHiriseTab(t)}
              aria-label={`${t} tab`}
              className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                hiriseTab === t
                  ? "border-b-2 border-primary bg-primary/5 text-white"
                  : "border-b-2 border-transparent text-slate-500 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}

        {isCRISM &&
          crismTabs.map((t) => (
            <button
              key={t}
              onClick={() => setCrismTab(t)}
              aria-label={`${t} tab`}
              className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                crismTab === t
                  ? "border-b-2 border-primary bg-primary/5 text-white"
                  : "border-b-2 border-transparent text-slate-500 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}

        {isCustom && (
          <button
            className="flex-1 py-3 text-[10px] font-bold uppercase tracking-widest border-b-2 border-primary bg-primary/5 text-white"
          >
            Metadata
          </button>
        )}

        {isDTM && (
          <button
            className="flex-1 py-3 text-[10px] font-bold uppercase tracking-widest border-b-2 border-amber-600 bg-amber-600/5 text-white"
          >
            HiRISE DTM
          </button>
        )}

        <div className="flex items-center shrink-0">
          {onCollapse && (
            <button
              onClick={onCollapse}
              className="flex items-center justify-center px-1.5 text-slate-500 hover:text-white transition-colors"
              title="Collapse panel"
              aria-label="Collapse inspector panel"
            >
              <span className="material-symbols-outlined text-base">chevron_right</span>
            </button>
          )}
          <button
            onClick={onClose}
            className="flex items-center justify-center px-3 text-slate-500 hover:text-red-400 transition-colors"
            aria-label="Close inspector"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>
      </div>

      {/* Recent Products Chip Bar */}
      {recentProducts.length > 1 && onSelectRecent && (
        <div className="flex gap-1.5 px-3 py-2 border-b border-border-dark overflow-x-auto scrollbar-dark">
          {recentProducts.map((p) => (
            <div
              key={p.productId}
              className={`flex-shrink-0 flex items-center rounded-full text-[10px] font-medium transition-all ${
                p.productId === selected.productId
                  ? "bg-primary/20 border border-primary/50 text-primary"
                  : "bg-surface-dark border border-border-dark text-slate-400 hover:text-white hover:border-slate-500"
              }`}
            >
              <button
                onClick={() => onSelectRecent(p)}
                className="pl-2.5 pr-1 py-1"
                title={`${p.instrument} — ${p.productId}`}
              >
                {p.productId.length > 16 ? p.productId.slice(0, 14) + "…" : p.productId}
              </button>
              {onRemoveRecent && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveRecent(p.productId);
                  }}
                  className="pr-1.5 pl-0.5 py-1 opacity-50 hover:opacity-100 hover:text-red-400 transition-all"
                  title="Remove from history"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 12 }}>close</span>
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-dark">
        {/* HiRISE Tabs */}
        {isHiRISE && hiriseTab === "Metadata" && <MetadataTab selected={selected} hasHighResData={hasHighResData} />}

        {isHiRISE && hiriseTab === "Pixel" && (
          <PixelTab
            selected={selected}
            stats={stats}
            loading={loading}
            windowSize={windowSize}
            onWindowSizeChange={setWindowSize}
          />
        )}

        {/* CRISM Tabs */}
        {isCRISM && crismTab === "Metadata" && <MetadataTab selected={selected} hasHighResData={hasHighResData} onOpenMineralSequence={onOpenMineralSequence} />}

        {isCRISM && crismTab === "Spectrum" && (
          <SpectrumTab
            selected={selected}
            spectrumData={spectrumData}
            dustAssessment={dustAssessment}
            loading={spectrumLoading}
            onPinSpectrum={onPinSpectrum}
          />
        )}

        {isCRISM && crismTab === "Bands" && (
          <BandsTab
            rgb={localRGB}
            onChange={handleRGBSliderChange}
            onApply={handleApplyRGB}
            isOverlayActive={activeOverlay?.type === "highres"}
          />
        )}

        {isCustom && customDataset && (
          <CustomMetadataTab dataset={customDataset} />
        )}

        {isDTM && <MetadataTab selected={selected} hasHighResData={hasHighResData} />}
      {/* ── Collapsible Overlay Section ── */}
        <div className="mt-4 border-t border-border-dark pt-3">
          <button
            onClick={() => setShowOverlaySection(!showOverlaySection)}
            className="flex w-full items-center justify-between text-[10px] font-bold uppercase tracking-widest text-slate-500 hover:text-slate-300 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <span className="material-symbols-outlined text-xs">layers</span>
              Overlays
              {activeOverlay && (
                <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[8px] bg-green-500/20 text-green-400 normal-case font-medium">
                  {OVERLAY_CONFIG[activeOverlay.type].label}
                </span>
              )}
            </span>
            <span className={`material-symbols-outlined text-xs transition-transform ${showOverlaySection ? 'rotate-180' : ''}`}>
              expand_more
            </span>
          </button>

          {showOverlaySection && (
            <div className="mt-2 space-y-2">
              {/* Custom dataset opacity control */}
              {isCustom && customDataset && (
                <div className="space-y-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                    Overlay Opacity
                  </h4>
                  <div className="flex items-center gap-2 px-2 py-2 bg-[#0a0f18] rounded border border-[#232f48]">
                    <span className="text-[9px] text-[#6b7c9c] uppercase">Opacity</span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={customDataset.opacity}
                      onChange={(e) => onCustomDatasetOpacity?.(customDataset.id, Number(e.target.value))}
                      className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                        [&::-webkit-slider-thumb]:appearance-none
                        [&::-webkit-slider-thumb]:h-3
                        [&::-webkit-slider-thumb]:w-3
                        [&::-webkit-slider-thumb]:rounded-full
                        [&::-webkit-slider-thumb]:bg-fuchsia-400
                        [&::-webkit-slider-thumb]:cursor-pointer"
                    />
                    <span className="text-[10px] text-white font-mono w-8 text-right">
                      {customDataset.opacity}%
                    </span>
                  </div>
                </div>
              )}

              {/* Standard overlay controls for non-custom instruments */}
              {!isCustom && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                      Overlay
                    </h4>
                    {activeOverlay && (
                      <button
                        onClick={() => setShowOpacity(!showOpacity)}
                        className={`p-1 rounded transition-colors ${
                          showOpacity ? "text-primary bg-primary/20" : "text-slate-500 hover:text-slate-300"
                        }`}
                        title="Adjust opacity"
                      >
                        <span className="material-symbols-outlined text-sm">opacity</span>
                      </button>
                    )}
                  </div>

                  {/* Overlay type buttons */}
                  <div className="grid grid-cols-3 gap-1.5">
                    {displayOverlays.map((type) => {
                      const config = OVERLAY_CONFIG[type];
                      const isActive = activeOverlay?.type === type;
                      const isDisabled = type === "highres" && !hasHighResData;

                      return (
                        <button
                          key={type}
                          onClick={() => {
                            if (isDisabled) return;
                            onSetOverlay(isActive ? null : type);
                          }}
                          disabled={isDisabled}
                          className={`flex items-center justify-center gap-1 px-2 py-2 rounded-lg text-[10px] font-bold uppercase transition-all ${
                            isDisabled
                              ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                              : isActive
                              ? config.activeClass
                              : "bg-surface-dark border border-border-dark text-slate-400 hover:border-slate-500"
                          }`}
                          title={isDisabled ? "No high-res data available" : config.label}
                        >
                          <span className="material-symbols-outlined text-xs">
                            {isActive ? "check_circle" : config.icon}
                          </span>
                          {config.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Opacity slider */}
                  {activeOverlay && showOpacity && (
                    <div className="flex items-center gap-2 px-2 py-2 bg-[#0a0f18] rounded border border-[#232f48]">
                      <span className="text-[9px] text-[#6b7c9c] uppercase">Opacity</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={activeOverlay.opacity}
                        onChange={(e) => onSetOpacity?.(Number(e.target.value))}
                        className="flex-1 h-1 bg-[#232f48] rounded-lg appearance-none cursor-pointer
                          [&::-webkit-slider-thumb]:appearance-none
                          [&::-webkit-slider-thumb]:h-3
                          [&::-webkit-slider-thumb]:w-3
                          [&::-webkit-slider-thumb]:rounded-full
                          [&::-webkit-slider-thumb]:bg-primary
                          [&::-webkit-slider-thumb]:cursor-pointer"
                      />
                      <span className="text-[10px] text-white font-mono w-8 text-right">
                        {activeOverlay.opacity}%
                      </span>
                    </div>
                  )}

                  {/* Current overlay status */}
                  {activeOverlay && (
                    <div className="flex items-center justify-between px-2 py-1.5 bg-green-500/10 rounded border border-green-500/30">
                      <span className="text-[10px] text-green-400">
                        {OVERLAY_CONFIG[activeOverlay.type].label} overlay active
                      </span>
                      <button
                        onClick={() => onSetOverlay(null)}
                        className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                      >
                        Turn Off
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Action Buttons ── */}
        <div className="mt-3 space-y-2">
          {/* Show 3D View button (HiRISE DTM) */}
          {isDTM && selected && onShow3DView && (
            <button
              onClick={() => onShow3DView(selected.productId, selected.lat, selected.lon)}
              className="flex w-full items-center justify-center gap-2 rounded-lg py-3 text-xs font-bold uppercase tracking-widest bg-amber-600/20 border border-amber-600/50 text-amber-500 hover:bg-amber-600/30 shadow-lg shadow-amber-600/10 active:scale-[0.98] transition-all"
            >
              <span className="material-symbols-outlined text-sm">terrain</span>
              Show 3D View
            </button>
          )}

          {/* Field Note button */}
          {selected && onOpenFieldNote && (
            <button
              onClick={() => onOpenFieldNote(selected.productId, selected.instrument, selected.lat, selected.lon)}
              className={`flex w-full items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-bold uppercase tracking-widest transition-all active:scale-[0.98] ${
                hasNote
                  ? "bg-amber-500/20 border border-amber-500/50 text-amber-400 hover:bg-amber-500/30"
                  : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:text-amber-400 hover:border-amber-500/30"
              }`}
            >
              <span className="material-symbols-outlined text-sm">
                {hasNote ? "description" : "note_add"}
              </span>
              {hasNote ? `Field Notes (${noteCount})` : "Add Field Note"}
            </button>
          )}

          {/* Classify Landform button (HiRISE only) */}
          {isHiRISE && (
            <button
              onClick={() => setShowLandformPanel(!showLandformPanel)}
              className="flex w-full items-center justify-center gap-2 rounded-lg py-2.5 text-[10px] font-bold uppercase tracking-widest bg-violet-500/20 border border-violet-500/50 text-violet-400 hover:bg-violet-500/30 transition-all active:scale-[0.98]"
            >
              <span className="material-symbols-outlined text-sm">image_search</span>
              Classify Landform
            </button>
          )}

          {/* Quick Actions Bar */}
          {selected && selected.instrument !== "CUSTOM" && (
            <div className="grid grid-cols-2 gap-2">
              {/* Download */}
              {onDownloadProduct && (
                <button
                  onClick={() => onDownloadProduct(selected.productId, selected.instrument)}
                  className="flex items-center justify-center gap-1.5 rounded-lg py-2.5 text-[10px] font-bold uppercase tracking-widest bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/30 transition-all active:scale-[0.98]"
                >
                  <span className="material-symbols-outlined text-sm">download</span>
                  Download
                </button>
              )}

              {/* Find Related */}
              {onFindRelated && (
                <button
                  onClick={() => onFindRelated(selected.productId, selected.instrument)}
                  className="flex items-center justify-center gap-1.5 rounded-lg py-2.5 text-[10px] font-bold uppercase tracking-widest bg-purple-500/20 border border-purple-500/50 text-purple-400 hover:bg-purple-500/30 transition-all active:scale-[0.98]"
                >
                  <span className="material-symbols-outlined text-sm">hub</span>
                  Related
                </button>
              )}
            </div>
          )}

          {/* Find Temporal Pairs */}
          {selected && selected.instrument !== "CUSTOM" && onFindTemporalPairs && (
            <button
              onClick={() => onFindTemporalPairs(selected.lat, selected.lon, selected.instrument)}
              className="flex w-full items-center justify-center gap-2 rounded-lg py-2.5 text-[10px] font-bold uppercase tracking-widest bg-amber-500/20 border border-amber-500/50 text-amber-400 hover:bg-amber-500/30 transition-all active:scale-[0.98]"
            >
              <span className="material-symbols-outlined text-sm">compare</span>
              Find Temporal Pairs
            </button>
          )}

          <button className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary py-3 text-xs font-bold uppercase tracking-widest text-white shadow-lg shadow-primary/20 hover:brightness-110 active:scale-[0.98] transition-all">
            <span className="material-symbols-outlined text-sm">ios_share</span>
            Export Statistics
          </button>
        </div>

        {/* ── HiRISE Landform Classification ── */}
        {showLandformPanel && isHiRISE && (
          <div className="border-t border-border-dark -mx-4 mt-4">
            <HiriseLandformPanel
              productId={selected.productId}
              onClose={() => setShowLandformPanel(false)}
            />
          </div>
        )}
      </div>

    </aside>
  );
}

/* =========================================================
 * Bands Tab (CRISM RGB Selection)
 * =======================================================*/
function BandsTab({
  rgb,
  onChange,
  onApply,
  isOverlayActive,
}: {
  rgb: RGBWavelengths;
  onChange: (channel: "r" | "g" | "b", value: number) => void;
  onApply: () => void;
  isOverlayActive: boolean;
}) {
  // CRISM wavelength range (approximately 1.0 - 4.0 micrometers for VNIR+IR)
  const minWavelength = 1.0;
  const maxWavelength = 4.0;
  const step = 0.01;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">palette</span>
        <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
          RGB Band Selection
        </h3>
      </div>

      <p className="text-[11px] text-slate-400">
        Select wavelengths (in micrometers) to create an RGB composite from CRISM spectral data.
      </p>

      {/* Red Channel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="inline-block w-3 h-3 rounded-full bg-red-500"></span>
            Red Channel
          </label>
          <span className="font-mono text-xs text-red-400">{rgb.r.toFixed(2)} μm</span>
        </div>
        <input
          type="range"
          min={minWavelength}
          max={maxWavelength}
          step={step}
          value={rgb.r}
          onChange={(e) => onChange("r", parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-red-500 bg-slate-700"
        />
        <div className="flex justify-between text-[9px] text-slate-600">
          <span>{minWavelength}</span>
          <span>{maxWavelength}</span>
        </div>
      </div>

      {/* Green Channel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="inline-block w-3 h-3 rounded-full bg-green-500"></span>
            Green Channel
          </label>
          <span className="font-mono text-xs text-green-400">{rgb.g.toFixed(2)} μm</span>
        </div>
        <input
          type="range"
          min={minWavelength}
          max={maxWavelength}
          step={step}
          value={rgb.g}
          onChange={(e) => onChange("g", parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-green-500 bg-slate-700"
        />
        <div className="flex justify-between text-[9px] text-slate-600">
          <span>{minWavelength}</span>
          <span>{maxWavelength}</span>
        </div>
      </div>

      {/* Blue Channel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="inline-block w-3 h-3 rounded-full bg-blue-500"></span>
            Blue Channel
          </label>
          <span className="font-mono text-xs text-blue-400">{rgb.b.toFixed(2)} μm</span>
        </div>
        <input
          type="range"
          min={minWavelength}
          max={maxWavelength}
          step={step}
          value={rgb.b}
          onChange={(e) => onChange("b", parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500 bg-slate-700"
        />
        <div className="flex justify-between text-[9px] text-slate-600">
          <span>{minWavelength}</span>
          <span>{maxWavelength}</span>
        </div>
      </div>

      {/* Presets */}
      <div className="space-y-2">
        <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
          Presets
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => {
              onChange("r", 2.53);
              onChange("g", 1.51);
              onChange("b", 1.08);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            True Color
          </button>
          <button
            onClick={() => {
              onChange("r", 2.3);
              onChange("g", 1.93);
              onChange("b", 1.08);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            Mineralogy
          </button>
          <button
            onClick={() => {
              onChange("r", 1.08);
              onChange("g", 1.51);
              onChange("b", 2.53);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            Inverted
          </button>
          <button
            onClick={() => {
              onChange("r", 2.53);
              onChange("g", 1.93);
              onChange("b", 1.51);
            }}
            className="px-3 py-2 text-[10px] rounded border border-border-dark bg-surface-dark/50 hover:bg-surface-dark text-slate-300 transition-colors"
          >
            Hydration
          </button>
        </div>
      </div>

      {/* Apply Button */}
      <button
        onClick={onApply}
        className="w-full py-2.5 rounded-lg bg-primary/20 border border-primary/30 text-primary text-xs font-bold uppercase tracking-widest hover:bg-primary/30 transition-colors"
      >
        <span className="material-symbols-outlined text-sm align-middle mr-1">
          {isOverlayActive ? "refresh" : "add_photo_alternate"}
        </span>
        {isOverlayActive ? "Apply RGB Changes" : "Create RGB Overlay"}
      </button>
    </div>
  );
}

/* =========================================================
 * Spectrum Tab (CRISM)
 * =======================================================*/
function SpectrumTab({
  selected,
  spectrumData,
  dustAssessment,
  loading,
  onPinSpectrum,
}: {
  selected: InspectorContext;
  spectrumData: SpectrumData | null;
  dustAssessment: DustAssessment | null;
  loading: boolean;
  onPinSpectrum?: (spectrum: { productId: string; lat: number; lon: number; wavelengths: number[]; reflectance: (number | null)[] }) => void;
}) {
  const [showBandRatios, setShowBandRatios] = useState(false);
  const hasPixel = selected.pixelLine !== undefined && selected.pixelSample !== undefined;

  if (!hasPixel) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">show_chart</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
            Pixel Spectrum
          </h3>
        </div>

        <div className="flex flex-col items-center justify-center h-64 text-center">
          <span className="material-symbols-outlined text-4xl text-slate-600 mb-3">
            touch_app
          </span>
          <p className="text-sm text-slate-400 mb-2">
            Click on the CRISM overlay to select a pixel
          </p>
          <p className="text-[11px] text-slate-500">
            The spectral profile will be displayed here
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <span className="material-symbols-outlined animate-spin text-2xl text-primary">
          progress_activity
        </span>
      </div>
    );
  }

  if (!spectrumData) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500 text-sm">
        Failed to load spectrum data
      </div>
    );
  }

  // Filter valid data points for the chart
  const validPoints = spectrumData.wavelengths
    .map((wl, i) => ({
      wavelength: wl,
      reflectance: spectrumData.reflectance[i],
    }))
    .filter((p) => p.reflectance !== null) as { wavelength: number; reflectance: number }[];

  // Calculate stats
  const reflValues = validPoints.map((p) => p.reflectance);
  const minRefl = Math.min(...reflValues);
  const maxRefl = Math.max(...reflValues);
  const meanRefl = reflValues.reduce((a, b) => a + b, 0) / reflValues.length;

  // Chart dimensions
  const chartWidth = 340;
  const chartHeight = 180;
  const padding = { top: 10, right: 10, bottom: 25, left: 45 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Scales
  const wlMin = Math.min(...validPoints.map((p) => p.wavelength));
  const wlMax = Math.max(...validPoints.map((p) => p.wavelength));
  const yMin = Math.max(0, minRefl - (maxRefl - minRefl) * 0.1);
  const yMax = maxRefl + (maxRefl - minRefl) * 0.1;

  const xScale = (wl: number) => padding.left + ((wl - wlMin) / (wlMax - wlMin)) * innerWidth;
  const yScale = (r: number) => padding.top + innerHeight - ((r - yMin) / (yMax - yMin)) * innerHeight;

  // Generate path
  const pathD = validPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.wavelength)} ${yScale(p.reflectance)}`)
    .join(" ");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">show_chart</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
            Pixel Spectrum
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {onPinSpectrum && (
            <button
              onClick={() =>
                onPinSpectrum({
                  productId: selected.productId,
                  lat: selected.lat,
                  lon: selected.lon,
                  wavelengths: spectrumData.wavelengths,
                  reflectance: spectrumData.reflectance,
                })
              }
              className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
              title="Pin spectrum for comparison"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                push_pin
              </span>
              Pin
            </button>
          )}
          <span className="text-[10px] text-slate-500">
            {spectrumData.validBands} bands
          </span>
        </div>
      </div>

      {/* Pixel coordinates */}
      <div className="flex gap-4 text-[11px]">
        <div>
          <span className="text-slate-500">Line: </span>
          <span className="font-mono text-white">{selected.pixelLine}</span>
        </div>
        <div>
          <span className="text-slate-500">Sample: </span>
          <span className="font-mono text-white">{selected.pixelSample}</span>
        </div>
      </div>

      {dustAssessment && dustAssessment.risk_level !== "LOW" && (
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-[10px] leading-relaxed ${
          dustAssessment.risk_level === "HIGH"
            ? "bg-red-500/10 border border-red-500/30 text-red-400"
            : "bg-amber-500/10 border border-amber-500/30 text-amber-400"
        }`}>
          <span className="material-symbols-outlined text-sm flex-shrink-0">
            {dustAssessment.risk_level === "HIGH" ? "warning" : "info"}
          </span>
          <div>
            <span className="font-semibold">
              {dustAssessment.risk_level === "HIGH" ? "High" : "Moderate"} Dust Risk
            </span>
            {" · "}tau~{dustAssessment.tau_estimated.toFixed(1)}
            {dustAssessment.band_depth_suppression_pct > 0 && (
              <> · Band depths suppressed ~{dustAssessment.band_depth_suppression_pct.toFixed(0)}%</>
            )}
            {dustAssessment.warning_message && (
              <div className="mt-0.5 opacity-80">{dustAssessment.warning_message}</div>
            )}
          </div>
        </div>
      )}

      {/* Spectrum Chart */}
      <div className="rounded-lg border border-border-dark bg-bg-dark/60 p-3">
        <svg width={chartWidth} height={chartHeight} className="overflow-visible">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padding.top + innerHeight * (1 - t);
            const val = yMin + (yMax - yMin) * t;
            return (
              <g key={t}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + innerWidth}
                  y2={y}
                  stroke="#334155"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
                <text
                  x={padding.left - 5}
                  y={y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-slate-500 text-[8px]"
                >
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* X axis labels */}
          {[wlMin, (wlMin + wlMax) / 2, wlMax].map((wl, i) => (
            <text
              key={i}
              x={xScale(wl)}
              y={chartHeight - 5}
              textAnchor="middle"
              className="fill-slate-500 text-[8px]"
            >
              {wl.toFixed(1)}
            </text>
          ))}

          {/* Axis labels */}
          <text
            x={chartWidth / 2}
            y={chartHeight}
            textAnchor="middle"
            className="fill-slate-400 text-[9px]"
          >
            Wavelength (μm)
          </text>
          <text
            x={10}
            y={chartHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90, 10, ${chartHeight / 2})`}
            className="fill-slate-400 text-[9px]"
          >
            I/F
          </text>

          {/* Spectrum line */}
          <path d={pathD} fill="none" stroke="#3b82f6" strokeWidth="1.5" />
        </svg>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
          <div className="text-[9px] uppercase text-slate-500">Min</div>
          <div className="font-mono text-xs text-white">{minRefl.toFixed(4)}</div>
        </div>
        <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
          <div className="text-[9px] uppercase text-slate-500">Max</div>
          <div className="font-mono text-xs text-white">{maxRefl.toFixed(4)}</div>
        </div>
        <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
          <div className="text-[9px] uppercase text-slate-500">Mean</div>
          <div className="font-mono text-xs text-white">{meanRefl.toFixed(4)}</div>
        </div>
      </div>

      {/* Band Ratios Section */}
      {!showBandRatios ? (
        <button
          onClick={() => setShowBandRatios(true)}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg py-2 text-[10px] font-bold uppercase tracking-widest bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20 transition-colors"
        >
          <span className="material-symbols-outlined text-sm">calculate</span>
          Band Ratios
        </button>
      ) : (
        <BandRatioCalculator
          wavelengths={spectrumData.wavelengths}
          reflectance={spectrumData.reflectance}
          onClose={() => setShowBandRatios(false)}
        />
      )}
    </div>
  );
}

/* =========================================================
 * TRR3 Mineral Classification Section
 * =======================================================*/
function TRR3MineralSection({ obsId, onOpenMineralSequence }: { obsId: string; onOpenMineralSequence?: (obsId: string) => void }) {
  const [status, setStatus] = useState<"checking" | "not_downloaded" | "idle" | "loading" | "done" | "error">("checking");
  const [stats, setStats] = useState<any>(null);
  const [legend, setLegend] = useState<any[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [progressMsg, setProgressMsg] = useState("");
  const [progressPct, setProgressPct] = useState<number | null>(null);

  // Check data availability on mount
  useEffect(() => {
    let cancelled = false;

    // First check if classification results exist
    fetch(`/api/mineral-cnn/result/${obsId}/stats`)
      .then((res) => res.ok ? res.json() : null)
      .then((data) => {
        if (cancelled) return;
        if (data) {
          setStats(data);
          setStatus("done");
          fetch(`/api/mineral-cnn/result/${obsId}/legend`)
            .then((r) => r.json())
            .then((d) => !cancelled && setLegend(d.legend || []))
            .catch(() => {});
          return;
        }
        // No results — check if data is downloaded
        return fetch(`/api/mineral-cnn/acquire/${obsId}/status`);
      })
      .then((res) => {
        if (!res || cancelled) return;
        return res.json();
      })
      .then((data) => {
        if (!data || cancelled) return;
        if (data.has_results) {
          setStatus("done");
        } else if (data.has_trr3_data) {
          setStatus("idle"); // Data exists, ready to classify
        } else {
          setStatus("not_downloaded"); // Need to download first
        }
      })
      .catch((e) => { if (!cancelled) { setStatus("not_downloaded"); toast.error(`CNN status check failed: ${e.message}`); } });

    return () => { cancelled = true; };
  }, [obsId]);

  // SSE stream reader — shared between classify-only and full acquire
  const streamSSE = async (url: string) => {
    setStatus("loading");
    setErrorMsg("");
    setProgressMsg("Starting...");
    setProgressPct(null);
    try {
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";
      let gotError = false;
      let errDetail = "Pipeline failed";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.replace(/^data:\s*/, "").trim();
          if (!trimmed.startsWith("{")) continue;
          try {
            const evt = JSON.parse(trimmed);
            if (evt.event === "error") {
              gotError = true;
              errDetail = evt.data?.error || "Pipeline failed";
              break;
            }
            if (evt.event === "status" && evt.data?.message) {
              setProgressMsg(evt.data.message);
              if (evt.data.step !== "jcat") setProgressPct(null);
            }
            if (evt.event === "progress" && evt.data?.percent != null) {
              setProgressPct(evt.data.percent);
              setProgressMsg(`JCAT atmospheric correction: ${evt.data.percent}%`);
            }
            if (evt.event === "download_progress" && evt.data?.percent != null) {
              setProgressPct(evt.data.percent);
              setProgressMsg(`Downloading ${evt.data.file}: ${evt.data.percent}%`);
            }
            if (evt.event === "discovery" && evt.data) {
              setProgressMsg(`Found ${evt.data.files} files (${evt.data.total_size_mb} MB)`);
            }
          } catch { /* skip */ }
        }
        if (gotError) break;
      }

      if (gotError) {
        setStatus("error");
        setErrorMsg(errDetail);
        return;
      }

      // Fetch results
      const statsRes = await fetch(`/api/mineral-cnn/result/${obsId}/stats`);
      if (statsRes.ok) {
        setStats(await statsRes.json());
        setStatus("done");
        const legendRes = await fetch(`/api/mineral-cnn/result/${obsId}/legend`);
        if (legendRes.ok) {
          const ld = await legendRes.json();
          setLegend(ld.legend || []);
        }
      } else {
        // Retry once after short delay
        await new Promise(r => setTimeout(r, 500));
        const retry = await fetch(`/api/mineral-cnn/result/${obsId}/stats`);
        if (retry.ok) {
          setStats(await retry.json());
          setStatus("done");
        } else {
          setStatus("error");
          setErrorMsg("Pipeline completed but results not available");
        }
      }
    } catch (e: unknown) {
      setStatus("error");
      const msg = e instanceof Error ? e.message : "Pipeline failed";
      setErrorMsg(msg);
    }
  };

  const runAcquire = () => streamSSE(`/api/mineral-cnn/acquire/${obsId}`);
  const runClassification = () => streamSSE(`/api/mineral-cnn/classify/${obsId}`);

  return (
    <div className="space-y-3">
      <h4 className="text-xs font-bold uppercase tracking-wider text-teal-400 flex items-center gap-1.5">
        <span className="material-symbols-outlined text-sm">science</span>
        CNN Mineral Classification
      </h4>

      {status === "checking" && (
        <div className="flex items-center gap-2 py-2 text-slate-400 text-[11px]">
          <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
          Checking data availability...
        </div>
      )}

      {status === "not_downloaded" && (
        <div className="space-y-2">
          <p className="text-[9px] text-slate-400/80 leading-relaxed">
            TRR3 data not downloaded yet. This will download L-sensor TRR3 + DDR from PDS, then run JCAT atmospheric correction and CNN classification.
          </p>
          <button
            onClick={runAcquire}
            className="w-full px-3 py-2 rounded text-[11px] font-medium bg-teal-500/20 border border-teal-500/30 text-teal-400 hover:bg-teal-500/30 transition-colors flex items-center justify-center gap-2"
          >
            <span className="material-symbols-outlined text-sm">download</span>
            Download & Classify
          </button>
        </div>
      )}

      {status === "idle" && (
        <div className="space-y-2">
          <p className="text-[9px] text-amber-400/80 flex items-center gap-1 leading-relaxed">
            <span className="material-symbols-outlined text-xs">info</span>
            TRR3 data available locally. Ready to classify.
          </p>
          <button
            onClick={runClassification}
            className="w-full px-3 py-2 rounded text-[11px] font-medium bg-teal-500/20 border border-teal-500/30 text-teal-400 hover:bg-teal-500/30 transition-colors flex items-center justify-center gap-2"
          >
            <span className="material-symbols-outlined text-sm">play_arrow</span>
            Run Classification
          </button>
        </div>
      )}

      {status === "loading" && (
        <div className="space-y-2 py-3">
          <div className="flex items-center gap-2 text-teal-400 text-[11px]">
            <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
            <span className="truncate">{progressMsg || "Running CNN inference..."}</span>
          </div>
          {progressPct != null && (
            <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-500 rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          )}
        </div>
      )}

      {status === "error" && (
        <div className="space-y-2">
          <p className="text-[11px] text-red-400 flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">error</span>
            {errorMsg}
          </p>
          <button
            onClick={runAcquire}
            className="px-3 py-1.5 rounded text-[10px] font-medium bg-teal-500/20 border border-teal-500/30 text-teal-400 hover:bg-teal-500/30 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {status === "done" && stats && (
        <div className="space-y-3">
          {/* Mineral map */}
          <div className="overflow-hidden rounded-lg border border-border-dark">
            <img
              src={`/api/mineral-cnn/result/${obsId}/mineral-map.png`}
              alt="Mineral Map"
              className="w-full bg-black"
              loading="lazy"
            />
          </div>

          {/* Stats summary */}
          <div className="grid grid-cols-3 gap-2 text-[10px]">
            <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
              <div className="text-[9px] uppercase text-slate-500">Classified</div>
              <div className="font-mono text-white">{stats.classified_pixels?.toLocaleString()}</div>
            </div>
            <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
              <div className="text-[9px] uppercase text-slate-500">Threshold</div>
              <div className="font-mono text-white">≥{((stats.confidence_threshold ?? 0.95) * 100).toFixed(0)}%</div>
            </div>
            <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
              <div className="text-[9px] uppercase text-slate-500">Mean Conf.</div>
              <div className={`font-mono ${ 
                (stats.mean_confidence ?? 0) >= 0.95 ? "text-emerald-400" :
                (stats.mean_confidence ?? 0) >= 0.80 ? "text-amber-400" : "text-red-400"
              }`}>{((stats.mean_confidence ?? 0) * 100).toFixed(1)}%</div>
            </div>
          </div>

          {/* Legend */}
          {legend.length > 0 && (
            <div className="space-y-1">
              <h5 className="text-[9px] uppercase text-slate-500 font-bold">Minerals Detected</h5>
              <div className="max-h-40 overflow-y-auto scrollbar-dark space-y-0.5">
                {legend.map((item: any) => {
                  const conf = item.avg_confidence as number | undefined;
                  const confPct = ((conf ?? 0) * 100);
                  const confColor = confPct >= 95 ? "text-emerald-400" : confPct >= 80 ? "text-amber-400" : "text-red-400";
                  return (
                    <div key={item.mineral_id} className="flex items-center gap-2 py-0.5">
                      <span
                        className="w-3 h-3 rounded-sm flex-shrink-0"
                        style={{ backgroundColor: item.color_hex }}
                      />
                      <span className="text-[10px] text-white flex-1 truncate">{item.name}</span>
                      {conf != null && conf > 0 && (
                        <span className={`text-[8px] font-mono px-1 py-0.5 rounded ${confColor} bg-surface-dark/60`}>
                          {confPct.toFixed(0)}%
                        </span>
                      )}
                      <span className="text-[9px] text-slate-500 font-mono">{item.pixel_count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Mineral Sequence button */}
          {onOpenMineralSequence && (
            <button
              onClick={() => onOpenMineralSequence(obsId)}
              className="w-full px-3 py-2 rounded text-[11px] font-medium bg-amber-500/20 border border-amber-500/30 text-amber-400 hover:bg-amber-500/30 transition-colors flex items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined text-sm">science</span>
              Mineral Sequence Analysis
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* =========================================================
 * Product Download Links Component
 * Fetches PDS download URLs from backend and displays them
 * =======================================================*/
function ProductDownloadLinks({ productId, instrument }: { productId: string; instrument: InstrumentType }) {
  const [urls, setUrls] = useState<{
    jp2_url?: string;
    jp2_size_bytes?: number;
    jp2_filename?: string;
    lbl_url?: string;
    img_url?: string;
    img_filename?: string;
    browse_urls?: Record<string, string>;
    product_type?: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const isHiRISE = instrument === "HIRISE";
  const isCRISM = instrument === "CRISM" || instrument === "CRISM_TRR3";

  useEffect(() => {
    if (!isHiRISE && !isCRISM) return;
    setLoading(true);
    setError(false);
    setUrls(null);

    const endpoint = isHiRISE
      ? `/api/product-urls/hirise/${productId}`
      : `/api/product-urls/crism/${productId}`;

    fetch(endpoint)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setUrls(data))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [productId, instrument, isHiRISE, isCRISM]);

  if (!isHiRISE && !isCRISM) return null;

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-slate-500 text-[11px]">
        <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
        Resolving PDS download URLs…
      </div>
    );
  }

  if (error || !urls) {
    return (
      <div className="text-[11px] text-slate-500">
        Could not resolve download URLs from ODE.
      </div>
    );
  }

  // CRISM browse labels — use full Tailwind class names (no dynamic interpolation for JIT)
  const CRISM_BROWSE_LABELS: Record<string, { label: string; className: string }> = {
    vna: { label: "VNIR", className: "bg-emerald-500/20 border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/30" },
    hyd: { label: "HYD", className: "bg-fuchsia-500/20 border-fuchsia-500/50 text-fuchsia-400 hover:bg-fuchsia-500/30" },
    ice: { label: "ICE", className: "bg-blue-500/20 border-blue-500/50 text-blue-400 hover:bg-blue-500/30" },
    ic2: { label: "CO\u2082", className: "bg-cyan-500/20 border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/30" },
  };

  const browseUrls = urls.browse_urls;

  return (
    <div className="space-y-2">
      {/* HiRISE JP2 download */}
      {isHiRISE && urls.jp2_url && (
        <a
          href={urls.jp2_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 w-full rounded-lg py-2.5 px-3 text-[10px] font-bold uppercase tracking-widest bg-sky-500/20 border border-sky-500/50 text-sky-400 hover:bg-sky-500/30 transition-all active:scale-[0.98] no-underline"
        >
          <span className="material-symbols-outlined text-sm">cloud_download</span>
          <span className="flex-1">Download RED JP2 from PDS</span>
          {urls.jp2_size_bytes && (
            <span className="text-[9px] font-normal text-sky-400/70">
              {formatBytes(urls.jp2_size_bytes)}
            </span>
          )}
        </a>
      )}

      {/* CRISM browse product downloads */}
      {isCRISM && browseUrls && Object.keys(browseUrls).length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {Object.entries(CRISM_BROWSE_LABELS).map(([key, cfg]) => {
            const url = browseUrls[key];
            if (!url) return null;
            return (
              <a
                key={key}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center justify-center gap-1 rounded-lg py-2 px-2 text-[10px] font-bold uppercase tracking-widest no-underline border transition-all active:scale-[0.98] ${cfg.className}`}
              >
                <span className="material-symbols-outlined text-xs">cloud_download</span>
                {cfg.label}
              </a>
            );
          })}
        </div>
      )}

      {/* CRISM core product download */}
      {isCRISM && urls.img_url && (
        <a
          href={urls.img_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 w-full rounded-lg py-2 px-3 text-[10px] font-bold uppercase tracking-widest bg-violet-500/20 border border-violet-500/50 text-violet-400 hover:bg-violet-500/30 transition-all active:scale-[0.98] no-underline"
        >
          <span className="material-symbols-outlined text-sm">cloud_download</span>
          <span className="flex-1">Download IMG from PDS</span>
        </a>
      )}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/* =========================================================
 * Metadata Tab
 * =======================================================*/
function MetadataTab({ selected, hasHighResData, onOpenMineralSequence }: { selected: InspectorContext; hasHighResData?: boolean; onOpenMineralSequence?: (obsId: string) => void }) {
  const isHiRISE = selected.instrument === "HIRISE";
  const isCRISM = selected.instrument === "CRISM" || selected.instrument === "CRISM_TRR3";

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">
          {selected.instrument === "CRISM" || selected.instrument === "CRISM_TRR3" ? "spectrum" : "satellite_alt"}
        </span>
        <span className="text-sm font-bold">{selected.instrument}</span>
        {/* Status badge */}
        {isHiRISE && (
          hasHighResData ? (
            <span className="ml-auto flex items-center gap-1 rounded-full bg-green-500/20 border border-green-500/40 px-2 py-0.5 text-[9px] font-bold uppercase text-green-400">
              <span className="material-symbols-outlined" style={{ fontSize: 10 }}>check_circle</span>
              Full Res
            </span>
          ) : (
            <span className="ml-auto flex items-center gap-1 rounded-full bg-amber-500/20 border border-amber-500/40 px-2 py-0.5 text-[9px] font-bold uppercase text-amber-400">
              <span className="material-symbols-outlined" style={{ fontSize: 10 }}>photo_camera</span>
              Quickview Only
            </span>
          )
        )}
      </div>

      {/* Title (if available) */}
      {selected.title && (
        <div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
          <span className="text-[10px] uppercase text-primary/70 block mb-1">Title</span>
          <span className="text-sm text-white font-medium">{selected.title}</span>
        </div>
      )}

      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Product ID</span>
          <span className="font-mono text-xs">{selected.productId}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Longitude</span>
          <span className="font-mono text-xs">{selected.lon.toFixed(6)}°</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Latitude</span>
          <span className="font-mono text-xs">{selected.lat.toFixed(6)}°</span>
        </div>
      </div>

      <div>
        <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-primary">
          Quickview
        </h4>
        <div className="overflow-hidden rounded-lg border border-border-dark">
          <CRISMQuickviewImage productId={selected.productId} instrument={selected.instrument} />
        </div>
      </div>

      {/* PDS Download Links */}
      {(isHiRISE || isCRISM) && (
        <div>
          <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-primary">
            PDS Downloads
          </h4>
          <ProductDownloadLinks productId={selected.productId} instrument={selected.instrument} />
        </div>
      )}

      {/* TRR3 Mineral Classification */}
      {selected.instrument === "CRISM_TRR3" && (
        <TRR3MineralSection obsId={selected.productId.replace(/_\d{2}$/, "")} onOpenMineralSequence={onOpenMineralSequence} />
      )}
    </div>
  );
}

/* =========================================================
 * Pixel Tab (HiRISE Analysis)
 * =======================================================*/
function PixelTab({
  selected,
  stats,
  loading,
  windowSize,
  onWindowSizeChange,
}: {
  selected: InspectorContext;
  stats: WindowStats | null;
  loading: boolean;
  windowSize: number;
  onWindowSizeChange: (size: number) => void;
}) {
  const hasPixel = selected.pixelLine !== undefined && selected.pixelSample !== undefined;

  if (!hasPixel) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">analytics</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-primary">
            Pixel Statistics
          </h3>
        </div>

        <div className="flex flex-col items-center justify-center h-64 text-center">
          <span className="material-symbols-outlined text-4xl text-slate-600 mb-3">
            touch_app
          </span>
          <p className="text-sm text-slate-400 mb-2">
            Click on the HiRISE overlay to select a pixel
          </p>
          <p className="text-[11px] text-slate-500">
            Enable the high-resolution overlay first, then click on it
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-primary">
          <span className="material-symbols-outlined text-sm">analytics</span>
          Neighborhood Statistics
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-[9px] uppercase text-slate-500">Window</span>
          <select
            value={windowSize}
            onChange={(e) => onWindowSizeChange(Number(e.target.value))}
            className="rounded border-border-dark bg-bg-dark py-0.5 pl-2 pr-6 text-[10px] focus:border-primary focus:ring-primary"
          >
            <option value={3}>3×3</option>
            <option value={5}>5×5</option>
            <option value={7}>7×7</option>
            <option value={11}>11×11</option>
            <option value={21}>21×21</option>
          </select>
        </div>
      </div>

      {/* Pixel coordinates */}
      <div className="flex gap-4 text-[11px]">
        <div>
          <span className="text-slate-500">Line: </span>
          <span className="font-mono text-white">{selected.pixelLine}</span>
        </div>
        <div>
          <span className="text-slate-500">Sample: </span>
          <span className="font-mono text-white">{selected.pixelSample}</span>
        </div>
      </div>

      {loading ? (
        <div className="flex h-40 items-center justify-center text-slate-500">
          <span className="material-symbols-outlined animate-spin">progress_activity</span>
        </div>
      ) : stats ? (
        <>
          <Histogram histogram={stats.histogram} binEdges={stats.binEdges} />

          <div className="grid grid-cols-2 gap-2">
            <StatCard label="Mean" value={stats.mean.toFixed(2)} />
            <StatCard label="Median" value={stats.median.toFixed(2)} />
            <StatCard label="Std Dev" value={`±${stats.std.toFixed(1)}`} highlight />
            <StatCard label="Sum" value={stats.sum.toLocaleString()} />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center justify-between rounded border border-border-dark/30 bg-bg-dark/20 p-2">
              <span className="text-[9px] uppercase text-slate-500">Min</span>
              <span className="font-mono text-xs text-slate-300">{stats.min.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between rounded border border-border-dark/30 bg-bg-dark/20 p-2">
              <span className="text-[9px] uppercase text-slate-500">Max</span>
              <span className="font-mono text-xs text-slate-300">{stats.max.toLocaleString()}</span>
            </div>
          </div>
        </>
      ) : (
        <div className="flex h-40 items-center justify-center text-slate-500 text-sm">
          Failed to load pixel data
        </div>
      )}
    </div>
  );
}

/* =========================================================
 * Histogram Component
 * =======================================================*/
function Histogram({
  histogram,
  binEdges,
}: {
  histogram: number[];
  binEdges: number[];
}) {
  const maxCount = Math.max(...histogram);

  return (
    <div className="rounded-lg border border-border-dark bg-bg-dark/60 p-3">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-[10px] font-bold uppercase tracking-tight text-slate-500">
          DN Distribution (Sampled Area)
        </span>
        <span className="font-mono text-[9px] text-slate-500">
          n={histogram.reduce((a, b) => a + b, 0)}
        </span>
      </div>

      <div className="flex h-32 items-end gap-1 px-1">
        {histogram.map((count, i) => {
          const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
          const opacity = 0.4 + (height / 100) * 0.6;
          return (
            <div
              key={i}
              className="flex-1 rounded-t-sm bg-gradient-to-t from-primary to-blue-400"
              style={{ height: `${height}%`, opacity }}
            />
          );
        })}
      </div>

      <div className="mt-2 flex justify-between font-mono text-[8px] text-slate-600">
        <span>{Math.round(binEdges[0]!)}</span>
        <span>{Math.round(binEdges[Math.floor(binEdges.length / 4)]!)}</span>
        <span>{Math.round(binEdges[Math.floor(binEdges.length / 2)]!)}</span>
        <span>{Math.round(binEdges[Math.floor((binEdges.length * 3) / 4)]!)}</span>
        <span>{Math.round(binEdges[binEdges.length - 1]!)}</span>
      </div>
    </div>
  );
}

/* =========================================================
 * Custom Dataset Metadata Tab
 * =======================================================*/
function CustomMetadataTab({ dataset }: { dataset: CustomDataset }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-fuchsia-400">upload_file</span>
        <span className="text-sm font-bold">Custom Dataset</span>
      </div>

      <div className="rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/10 p-3">
        <span className="text-[10px] uppercase text-fuchsia-400/70 block mb-1">Name</span>
        <span className="text-sm text-white font-medium">{dataset.name}</span>
      </div>

      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Dataset ID</span>
          <span className="font-mono text-xs">{dataset.id}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">CRS</span>
          <span className="font-mono text-xs">{dataset.crs}</span>
        </div>
        {dataset.crs_warning && (
          <div className="text-[10px] text-amber-400 bg-amber-500/10 rounded p-2 border border-amber-500/30">
            {dataset.crs_warning}
          </div>
        )}
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Dimensions</span>
          <span className="font-mono text-xs">{dataset.width} x {dataset.height}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Bands</span>
          <span className="font-mono text-xs">{dataset.bands}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[10px] uppercase text-slate-500">Data Type</span>
          <span className="font-mono text-xs">{dataset.dtype}</span>
        </div>
        {dataset.nodata !== null && (
          <div className="flex justify-between">
            <span className="text-[10px] uppercase text-slate-500">NoData</span>
            <span className="font-mono text-xs">{dataset.nodata}</span>
          </div>
        )}
      </div>

      <div className="space-y-3 rounded-lg border border-border-dark bg-bg-dark/60 p-4">
        <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">
          Geographic Bounds
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-[9px] text-slate-500 block">West</span>
            <span className="font-mono text-xs">{dataset.bounds.west.toFixed(4)}&deg;</span>
          </div>
          <div>
            <span className="text-[9px] text-slate-500 block">East</span>
            <span className="font-mono text-xs">{dataset.bounds.east.toFixed(4)}&deg;</span>
          </div>
          <div>
            <span className="text-[9px] text-slate-500 block">South</span>
            <span className="font-mono text-xs">{dataset.bounds.south.toFixed(4)}&deg;</span>
          </div>
          <div>
            <span className="text-[9px] text-slate-500 block">North</span>
            <span className="font-mono text-xs">{dataset.bounds.north.toFixed(4)}&deg;</span>
          </div>
        </div>
      </div>

      <div className="text-[9px] text-slate-500">
        Uploaded: {new Date(dataset.created_at).toLocaleString()}
      </div>
    </div>
  );
}

/* =========================================================
 * Stat Card Component
 * =======================================================*/
function StatCard({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2.5">
      <div className="mb-0.5 text-[9px] uppercase text-slate-500">{label}</div>
      <div
        className={`font-mono text-sm font-bold ${
          highlight ? "text-blue-400" : "text-white"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
