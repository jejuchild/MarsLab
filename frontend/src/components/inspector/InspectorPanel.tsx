import { useEffect, useState, useCallback } from "react";
import { fetchHiRISEWindow } from "../../api/hirise";
import type { InspectorPanelProps, RGBWavelengths, WindowStats, HiRISETabKey, CRISMTabKey } from "./types";
import { DEFAULT_RGB } from "./types";
import useProductMetadata from "./hooks/useProductMetadata";
import useSpectrumData from "./hooks/useSpectrumData";
import InspectorHeader from "./InspectorHeader";
import MetadataTab from "./tabs/MetadataTab";
import HiRISEPixelTab from "./tabs/HiRISEPixelTab";
import CRISMSpectrumTab from "./tabs/CRISMSpectrumTab";
import CRISMBandsTab from "./tabs/CRISMBandsTab";
import CustomDataTab from "./tabs/CustomDataTab";
import OverlayControls from "./overlays/OverlayControls";
import ActionBar from "./actions/ActionBar";
import HiResImageViewer from "./HiResImageViewer";

export default function InspectorPanel({
  selected,
  onClose,
  onCollapse,
  activeOverlay,
  onSetOverlay,
  onSetOpacity,
  rgbWavelengths,
  onRGBChange,
  hasHighResData: hasHighResDataProp = true,
  customDataset = null,
  onCustomDatasetOpacity,
  fieldNotes = [],
  onOpenFieldNote,
  onShow3DView,
  onFindRelated: _onFindRelated,
  recentProducts = [],
  onSelectRecent,
  onRemoveRecent,
  onDownloadProduct: _onDownloadProduct,
  onPinSpectrum,
  onFindTemporalPairs: _onFindTemporalPairs,
  onOpenMineralSequence,
  isMobile = false,
}: InspectorPanelProps) {
  // ── Check high-res availability directly for selected product ──
  const [localHighRes, setLocalHighRes] = useState(false);
  useEffect(() => {
    if (!selected) { setLocalHighRes(false); return; }
    const inst = selected.instrument.toLowerCase();
    fetch(`/api/exists/${inst}/${encodeURIComponent(selected.productId)}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setLocalHighRes(d?.has_core ?? false))
      .catch(() => setLocalHighRes(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.productId, selected?.instrument]);
  const hasHighResData = hasHighResDataProp || localHighRes;

  // ── Keyboard: Escape to close ──
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // ── Tab State ──
  const [hiriseTab, setHiriseTab] = useState<HiRISETabKey>("Metadata");
  const [crismTab, setCrismTab] = useState<CRISMTabKey>("Metadata");

  // ── Panel Width (resizable, desktop only) ──
  const [panelWidth, setPanelWidth] = useState(384);
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
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
    },
    [panelWidth],
  );

  // ── HiRISE Pixel Stats ──
  const [windowSize, setWindowSize] = useState(5);
  const [pixelStats, setPixelStats] = useState<WindowStats | null>(null);
  const [pixelLoading, setPixelLoading] = useState(false);

  useEffect(() => {
    if (!selected || selected.instrument !== "HIRISE") {
      setPixelStats(null);
      return;
    }
    if (selected.pixelLine === undefined || selected.pixelSample === undefined) {
      setPixelStats(null);
      return;
    }

    const { productId, pixelSample, pixelLine } = selected;
    const controller = new AbortController();

    async function fetchStats() {
      setPixelLoading(true);
      try {
        const halfSize = Math.floor(windowSize / 2);
        const data = await fetchHiRISEWindow(productId, pixelSample!, pixelLine!, halfSize);
        const dn = data.dn.flat() as number[];
        if (dn.length === 0) { setPixelStats(null); return; }

        const mean = dn.reduce((a, b) => a + b, 0) / dn.length;
        const sorted = [...dn].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)]!;
        const std = Math.sqrt(dn.reduce((acc, v) => acc + (v - mean) ** 2, 0) / dn.length);
        const min = Math.min(...dn);
        const max = Math.max(...dn);
        const sum = dn.reduce((a, b) => a + b, 0);

        const bins = 10;
        const binWidth = (max - min) / bins || 1;
        const histogram = new Array(bins).fill(0) as number[];
        const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + i * binWidth);
        dn.forEach((v) => {
          const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
          histogram[idx] = (histogram[idx] ?? 0) + 1;
        });

        setPixelStats({ mean, median, std, min, max, sum, histogram, binEdges });
      } catch (e) {
        console.error("Failed to fetch stats:", e);
        setPixelStats(null);
      } finally {
        setPixelLoading(false);
      }
    }

    fetchStats();
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.productId, selected?.pixelLine, selected?.pixelSample, windowSize]);

  // ── CRISM Spectrum ──
  const { spectrumData, dustAssessment, loading: spectrumLoading } = useSpectrumData(
    selected?.productId ?? null,
    selected?.instrument ?? null,
    selected?.pixelLine,
    selected?.pixelSample,
    selected?.lat ?? 0,
    selected?.lon ?? 0,
  );

  // ── Product Metadata (enriched from GeoJSON) ──
  const { metadata, loading: metadataLoading } = useProductMetadata(
    selected?.productId ?? null,
    selected?.instrument ?? null,
  );

  // ── RGB State ──
  const [localRGB, setLocalRGB] = useState<RGBWavelengths>(rgbWavelengths || DEFAULT_RGB);
  useEffect(() => {
    if (rgbWavelengths) setLocalRGB(rgbWavelengths);
  }, [rgbWavelengths]);

  const handleRGBSliderChange = (channel: "r" | "g" | "b", value: number) => {
    setLocalRGB((prev) => ({ ...prev, [channel]: value }));
  };

  const handleApplyRGB = () => {
    if (activeOverlay?.type !== "highres") onSetOverlay("highres");
    onRGBChange?.(localRGB);
  };

  // ── High-Res Image Viewer ──
  const [showHiResViewer, setShowHiResViewer] = useState(false);

  if (!selected) return null;

  const isHiRISE = selected.instrument === "HIRISE";
  const isCRISM = selected.instrument === "CRISM" || selected.instrument === "CRISM_TRR3";
  const isTRR3 = selected.instrument === "CRISM_TRR3";
  const isCustom = selected.instrument === "CUSTOM";
  const isDTM = selected.instrument === "HIRISE_DTM";

  return (
    <aside
      className={`relative flex flex-col bg-surface-dark/40 ${isMobile ? "w-full" : "h-full border-l border-border-dark"}`}
      style={isMobile ? undefined : { width: panelWidth }}
      role="complementary"
      aria-label="Product inspector"
    >
      {/* Resize handle (desktop) */}
      {!isMobile && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 hover:bg-primary/30 active:bg-primary/50 transition-colors"
          onMouseDown={handleResizeStart}
        />
      )}

      {/* Header: Tabs + Recent Products */}
      <InspectorHeader
        selected={selected}
        hiriseTab={hiriseTab}
        onHiriseTabChange={setHiriseTab}
        crismTab={crismTab}
        onCrismTabChange={setCrismTab}
        onClose={onClose}
        onCollapse={onCollapse}
        recentProducts={recentProducts}
        onSelectRecent={onSelectRecent}
        onRemoveRecent={onRemoveRecent}
      />

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-dark space-y-0">
        {/* ── Tab Content ── */}

        {/* HiRISE */}
        {isHiRISE && hiriseTab === "Metadata" && (
          <MetadataTab selected={selected} metadata={metadata} metadataLoading={metadataLoading} hasHighResData={hasHighResData} />
        )}
        {isHiRISE && hiriseTab === "Pixel" && (
          <HiRISEPixelTab
            selected={selected}
            stats={pixelStats}
            loading={pixelLoading}
            windowSize={windowSize}
            onWindowSizeChange={setWindowSize}
            activeOverlayType={activeOverlay?.type ?? null}
            onSetOverlay={onSetOverlay}
          />
        )}

        {/* CRISM */}
        {isCRISM && crismTab === "Metadata" && (
          <MetadataTab
            selected={selected}
            metadata={metadata}
            metadataLoading={metadataLoading}
            hasHighResData={hasHighResData}
            onOpenMineralSequence={onOpenMineralSequence}
          />
        )}
        {isCRISM && crismTab === "Spectrum" && (
          <CRISMSpectrumTab
            selected={selected}
            spectrumData={spectrumData}
            dustAssessment={dustAssessment}
            loading={spectrumLoading}
            onPinSpectrum={onPinSpectrum}
            activeOverlayType={activeOverlay?.type ?? null}
            onSetOverlay={onSetOverlay}
          />
        )}
        {isCRISM && crismTab === "Bands" && (
          <CRISMBandsTab
            rgb={localRGB}
            onChange={handleRGBSliderChange}
            onApply={handleApplyRGB}
            isOverlayActive={activeOverlay?.type === "highres"}
          />
        )}

        {/* Custom */}
        {isCustom && customDataset && <CustomDataTab dataset={customDataset} />}

        {/* DTM */}
        {isDTM && <MetadataTab selected={selected} metadata={metadata} metadataLoading={metadataLoading} hasHighResData={hasHighResData} />}

        {/* Generic instruments (SHARAD, SHARAD_HIGHRES, CTX) */}
        {!isHiRISE && !isCRISM && !isCustom && !isDTM && (
          <MetadataTab selected={selected} metadata={metadata} metadataLoading={metadataLoading} hasHighResData={false} />
        )}

        {/* ── Overlay Controls (default OPEN) ── */}
        <div className="mt-4">
          <OverlayControls
            productId={selected.productId}
            instrument={selected.instrument}
            isTRR3={isTRR3}
            isCRISM={isCRISM}
            isCustom={isCustom}
            activeOverlay={activeOverlay}
            onSetOverlay={onSetOverlay}
            onSetOpacity={onSetOpacity}
            hasHighResData={hasHighResData}
            customDataset={customDataset}
            onCustomDatasetOpacity={onCustomDatasetOpacity}
          />
        </div>

        {/* ── View High-Res Image Button ── */}
        {(isHiRISE || isDTM) && (
          <button
            onClick={() => setShowHiResViewer(true)}
            className="flex w-full items-center justify-center gap-2 mt-4 py-3 rounded-lg bg-purple-500/20 border border-purple-500/50 text-purple-300 text-[11px] font-bold uppercase tracking-widest hover:bg-purple-500/30 active:scale-[0.98] transition-all"
          >
            <span className="material-symbols-outlined text-base">hd</span>
            View High-Res Image
          </button>
        )}

        {/* ── Action Bar ── */}
        <ActionBar
          productId={selected.productId}
          instrument={selected.instrument}
          lat={selected.lat}
          lon={selected.lon}
          isDTM={isDTM}
          onShow3DView={onShow3DView}
          fieldNotes={fieldNotes}
          onOpenFieldNote={onOpenFieldNote}
        />

      </div>

      {/* ── High-Res Image Viewer (fullscreen popup) ── */}
      {showHiResViewer && (
        <HiResImageViewer
          productId={selected.productId}
          onClose={() => setShowHiResViewer(false)}
        />
      )}
    </aside>
  );
}
