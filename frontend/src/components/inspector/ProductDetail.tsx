import { useEffect, useState } from "react";
import { fetchHiRISEWindow } from "../../api/hirise";
import type { OverlayType, ProductOverlay, CustomDataset } from "../../pages/MainPage";
import type { FieldNote } from "../../api/fieldnotes";
import type { InspectorContext, RGBWavelengths, WindowStats } from "./types";
import { DEFAULT_RGB } from "./types";
import useProductMetadata from "./hooks/useProductMetadata";
import useSpectrumData from "./hooks/useSpectrumData";
import usePixelAtPoint from "./hooks/usePixelAtPoint";
import MetadataTab from "./tabs/MetadataTab";
import HiRISEPixelTab from "./tabs/HiRISEPixelTab";
import CRISMSpectrumTab from "./tabs/CRISMSpectrumTab";
import CRISMBandsTab from "./tabs/CRISMBandsTab";
import CustomDataTab from "./tabs/CustomDataTab";
import OverlayControls from "./overlays/OverlayControls";
import ActionBar from "./actions/ActionBar";
import HiResImageViewer from "./HiResImageViewer";

/** Map-side handlers shared by every product shown in the Inspector. */
export interface ProductDetailHandlers {
  getActiveOverlay: (productId: string) => ProductOverlay | null;
  onSetOverlay: (productId: string, type: OverlayType | null) => void;
  onSetOpacity?: (productId: string, opacity: number) => void;
  rgbWavelengths?: RGBWavelengths;
  onRGBChange?: (rgb: RGBWavelengths) => void;
  hasHighResData: (productId: string) => boolean;
  getCustomDataset: (productId: string) => CustomDataset | null;
  onCustomDatasetOpacity?: (id: string, opacity: number) => void;
  fieldNotes?: FieldNote[];
  onOpenFieldNote?: (productId: string, instrument: string, lat: number, lon: number) => void;
  onShow3DView?: (productId: string, lat: number, lon: number) => void;
  onPinSpectrum?: (spectrum: { productId: string; lat: number; lon: number; wavelengths: number[]; reflectance: (number | null)[] }) => void;
  onOpenMineralSequence?: (obsId: string) => void;
}

type DetailTab = "Metadata" | "Pixel" | "Spectrum" | "Bands";

function tabsFor(instrument: InspectorContext["instrument"]): DetailTab[] {
  if (instrument === "HIRISE") return ["Metadata", "Pixel"];
  if (instrument === "CRISM" || instrument === "CRISM_TRR3") return ["Metadata", "Spectrum", "Bands"];
  return ["Metadata"];
}

/**
 * Product-level detail body (metadata, spectrum, pixel stats, overlays,
 * actions) rendered inside an Inspector lane. Formerly the legacy
 * InspectorPanel, which required a full page reload to reach.
 */
export default function ProductDetail({
  product,
  handlers,
}: {
  product: InspectorContext;
  handlers: ProductDetailHandlers;
}) {
  const { productId, instrument } = product;
  const isHiRISE = instrument === "HIRISE";
  const isCRISM = instrument === "CRISM" || instrument === "CRISM_TRR3";
  const isTRR3 = instrument === "CRISM_TRR3";
  const isCustom = instrument === "CUSTOM";
  const isDTM = instrument === "HIRISE_DTM";
  const activeOverlay = handlers.getActiveOverlay(productId);
  const customDataset = isCustom ? handlers.getCustomDataset(productId) : null;

  // ── Tabs (reset when the product changes) ──
  const tabs = tabsFor(instrument);
  const [tab, setTab] = useState<DetailTab>("Metadata");
  useEffect(() => setTab("Metadata"), [productId, instrument]);

  // ── High-res availability: map-provided flag or on-disk check ──
  const [localHighRes, setLocalHighRes] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setLocalHighRes(false);
    fetch(`/api/exists/${instrument.toLowerCase()}/${encodeURIComponent(productId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled) setLocalHighRes(d?.has_core ?? false); })
      .catch(() => { if (!cancelled) setLocalHighRes(false); });
    return () => { cancelled = true; };
  }, [productId, instrument]);
  const hasHighResData = handlers.hasHighResData(productId) || localHighRes;

  // ── HiRISE pixel window stats ──
  const [windowSize, setWindowSize] = useState(5);
  const [pixelStats, setPixelStats] = useState<WindowStats | null>(null);
  const [pixelLoading, setPixelLoading] = useState(false);
  const { pixelLine, pixelSample } = product;
  useEffect(() => {
    if (!isHiRISE || pixelLine === undefined || pixelSample === undefined) {
      setPixelStats(null);
      return;
    }
    let cancelled = false;
    setPixelLoading(true);
    fetchHiRISEWindow(productId, pixelSample, pixelLine, Math.floor(windowSize / 2))
      .then((data) => {
        if (cancelled) return;
        setPixelStats(computeWindowStats(data.dn.flat() as number[]));
      })
      .catch((e) => {
        if (cancelled) return;
        console.error("Failed to fetch stats:", e);
        setPixelStats(null);
      })
      .finally(() => { if (!cancelled) setPixelLoading(false); });
    return () => { cancelled = true; };
  }, [isHiRISE, productId, pixelLine, pixelSample, windowSize]);

  // ── CRISM pixel at the inspected point (clicked or resolved) ──
  const pixel = usePixelAtPoint(product);

  // ── CRISM spectrum + metadata ──
  const { spectrumData, dustAssessment, loading: spectrumLoading } = useSpectrumData(
    productId, instrument,
    pixel.line, pixel.sample,
    product.lat, product.lon,
  );
  const { metadata, loading: metadataLoading } = useProductMetadata(productId, instrument);

  // ── RGB composite ──
  const [localRGB, setLocalRGB] = useState<RGBWavelengths>(handlers.rgbWavelengths || DEFAULT_RGB);
  useEffect(() => {
    if (handlers.rgbWavelengths) setLocalRGB(handlers.rgbWavelengths);
  }, [handlers.rgbWavelengths]);

  const [showHiResViewer, setShowHiResViewer] = useState(false);
  const setOverlay = (type: OverlayType | null) => handlers.onSetOverlay(productId, type);

  return (
    <div className="border-t border-border-dark/60">
      {tabs.length > 1 && (
        <div className="flex border-b border-border-dark" role="tablist">
          {tabs.map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`flex-1 py-2.5 text-[10px] font-bold uppercase tracking-widest border-b-2 transition-colors ${
                tab === t ? "border-primary bg-primary/5 text-white" : "border-transparent text-slate-500 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      )}

      <div className="p-4">
        {tab === "Metadata" && (
          isCustom
            ? customDataset && <CustomDataTab dataset={customDataset} />
            : (
              <MetadataTab
                selected={product}
                metadata={metadata}
                metadataLoading={metadataLoading}
                hasHighResData={isHiRISE || isCRISM || isDTM ? hasHighResData : false}
                onOpenMineralSequence={isCRISM ? handlers.onOpenMineralSequence : undefined}
              />
            )
        )}
        {tab === "Pixel" && isHiRISE && (
          <HiRISEPixelTab
            selected={product}
            stats={pixelStats}
            loading={pixelLoading}
            windowSize={windowSize}
            onWindowSizeChange={setWindowSize}
            activeOverlayType={activeOverlay?.type ?? null}
            onSetOverlay={setOverlay}
          />
        )}
        {tab === "Spectrum" && isCRISM && (
          <CRISMSpectrumTab
            selected={product}
            spectrumData={spectrumData}
            dustAssessment={dustAssessment}
            loading={spectrumLoading}
            onPinSpectrum={handlers.onPinSpectrum}
            activeOverlayType={activeOverlay?.type ?? null}
            onSetOverlay={setOverlay}
          />
        )}
        {tab === "Bands" && isCRISM && (
          <CRISMBandsTab
            rgb={localRGB}
            onChange={(channel, value) => setLocalRGB((prev) => ({ ...prev, [channel]: value }))}
            onApply={() => {
              if (activeOverlay?.type !== "highres") setOverlay("highres");
              handlers.onRGBChange?.(localRGB);
            }}
            isOverlayActive={activeOverlay?.type === "highres"}
          />
        )}

        <div className="mt-4">
          <OverlayControls
            productId={productId}
            instrument={instrument}
            isTRR3={isTRR3}
            isCRISM={isCRISM}
            isCustom={isCustom}
            activeOverlay={activeOverlay}
            onSetOverlay={setOverlay}
            onSetOpacity={handlers.onSetOpacity ? (o) => handlers.onSetOpacity!(productId, o) : undefined}
            hasHighResData={hasHighResData}
            customDataset={customDataset}
            onCustomDatasetOpacity={handlers.onCustomDatasetOpacity}
          />
        </div>

        {isHiRISE && (
          <button
            onClick={() => setShowHiResViewer(true)}
            className="flex w-full items-center justify-center gap-2 mt-4 py-3 rounded-lg bg-purple-500/20 border border-purple-500/50 text-purple-300 text-[11px] font-bold uppercase tracking-widest hover:bg-purple-500/30 active:scale-[0.98] transition-all"
          >
            <span className="material-symbols-outlined text-base">hd</span>
            View High-Res Image
          </button>
        )}

        <ActionBar
          productId={productId}
          instrument={instrument}
          lat={product.lat}
          lon={product.lon}
          isDTM={isDTM}
          onShow3DView={handlers.onShow3DView}
          fieldNotes={handlers.fieldNotes ?? []}
          onOpenFieldNote={handlers.onOpenFieldNote}
        />
      </div>

      {showHiResViewer && (
        <HiResImageViewer
          productId={productId}
          lat={product.lat}
          lon={product.lon}
          onClose={() => setShowHiResViewer(false)}
        />
      )}
    </div>
  );
}

function computeWindowStats(dn: number[]): WindowStats | null {
  if (dn.length === 0) return null;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of dn) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / dn.length;
  const sorted = [...dn].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const std = Math.sqrt(dn.reduce((acc, v) => acc + (v - mean) ** 2, 0) / dn.length);
  const bins = 10;
  const binWidth = (max - min) / bins || 1;
  const histogram = new Array(bins).fill(0) as number[];
  const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + i * binWidth);
  for (const v of dn) {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    histogram[idx] = (histogram[idx] ?? 0) + 1;
  }
  return { mean, median, std, min, max, sum, histogram, binEdges };
}
