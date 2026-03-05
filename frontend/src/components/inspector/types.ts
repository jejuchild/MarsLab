import type { OverlayType, ProductOverlay, CustomDataset } from "../../pages/MainPage";
import type { FieldNote } from "../../api/fieldnotes";

/* ── Instrument Types ── */
export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

/* ── Inspector Context (what product is selected) ── */
export type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  pixelLine?: number;
  pixelSample?: number;
  title?: string;
};

/* ── RGB Wavelengths ── */
export type RGBWavelengths = {
  r: number;
  g: number;
  b: number;
};

/* ── Tab Keys ── */
export type HiRISETabKey = "Metadata" | "Pixel";
export type CRISMTabKey = "Metadata" | "Spectrum" | "Bands";

/* ── Spectrum Data ── */
export type SpectrumData = {
  wavelengths: number[];
  reflectance: (number | null)[];
  validBands: number;
};

/* ── Dust Assessment ── */
export type DustAssessment = {
  tau_estimated: number;
  risk_level: "LOW" | "MODERATE" | "HIGH";
  spectral_slope: number | null;
  band_depth_suppression_pct: number;
  warning_message: string | null;
};

/* ── Window Stats (HiRISE pixel analysis) ── */
export type WindowStats = {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  sum: number;
  histogram: number[];
  binEdges: number[];
};

/* ── Product Metadata (enriched from GeoJSON index) ── */
export type ProductMetadata = {
  title?: string;
  centerLatitude?: number;
  centerLongitude?: number;
  observationDate?: string;
  solarIncidence?: number;
  emissionAngle?: number;
  phaseAngle?: number;
  resolution?: number;        // m/pixel
  imageLines?: number;
  imageSamples?: number;
  orbitNumber?: number;
  productType?: string;
  targetName?: string;
  description?: string;
  // CRISM-specific
  sensorId?: string;
  wavelengthRange?: string;
  // HiRISE-specific
  hasColorData?: boolean;
  mapScale?: number;
};

/* ── Recent Product ── */
export type RecentProduct = {
  productId: string;
  instrument: InstrumentType;
  lat: number;
  lon: number;
  title?: string;
};

/* ── Overlay display configuration ── */
export const OVERLAY_CONFIG: Record<OverlayType, { label: string; activeClass: string; icon: string; description: string }> = {
  quickview: {
    label: "Quickview",
    activeClass: "bg-emerald-500/20 border border-emerald-500/50 text-emerald-400",
    icon: "visibility",
    description: "Low-resolution preview image",
  },
  highres: {
    label: "High-Res",
    activeClass: "bg-purple-500/20 border border-purple-500/50 text-purple-400",
    icon: "hd",
    description: "Full-resolution image overlay",
  },
  browse_HYD: {
    label: "HYD",
    activeClass: "bg-fuchsia-500/20 border border-fuchsia-500/50 text-fuchsia-400",
    icon: "water_drop",
    description: "Hydrated mineral browse product",
  },
  browse_ICE: {
    label: "ICE",
    activeClass: "bg-blue-500/20 border border-blue-500/50 text-blue-400",
    icon: "ac_unit",
    description: "H₂O ice browse product",
  },
  browse_IC2: {
    label: "IC2",
    activeClass: "bg-cyan-500/20 border border-cyan-500/50 text-cyan-400",
    icon: "ac_unit",
    description: "CO₂ ice browse product",
  },
  score_ice: {
    label: "S-ICE",
    activeClass: "bg-sky-500/20 border border-sky-500/50 text-sky-400",
    icon: "analytics",
    description: "Ice detection score map",
  },
  score_hyd: {
    label: "S-HYD",
    activeClass: "bg-rose-500/20 border border-rose-500/50 text-rose-400",
    icon: "analytics",
    description: "Hydration detection score map",
  },
  mineral_cnn: {
    label: "Minerals",
    activeClass: "bg-amber-500/20 border border-amber-500/50 text-amber-400",
    icon: "science",
    description: "CNN mineral classification map",
  },
};

/* ── Default CRISM Wavelengths ── */
export const DEFAULT_RGB: RGBWavelengths = {
  r: 2.53,
  g: 1.51,
  b: 1.08,
};

/* ── All Inspector Props ── */
export type InspectorPanelProps = {
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
};

/* ── Helpers ── */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
