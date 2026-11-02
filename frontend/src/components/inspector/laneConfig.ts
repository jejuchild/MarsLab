import type { Lane, LaneVariant } from "../../api/inspector";
import type { InstrumentType } from "./types";

/**
 * Static description of the 4 Inspector lanes (docs/identity.md).
 *
 * Each lane variant maps 1:1 to the backend `/api/inspector/at-point`
 * variant and to the InstrumentType used by product-level tabs.
 */
export interface LaneVariantConfig {
  value: LaneVariant;
  label: string;
  description: string;
  instrument: InstrumentType;
}

export interface LaneConfig {
  label: string;
  accent: string;
  variants: LaneVariantConfig[];
}

export const LANES: Lane[] = ["SHARAD", "CRISM", "HIRISE", "CTX"];

export const LANE_CONFIG: Record<Lane, LaneConfig> = {
  SHARAD: {
    label: "SHARAD",
    accent: "#ef4444",
    variants: [
      { value: "standard", label: "Standard", description: "SHARAD radargram trace coverage", instrument: "SHARAD" },
      { value: "highres", label: "Hi-res", description: "High-resolution radargram", instrument: "SHARAD_HIGHRES" },
    ],
  },
  CRISM: {
    label: "CRISM",
    accent: "#fbbf24",
    variants: [
      { value: "standard", label: "Standard", description: "Standard CRISM cube", instrument: "CRISM" },
      { value: "trr3", label: "TRR3", description: "TRR3 mineral classification", instrument: "CRISM_TRR3" },
    ],
  },
  HIRISE: {
    label: "HiRISE",
    accent: "#22c55e",
    variants: [
      { value: "image", label: "Image", description: "High-resolution image", instrument: "HIRISE" },
      { value: "dtm", label: "DTM", description: "Digital terrain model (elevation)", instrument: "HIRISE_DTM" },
    ],
  },
  CTX: {
    label: "CTX",
    accent: "#60a5fa",
    variants: [
      { value: "image", label: "Image", description: "Wide-area context image", instrument: "CTX" },
      { value: "mosaic", label: "Mosaic", description: "Murray Lab global CTX mosaic", instrument: "CTX" },
    ],
  },
};

/** Map a product-level instrument (e.g. from a footprint click) to its lane + variant. */
export function laneForInstrument(
  instrument: InstrumentType,
): { lane: Lane; variant: LaneVariant } | null {
  for (const lane of LANES) {
    const v = LANE_CONFIG[lane].variants.find((cfg) => cfg.instrument === instrument);
    if (v) return { lane, variant: v.value };
  }
  return null; // e.g. CUSTOM datasets have no lane
}

export function instrumentFor(lane: Lane, variant: LaneVariant): InstrumentType {
  const cfg = LANE_CONFIG[lane].variants.find((v) => v.value === variant);
  return cfg?.instrument ?? LANE_CONFIG[lane].variants[0]!.instrument;
}
