import { useCallback, useState } from "react";
import { getInstrumentIds, type InstrumentId } from "../config/instrumentRegistry";

/* =========================================================
 * Types
 * =======================================================*/

export type FootprintCount = { count: number; truncated: boolean; total: number } | null;

/** Explicit instrument names used by the legacy load trigger.
 *  These map 1:1 to instrumentRegistry.ts but uppercase. */
export type ExplicitLoadInstrument =
  | "CRISM"
  | "HIRISE"
  | "SHARAD"
  | "SHARAD_HIGHRES"
  | "CTX"
  | "CTX_MOSAIC"
  | "HIRISE_DTM"
  | "CRISM_TRR3";

export type FootprintLoadTrigger = {
  instrument: ExplicitLoadInstrument;
  timestamp: number;
} | null;

/* =========================================================
 * Hook
 * =======================================================*/

export interface UseFootprintLayersReturn {
  /** Per-instrument visibility (toggles only — does NOT trigger loading). */
  visibility: Record<InstrumentId, boolean>;
  /** Toggle one instrument on/off. */
  toggleInstrument: (id: InstrumentId, v: boolean) => void;
  /** Set one instrument's visibility directly. */
  setInstrumentVisible: (id: InstrumentId, v: boolean) => void;

  /** Per-instrument loaded counts. */
  counts: Record<string, FootprintCount>;
  setCounts: React.Dispatch<React.SetStateAction<Record<string, FootprintCount>>>;

  /** Per-instrument loading flags. */
  loading: Record<string, boolean>;
  setLoading: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;

  /** Explicit load trigger consumed by MapView. Set this to request a load. */
  loadTrigger: FootprintLoadTrigger;
  /** Trigger an explicit footprint load for an instrument (one-shot). */
  loadFootprints: (instrument: ExplicitLoadInstrument) => void;

  /** High-resolution-only filter (cross-cutting). */
  highResOnly: boolean;
  setHighResOnly: (v: boolean) => void;
}

/**
 * useFootprintLayers — owns instrument footprint visibility, counts, and load triggers.
 *
 * Phase 2: lifts state out of MainPage with the same shape used today.
 * Phase 3: this will be reorganized into the 4-lane variant model
 *   (SHARAD: standard|highres, CRISM: standard|trr3, HIRISE: image|dtm, CTX).
 */
export default function useFootprintLayers(): UseFootprintLayersReturn {
  const [visibility, setVisibility] = useState<Record<InstrumentId, boolean>>(
    () =>
      Object.fromEntries(getInstrumentIds().map((id) => [id, false])) as Record<
        InstrumentId,
        boolean
      >
  );

  const setInstrumentVisible = useCallback((id: InstrumentId, v: boolean) => {
    setVisibility((prev) => ({ ...prev, [id]: v }));
  }, []);

  const toggleInstrument = useCallback((id: InstrumentId, v: boolean) => {
    setVisibility((prev) => ({ ...prev, [id]: v }));
  }, []);

  const [counts, setCounts] = useState<Record<string, FootprintCount>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  const [loadTrigger, setLoadTrigger] = useState<FootprintLoadTrigger>(null);
  const loadFootprints = useCallback((instrument: ExplicitLoadInstrument) => {
    setLoadTrigger({ instrument, timestamp: Date.now() });
  }, []);

  const [highResOnly, setHighResOnly] = useState(false);

  return {
    visibility,
    toggleInstrument,
    setInstrumentVisible,
    counts,
    setCounts,
    loading,
    setLoading,
    loadTrigger,
    loadFootprints,
    highResOnly,
    setHighResOnly,
  };
}
