import { useCallback, useEffect, useRef, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { isInstrumentId, type InstrumentId } from "../config/instrumentRegistry";

/**
 * Supported URL state parameters for bookmarkable/shareable links.
 *
 * Example URL: /?lat=18.44&lon=77.45&instruments=crism,hirise&product=FRT0001&base=MOLA&view=2D
 *
 * Phase 3 follow-up D removed legacy `mode=` parameter (was source of
 * a crash-loop that required a sessionStorage workaround).
 */
export interface UrlState {
  lat?: number;
  lon?: number;
  camLat?: number;
  camLon?: number;
  camHeight?: number;
  instruments?: InstrumentId[];
  product?: string;
  base?: string;
  view?: string;
}

/** Debounce interval for URL writes (ms) */
const DEBOUNCE_MS = 500;

/**
 * Parse current URL search params into a typed UrlState object.
 * - lat/lon: parsed as floats, ignored if NaN
 * - instruments: comma-separated, filtered to valid InstrumentIds
 * - product: string (also reads legacy "flyTo" param for backward compat)
 * - mode: string
 * - base: string
 */
function parseParams(params: URLSearchParams): UrlState {
  const state: UrlState = {};

  // lat
  const latStr = params.get("lat");
  if (latStr !== null) {
    const lat = parseFloat(latStr);
    if (!Number.isNaN(lat)) state.lat = lat;
  }

  // lon
  const lonStr = params.get("lon");
  if (lonStr !== null) {
    const lon = parseFloat(lonStr);
    if (!Number.isNaN(lon)) state.lon = lon;
  }

  const camLatStr = params.get("camLat");
  if (camLatStr !== null) {
    const camLat = parseFloat(camLatStr);
    if (!Number.isNaN(camLat)) state.camLat = camLat;
  }

  const camLonStr = params.get("camLon");
  if (camLonStr !== null) {
    const camLon = parseFloat(camLonStr);
    if (!Number.isNaN(camLon)) state.camLon = camLon;
  }

  const camHeightStr = params.get("camHeight");
  if (camHeightStr !== null) {
    const camHeight = parseFloat(camHeightStr);
    if (!Number.isNaN(camHeight)) state.camHeight = camHeight;
  }

  // instruments (comma-separated)
  const instStr = params.get("instruments");
  if (instStr) {
    const ids = instStr
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(isInstrumentId) as InstrumentId[];
    if (ids.length > 0) state.instruments = ids;
  }

  // product (new param) OR flyTo (legacy param)
  const product = params.get("product");
  const flyTo = params.get("flyTo");
  if (product) {
    state.product = product;
  } else if (flyTo) {
    state.product = flyTo;
  }

  // base
  const base = params.get("base");
  if (base) state.base = base;
  // view (2D / 3D)
  const view = params.get("view");
  if (view) state.view = view;

  return state;
}

/**
 * Serialize a UrlState into URLSearchParams, omitting undefined/empty values.
 */
function serializeState(state: UrlState): URLSearchParams {
  const params = new URLSearchParams();

  if (state.lat !== undefined) params.set("lat", state.lat.toFixed(4));
  if (state.lon !== undefined) params.set("lon", state.lon.toFixed(4));
  if (state.camLat !== undefined) params.set("camLat", state.camLat.toFixed(2));
  if (state.camLon !== undefined) params.set("camLon", state.camLon.toFixed(2));
  if (state.camHeight !== undefined) params.set("camHeight", state.camHeight.toFixed(2));
  if (state.instruments && state.instruments.length > 0) {
    params.set("instruments", state.instruments.join(","));
  }
  if (state.product) params.set("product", state.product);
  if (state.base) params.set("base", state.base);
  if (state.view) params.set("view", state.view);

  return params;
}

/**
 * Custom hook that syncs app state with URL query parameters.
 *
 * - Reads state from URL on mount (returned as `urlState`).
 * - Provides `updateUrl(partial)` which debounces writes by 500ms
 *   and uses `replaceState` to avoid polluting browser history.
 * - Handles legacy `?flyTo=PRODUCT_ID` by mapping it to `product`.
 */
export default function useUrlState() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Parse current params on every render (cheap operation).
  // But `urlState` is only used for the initial mount read in MainPage,
  // so we memoize based on the raw search string.
  const urlState = useMemo(
    () => parseParams(searchParams),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [searchParams.toString()],
  );

  // Internal accumulated state for debounced writes.
  // This merges successive updateUrl calls into one URL write.
  const pendingRef = useRef<UrlState>({});
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep a ref to the latest setSearchParams to avoid stale closures
  const setParamsRef = useRef(setSearchParams);
  setParamsRef.current = setSearchParams;

  // On mount: if legacy "flyTo" param is present, clean it up
  // (the parsed urlState.product already captured the value above).
  useEffect(() => {
    if (searchParams.has("flyTo")) {
      const next = new URLSearchParams(searchParams);
      const flyTo = next.get("flyTo");
      next.delete("flyTo");
      if (flyTo) next.set("product", flyTo);
      setSearchParams(next, { replace: true });
    }
    // Run only once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Merge partial state into the URL. Debounced by 500ms.
   * Uses `replace: true` so the back button is not polluted.
   *
   * Pass `undefined` for a key to remove it from the URL.
   */
  const updateUrl = useCallback((partial: Partial<UrlState>) => {
    // Merge into pending state
    pendingRef.current = { ...pendingRef.current, ...partial };

    // Reset debounce timer
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }

    timerRef.current = setTimeout(() => {
      const merged = pendingRef.current;
      const params = serializeState(merged);
      setParamsRef.current(params, { replace: true });
      // Don't clear pendingRef -- it represents the full accumulated state.
      // Next updateUrl calls will merge on top of it.
      timerRef.current = null;
    }, DEBOUNCE_MS);
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  return { urlState, updateUrl };
}
