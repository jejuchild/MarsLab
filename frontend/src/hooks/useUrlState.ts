import { useCallback, useEffect, useRef } from "react";
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
 * One-time deep links + explicit share links.
 *
 * The app used to mirror its state into the address bar continuously
 * (debounced replaceState). A plain reload or revisit then replayed stale
 * state: fly to the last product, auto-load every layer that had been on.
 * Now:
 *   - `urlState` is read ONCE on mount (shared links, `/?flyTo=` from the
 *     download page) and the params are immediately stripped from the URL,
 *     so the next reload starts clean.
 *   - `updateUrl(partial)` only records the current view in memory.
 *   - `getShareUrl()` builds a link to the current view on demand
 *     (the "Copy link" button).
 * The legacy `?flyTo=PRODUCT_ID` param is read as `product`.
 */
export default function useUrlState() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Frozen initial state: later URL changes never re-trigger a restore.
  const initialRef = useRef<UrlState | null>(null);
  if (initialRef.current === null) initialRef.current = parseParams(searchParams);
  const urlState = initialRef.current;

  // Strip state params once they have been read.
  useEffect(() => {
    const STATE_KEYS = ["lat", "lon", "camLat", "camLon", "camHeight", "instruments", "product", "flyTo", "base", "view", "instrument"];
    if (!STATE_KEYS.some((k) => searchParams.has(k))) return;
    const next = new URLSearchParams(searchParams);
    for (const k of STATE_KEYS) next.delete(k);
    setSearchParams(next, { replace: true });
    // Run only once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Current view, accumulated from updateUrl calls (memory only).
  const viewRef = useRef<UrlState>({});

  /** Record part of the current view (no address-bar write). `undefined` clears a key. */
  const updateUrl = useCallback((partial: Partial<UrlState>) => {
    viewRef.current = { ...viewRef.current, ...partial };
  }, []);

  /** Absolute link that restores the current view once when opened. */
  const getShareUrl = useCallback((): string => {
    const params = serializeState(viewRef.current);
    const query = params.toString();
    return `${window.location.origin}${window.location.pathname}${query ? `?${query}` : ""}`;
  }, []);

  return { urlState, updateUrl, getShareUrl };
}
