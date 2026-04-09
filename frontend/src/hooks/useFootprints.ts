import { useCallback, useEffect, useRef, useState } from "react";
import toast from "react-hot-toast";
import * as Cesium from "cesium";
import FootprintManager, { type FootprintFeature } from "../utils/FootprintManager";
import {
  computeOverlapFilter,
  type OverlapResult,
  type OverlapStats,
} from "../utils/overlapFilter";
import type { InstrumentType as FPInstrumentType } from "../utils/FootprintManager";

type InstrumentType =
  | "CRISM"
  | "HIRISE"
  | "SHARAD"
  | "SHARAD_HIGHRES"
  | "CTX"
  | "CUSTOM"
  | "HIRISE_DTM"
  | "CRISM_TRR3";

type VisibleProduct = {
  productId: string;
  instrument: InstrumentType;
  title?: string;
  lat?: number;
  lon?: number;
};

type ExplicitLoadInstrument =
  | "CRISM"
  | "HIRISE"
  | "SHARAD"
  | "SHARAD_HIGHRES"
  | "CTX"
  | "CTX_MOSAIC"
  | "HIRISE_DTM"
  | "CRISM_TRR3";

type FootprintLoadResult = {
  instrument: ExplicitLoadInstrument;
  count: number;
  truncated: boolean;
  total: number;
};

type UseFootprintsParams = {
  footprintManagerRef?: React.MutableRefObject<FootprintManager | null>;
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  showCRISM: boolean;
  showHiRISE: boolean;
  showSHARAD: boolean;
  showSharadHighres: boolean;
  showCTX: boolean;
  showHiRISEDTM: boolean;
  showCRISM_TRR3: boolean;
  crismFilteredIds: Set<string> | null;
  loadFootprintsTrigger: {
    instrument: ExplicitLoadInstrument;
    timestamp: number;
  } | null;
  onFootprintsLoaded?: (result: FootprintLoadResult) => void;
  onFootprintsLoading?: (instrument: ExplicitLoadInstrument, loading: boolean) => void;
  overlapFilter?: { enabled: boolean; instruments: string[] };
  onOverlapStatsChange?: (stats: OverlapStats | null) => void;
  inspectedProductId?: string | null;
  onVisibleProductsChange?: (products: VisibleProduct[]) => void;
  showCustomData: boolean;
  customDatasets: Array<{
    id: string;
    name: string;
    bounds: { west: number; south: number; east: number; north: number };
    visible: boolean;
    opacity: number;
  }>;
  showRegionLayer: boolean;
  extractCrismObsId: (productId: string) => string;
  highResOnly?: boolean;
};

type UseFootprintsResult = {
  footprintManagerRef: React.MutableRefObject<FootprintManager | null>;
};

export default function useFootprints({
  footprintManagerRef: externalFootprintManagerRef,
  viewerRef,
  marsEllipsoid,
  showCRISM,
  showHiRISE,
  showSHARAD,
  showSharadHighres,
  showCTX,
  showHiRISEDTM,
  showCRISM_TRR3,
  crismFilteredIds,
  loadFootprintsTrigger,
  onFootprintsLoaded,
  onFootprintsLoading,
  overlapFilter,
  onOverlapStatsChange,
  inspectedProductId,
  onVisibleProductsChange,
  showCustomData,
  customDatasets,
  showRegionLayer,
  extractCrismObsId,
  highResOnly = false,
}: UseFootprintsParams): UseFootprintsResult {
  const internalFootprintManagerRef = useRef<FootprintManager | null>(null);
  const footprintManagerRef = externalFootprintManagerRef ?? internalFootprintManagerRef;

  // Overlap filter state
  const overlapResultRef = useRef<OverlapResult | null>(null);
  const inspectedProductIdRef = useRef<string | null>(null);
  const [footprintVersion, setFootprintVersion] = useState(0);
  const overlapEnabled = overlapFilter?.enabled ?? false;

  const onFootprintsLoadedRef = useRef(onFootprintsLoaded);
  const onFootprintsLoadingRef = useRef(onFootprintsLoading);

  useEffect(() => {
    onFootprintsLoadedRef.current = onFootprintsLoaded;
  }, [onFootprintsLoaded]);

  useEffect(() => {
    onFootprintsLoadingRef.current = onFootprintsLoading;
  }, [onFootprintsLoading]);

  useEffect(() => {
    let cancelled = false;
    let poll: ReturnType<typeof setInterval> | null = null;

    const tryInitialize = (): boolean => {
      const viewer = viewerRef.current;
      if (!viewer || footprintManagerRef.current) return false;

      // Initialize FootprintManager for explicit snapshot-based loading
      // NO automatic camera-based updates - footprints load only on explicit button click
      const footprintManager = new FootprintManager({
        viewer,
        ellipsoid: marsEllipsoid,
        onLoadStart: (instrument) => {
          onFootprintsLoadingRef.current?.(instrument, true);
        },
        onLoadEnd: (instrument, result) => {
          onFootprintsLoadingRef.current?.(instrument, false);
          onFootprintsLoadedRef.current?.({
            instrument,
            count: result.count,
            truncated: result.truncated,
            total: result.total,
          });
        },
        onError: (instrument, error) => {
          console.error(`[FootprintManager] Error loading ${instrument}:`, error);
          onFootprintsLoadingRef.current?.(instrument, false);
        },
      });

      if (cancelled) {
        footprintManager.dispose();
        return true;
      }

      footprintManagerRef.current = footprintManager;
      return true;
    };

    if (!tryInitialize()) {
      poll = setInterval(() => {
        if (tryInitialize() && poll) {
          clearInterval(poll);
        }
      }, 150);
    }

    return () => {
      cancelled = true;
      if (poll) {
        clearInterval(poll);
      }
      // Guard: only dispose if manager exists and viewer is still alive
      // (viewer may already be destroyed during render error cascade)
      if (footprintManagerRef.current) {
        try {
          footprintManagerRef.current.dispose();
        } catch {
          // Swallow errors during teardown — viewer may be in bad state
        }
        footprintManagerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [marsEllipsoid, viewerRef]);

  // Sync highResOnly to FootprintManager
  useEffect(() => {
    const fm = footprintManagerRef.current;
    if (fm) fm.highResOnly = highResOnly;
  }, [highResOnly, footprintManagerRef]);

  // Keep inspected product ref in sync
  inspectedProductIdRef.current = inspectedProductId ?? null;

  // Helper: apply per-entity visibility for an instrument considering overlap filter
  // Always force-shows the inspected product's footprint regardless of visibility/filter state
  const applyInstrumentVisibility = useCallback((instrument: FPInstrumentType, show: boolean) => {
    const fm = footprintManagerRef.current;
    const viewer = viewerRef.current;
    if (!fm || !viewer) return;

    const overlapPassing = overlapResultRef.current?.get(instrument);
    if (show && overlapPassing) {
      // Overlap filter active: per-entity visibility
      const features = fm.getFeatures(instrument);
      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;
        const visible = overlapPassing.has(pid);
        fm.setFeatureVisible(instrument, pid, visible);
      }
    } else {
      fm.setVisible(instrument, show);
    }

    // Force-show inspected product's footprint regardless of layer/filter state
    const ipid = inspectedProductIdRef.current;
    if (ipid) {
      fm.setFeatureVisible(instrument, ipid, true);
    }

    viewer.scene.requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewerRef]);

  // Toggle footprint visibility (does NOT load new footprints, just shows/hides existing ones)
  useEffect(() => {
    applyInstrumentVisibility("HIRISE", showHiRISE);
  }, [showHiRISE, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("CRISM", showCRISM);
  }, [showCRISM, applyInstrumentVisibility]);

  // Explicit footprint loading - triggered by loadFootprintsTrigger prop
  // After loading completes, ensure visibility is set correctly (fixes reload visibility bug)
  useEffect(() => {
    if (!loadFootprintsTrigger || !footprintManagerRef.current) return;

    const { instrument } = loadFootprintsTrigger;
    const fm = footprintManagerRef.current;
    let cancelled = false;

    // Load footprints and then ensure visibility is set correctly
    // This fixes the issue where reloading doesn't show footprints because
    // the visibility effect doesn't re-run (show state unchanged)
    (async () => {
      try {
        const result = await fm.loadFootprints(instrument);
        if (cancelled) return;
        if (result && result.count > 0) {
          fm.setVisible(instrument, true);
          setFootprintVersion((v) => v + 1);
        }
      } catch (e) {
        if (!cancelled) toast.error(`Failed to load ${instrument} footprints`);
        console.error("Footprint load error:", e);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadFootprintsTrigger]);

  // Render loaded footprint bounding boxes as dashed rectangles on the map
  useEffect(() => {
    const viewer = viewerRef.current;
    const fm = footprintManagerRef.current;
    if (!viewer || !fm) return;

    // Remove old bbox entities
    const toRemove: string[] = [];
    for (let i = 0; i < viewer.entities.values.length; i++) {
      const e = viewer.entities.values[i]!;
      if (e.id && typeof e.id === "string" && e.id.startsWith("BBOX_")) {
        toRemove.push(e.id);
      }
    }
    for (const id of toRemove) {
      viewer.entities.removeById(id);
    }

    // Add bbox rectangles for all loaded instruments
    const bboxes = fm.getAllLoadedBboxes();
    const BBOX_COLORS: Record<string, Cesium.Color> = {
      CRISM: Cesium.Color.CYAN.withAlpha(0.35),
      HIRISE: Cesium.Color.YELLOW.withAlpha(0.35),
      SHARAD: Cesium.Color.ORANGE.withAlpha(0.35),
      SHARAD_HIGHRES: Cesium.Color.fromCssColorString("#FFD700").withAlpha(0.35),
      CTX: Cesium.Color.fromCssColorString("#FF1493").withAlpha(0.35),
      HIRISE_DTM: Cesium.Color.fromCssColorString("#B8860B").withAlpha(0.35),
      CRISM_TRR3: Cesium.Color.TEAL.withAlpha(0.35),
    };

    for (const [instrument, bbox] of bboxes) {
      const [west, south, east, north] = bbox;
      const color = BBOX_COLORS[instrument] || Cesium.Color.WHITE.withAlpha(0.3);
      viewer.entities.add({
        id: `BBOX_${instrument}`,
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
          material: Cesium.Color.TRANSPARENT,
          outline: true,
          outlineColor: color,
          outlineWidth: 2,
          height: 0,
        },
        properties: new Cesium.PropertyBag({ isBbox: true, instrument }),
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [footprintVersion, viewerRef]);

  // Update CRISM footprint visibility when ice score filter changes
  useEffect(() => {
    const viewer = viewerRef.current;
    const footprintManager = footprintManagerRef.current;
    if (!viewer || !footprintManager || !showCRISM) return;
    if (crismFilteredIds === null) return; // No ice filter active — let applyInstrumentVisibility handle it

    const crismFeatures = footprintManager.getFeatures("CRISM");
    const overlapPassingCrism = overlapResultRef.current?.get("CRISM" as FPInstrumentType);

    for (const feature of crismFeatures) {
      const pid = feature.properties.product_id;
      if (!pid) continue;

      // Compose filters: ice score AND overlap
      let visible = true;
      const obsId = extractCrismObsId(pid);
      visible = crismFilteredIds.has(obsId);
      if (visible && overlapPassingCrism) {
        visible = overlapPassingCrism.has(pid);
      }

      footprintManager.setFeatureVisible("CRISM", pid, visible);
    }

    // Force-show inspected product's CRISM footprint
    const ipid = inspectedProductIdRef.current;
    if (ipid) {
      footprintManager.setFeatureVisible("CRISM", ipid, true);
    }

    viewer.scene.requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [crismFilteredIds, showCRISM, applyInstrumentVisibility, extractCrismObsId, viewerRef]);

  useEffect(() => {
    applyInstrumentVisibility("SHARAD", showSHARAD);
  }, [showSHARAD, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("SHARAD_HIGHRES", showSharadHighres);
  }, [showSharadHighres, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("CTX", showCTX);
  }, [showCTX, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("CRISM_TRR3", showCRISM_TRR3);
  }, [showCRISM_TRR3, applyInstrumentVisibility]);

  useEffect(() => {
    applyInstrumentVisibility("HIRISE_DTM", showHiRISEDTM);
  }, [showHiRISEDTM, applyInstrumentVisibility]);

  // Force-show the inspected product's footprint when inspector selection changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !inspectedProductId) return;

    const instruments = [
      "HIRISE",
      "CRISM",
      "CTX",
      "SHARAD",
      "SHARAD_HIGHRES",
      "HIRISE_DTM",
      "CRISM_TRR3",
    ];
    for (const inst of instruments) {
      if (footprintManagerRef.current?.hasFeature(`${inst}_FP_${inspectedProductId}`)) {
        footprintManagerRef.current.setFeatureVisible(inst as FPInstrumentType, inspectedProductId, true);
        break;
      }
    }
    viewer.scene.requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspectedProductId, viewerRef]);

  // Note: Legacy footprint overlay hiding is no longer needed since
  // footprints are now managed by FootprintManager (viewport-based loading)

  // Multi-Instrument Overlap Filter: compute and apply visibility
  useEffect(() => {
    const fm = footprintManagerRef.current;
    const viewer = viewerRef.current;
    if (!fm || !viewer) return;

    // If filter disabled, clear and restore normal visibility
    if (!overlapEnabled) {
      overlapResultRef.current = null;
      onOverlapStatsChange?.(null);
      // Restore visibility for all instruments to match their toggle state
      if (showCRISM) fm.setVisible("CRISM", true);
      if (showHiRISE) fm.setVisible("HIRISE", true);
      if (showSHARAD) fm.setVisible("SHARAD", true);
      if (showSharadHighres) fm.setVisible("SHARAD_HIGHRES", true);
      if (showCTX) fm.setVisible("CTX", true);
      if (showHiRISEDTM) fm.setVisible("HIRISE_DTM", true);
      if (showCRISM_TRR3) fm.setVisible("CRISM_TRR3", true);
      // Re-apply CRISM ice score filter if active
      if (crismFilteredIds !== null && showCRISM) {
        const crismFeatures = fm.getFeatures("CRISM");
        for (const feature of crismFeatures) {
          const pid = feature.properties.product_id;
          if (!pid) continue;
          const obsId = extractCrismObsId(pid);
          const visible = crismFilteredIds.has(obsId);
          fm.setFeatureVisible("CRISM", pid, visible);
        }
      }
      viewer.scene.requestRender();
      return;
    }

    // Auto-detect all instruments with loaded features
    const allInstruments: FPInstrumentType[] = [
      "CRISM",
      "HIRISE",
      "SHARAD",
      "SHARAD_HIGHRES",
      "CTX",
      "HIRISE_DTM",
      "CRISM_TRR3",
    ];
    const featuresByInstrument = new Map<FPInstrumentType, FootprintFeature[]>();
    for (const inst of allInstruments) {
      const features = fm.getFeatures(inst);
      if (features.length > 0) {
        featuresByInstrument.set(inst, features);
      }
    }
    const selectedInstruments = Array.from(featuresByInstrument.keys());

    // Need at least 2 instruments with loaded features
    if (selectedInstruments.length < 2) {
      overlapResultRef.current = null;
      onOverlapStatsChange?.({
        totalChecked: 0,
        totalPassing: 0,
        perInstrument: new Map(),
      });
      viewer.scene.requestRender();
      return;
    }

    // Compute overlap
    const { result, stats } = computeOverlapFilter(featuresByInstrument, selectedInstruments);
    overlapResultRef.current = result;
    onOverlapStatsChange?.(stats);

    // Apply visibility for each selected instrument (batched)
    for (const inst of selectedInstruments) {
      const passingIds = result.get(inst);
      const features = fm.getFeatures(inst);
      // Check if this instrument is toggled on
      const instrumentOn = {
        CRISM: showCRISM,
        HIRISE: showHiRISE,
        SHARAD: showSHARAD,
        SHARAD_HIGHRES: showSharadHighres,
        CTX: showCTX,
        HIRISE_DTM: showHiRISEDTM,
        CRISM_TRR3: showCRISM_TRR3,
      }[inst] ?? false;

      if (!instrumentOn) continue;

      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;
        let visible = passingIds ? passingIds.has(pid) : false;
        // Also compose with CRISM ice score filter
        if (visible && inst === "CRISM" && crismFilteredIds !== null) {
          const obsId = extractCrismObsId(pid);
          visible = crismFilteredIds.has(obsId);
        }
        fm.setFeatureVisible(inst, pid, visible);
      }
    }

    viewer.scene.requestRender();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    overlapEnabled,
    footprintVersion,
    crismFilteredIds,
    showCRISM,
    showHiRISE,
    showSHARAD,
    showSharadHighres,
    showCTX,
    showHiRISEDTM,
    showCRISM_TRR3,
    extractCrismObsId,
    onOverlapStatsChange,
    viewerRef,
  ]);

  // PERFORMANCE OPTIMIZED: Track visible products in current view
  // Reduced polling frequency and removed excessive logging
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !onVisibleProductsChange) return;

    // Track last result to avoid unnecessary updates
    let lastResultHash = "";

    const updateVisibleProducts = () => {
      const footprintManager = footprintManagerRef.current;
      if (!footprintManager) return;

      const visible: VisibleProduct[] = [];
      const seen = new Set<string>();

      // Helper to extract center coordinates from feature
      const getFeatureCenter = (feature: FootprintFeature): { lat: number; lon: number } | null => {
        const props = feature.properties || {};
        // HiRISE DTM has explicit bounds in properties
        if (
          props.west != null &&
          props.east != null &&
          props.south != null &&
          props.north != null
        ) {
          return {
            lat: (props.south + props.north) / 2,
            lon: (props.west + props.east) / 2,
          };
        }
        // Try geometry
        const geom = feature.geometry;
        if (!geom) return null;
        const coords = geom.coordinates;
        if (!coords) return null;

        if (geom.type === "Point") {
          return { lon: coords[0], lat: coords[1] };
        }
        if (geom.type === "Polygon" && coords[0]) {
          const ring = coords[0];
          let sumLat = 0;
          let sumLon = 0;
          for (const [lon, lat] of ring) {
            sumLon += lon;
            sumLat += lat;
          }
          return { lat: sumLat / ring.length, lon: sumLon / ring.length };
        }
        if (geom.type === "LineString" && coords.length > 0) {
          const midIdx = Math.floor(coords.length / 2);
          return { lon: coords[midIdx][0], lat: coords[midIdx][1] };
        }
        return null;
      };

      // Helper: check if product passes the overlap filter
      const overlapResult = overlapResultRef.current;
      const passesOverlap = (inst: string, pid: string): boolean => {
        if (!overlapResult) return true; // No filter active
        const passing = overlapResult.get(inst as FPInstrumentType);
        return passing ? passing.has(pid) : true; // If instrument not in filter, pass
      };

      // Get products from FootprintManager for HiRISE
      if (showHiRISE && footprintManager.hasFootprints("HIRISE")) {
        const hiriseFeatures = footprintManager.getFeatures("HIRISE");
        for (const feature of hiriseFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid) && passesOverlap("HIRISE", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({
              productId: pid,
              instrument: "HIRISE",
              title,
              lat: center?.lat,
              lon: center?.lon,
            });
          }
        }
      }

      // Get products from FootprintManager for CRISM (with optional filter)
      if (showCRISM && footprintManager.hasFootprints("CRISM")) {
        const crismFeatures = footprintManager.getFeatures("CRISM");
        for (const feature of crismFeatures) {
          const pid = feature.properties.product_id;
          if (pid && !seen.has(pid)) {
            // Ice score filter
            if (crismFilteredIds !== null) {
              const obsId = extractCrismObsId(pid);
              if (!crismFilteredIds.has(obsId)) continue;
            }
            // Overlap filter
            if (!passesOverlap("CRISM", pid)) continue;
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({
              productId: pid,
              instrument: "CRISM",
              lat: center?.lat,
              lon: center?.lon,
            });
          }
        }
      }

      // Get products from FootprintManager for CTX
      if (showCTX && footprintManager.hasFootprints("CTX")) {
        const ctxFeatures = footprintManager.getFeatures("CTX");
        for (const feature of ctxFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid) && passesOverlap("CTX", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({
              productId: pid,
              instrument: "CTX",
              title,
              lat: center?.lat,
              lon: center?.lon,
            });
          }
        }
      }

      // Get products from FootprintManager for HiRISE DTM
      if (showHiRISEDTM && footprintManager.hasFootprints("HIRISE_DTM")) {
        const dtmFeatures = footprintManager.getFeatures("HIRISE_DTM");
        for (const feature of dtmFeatures) {
          const pid = feature.properties.product_id;
          const title = feature.properties.title;
          if (pid && !seen.has(pid) && passesOverlap("HIRISE_DTM", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({
              productId: pid,
              instrument: "HIRISE_DTM",
              title,
              lat: center?.lat,
              lon: center?.lon,
            });
          }
        }
      }

      // Get products from FootprintManager for CRISM TRR3
      if (showCRISM_TRR3 && footprintManager.hasFootprints("CRISM_TRR3")) {
        const trr3Features = footprintManager.getFeatures("CRISM_TRR3");
        for (const feature of trr3Features) {
          const pid = feature.properties.product_id;
          if (pid && !seen.has(pid) && passesOverlap("CRISM_TRR3", pid)) {
            seen.add(pid);
            const center = getFeatureCenter(feature);
            visible.push({
              productId: pid,
              instrument: "CRISM_TRR3",
              lat: center?.lat,
              lon: center?.lon,
            });
          }
        }
      }

      // Include custom datasets that are loaded and visible
      if (showCustomData) {
        for (const dataset of customDatasets) {
          if (dataset.visible && !seen.has(dataset.id)) {
            seen.add(dataset.id);
            const center = {
              lat: (dataset.bounds.south + dataset.bounds.north) / 2,
              lon: (dataset.bounds.west + dataset.bounds.east) / 2,
            };
            visible.push({
              productId: dataset.id,
              instrument: "CUSTOM",
              title: dataset.name,
              lat: center.lat,
              lon: center.lon,
            });
          }
        }
      }

      // Only update if results changed (avoid unnecessary re-renders)
      const newHash = visible.map((p) => p.productId).join(",");
      if (newHash !== lastResultHash) {
        lastResultHash = newHash;
        onVisibleProductsChange(visible);
      }
    };

    // Update on camera move end (main trigger)
    const removeListener = viewer.camera.moveEnd.addEventListener(updateVisibleProducts);

    // Initial update with delay for FootprintManager initialization
    const initTimeout = setTimeout(updateVisibleProducts, 500);

    // Run once immediately when deps change (e.g., footprint load, filter toggle)
    updateVisibleProducts();

    return () => {
      removeListener();
      clearTimeout(initTimeout);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    showHiRISE,
    showCRISM,
    showCTX,
    showHiRISEDTM,
    showCRISM_TRR3,
    showCustomData,
    customDatasets,
    onVisibleProductsChange,
    crismFilteredIds,
    overlapEnabled,
    footprintVersion,
    extractCrismObsId,
    viewerRef,
  ]);

  // Named region layer overlay
  const regionCacheRef = useRef<
    Array<{
      region_id: string;
      display_name: string;
      lat_min: number;
      lat_max: number;
      lon_min: number;
      lon_max: number;
      tags: string[];
    }> | null
  >(null);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const REGION_PREFIX = "REGION_";

    if (!showRegionLayer) {
      // Remove all region entities
      const toRemove = viewer.entities.values.filter(
        (e) => typeof e.id === "string" && e.id.startsWith(REGION_PREFIX),
      );
      for (const ent of toRemove) viewer.entities.remove(ent);
      if (toRemove.length > 0) viewer.scene.requestRender();
      return;
    }

    // Tag → color mapping
    const TAG_COLORS: Record<string, [number, number, number]> = {
      volcanic: [0.93, 0.35, 0.05], // orange
      shield_volcano: [0.93, 0.35, 0.05],
      ice: [0.0, 0.8, 0.85], // cyan
      polar: [0.0, 0.8, 0.85],
      landing_site: [0.96, 0.75, 0.14], // gold
      canyon: [0.87, 0.25, 0.25], // red
      planitia: [0.3, 0.5, 0.9], // blue
      lowland: [0.3, 0.5, 0.9],
      terra: [0.85, 0.65, 0.3], // amber
      highland: [0.85, 0.65, 0.3],
      crater: [0.2, 0.8, 0.5], // emerald
      impact: [0.2, 0.8, 0.5],
    };

    const getRegionColor = (tags: string[]): Cesium.Color => {
      for (const tag of tags) {
        const c = TAG_COLORS[tag];
        if (c) return new Cesium.Color(c[0], c[1], c[2], 1.0);
      }
      return new Cesium.Color(0.55, 0.65, 0.8, 1.0); // slate default
    };

    const renderRegions = (regions: typeof regionCacheRef.current) => {
      if (!regions || !viewerRef.current) return;
      const v = viewerRef.current;

      v.entities.suspendEvents();
      for (const r of regions) {
        if (v.entities.getById(`${REGION_PREFIX}${r.region_id}`)) continue;

        const color = getRegionColor(r.tags);
        const centerLat = (r.lat_min + r.lat_max) / 2;
        const centerLon = (r.lon_min + r.lon_max) / 2;

        v.entities.add({
          id: `${REGION_PREFIX}${r.region_id}`,
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(
              r.lon_min,
              r.lat_min,
              r.lon_max,
              r.lat_max,
            ),
            material: new Cesium.ColorMaterialProperty(color.withAlpha(0.12)),
            outline: true,
            outlineColor: new Cesium.ConstantProperty(color.withAlpha(0.4)),
            outlineWidth: new Cesium.ConstantProperty(1),
            height: 0,
          },
          position: Cesium.Cartesian3.fromDegrees(centerLon, centerLat),
          label: {
            text: r.display_name,
            font: "11px monospace",
            fillColor: color.withAlpha(0.85),
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.CENTER,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(500000, 1.0, 5000000, 0.4),
            translucencyByDistance: new Cesium.NearFarScalar(1000000, 1.0, 8000000, 0.3),
          },
        });
      }
      v.entities.resumeEvents();
      v.scene.requestRender();
    };

    // Use cached data if available
    if (regionCacheRef.current) {
      renderRegions(regionCacheRef.current);
      return;
    }

    // Fetch regions from API
    fetch("/api/proximity/regions")
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        regionCacheRef.current = data;
        renderRegions(data);
      })
      .catch(() => {});
  }, [showRegionLayer, viewerRef]);

  return {
    footprintManagerRef,
  };
}
