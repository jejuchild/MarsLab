import { useCallback, useEffect, useRef } from "react";
import toast from "react-hot-toast";
import type React from "react";
import * as Cesium from "cesium";
import type FootprintManager from "../utils/FootprintManager";
import {
  loadDTMElevationGrid,
  type DTMElevationGrid,
} from "../utils/dtmHover";
import type { DTMHoverReadoutHandle } from "../components/DTMHoverReadout";

type BrowseProductType = "HYD" | "ICE" | "IC2";
type ScoreProductType = "score_ice" | "score_hyd";

type RGBWavelengths = {
  r: number;
  g: number;
  b: number;
};

type ProductBounds = {
  west: number;
  south: number;
  east: number;
  north: number;
  lines?: number;
  samples?: number;
};

type UseOverlaysParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  footprintManagerRef: React.MutableRefObject<FootprintManager | null>;
  quickviewOverlays: string[];
  highResOverlays: string[];
  mineralOverlays: string[];
  browseOverlays: Map<string, Set<BrowseProductType>>;
  scoreOverlays: Map<string, Set<ScoreProductType>>;
  overlayOpacities: Map<string, number>;
  rgbWavelengths: RGBWavelengths;
  showHiRISEDTM: boolean;
  showHiRISE: boolean;
  showCRISM: boolean;
  activeDTMProductRef: React.MutableRefObject<string | null>;
  dtmGridCacheRef: React.MutableRefObject<Map<string, DTMElevationGrid>>;
  dtmHoverReadoutRef: React.RefObject<DTMHoverReadoutHandle | null>;
  setDtmGrid: (productId: string, grid: DTMElevationGrid) => void;
  marsEllipsoid: Cesium.Ellipsoid;
  getProductBounds: (productId: string) => Promise<ProductBounds | null>;
  getFootprintBounds: (viewer: Cesium.Viewer, productId: string) => ProductBounds | null;
  loadCRISMLBL: (id: string) => Promise<string | null>;
  parseLBLValue: (block: string | null | undefined, key: string) => number | null;
  normalizeLonTo180: (lon: number) => number;
};

export default function useOverlays({
  viewerRef,
  footprintManagerRef,
  quickviewOverlays,
  highResOverlays,
  mineralOverlays,
  browseOverlays,
  scoreOverlays,
  overlayOpacities,
  rgbWavelengths,
  showHiRISEDTM,
  showHiRISE,
  showCRISM,
  activeDTMProductRef,
  dtmGridCacheRef,
  dtmHoverReadoutRef,
  setDtmGrid,
  marsEllipsoid,
  getProductBounds,
  getFootprintBounds,
  loadCRISMLBL,
  parseLBLValue,
  normalizeLonTo180,
}: UseOverlaysParams): void {
  // Default opacity for overlays
  const DEFAULT_OPACITY = 0.8;

  // Helper to get opacity for a specific product
  const getProductOpacity = useCallback((productId: string): number => {
    return overlayOpacities.get(productId) ?? DEFAULT_OPACITY;
  }, [overlayOpacities]);

  // Helper: make footprint fill transparent when an overlay is active,
  // or restore it when overlay is removed. This prevents the semi-transparent
  // footprint fill from tinting the overlay image.
  const setFootprintTransparent = useCallback((viewer: Cesium.Viewer, productId: string, transparent: boolean) => {
    const footprintManager = footprintManagerRef.current;
    const isHiRISE = productId.startsWith("ESP_");
    const isHiRISEDTM = productId.startsWith("DTE");
    const isTRR3 = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);
    const instrument = isHiRISEDTM
      ? "HIRISE_DTM"
      : isHiRISE
        ? "HIRISE"
        : isTRR3
          ? "CRISM_TRR3"
          : "CRISM";
    // Try FootprintManager ID
    const fpEnt = viewer.entities.getById(`${instrument}_FP_${productId}`);
    if (fpEnt?.rectangle) {
      if (transparent) {
        fpEnt.rectangle.material = new Cesium.ColorMaterialProperty(
          Cesium.Color.TRANSPARENT,
        );
        fpEnt.rectangle.outlineColor = new Cesium.ConstantProperty(
          Cesium.Color.TRANSPARENT,
        );
      } else {
        const color = isHiRISEDTM
          ? Cesium.Color.fromCssColorString("#d97706")
          : isHiRISE
            ? Cesium.Color.YELLOW
            : isTRR3
              ? Cesium.Color.fromCssColorString("#00CED1")
              : Cesium.Color.CYAN;
        fpEnt.rectangle.material = new Cesium.ColorMaterialProperty(
          color.withAlpha(0.10),
        );
        fpEnt.rectangle.outlineColor = new Cesium.ConstantProperty(color);
      }
    } else if (footprintManager?.hasFeature(`${instrument}_FP_${productId}`)) {
      footprintManager.setFeatureVisible(instrument, productId, !transparent);
    }
  }, [footprintManagerRef]);

  const showHiRISEDTMRef = useRef(showHiRISEDTM);
  useEffect(() => {
    showHiRISEDTMRef.current = showHiRISEDTM;
  }, [showHiRISEDTM]);

  // Refs to track current overlays
  const quickviewOverlayIdsRef = useRef<Set<string>>(new Set());
  const quickviewBlobUrlsRef = useRef<Map<string, string>>(new Map());
  const highResOverlayIdsRef = useRef<Set<string>>(new Set());
  const mineralOverlayIdsRef = useRef<Set<string>>(new Set());
  const mineralBlobUrlsRef = useRef<Map<string, string>>(new Map());
  const browseOverlayIdsRef = useRef<Map<string, Set<BrowseProductType>>>(new Map());
  const scoreOverlayIdsRef = useRef<Map<string, Set<ScoreProductType>>>(new Map());

  // Track CTX tile imagery layers for cleanup
  const ctxTileLayersRef = useRef<Map<string, Cesium.ImageryLayer>>(new Map());

  // Guard against duplicate quickview fetches (prevents 429 retry loops)
  const quickviewFetchingRef = useRef<Set<string>>(new Set());

  // Track blob URLs for CRISM RGB images to clean up later
  const crismBlobUrlsRef = useRef<Map<string, string>>(new Map());

  // Track previous RGB wavelengths to detect changes
  const prevRgbRef = useRef<RGBWavelengths>(rgbWavelengths);

  // Hide footprint boxes when high-res overlay is active
  useEffect(() => {
    const viewer = viewerRef.current;
    const footprintManager = footprintManagerRef.current;
    if (!viewer || !footprintManager) return;

    const highResSet = new Set(highResOverlays);

    // Helper to update footprint visibility for FootprintManager entities
    const updateFootprintVisibility = (instrument: "HIRISE" | "CRISM") => {
      const features = footprintManager.getFeatures(instrument);
      const isVisible = instrument === "HIRISE" ? showHiRISE : showCRISM;

      for (const feature of features) {
        const pid = feature.properties.product_id;
        if (!pid) continue;

        // Find the main footprint entity (new ID format)
        const entityId = `${instrument}_FP_${pid}`;
        const entity = viewer.entities.getById(entityId);

        if (entity) {
          if (highResSet.has(pid)) {
            // Hide footprint when high-res is active (so clicks go through to overlay)
            entity.show = false;
          } else {
            entity.show = isVisible;
          }
        } else {
          footprintManager.setFeatureVisible(instrument, pid, highResSet.has(pid) ? false : isVisible);
        }

        // Update label entity (FootprintManager uses _LBL_ for labels)
        const labelId = `${instrument}_LBL_${pid}`;
        const labelEnt = viewer.entities.getById(labelId);

        if (labelEnt) {
          labelEnt.show = highResSet.has(pid) ? false : isVisible;
        }
      }
    };

    // Update HiRISE footprint visibility
    if (footprintManager.hasFootprints("HIRISE") || highResSet.size > 0) {
      updateFootprintVisibility("HIRISE");
    }

    // Update CRISM footprint visibility
    if (footprintManager.hasFootprints("CRISM") || highResSet.size > 0) {
      updateFootprintVisibility("CRISM");
    }

    viewer.scene.requestRender();
  }, [highResOverlays, showHiRISE, showCRISM, viewerRef, footprintManagerRef]);

  // Keep DTM quickview overlays in sync with layer visibility toggle
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    for (const id of quickviewOverlayIdsRef.current) {
      if (!id.startsWith("DTE")) continue;
      const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${id}`);
      if (ent) ent.show = showHiRISEDTM;
    }

    viewer.scene.requestRender();
  }, [showHiRISEDTM, viewerRef]);

  // PERFORMANCE OPTIMIZED: Quickview overlays effect
  // Key optimizations:
  // 1. Toggle visibility instead of add/remove for existing entities
  // 2. Use cached bounds (getProductBounds)
  // 3. Batch entity operations
  // 4. Single requestRender at end
  // 5. Process additions in parallel
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(quickviewOverlays);
    const existingIds = quickviewOverlayIdsRef.current;

    let needsRender = false;

    // Helper: check if a product is CTX by looking up its footprint entity
    const isCTXProduct = (productId: string): boolean => {
      const fpEnt = viewer.entities.getById(`CTX_FP_${productId}`);
      return !!fpEnt || !!footprintManagerRef.current?.getFeatureMetadata(`CTX_FP_${productId}`);
    };

    // Helper: get CTX tile info from footprint entity properties
    const getCTXTileInfo = (productId: string): {
      tileUrl: string;
      west: number;
      south: number;
      east: number;
      north: number;
    } | null => {
      const fpEnt = viewer.entities.getById(`CTX_FP_${productId}`);
      const meta = footprintManagerRef.current?.getFeatureMetadata(`CTX_FP_${productId}`);

      if (!fpEnt?.properties && !meta) return null;

      if (!fpEnt?.properties && meta) {
        const tileUrl = meta.properties.tile_url;
        const west = meta.properties.bbox_west;
        const south = meta.properties.bbox_south;
        const east = meta.properties.bbox_east;
        const north = meta.properties.bbox_north;
        if (
          typeof tileUrl !== "string" ||
          typeof west !== "number" ||
          typeof south !== "number" ||
          typeof east !== "number" ||
          typeof north !== "number"
        ) {
          return null;
        }
        return { tileUrl, west, south, east, north };
      }

      if (!fpEnt?.properties) return null;

      const props = fpEnt.properties as unknown as {
        tile_url?: Cesium.Property;
        bbox_west?: Cesium.Property;
        bbox_south?: Cesium.Property;
        bbox_east?: Cesium.Property;
        bbox_north?: Cesium.Property;
      };
      const now = Cesium.JulianDate.now();
      const tileUrl = props.tile_url?.getValue(now) as string | undefined;
      const west = props.bbox_west?.getValue(now) as number | undefined;
      const south = props.bbox_south?.getValue(now) as number | undefined;
      const east = props.bbox_east?.getValue(now) as number | undefined;
      const north = props.bbox_north?.getValue(now) as number | undefined;
      if (
        !tileUrl ||
        west == null ||
        south == null ||
        east == null ||
        north == null
      ) {
        return null;
      }
      return { tileUrl, west, south, east, north };
    };

    // STEP 1: Hide/remove overlays that are no longer in the list
    const toHide = Array.from(existingIds).filter((id) => !currentIds.has(id));
    for (const id of toHide) {
      // Remove CTX tile layer if it exists
      const ctxLayer = ctxTileLayersRef.current.get(id);
      if (ctxLayer) {
        viewer.imageryLayers.remove(ctxLayer);
        ctxTileLayersRef.current.delete(id);
        needsRender = true;
      }
      // Hide entity overlay (CRISM/HiRISE/HiRISE DTM)
      const ent = viewer.entities.getById(`QUICKVIEW_OVERLAY_${id}`);
      if (ent) {
        ent.show = false;
        needsRender = true;
      }
      // Clean up quickview blob URL
      const qvBlobUrl = quickviewBlobUrlsRef.current.get(id);
      if (qvBlobUrl) {
        URL.revokeObjectURL(qvBlobUrl);
        quickviewBlobUrlsRef.current.delete(id);
      }
      // Clear active DTM if this was a DTM product being hidden
      if (id.startsWith("DTE") && activeDTMProductRef.current === id) {
        activeDTMProductRef.current = null;
        dtmHoverReadoutRef.current?.hide();
      }
      // Restore footprint fill when overlay is removed
      setFootprintTransparent(viewer, id, false);
      existingIds.delete(id);
    }

    // STEP 2: Show existing overlays that are back in the list
    const toCreate: string[] = [];

    for (const productId of quickviewOverlays) {
      if (existingIds.has(productId)) continue; // Already tracked and visible

      // Check for existing CTX tile layer
      const ctxLayer = ctxTileLayersRef.current.get(productId);
      if (ctxLayer) {
        ctxLayer.show = true;
        existingIds.add(productId);
        needsRender = true;
        continue;
      }

      const existingEnt = viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`);
      if (existingEnt) {
        // Entity exists but was hidden - show it (but respect DTM layer toggle)
        const isDTM = productId.startsWith("DTE");
        existingEnt.show = isDTM ? showHiRISEDTM : true;
        setFootprintTransparent(viewer, productId, true);
        existingIds.add(productId);
        needsRender = true;

        // For HiRISE DTM, ensure elevation grid is loaded for hover
        if (isDTM) {
          activeDTMProductRef.current = productId;
          if (!dtmGridCacheRef.current.has(productId)) {
            loadDTMElevationGrid(productId).then((grid) => {
              if (grid) {
                setDtmGrid(productId, grid);
              }
            });
          }
        }
      } else {
        toCreate.push(productId);
      }
    }

    // STEP 3: Create new overlays
    // Skip products already being fetched (prevents duplicate requests / 429 loops)
    const deduped = toCreate.filter((id) => !quickviewFetchingRef.current.has(id));
    // Separate CTX products (tile layers) from CRISM/HiRISE (single images)
    const ctxToCreate = deduped.filter((id) => isCTXProduct(id));
    const imageToCreate = deduped.filter((id) => !isCTXProduct(id));

    // Create CTX tile overlays (synchronous - no network fetch needed)
    for (const productId of ctxToCreate) {
      const info = getCTXTileInfo(productId);
      if (!info) continue;

      const provider = new Cesium.UrlTemplateImageryProvider({
        url: info.tileUrl,
        rectangle: Cesium.Rectangle.fromDegrees(
          info.west,
          info.south,
          info.east,
          info.north,
        ),
        tilingScheme: new Cesium.GeographicTilingScheme({
          ellipsoid: marsEllipsoid,
          numberOfLevelZeroTilesX: 2,
          numberOfLevelZeroTilesY: 1,
        }),
        minimumLevel: 0,
        maximumLevel: 12,
        credit: "NASA/JPL/MSSS - MRO CTX",
      });

      const layer = viewer.imageryLayers.addImageryProvider(provider);
      layer.alpha = getProductOpacity(productId);
      ctxTileLayersRef.current.set(productId, layer);
      quickviewOverlayIdsRef.current.add(productId);
      needsRender = true;
    }

    // Create CRISM/HiRISE/HiRISE DTM image overlays (async)
    if (imageToCreate.length > 0) {
      // Mark all as in-flight to prevent duplicate fetches from re-renders
      for (const id of imageToCreate) quickviewFetchingRef.current.add(id);

      // Pre-fetch bounds in parallel for faster creation
      Promise.all(
        imageToCreate.map(async (productId) => {
          try {
            let bounds = await getProductBounds(productId);
            // Fallback: extract bounds from existing footprint entity
            if (!bounds && viewerRef.current) {
              bounds = getFootprintBounds(viewerRef.current, productId);
            }
            if (!bounds || !viewerRef.current) {
              return null;
            }

            const isHiRISE = productId.startsWith("ESP_") || productId.startsWith("PSP_") || productId.startsWith("TRA_");
            const isHiRISEDTM = productId.startsWith("DTE");
            // TRR3 IDs: {type}{hex}_{nn} (no _mtr3 suffix), e.g. frs0005bad3_01
            const isTRR3 = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);

            // Derive quickview URL
            let imageUrl: string;
            const isCtxMosaic = productId.startsWith("CTX_MOSAIC_");
            let instrument: "HIRISE" | "CRISM" | "HIRISE_DTM" | "CRISM_TRR3" | "CTX_MOSAIC" = "CRISM";

            if (isCtxMosaic) {
              imageUrl = `/api/ctx-mosaic/quickview/${productId}.png`;
              instrument = "CTX_MOSAIC";
            } else if (isHiRISEDTM) {
              imageUrl = `/hirise_dtm/overlay/${productId}.png`;
              instrument = "HIRISE_DTM";
            } else if (isHiRISE) {
              imageUrl = `/hirise/quickview/${productId}.png`;
              instrument = "HIRISE";
            } else if (isTRR3) {
              imageUrl = `/api/mineral-cnn/quickview/${productId}`;
              instrument = "CRISM_TRR3";
            } else {
              // Smart backend endpoint handles all naming patterns
              // Pass obs_id (e.g. frt00008a1e_07) so backend can search crism_quickview/ + crism_data/
              const parts = productId.split("_");
              const base = parts.length >= 2 ? `${parts[0]}_${parts[1]}` : productId;
              imageUrl = `/crism/quickview/${base}.png`;
            }

            // Fetch image first to avoid gray rectangles from 404 URLs
            const imgRes = await fetch(imageUrl);
            if (!imgRes.ok) {
              toast.error(`Quickview not available for ${productId}`);
              return null;
            }
            const blob = await imgRes.blob();
            const blobUrl = URL.createObjectURL(blob);

            return { productId, bounds, imageUrl: blobUrl, instrument };
          } catch (e) {
            console.error("[Quickview] Failed to add overlay:", productId, e);
            toast.error(`Failed to load quickview for ${productId}`);
            return null;
          } finally {
            quickviewFetchingRef.current.delete(productId);
          }
        }),
      ).then((results) => {
        const v = viewerRef.current;
        if (!v) return;

        // Batch entity creation
        v.entities.suspendEvents();

        for (const result of results) {
          if (!result) continue;
          const { productId, bounds, imageUrl, instrument } = result;

          const isDTM = instrument === "HIRISE_DTM";
          v.entities.add({
            id: `QUICKVIEW_OVERLAY_${productId}`,
            show: isDTM ? showHiRISEDTMRef.current : true,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(
                bounds.west,
                bounds.south,
                bounds.east,
                bounds.north,
              ),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument,
              kind: "OVERLAY",
            },
          });

          // Track blob URL for cleanup
          if (imageUrl.startsWith("blob:")) {
            quickviewBlobUrlsRef.current.set(productId, imageUrl);
          }

          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(v, productId, true);
          quickviewOverlayIdsRef.current.add(productId);

          // For HiRISE DTM products, load elevation grid for hover readout
          if (instrument === "HIRISE_DTM") {
            activeDTMProductRef.current = productId;
            loadDTMElevationGrid(productId).then((grid) => {
              if (grid) {
                setDtmGrid(productId, grid);
              }
            });
          }
        }

        v.entities.resumeEvents();
        v.scene.requestRender();
      });
    }

    if (needsRender) {
      viewer.scene.requestRender();
    }
  }, [
    quickviewOverlays,
    showHiRISEDTM,
    marsEllipsoid,
    getProductBounds,
    getFootprintBounds,
    setFootprintTransparent,
    getProductOpacity,
    activeDTMProductRef,
    dtmGridCacheRef,
    dtmHoverReadoutRef,
    setDtmGrid,
    viewerRef,
  ]);

  // PERFORMANCE OPTIMIZED: High-resolution overlays effect
  // Uses visibility toggling and batched operations
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(highResOverlays);
    const existingIds = highResOverlayIdsRef.current;

    let needsRender = false;

    // STEP 1: Hide overlays no longer in list (keep entity for potential re-show)
    const toHide = Array.from(existingIds).filter((id) => !currentIds.has(id));
    for (const id of toHide) {
      const ent = viewer.entities.getById(`HIGHRES_OVERLAY_${id}`);
      if (ent) {
        ent.show = false;
        needsRender = true;
      }
      // Restore footprint fill when overlay is removed
      setFootprintTransparent(viewer, id, false);
      existingIds.delete(id);

      // Clean up CRISM blob URLs to free memory
      const blobUrl = crismBlobUrlsRef.current.get(id);
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        crismBlobUrlsRef.current.delete(id);
        // Also remove the hidden entity for CRISM to force reload with potentially new wavelengths
        if (!id.startsWith("ESP_") && ent) {
          viewer.entities.remove(ent);
        }
      }
    }

    // STEP 2: Show existing or create new overlays
    const toCreate: string[] = [];
    for (const productId of highResOverlays) {
      if (existingIds.has(productId)) continue;

      const existingEnt = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
      if (existingEnt && productId.startsWith("ESP_")) {
        // HiRISE entity exists - just show it
        existingEnt.show = true;
        setFootprintTransparent(viewer, productId, true);
        existingIds.add(productId);
        needsRender = true;
      } else {
        toCreate.push(productId);
      }
    }

    // STEP 3: Create new overlays (async)
    if (toCreate.length > 0) {
      Promise.all(
        toCreate.map(async (productId) => {
          try {
            let bounds = await getProductBounds(productId);
            if (!bounds && viewerRef.current) {
              bounds = getFootprintBounds(viewerRef.current, productId);
            }
            if (!bounds || !viewerRef.current) {
              return null;
            }

            const isHiRISE = productId.startsWith("ESP_") || productId.startsWith("PSP_") || productId.startsWith("TRA_");
            let imageUrl: string;

            if (isHiRISE) {
              imageUrl = `/hirise/overlay/${productId}.png`;
            } else {
              // CRISM / TRR3 RGB request
              const isTRR3 = /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId);
              let rgbUrl: string;
              if (isTRR3) {
                const obsId = productId.replace(/_\d{2}$/, "");
                rgbUrl = `/api/crism-trr3/${obsId}/rgb`;
              } else {
                rgbUrl = `/crism/${productId}/rgb`;
              }

              const response = await fetch(rgbUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  r_um: rgbWavelengths.r,
                  g_um: rgbWavelengths.g,
                  b_um: rgbWavelengths.b,
                  vmin: 0.02,
                  vmax: 0.25,
                }),
              });

              if (!response.ok) {
                toast.error(`High-res overlay not available for ${productId}`);
                return null;
              }

              const blob = await response.blob();
              imageUrl = URL.createObjectURL(blob);
              crismBlobUrlsRef.current.set(productId, imageUrl);
            }

            return { productId, bounds, imageUrl, isHiRISE };
          } catch (e) {
            console.error("[HighRes] Failed to add overlay:", productId, e);
            toast.error(`Failed to load high-res overlay for ${productId}`);
            return null;
          }
        }),
      ).then((results) => {
        const v = viewerRef.current;
        if (!v) return;

        v.entities.suspendEvents();

        for (const result of results) {
          if (!result) continue;
          const { productId, bounds, imageUrl, isHiRISE } = result;

          v.entities.add({
            id: `HIGHRES_OVERLAY_${productId}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(
                bounds.west,
                bounds.south,
                bounds.east,
                bounds.north,
              ),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: isHiRISE
                ? "HIRISE"
                : /^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(productId)
                  ? "CRISM_TRR3"
                  : "CRISM",
              kind: "OVERLAY",
            },
          });

          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(v, productId, true);
          highResOverlayIdsRef.current.add(productId);
        }

        v.entities.resumeEvents();
        v.scene.requestRender();
      });
    }

    if (needsRender) {
      viewer.scene.requestRender();
    }
  }, [
    highResOverlays,
    rgbWavelengths,
    getProductBounds,
    getFootprintBounds,
    setFootprintTransparent,
    getProductOpacity,
    viewerRef,
  ]);

  // Effect to refresh CRISM overlays when RGB wavelengths change
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Check if RGB wavelengths actually changed
    const prev = prevRgbRef.current;
    if (
      prev.r === rgbWavelengths.r &&
      prev.g === rgbWavelengths.g &&
      prev.b === rgbWavelengths.b
    ) {
      return;
    }

    prevRgbRef.current = rgbWavelengths;

    // Find CRISM products in highResOverlays and refresh them
    const crismProducts = highResOverlays.filter((id) => !id.startsWith("ESP_"));

    crismProducts.forEach((productId) => {
      // Remove existing entity
      const ent = viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`);
      if (ent) viewer.entities.remove(ent);

      // Clean up blob URL
      const blobUrl = crismBlobUrlsRef.current.get(productId);
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        crismBlobUrlsRef.current.delete(productId);
      }

      // Remove from tracking so main effect will re-add it
      highResOverlayIdsRef.current.delete(productId);
    });

    viewer.scene.requestRender();
  }, [rgbWavelengths, highResOverlays, viewerRef]);

  // Browse product overlays effect (HYD, ICE, IC2)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const existingOverlays = browseOverlayIdsRef.current;

    // Get all product IDs that should have browse overlays
    const newProductIds = new Set(browseOverlays.keys());
    const existingProductIds = new Set(existingOverlays.keys());

    // Remove overlays for products no longer in the list
    existingProductIds.forEach((productId) => {
      if (!newProductIds.has(productId)) {
        // Remove all browse overlays for this product
        const types = existingOverlays.get(productId);
        types?.forEach((browseType) => {
          const ent = viewer.entities.getById(
            `BROWSE_OVERLAY_${productId}_${browseType}`,
          );
          if (ent) viewer.entities.remove(ent);
        });
        // Restore footprint fill when all browse overlays are removed
        setFootprintTransparent(viewer, productId, false);
        existingOverlays.delete(productId);
      }
    });

    // Update or add overlays for current products
    browseOverlays.forEach(async (types, productId) => {
      const existingTypes = existingOverlays.get(productId) || new Set();

      // Remove types that are no longer active
      existingTypes.forEach((browseType) => {
        if (!types.has(browseType)) {
          const ent = viewer.entities.getById(
            `BROWSE_OVERLAY_${productId}_${browseType}`,
          );
          if (ent) viewer.entities.remove(ent);
        }
      });

      // Add new types
      for (const browseType of types) {
        if (existingTypes.has(browseType)) continue;

        try {
          // Try getProductBounds first, then footprint fallback
          let west: number;
          let east: number;
          let south: number;
          let north: number;
          let bounds = await getProductBounds(productId);
          if (!bounds && viewerRef.current) {
            bounds = getFootprintBounds(viewerRef.current, productId);
          }

          if (bounds) {
            west = bounds.west;
            east = bounds.east;
            south = bounds.south;
            north = bounds.north;
          } else {
            // Last resort: try LBL directly
            const lbl = await loadCRISMLBL(productId);
            if (!lbl) {
              continue;
            }

            const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
            const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
            const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
            const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

            if (
              minLat == null ||
              maxLat == null ||
              westLon360 == null ||
              eastLon360 == null
            ) {
              continue;
            }

            west = normalizeLonTo180(westLon360);
            east = normalizeLonTo180(eastLon360);
            south = Math.min(minLat, maxLat);
            north = Math.max(minLat, maxLat);
          }

          // Construct browse image URL
          // Arcadia products: frt00003156_07_brcarj_mtr3 -> frt00003156_HYD.png
          const baseObsId = productId.split("_")[0];
          const imageUrl = `/crism/browse/${baseObsId}_${browseType}.png`;

          // Pre-validate image exists to avoid gray rectangles
          const headRes = await fetch(imageUrl, { method: "HEAD" });
          if (!headRes.ok) {
            continue;
          }

          if (!viewerRef.current) return;

          viewer.entities.add({
            id: `BROWSE_OVERLAY_${productId}_${browseType}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: "CRISM",
              kind: "BROWSE_OVERLAY",
              browse_type: browseType,
            },
          });
          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(viewer, productId, true);
        } catch (e) {
          console.error("[Browse] Failed to add overlay:", productId, browseType, e);
          toast.error(`Failed to load ${browseType} browse overlay for ${productId}`);
        }
      }
      // Update tracking
      existingOverlays.set(productId, new Set(types));
    });

    viewer.scene.requestRender();
  }, [
    browseOverlays,
    getProductBounds,
    getFootprintBounds,
    loadCRISMLBL,
    parseLBLValue,
    normalizeLonTo180,
    getProductOpacity,
    setFootprintTransparent,
    viewerRef,
  ]);

  // Score product overlays effect (score_ice, score_hyd)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const existingOverlays = scoreOverlayIdsRef.current;

    // Get all product IDs that should have score overlays
    const newProductIds = new Set(scoreOverlays.keys());
    const existingProductIds = new Set(existingOverlays.keys());

    // Remove overlays for products no longer in the list
    existingProductIds.forEach((productId) => {
      if (!newProductIds.has(productId)) {
        // Remove all score overlays for this product
        const types = existingOverlays.get(productId);
        types?.forEach((scoreType) => {
          const ent = viewer.entities.getById(
            `SCORE_OVERLAY_${productId}_${scoreType}`,
          );
          if (ent) viewer.entities.remove(ent);
        });
        // Restore footprint fill when all score overlays are removed
        setFootprintTransparent(viewer, productId, false);
        existingOverlays.delete(productId);
      }
    });

    // Update or add overlays for current products
    scoreOverlays.forEach(async (types, productId) => {
      const existingTypes = existingOverlays.get(productId) || new Set();

      // Remove types that are no longer active
      existingTypes.forEach((scoreType) => {
        if (!types.has(scoreType)) {
          const ent = viewer.entities.getById(
            `SCORE_OVERLAY_${productId}_${scoreType}`,
          );
          if (ent) viewer.entities.remove(ent);
        }
      });

      // Add new types
      for (const scoreType of types) {
        if (existingTypes.has(scoreType)) continue;

        try {
          // Try getProductBounds first, then footprint fallback, then LBL as last resort
          let west: number;
          let east: number;
          let south: number;
          let north: number;
          let bounds = await getProductBounds(productId);
          if (!bounds && viewerRef.current) {
            bounds = getFootprintBounds(viewerRef.current, productId);
          }

          if (bounds) {
            west = bounds.west;
            east = bounds.east;
            south = bounds.south;
            north = bounds.north;
          } else {
            // Last resort: try LBL directly
            const lbl = await loadCRISMLBL(productId);
            if (!lbl) {
              toast.error(`Cannot determine bounds for score overlay: ${productId}`);
              continue;
            }

            const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
            const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
            const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
            const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

            if (
              minLat == null ||
              maxLat == null ||
              westLon360 == null ||
              eastLon360 == null
            ) {
              toast.error(`Incomplete bounds in LBL for ${productId}`);
              continue;
            }

            west = normalizeLonTo180(westLon360);
            east = normalizeLonTo180(eastLon360);
            south = Math.min(minLat, maxLat);
            north = Math.max(minLat, maxLat);
          }

          // Construct score image URL
          // Score files: frt00003156_score_ice.png, frt00003156_score_hyd.png
          const baseObsId = productId.split("_")[0];
          const imageUrl = `/crism/browse/${baseObsId}_${scoreType}.png`;

          // Pre-validate image exists to avoid gray rectangles
          const headRes = await fetch(imageUrl, { method: "HEAD" });
          if (!headRes.ok) {
            continue;
          }

          if (!viewerRef.current) return;

          viewer.entities.add({
            id: `SCORE_OVERLAY_${productId}_${scoreType}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
              material: new Cesium.ImageMaterialProperty({
                image: imageUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: "CRISM",
              kind: "SCORE_OVERLAY",
              score_type: scoreType,
            },
          });
          // Make footprint fill transparent so overlay image is clearly visible
          setFootprintTransparent(viewer, productId, true);
        } catch (e) {
          console.error("[Score] Failed to add overlay:", productId, scoreType, e);
          toast.error(`Failed to load ${scoreType} score overlay for ${productId}`);
        }
      }
      // Update tracking
      existingOverlays.set(productId, new Set(types));
    });

    viewer.scene.requestRender();
  }, [
    scoreOverlays,
    getProductBounds,
    getFootprintBounds,
    loadCRISMLBL,
    parseLBLValue,
    normalizeLonTo180,
    getProductOpacity,
    setFootprintTransparent,
    viewerRef,
  ]);

  // Mineral classification overlays effect (CNN mineral map on TRR3)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const currentIds = new Set(mineralOverlays);
    const existingIds = mineralOverlayIdsRef.current;

    let needsRender = false;

    // STEP 1: Remove overlays no longer in list
    for (const id of Array.from(existingIds)) {
      if (!currentIds.has(id)) {
        const ent = viewer.entities.getById(`MINERAL_OVERLAY_${id}`);
        if (ent) {
          viewer.entities.remove(ent);
          needsRender = true;
        }
        setFootprintTransparent(viewer, id, false);
        existingIds.delete(id);
        // Clean up blob URL
        const blobUrl = mineralBlobUrlsRef.current.get(id);
        if (blobUrl) {
          URL.revokeObjectURL(blobUrl);
          mineralBlobUrlsRef.current.delete(id);
        }
      }
    }

    // STEP 2: Create new overlays
    const toCreate = mineralOverlays.filter((id) => !existingIds.has(id));
    if (toCreate.length > 0) {
      Promise.all(
        toCreate.map(async (productId) => {
          try {
            let bounds = await getProductBounds(productId);
            if (!bounds && viewerRef.current) {
              bounds = getFootprintBounds(viewerRef.current, productId);
            }
            if (!bounds || !viewerRef.current) {
              toast.error(`Cannot determine bounds for mineral overlay: ${productId}`);
              return null;
            }

            const mapUrl = `/api/mineral-cnn/result/${productId}/mineral-map.png`;

            const res = await fetch(mapUrl);
            if (!res.ok) {
              toast.error(`Mineral map not available for ${productId}`);
              return null;
            }

            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);

            return { productId, bounds, blobUrl };
          } catch (e) {
            console.error("[Mineral] Failed to add overlay:", productId, e);
            toast.error(`Failed to load mineral overlay for ${productId}`);
            return null;
          }
        }),
      ).then((results) => {
        const v = viewerRef.current;
        if (!v) return;

        v.entities.suspendEvents();
        for (const result of results) {
          if (!result) continue;
          const { productId, bounds, blobUrl } = result;

          v.entities.add({
            id: `MINERAL_OVERLAY_${productId}`,
            rectangle: {
              coordinates: Cesium.Rectangle.fromDegrees(
                bounds.west,
                bounds.south,
                bounds.east,
                bounds.north,
              ),
              material: new Cesium.ImageMaterialProperty({
                image: blobUrl,
                transparent: true,
                color: Cesium.Color.WHITE.withAlpha(getProductOpacity(productId)),
              }),
              height: 0,
            },
            properties: {
              product_id: productId,
              instrument: "CRISM_TRR3",
              kind: "MINERAL_OVERLAY",
            },
          });

          mineralBlobUrlsRef.current.set(productId, blobUrl);
          setFootprintTransparent(v, productId, true);
          mineralOverlayIdsRef.current.add(productId);
        }
        v.entities.resumeEvents();
        v.scene.requestRender();
      });
    }

    if (needsRender) {
      viewer.scene.requestRender();
    }
  }, [
    mineralOverlays,
    getProductBounds,
    getFootprintBounds,
    getProductOpacity,
    setFootprintTransparent,
    viewerRef,
  ]);

  // PERFORMANCE OPTIMIZED: Update overlay opacity when per-product opacities change
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Update material opacity for a specific product
    const updateMaterial = (ent: Cesium.Entity | undefined, productId: string) => {
      if (!ent?.rectangle?.material) return;
      const material = ent.rectangle.material as Cesium.ImageMaterialProperty;
      if (material.color) {
        const opacity = getProductOpacity(productId);
        material.color = new Cesium.ConstantProperty(
          Cesium.Color.WHITE.withAlpha(opacity),
        );
      }
    };

    // Update quickview overlays (entities)
    for (const productId of quickviewOverlayIdsRef.current) {
      // CTX uses imagery layers, not entities
      const ctxLayer = ctxTileLayersRef.current.get(productId);
      if (ctxLayer) {
        ctxLayer.alpha = getProductOpacity(productId);
        continue;
      }
      updateMaterial(viewer.entities.getById(`QUICKVIEW_OVERLAY_${productId}`), productId);
    }

    // Update high-res overlays
    for (const productId of highResOverlayIdsRef.current) {
      updateMaterial(viewer.entities.getById(`HIGHRES_OVERLAY_${productId}`), productId);
    }

    // Update browse overlays
    for (const [productId, types] of browseOverlayIdsRef.current) {
      for (const browseType of types) {
        updateMaterial(
          viewer.entities.getById(`BROWSE_OVERLAY_${productId}_${browseType}`),
          productId,
        );
      }
    }

    // Update score overlays
    for (const [productId, types] of scoreOverlayIdsRef.current) {
      for (const scoreType of types) {
        updateMaterial(
          viewer.entities.getById(`SCORE_OVERLAY_${productId}_${scoreType}`),
          productId,
        );
      }
    }

    // Update mineral classification overlays
    for (const productId of mineralOverlayIdsRef.current) {
      updateMaterial(viewer.entities.getById(`MINERAL_OVERLAY_${productId}`), productId);
    }

    viewer.scene.requestRender();
  }, [overlayOpacities, getProductOpacity, viewerRef]);
}
