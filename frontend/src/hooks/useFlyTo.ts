import { useEffect } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { findEntityByProductId } from "../utils/cesiumEntityUtils";

type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

type InspectorContext = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  pixelLine?: number;
  pixelSample?: number;
  title?: string;
};

type UseFlyToParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  flyToProductId: string | null;
  onFlyToComplete?: () => void;
  flyToCoords: { lat: number; lon: number } | null;
  onFlyToCoordsComplete?: () => void;
  bringToFrontId: string | null;
  onBringToFrontComplete?: () => void;
  highlightProductId: string | null;
  onHighlightComplete?: () => void;
  onSelect: (ctx: InspectorContext | null) => void;
  onToggleOverlay?: (productId: string, type: "quickview" | null) => void;
  paddedRectangle: (rect: Cesium.Rectangle, padRatio?: number) => Cesium.Rectangle;
  normalizeLonTo180: (lon: number) => number;
  parseLBLValue: (block: string | null | undefined, key: string) => number | null;
  loadHiRISELBL: (id: string) => Promise<string | null>;
  loadCRISMLBL: (id: string) => Promise<string | null>;
};

export default function useFlyTo({
  viewerRef,
  marsEllipsoid,
  flyToProductId,
  onFlyToComplete,
  flyToCoords,
  onFlyToCoordsComplete,
  bringToFrontId,
  onBringToFrontComplete,
  highlightProductId,
  onHighlightComplete,
  onSelect,
  onToggleOverlay,
  paddedRectangle,
  normalizeLonTo180,
  parseLBLValue,
  loadHiRISELBL,
  loadCRISMLBL,
}: UseFlyToParams): void {

  // Fly to product when flyToProductId changes
  useEffect(() => {
    if (!flyToProductId) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    async function flyTo() {
      const v = viewerRef.current;
      if (!v) return;

      const pid = flyToProductId!;

      // Try to find entity from FootprintManager (works for all instruments)
      const result = findEntityByProductId(v, pid);
      const foundEntity = result?.entity ?? null;

      // If found entity with rectangle, fly to its bounds
      if (foundEntity?.rectangle?.coordinates) {
        const rectCoords = foundEntity.rectangle.coordinates.getValue(Cesium.JulianDate.now());
        if (rectCoords) {
          const padded = paddedRectangle(rectCoords, 0.3);
          v.camera.flyTo({
            destination: padded,
            duration: 0.8,
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // If found entity with position (point), fly above it
      if (foundEntity?.position) {
        const pos = foundEntity.position.getValue(Cesium.JulianDate.now());
        if (pos) {
          const carto = Cesium.Cartographic.fromCartesian(pos, marsEllipsoid);
          v.camera.flyTo({
            destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, 50000, marsEllipsoid),
            orientation: {
              heading: 0,
              pitch: Cesium.Math.toRadians(-90),
              roll: 0,
            },
            duration: 0.8,
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // If found entity with polyline, fly above its center
      if (foundEntity?.polyline?.positions) {
        const positions = foundEntity.polyline.positions.getValue(Cesium.JulianDate.now());
        if (positions && positions.length > 0) {
          const midIdx = Math.floor(positions.length / 2);
          const carto = Cesium.Cartographic.fromCartesian(positions[midIdx], marsEllipsoid);
          v.camera.flyTo({
            destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, 80000, marsEllipsoid),
            orientation: {
              heading: 0,
              pitch: Cesium.Math.toRadians(-90),
              roll: 0,
            },
            duration: 0.8,
            complete: () => onFlyToComplete?.(),
          });
          return;
        }
      }

      // Fallback: try LBL-based fly-to for HiRISE/CRISM
      const isHiRISE = pid.startsWith("ESP_") || pid.startsWith("PSP_");
      const isCRISM = pid.toLowerCase().match(/^(frt|hrl|hrs|frs)/);

      if (isHiRISE || isCRISM) {
        const lbl = isHiRISE
          ? await loadHiRISELBL(pid)
          : await loadCRISMLBL(pid);

        if (!lbl) {
          onFlyToComplete?.();
          return;
        }

        const minLat = parseLBLValue(lbl, "MINIMUM_LATITUDE");
        const maxLat = parseLBLValue(lbl, "MAXIMUM_LATITUDE");
        const westLon360 = parseLBLValue(lbl, "WESTERNMOST_LONGITUDE");
        const eastLon360 = parseLBLValue(lbl, "EASTERNMOST_LONGITUDE");

        if (minLat == null || maxLat == null || westLon360 == null || eastLon360 == null) {
          onFlyToComplete?.();
          return;
        }

        const west = normalizeLonTo180(westLon360);
        const east = normalizeLonTo180(eastLon360);
        const south = Math.min(minLat, maxLat);
        const north = Math.max(minLat, maxLat);

        const rect = Cesium.Rectangle.fromDegrees(west, south, east, north);
        const padded = paddedRectangle(rect, 0.3);

        const v2 = viewerRef.current;
        if (!v2) return;

        v2.camera.flyTo({
          destination: padded,
          duration: 0.8,
          complete: () => onFlyToComplete?.(),
        });
        return;
      }

      // Fallback: fetch coordinates from backend footprints API for any instrument
      const fallbackInstruments = ["SHARAD_HIGHRES", "SHARAD", "HIRISE_DTM", "CTX", "HIRISE", "CRISM"];
      for (const inst of fallbackInstruments) {
        try {
          const res = await fetch(`/api/footprints?instrument=${inst}&bbox=-180,-90,180,90&limit=5000&lod=poly`);
          if (!res.ok) continue;
          const data = await res.json();
          const feat = data.features?.find((f: { properties?: { product_id?: string }; geometry?: { type: string; coordinates: number[][] | number[][][] } }) => f.properties?.product_id === pid);
          if (!feat?.geometry?.coordinates) continue;

          const coords = feat.geometry.coordinates;
          if (feat.geometry.type === "LineString" && coords.length >= 2) {
            const rect = Cesium.Rectangle.fromDegrees(
              Math.min(...coords.map((c: number[]) => c[0])),
              Math.min(...coords.map((c: number[]) => c[1])),
              Math.max(...coords.map((c: number[]) => c[0])),
              Math.max(...coords.map((c: number[]) => c[1]))
            );
            v.camera.flyTo({
              destination: paddedRectangle(rect, 0.3),
              duration: 0.8,
              complete: () => onFlyToComplete?.(),
            });
            return;
          } else if (feat.geometry.type === "Polygon" && coords[0]?.length >= 4) {
            const ring = coords[0];
            const rect = Cesium.Rectangle.fromDegrees(
              Math.min(...ring.map((c: number[]) => c[0])),
              Math.min(...ring.map((c: number[]) => c[1])),
              Math.max(...ring.map((c: number[]) => c[0])),
              Math.max(...ring.map((c: number[]) => c[1]))
            );
            v.camera.flyTo({
              destination: paddedRectangle(rect, 0.3),
              duration: 0.8,
              complete: () => onFlyToComplete?.(),
            });
            return;
          }
        } catch {
          // Try next instrument
        }
      }

      onFlyToComplete?.();
    }

    flyTo();
  }, [flyToProductId, onFlyToComplete, viewerRef, marsEllipsoid, paddedRectangle, normalizeLonTo180, parseLBLValue, loadHiRISELBL, loadCRISMLBL]);

  // Fly to lat/lon coordinates (for search results not on map)
  useEffect(() => {
    if (!flyToCoords) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    const { lat, lon } = flyToCoords;
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, 50000, marsEllipsoid),
      duration: 1.0,
    });

    onFlyToCoordsComplete?.();
  }, [flyToCoords, onFlyToCoordsComplete, viewerRef, marsEllipsoid]);

  // Temporarily highlight a product after fly-to (deep-link from DataDownloadPage)
  // Uses retry logic because footprints may still be loading when this fires.
  useEffect(() => {
    if (!highlightProductId) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    const pid = highlightProductId;
    let cancelled = false;
    let cleanupTimer: ReturnType<typeof setTimeout> | null = null;

    function tryHighlight(attemptsLeft: number) {
      if (cancelled) return;
      const v = viewerRef.current;
      if (!v) return;

      const result = findEntityByProductId(v, pid);
      const entity = result?.entity ?? null;
      const foundInst = result?.instrument ?? "";

      if (!entity) {
        if (attemptsLeft > 0) {
          // Footprints still loading — retry after a short delay
          cleanupTimer = setTimeout(() => tryHighlight(attemptsLeft - 1), 400);
        } else {
          onHighlightComplete?.();
        }
        return;
      }

      // Auto-select the product so the Inspector opens (deep-link UX)
      if (foundInst) {
        let selectLat = 0;
        let selectLon = 0;
        if (entity.rectangle?.coordinates) {
          const rect = entity.rectangle.coordinates.getValue(Cesium.JulianDate.now());
          if (rect) {
            selectLat = Cesium.Math.toDegrees((rect.south + rect.north) / 2);
            selectLon = Cesium.Math.toDegrees((rect.west + rect.east) / 2);
          }
        } else if (entity.position) {
          const pos = entity.position.getValue(Cesium.JulianDate.now());
          if (pos) {
            const carto = Cesium.Cartographic.fromCartesian(pos, marsEllipsoid);
            selectLat = Cesium.Math.toDegrees(carto.latitude);
            selectLon = Cesium.Math.toDegrees(carto.longitude);
          }
        }
        const title = entity.properties?.title?.getValue?.() as string | undefined;
        onSelect({
          instrument: foundInst as InspectorContext["instrument"],
          productId: pid,
          lat: selectLat,
          lon: selectLon,
          title,
        });

        // Auto-activate quickview overlay for deep-link products
        onToggleOverlay?.(pid, "quickview");
      }

      // Save original material
      const origMaterial = entity.rectangle?.material;
      const origOutline = entity.rectangle?.outlineColor;
      const origOutlineWidth = entity.rectangle?.outlineWidth;

      // Apply bright highlight
      if (entity.rectangle) {
        entity.rectangle.material = new Cesium.ColorMaterialProperty(
          Cesium.Color.MAGENTA.withAlpha(0.7)
        );
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE);
        entity.rectangle.outlineWidth = new Cesium.ConstantProperty(3);
      }
      // Also handle polyline entities (SHARAD)
      if (entity.polyline) {
        const origPolyMaterial = entity.polyline.material;
        const origPolyWidth = entity.polyline.width;
        entity.polyline.material = new Cesium.ColorMaterialProperty(Cesium.Color.MAGENTA);
        entity.polyline.width = new Cesium.ConstantProperty(5);

        v.scene.requestRender();

        cleanupTimer = setTimeout(() => {
          if (entity?.polyline) {
            entity.polyline.material = origPolyMaterial;
            entity.polyline.width = origPolyWidth;
          }
          v.scene.requestRender();
          onHighlightComplete?.();
        }, 3000);
        return;
      }

      v.scene.requestRender();

      // Restore after 3 seconds
      cleanupTimer = setTimeout(() => {
        if (entity?.rectangle) {
          if (origMaterial) entity.rectangle.material = origMaterial;
          if (origOutline) entity.rectangle.outlineColor = origOutline;
          if (origOutlineWidth) entity.rectangle.outlineWidth = origOutlineWidth;
        }
        v.scene.requestRender();
        onHighlightComplete?.();
      }, 3000);
    }

    tryHighlight(8); // Up to 8 retries × 400ms = 3.2s wait for footprints

    return () => {
      cancelled = true;
      if (cleanupTimer) clearTimeout(cleanupTimer);
    };
  }, [highlightProductId, onHighlightComplete, onSelect, onToggleOverlay, viewerRef, marsEllipsoid]);

  // Bring high-res overlay to front when bringToFrontId changes
  useEffect(() => {
    if (!bringToFrontId) return;
    const viewer = viewerRef.current;
    if (!viewer) return;

    const entityId = `HIGHRES_OVERLAY_${bringToFrontId}`;
    const entity = viewer.entities.getById(entityId);

    if (entity) {
      // Remove and re-add to bring to front
      const savedProps = {
        id: entity.id,
        rectangle: entity.rectangle,
        properties: entity.properties,
      };

      viewer.entities.remove(entity);
      viewer.entities.add({
        id: savedProps.id,
        rectangle: savedProps.rectangle,
        properties: savedProps.properties,
      });

      viewer.scene.requestRender();
    }

    onBringToFrontComplete?.();
  }, [bringToFrontId, onBringToFrontComplete, viewerRef]);
}
