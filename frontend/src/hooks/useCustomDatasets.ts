import { useEffect, useRef, useState } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { decodeImage, renderUntilSettled, OVERLAY_HEIGHT_M } from "../utils/overlayImage";

type UseCustomDatasetsParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  showCustomData: boolean;
  customDatasets: Array<{
    id: string;
    name: string;
    bounds: { west: number; south: number; east: number; north: number };
    visible: boolean;
    opacity: number; // 0-100
  }>;
};

export default function useCustomDatasets({
  viewerRef,
  showCustomData,
  customDatasets,
}: UseCustomDatasetsParams): void {
  // Decoded overlay images per dataset id. The uploaded raster used to be
  // drawn only as an outline + label, so the opacity slider had no effect.
  const imagesRef = useRef<Map<string, HTMLImageElement | "loading" | "missing">>(new Map());
  const [imagesVersion, setImagesVersion] = useState(0);

  useEffect(() => {
    for (const dataset of customDatasets) {
      if (imagesRef.current.has(dataset.id)) continue;
      imagesRef.current.set(dataset.id, "loading");
      decodeImage(`/api/custom/${encodeURIComponent(dataset.id)}/overlay.png`).then((img) => {
        imagesRef.current.set(dataset.id, img ?? "missing");
        setImagesVersion((v) => v + 1);
      });
    }
  }, [customDatasets]);

  // Custom dataset overlay rendering
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Track which custom entity IDs should exist
    const desiredIds = new Set<string>();

    viewer.entities.suspendEvents();
    try {
      for (const dataset of customDatasets) {
        const outlineId = `CUSTOM_FP_${dataset.id}`;
        desiredIds.add(outlineId);

        if (showCustomData && dataset.visible) {
          const rect = Cesium.Rectangle.fromDegrees(
            dataset.bounds.west,
            dataset.bounds.south,
            dataset.bounds.east,
            dataset.bounds.north
          );

          // Footprint rectangle: fully transparent fill (alpha=0) so hover pick works,
          // outline only visible. Highlight fill shown on hover via mouse handler.
          const outlineEnt = viewer.entities.getById(outlineId);
          if (!outlineEnt) {
            viewer.entities.add({
              id: outlineId,
              rectangle: {
                coordinates: rect,
                material: Cesium.Color.FUCHSIA.withAlpha(0.0),
                outline: true,
                outlineColor: Cesium.Color.FUCHSIA,
                outlineWidth: 2,
                height: OVERLAY_HEIGHT_M,
              },
              properties: {
                product_id: dataset.id,
                instrument: "CUSTOM",
                kind: "FOOTPRINT",
                dataset_name: dataset.name,
              },
            });
          } else if (outlineEnt.rectangle) {
            // Force transparent fill on existing entity
            outlineEnt.rectangle.material = new Cesium.ColorMaterialProperty(
              Cesium.Color.FUCHSIA.withAlpha(0.0)
            );
          }

          // Raster image overlay (respects the per-dataset opacity slider)
          const overlayId = `CUSTOM_OVERLAY_${dataset.id}`;
          const image = imagesRef.current.get(dataset.id);
          if (image instanceof HTMLImageElement) {
            desiredIds.add(overlayId);
            const color = Cesium.Color.WHITE.withAlpha(Math.max(0, Math.min(100, dataset.opacity)) / 100);
            const overlayEnt = viewer.entities.getById(overlayId);
            if (!overlayEnt) {
              viewer.entities.add({
                id: overlayId,
                rectangle: {
                  coordinates: rect,
                  material: new Cesium.ImageMaterialProperty({ image, transparent: true, color }),
                  height: OVERLAY_HEIGHT_M,
                },
                properties: { product_id: dataset.id, instrument: "CUSTOM", kind: "OVERLAY" },
              });
              renderUntilSettled(viewer);
            } else {
              const material = overlayEnt.rectangle?.material as Cesium.ImageMaterialProperty | undefined;
              const current = material?.color?.getValue(Cesium.JulianDate.now()) as Cesium.Color | undefined;
              if (material && (!current || Math.abs(current.alpha - color.alpha) > 1e-3)) {
                material.color = new Cesium.ConstantProperty(color);
              }
            }
          }

          // Add or update label entity
          const labelId = `CUSTOM_LABEL_${dataset.id}`;
          desiredIds.add(labelId);
          const labelEnt = viewer.entities.getById(labelId);
          if (!labelEnt) {
            const centerLon = (dataset.bounds.west + dataset.bounds.east) / 2;
            const centerLat = (dataset.bounds.south + dataset.bounds.north) / 2;
            const carto = Cesium.Cartographic.fromDegrees(centerLon, centerLat, 100);
            const pos = viewer.scene.globe.ellipsoid.cartographicToCartesian(carto);

            viewer.entities.add({
              id: labelId,
              position: pos,
              label: {
                text: dataset.name,
                font: "12px sans-serif",
                fillColor: Cesium.Color.FUCHSIA,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 3,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              },
              properties: {
                product_id: dataset.id,
                instrument: "CUSTOM",
                kind: "LABEL",
              },
            });
          }
        } else {
          // Not visible - remove if exists
          const outlineEnt = viewer.entities.getById(outlineId);
          if (outlineEnt) viewer.entities.remove(outlineEnt);
          const labelEnt = viewer.entities.getById(`CUSTOM_LABEL_${dataset.id}`);
          if (labelEnt) viewer.entities.remove(labelEnt);
          const overlayEnt = viewer.entities.getById(`CUSTOM_OVERLAY_${dataset.id}`);
          if (overlayEnt) viewer.entities.remove(overlayEnt);
        }
      }

      // Remove entities for datasets that no longer exist
      const toRemove: Cesium.Entity[] = [];
      for (let i = 0; i < viewer.entities.values.length; i++) {
        const ent = viewer.entities.values[i]!;
        if (ent.id.startsWith("CUSTOM_OVERLAY_") || ent.id.startsWith("CUSTOM_FP_") || ent.id.startsWith("CUSTOM_LABEL_")) {
          if (!desiredIds.has(ent.id)) {
            toRemove.push(ent);
          }
        }
      }
      for (const ent of toRemove) {
        viewer.entities.remove(ent);
      }
    } finally {
      viewer.entities.resumeEvents();
    }

    viewer.scene.requestRender();
  }, [showCustomData, customDatasets, viewerRef, imagesVersion]);
}
