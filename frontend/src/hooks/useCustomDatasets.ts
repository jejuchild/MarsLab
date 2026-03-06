import { useEffect } from "react";
import type React from "react";
import * as Cesium from "cesium";

type UseCustomDatasetsParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  showCustomData: boolean;
  customDatasets: Array<{
    id: string;
    name: string;
    bounds: { west: number; south: number; east: number; north: number };
    visible: boolean;
    opacity: number;
  }>;
};

export default function useCustomDatasets({
  viewerRef,
  showCustomData,
  customDatasets,
}: UseCustomDatasetsParams): void {

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
                height: 0,
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
  }, [showCustomData, customDatasets, viewerRef]);
}
