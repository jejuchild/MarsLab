import { useEffect, useRef } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { findEntityByProductId } from "../utils/cesiumEntityUtils";
import { getInstrumentCesiumColor } from "../config/instrumentRegistry";
import type FootprintManager from "../utils/FootprintManager";

type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

// Pre-computed highlight materials
const HILITE_RECT_MATERIAL_HIRISE = new Cesium.ColorMaterialProperty(
  Cesium.Color.YELLOW.withAlpha(0.7)
);
const HILITE_RECT_MATERIAL_CRISM = new Cesium.ColorMaterialProperty(
  Cesium.Color.CYAN.withAlpha(0.6)
);
const HILITE_RECT_MATERIAL_CUSTOM = new Cesium.ColorMaterialProperty(
  Cesium.Color.FUCHSIA.withAlpha(0.3)
);
const HILITE_RECT_MATERIAL_CTX = new Cesium.ColorMaterialProperty(
  Cesium.Color.fromCssColorString("#FF69B4").withAlpha(0.6)
);
const HILITE_RECT_MATERIAL_DTM = new Cesium.ColorMaterialProperty(
  Cesium.Color.fromCssColorString("#d97706").withAlpha(0.6)
);
const HILITE_RECT_MATERIAL_TRR3 = new Cesium.ColorMaterialProperty(
  Cesium.Color.fromCssColorString("#00CED1").withAlpha(0.6)
);

function getHiliteMaterial(inst: string): Cesium.ColorMaterialProperty {
  switch (inst) {
    case "HIRISE": return HILITE_RECT_MATERIAL_HIRISE;
    case "CUSTOM": return HILITE_RECT_MATERIAL_CUSTOM;
    case "CTX": return HILITE_RECT_MATERIAL_CTX;
    case "HIRISE_DTM": return HILITE_RECT_MATERIAL_DTM;
    case "CRISM_TRR3": return HILITE_RECT_MATERIAL_TRR3;
    default: return HILITE_RECT_MATERIAL_CRISM;
  }
}

type UseHoverHighlightParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  footprintManagerRef?: React.MutableRefObject<FootprintManager | null>;
  hoveredProductId: string | null;
  onHoverProduct?: (productId: string | null) => void;
};

export default function useHoverHighlight({
  viewerRef,
  footprintManagerRef,
  hoveredProductId,
  onHoverProduct,
}: UseHoverHighlightParams): void {
  // Store onHoverProduct in ref to access in hover handler
  const onHoverProductRef = useRef(onHoverProduct);
  useEffect(() => {
    onHoverProductRef.current = onHoverProduct;
  }, [onHoverProduct]);

  // Bidirectional highlight: highlight footprint when hovering in ActiveProductsPanel
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Detect instrument by trying _FP_ entity lookup across all instruments
    const detectInstrumentLocal = (pid: string): InstrumentType => {
      const result = findEntityByProductId(viewer, pid);
      if (result) return result.instrument;
      // Fallback heuristics
      if (pid.startsWith("ESP_")) return "HIRISE";
      if (pid.startsWith("DTE")) return "HIRISE_DTM";
      if (/^(frt|hrl|hrs|frs)[0-9a-f]+_\d{2}$/i.test(pid)) return "CRISM_TRR3";
      return "CRISM";
    };

    // Helper to apply/remove highlight to an entity
    const setEntityHighlight = (entity: Cesium.Entity | undefined, highlighted: boolean, instrument: InstrumentType) => {
      if (!entity?.rectangle) return;

      // Check if this product has an active quickview overlay
      const pid = entity.id?.replace(/^[A-Z_]+_FP_/, "") ?? "";
      const hasOverlay = !!viewer.entities.getById(`QUICKVIEW_OVERLAY_${pid}`);

      if (highlighted && !hasOverlay) {
        // Only apply highlight when no quickview overlay is present
        entity.rectangle.material = getHiliteMaterial(instrument);
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.WHITE);
      } else if (hasOverlay) {
        // Quickview active — keep footprint fully transparent
        entity.rectangle.material = new Cesium.ColorMaterialProperty(Cesium.Color.TRANSPARENT);
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(Cesium.Color.TRANSPARENT);
      } else {
        // Restore to original light fill using instrument color
        const rgb = getInstrumentCesiumColor(instrument.toLowerCase());
        const baseColor = new Cesium.Color(rgb.r, rgb.g, rgb.b, 1.0);
        entity.rectangle.material = new Cesium.ColorMaterialProperty(baseColor.withAlpha(0.10));
        entity.rectangle.outlineColor = new Cesium.ConstantProperty(baseColor);
      }
    };

    // Clear previous highlight if any
    if (!hoveredProductId) {
      footprintManagerRef?.current?.hideHoverLabel();
      viewer.scene.requestRender();
      return;
    }

    // Find and highlight the hovered product
    const instrument = detectInstrumentLocal(hoveredProductId);

    // Try FootprintManager entity IDs first (_FP_), then legacy _VP_ IDs
    const entityIds: string[] = [];
    const fpId = `${instrument}_FP_${hoveredProductId}`;
    const fpEntity = viewer.entities.getById(fpId);
    if (fpEntity) {
      setEntityHighlight(fpEntity, true, instrument);
      entityIds.push(fpId);
    } else if (footprintManagerRef?.current?.hasFeature(fpId)) {
      const metadata = footprintManagerRef.current.getFeatureMetadata(fpId);
      if (metadata) {
        const pos = Cesium.Cartesian3.fromDegrees(
          (metadata.bounds.west + metadata.bounds.east) / 2,
          (metadata.bounds.south + metadata.bounds.north) / 2,
          0,
          viewer.scene.globe.ellipsoid,
        );
        footprintManagerRef.current.showHoverLabel(pos, metadata.productId);
      }
    }

    // Fallback to legacy VP IDs if no FP entity found
    if (entityIds.length === 0) {
      const vpPrefix = `${instrument}_VP_${hoveredProductId}`;
      for (const id of [vpPrefix, `${vpPrefix}_1`, `${vpPrefix}_2`, `${vpPrefix}_3`]) {
        const entity = viewer.entities.getById(id);
        if (entity) {
          setEntityHighlight(entity, true, instrument);
          entityIds.push(id);
        }
      }
    }

    // Also highlight label if it exists (try _LBL_ first, then legacy)
    const labelEnt = viewer.entities.getById(`${instrument}_LBL_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_VP_LABEL_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_LABEL_${hoveredProductId}`);
    if (labelEnt) {
      labelEnt.show = true;
      if (labelEnt.label) labelEnt.label.scale = new Cesium.ConstantProperty(1.3);
    }

    const pointEnt = viewer.entities.getById(`${instrument}_VP_POINT_${hoveredProductId}`) ||
                     viewer.entities.getById(`${instrument}_POINT_${hoveredProductId}`);
    if (pointEnt?.point) {
      pointEnt.point.pixelSize = new Cesium.ConstantProperty(10);
    }

    viewer.scene.requestRender();

    // Cleanup function to restore original appearance
    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      for (const id of entityIds) {
        const entity = viewer.entities.getById(id);
        if (entity) {
          setEntityHighlight(entity, false, instrument);
        }
      }

      if (labelEnt) {
        labelEnt.show = false;
        if (labelEnt.label) labelEnt.label.scale = new Cesium.ConstantProperty(1.0);
      }

      if (pointEnt?.point) {
        pointEnt.point.pixelSize = new Cesium.ConstantProperty(6);
      }

      footprintManagerRef?.current?.hideHoverLabel();

      viewer.scene.requestRender();
    };
  }, [hoveredProductId, viewerRef, footprintManagerRef]);
}
