import { useEffect, useRef } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { SWIM_METHODS } from "../api/swim_ice";

type UseMapLayersParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  swimLayer: string | false;
  scienceLayerVisibility: Record<string, boolean>;
  scienceLayerDepth: string;
  scienceLayerOpacities: Record<string, number>;
  accessibilityVisible: boolean;
  accessibilityOpacity: number;
  fusionVisible: boolean;
  fusionOpacity: number;
};

export default function useMapLayers({
  viewerRef,
  swimLayer,
  scienceLayerVisibility,
  scienceLayerDepth,
  scienceLayerOpacities,
  accessibilityVisible,
  accessibilityOpacity,
  fusionVisible,
  fusionOpacity,
}: UseMapLayersParams): void {
  const swimLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const scienceLayerRefs = useRef<Map<string, Cesium.ImageryLayer>>(new Map());
  const scienceLayerDepthRefs = useRef<Map<string, string | null>>(new Map());
  const accessibilityLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const fusionLayerRef = useRef<Cesium.ImageryLayer | null>(null);

  // SWIM real data imagery overlay
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove previous SWIM imagery layer
    if (swimLayerRef.current) {
      viewer.imageryLayers.remove(swimLayerRef.current, false);
      swimLayerRef.current = null;
    }

    // Also clean up old entity-based SWIM (migration cleanup)
    const oldEntities = viewer.entities.values.filter(
      (e: Cesium.Entity) => e.id?.startsWith("SWIM_")
    );
    for (const e of oldEntities) viewer.entities.remove(e);

    if (!swimLayer) {
      viewer.scene.requestRender();
      return;
    }

    // Map layer ID to backend tile URL
    const layerMap: Record<string, string> = {
      "0-1m": "/api/swim/tile/0-1m",
      "1-5m": "/api/swim/tile/1-5m",
      ">5m": "/api/swim/tile/%3E5m",
    };

    const tileUrl = layerMap[swimLayer];
    if (!tileUrl) return;

    // Use async fromUrl (constructor deprecated in Cesium 1.104+)
    let cancelled = false;
    Cesium.SingleTileImageryProvider.fromUrl(tileUrl, {
      rectangle: Cesium.Rectangle.fromDegrees(-180, -60, 180, 60),
    }).then((provider) => {
      if (cancelled || !viewer || viewer.isDestroyed()) return;
      const layer = viewer.imageryLayers.addImageryProvider(provider);
      layer.alpha = 0.75;
      swimLayerRef.current = layer;
      viewer.scene.requestRender();
    }).catch((err) => {
      console.warn("Failed to load SWIM overlay:", err);
    });

    return () => {
      cancelled = true;
      if (!viewer || viewer.isDestroyed()) return;
      if (swimLayerRef.current) {
        viewer.imageryLayers.remove(swimLayerRef.current, false);
        swimLayerRef.current = null;
      }
    };
  }, [swimLayer, viewerRef]);

  // Science layers (neutron, thermal, radar, etc.)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const urlSegments: Record<string, string> = {
      neutron: "neutron",
      thermal: "thermal",
      radar_surface: "radar-surface",
      radar_dielectric: "radar-dielectric",
      geomorphic: "geomorphic",
    };
    const currentRefs = scienceLayerRefs.current;
    const currentDepths = scienceLayerDepthRefs.current;
    let cancelled = false;

    const getDepthParam = (method: string): string | null => {
      if (method === "radar_dielectric") {
        return scienceLayerDepth === "5m-plus" ? "5m-plus" : "1-5m";
      }
      if (method === "geomorphic") {
        return scienceLayerDepth === "0-1m" || scienceLayerDepth === "5m-plus" ? scienceLayerDepth : "1-5m";
      }
      return null;
    };

    for (const method of SWIM_METHODS) {
      const visible = scienceLayerVisibility[method] ?? false;
      const existing = currentRefs.get(method);
      if (!existing) continue;
      const nextDepth = getDepthParam(method);
      const prevDepth = currentDepths.get(method) ?? null;
      if (!visible || nextDepth !== prevDepth) {
        viewer.imageryLayers.remove(existing, false);
        currentRefs.delete(method);
        currentDepths.delete(method);
      }
    }

    for (const method of SWIM_METHODS) {
      const visible = scienceLayerVisibility[method] ?? false;
      const opacity = scienceLayerOpacities[method] ?? 0.7;
      const existing = currentRefs.get(method);
      if (visible && existing) {
        existing.alpha = opacity;
        continue;
      }
      if (!visible || existing) continue;

      const segment = urlSegments[method] ?? method;
      const depthParam = getDepthParam(method);
      const suffix = depthParam ? `?depth=${depthParam}` : "";
      const tileUrl = `/api/swim/method-tile/${segment}${suffix}`;

      Cesium.SingleTileImageryProvider.fromUrl(tileUrl, {
        rectangle: Cesium.Rectangle.fromDegrees(-180, -60, 180, 60),
      }).then((provider) => {
        const v = viewerRef.current;
        if (cancelled || !v || v.isDestroyed()) return;
        if (currentRefs.has(method)) return;
        const layer = v.imageryLayers.addImageryProvider(provider);
        layer.alpha = scienceLayerOpacities[method] ?? 0.7;
        currentRefs.set(method, layer);
        currentDepths.set(method, depthParam);
        v.scene.requestRender();
      }).catch((err) => {
        console.warn(`Failed to load science layer ${method}:`, err);
      });
    }

    viewer.scene.requestRender();

    return () => {
      cancelled = true;
    };
  }, [scienceLayerVisibility, scienceLayerDepth, scienceLayerOpacities, viewerRef]);

  // Science layer cleanup on unmount
  useEffect(() => {
    return () => {
      const viewer = viewerRef.current;
      if (!viewer || viewer.isDestroyed()) return;
      for (const layer of scienceLayerRefs.current.values()) {
        viewer.imageryLayers.remove(layer, false);
      }
      scienceLayerRefs.current.clear();
      scienceLayerDepthRefs.current.clear();
    };
  }, [viewerRef]);

  // Ice Accessibility heatmap layer
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove existing layer
    if (accessibilityLayerRef.current) {
      viewer.imageryLayers.remove(accessibilityLayerRef.current, false);
      accessibilityLayerRef.current = null;
    }

    if (!accessibilityVisible) {
      viewer.scene.requestRender();
      return;
    }

    // Use UrlTemplateImageryProvider for z/x/y dynamic tiles
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: "/api/accessibility/tile/{z}/{x}/{y}.png",
      tilingScheme: new Cesium.GeographicTilingScheme(),
      minimumLevel: 0,
      maximumLevel: 5,
    });
    const layer = viewer.imageryLayers.addImageryProvider(provider);
    layer.alpha = accessibilityOpacity;
    accessibilityLayerRef.current = layer;
    viewer.scene.requestRender();

    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      if (accessibilityLayerRef.current) {
        viewer.imageryLayers.remove(accessibilityLayerRef.current, false);
        accessibilityLayerRef.current = null;
      }
    };
  }, [accessibilityVisible, viewerRef]);

  // Update accessibility layer opacity without recreating
  useEffect(() => {
    if (accessibilityLayerRef.current) {
      accessibilityLayerRef.current.alpha = accessibilityOpacity;
      viewerRef.current?.scene.requestRender();
    }
  }, [accessibilityOpacity, viewerRef]);

  // Ice Prospecting Fusion heatmap layer
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove existing layer
    if (fusionLayerRef.current) {
      viewer.imageryLayers.remove(fusionLayerRef.current, false);
      fusionLayerRef.current = null;
    }

    if (!fusionVisible) {
      viewer.scene.requestRender();
      return;
    }

    // Use UrlTemplateImageryProvider for z/x/y dynamic tiles
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: "/api/accessibility/fusion-tile/{z}/{x}/{y}.png",
      tilingScheme: new Cesium.GeographicTilingScheme(),
      minimumLevel: 0,
      maximumLevel: 5,
    });
    const layer = viewer.imageryLayers.addImageryProvider(provider);
    layer.alpha = fusionOpacity;
    fusionLayerRef.current = layer;
    viewer.scene.requestRender();

    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      if (fusionLayerRef.current) {
        viewer.imageryLayers.remove(fusionLayerRef.current, false);
        fusionLayerRef.current = null;
      }
    };
  }, [fusionVisible, viewerRef]);

  // Update fusion layer opacity without recreating
  useEffect(() => {
    if (fusionLayerRef.current) {
      fusionLayerRef.current.alpha = fusionOpacity;
      viewerRef.current?.scene.requestRender();
    }
  }, [fusionOpacity, viewerRef]);

}
