import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import * as Cesium from "cesium";
import { splitAndDensify } from "../utils/FootprintManager";
import { onSharadHover } from "../utils/sharadLinkBus";

/** Above image overlays (60 m) and footprints (150 m) so it stays visible on HiRISE. */
const TRACK_HEIGHT_M = 400;
const TRACK_ID = "SHARAD_ACTIVE_TRACK";
const HOVER_ID = "SHARAD_HOVER_CURSOR";

/**
 * While a radargram is open: draw its ground track on top of every overlay,
 * and move a cursor along it as the mouse moves over the radargram.
 */
export default function useSharadTrackLink({
  viewerRef,
  marsEllipsoid,
  productId,
}: {
  viewerRef: RefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  productId: string | null;
}) {
  const hoverPos = useRef<Cesium.Cartesian3 | null>(null);

  // Ground track of the open radargram.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !productId) return;
    let cancelled = false;
    const added: Cesium.Entity[] = [];

    fetch(`/api/sharad_highres/track?product_id=${encodeURIComponent(productId)}&downsample=20`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { lats: number[]; lons: number[] } | null) => {
        if (cancelled || !d || viewer.isDestroyed()) return;
        const coords = d.lons.map((lon, i) => [lon, d.lats[i]!] as [number, number]);
        splitAndDensify(coords).forEach((part, i) => {
          added.push(
            viewer.entities.add({
              id: `${TRACK_ID}_${i}`,
              polyline: {
                positions: Cesium.Cartesian3.fromDegreesArrayHeights(
                  part.flatMap(([lon, lat]) => [lon, lat, TRACK_HEIGHT_M]),
                  marsEllipsoid,
                ),
                width: 4,
                material: new Cesium.PolylineOutlineMaterialProperty({
                  color: Cesium.Color.fromCssColorString("#fbbf24"),
                  outlineColor: Cesium.Color.BLACK.withAlpha(0.8),
                  outlineWidth: 1.5,
                }),
              },
            }),
          );
        });
        viewer.scene.requestRender();
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      if (viewer.isDestroyed()) return;
      for (const e of added) viewer.entities.remove(e);
      viewer.scene.requestRender();
    };
  }, [viewerRef, marsEllipsoid, productId]);

  // Cursor that follows the mouse over the radargram.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !productId) return;
    hoverPos.current = null;
    const entity = viewer.entities.add({
      id: HOVER_ID,
      position: new Cesium.CallbackPositionProperty(() => hoverPos.current ?? undefined, false),
      point: {
        pixelSize: 12,
        color: Cesium.Color.fromCssColorString("#22d3ee"),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        show: new Cesium.CallbackProperty(() => hoverPos.current !== null, false),
      },
    });
    const off = onSharadHover((p) => {
      hoverPos.current = p ? Cesium.Cartesian3.fromDegrees(p.lon, p.lat, TRACK_HEIGHT_M, marsEllipsoid) : null;
      if (!viewer.isDestroyed()) viewer.scene.requestRender();
    });
    return () => {
      off();
      if (!viewer.isDestroyed()) {
        viewer.entities.remove(entity);
        viewer.scene.requestRender();
      }
    };
  }, [viewerRef, marsEllipsoid, productId]);
}
