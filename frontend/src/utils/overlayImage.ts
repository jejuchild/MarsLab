import * as Cesium from "cesium";

/**
 * Load and decode an overlay image before handing it to Cesium.
 * Doubles as an existence check: resolves null when the image is missing
 * or undecodable, so no grey placeholder entity is ever created for it.
 */
export async function decodeImage(url: string): Promise<HTMLImageElement | null> {
  const img = new Image();
  img.src = url;
  try {
    await img.decode();
    return img;
  } catch {
    return null;
  }
}

/**
 * Cesium prepares entity geometry and material textures asynchronously and,
 * in requestRenderMode, does not request a frame when they become ready — the
 * overlay showed as a grey placeholder until the user moved the map. Request a
 * few frames over the next ~10 s (slow GPUs / big images) so newly added
 * overlays actually appear.
 */
export function renderUntilSettled(viewer: Cesium.Viewer): void {
  for (const ms of [0, 100, 300, 700, 1500, 3000, 6000, 10000]) {
    setTimeout(() => {
      if (!viewer.isDestroyed()) viewer.scene.requestRender();
    }, ms);
  }
}

/**
 * Image overlays sit a little above the ellipsoid (and below footprint
 * outlines at 150 m) so they don't z-fight with the globe in 3D.
 */
export const OVERLAY_HEIGHT_M = 60;
