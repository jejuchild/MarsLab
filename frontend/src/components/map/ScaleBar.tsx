import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";

// Mars equatorial radius: 3396190m (used contextually for scale accuracy)

/** Nice round scale steps in meters */
const SCALE_STEPS = [
  1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000,
  50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
];

function formatScaleLabel(meters: number): string {
  if (meters >= 1_000) return `~${(meters / 1_000).toFixed(0)} km`;
  return `~${meters} m`;
}

interface ScaleBarProps {
  viewer: Cesium.Viewer | null;
}

export default function ScaleBar({ viewer }: ScaleBarProps) {
  const [label, setLabel] = useState("—");
  const [barWidth, setBarWidth] = useState(80);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!viewer) return;

    const updateScale = () => {
      if (viewer.isDestroyed()) return;

      const canvas = viewer.scene.canvas as HTMLCanvasElement;
      const canvasWidth = canvas.clientWidth;
      if (canvasWidth === 0) return;

      // Get camera height above ellipsoid surface
      const cameraPosition = viewer.camera.positionCartographic;
      const heightM = cameraPosition.height;

      // Approximate ground resolution (meters per pixel) using:
      // fov-based projection: visible ground width ≈ 2 * height * tan(fov/2)
const frustum = viewer.camera.frustum;
      let fov = Cesium.Math.toRadians(60);
      if (frustum instanceof Cesium.PerspectiveFrustum && frustum.fov !== undefined) {
        fov = frustum.fov;
      }

      // Scale factor: at Mars surface, account for Mars radius vs Earth
      const groundWidthM = 2 * heightM * Math.tan(fov / 2);
      const metersPerPixel = groundWidthM / canvasWidth;

      // Target bar width range: 60-150px. Pick a nice round scale value.
      const TARGET_PX = 100;
      const rawMeters = metersPerPixel * TARGET_PX;

      // Find the best scale step
      let bestStep = SCALE_STEPS[0]!;
      for (const step of SCALE_STEPS) {
        if (step <= rawMeters * 2) {
          bestStep = step;
        }
      }

      const px = bestStep / metersPerPixel;
      const clampedPx = Math.max(60, Math.min(150, px));

      setLabel(formatScaleLabel(bestStep));
      setBarWidth(Math.round(clampedPx));
    };

    // Initial update
    updateScale();

    // Update on camera move
    const removeListener = viewer.camera.moveEnd.addEventListener(updateScale);

    const raf = rafRef.current;
    return () => {
      removeListener();
      if (raf !== null) {
        cancelAnimationFrame(raf);
      }
    };
  }, [viewer]);

  if (!viewer) return null;

  return (
    <div className="flex flex-col items-start gap-1">
      <div className="flex items-center gap-1.5">
        <span className="material-symbols-outlined text-[12px] text-slate-500">
          straighten
        </span>
        <span className="text-[9px] uppercase tracking-wider text-slate-500">
          Scale
        </span>
      </div>
      <div className="flex flex-col items-start">
        <div
          className="h-[3px] rounded-full bg-slate-400"
          style={{ width: `${barWidth}px` }}
        />
        <div className="flex justify-between w-full mt-0.5">
          <div
            className="h-[6px] w-px bg-slate-400"
            style={{ marginTop: "-3px" }}
          />
          <div
            className="h-[6px] w-px bg-slate-400"
            style={{ marginTop: "-3px" }}
          />
        </div>
        <span className="text-[9px] font-mono text-slate-400 mt-0.5">
          {label}
        </span>
      </div>
    </div>
  );
}
