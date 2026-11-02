import { useEffect, useState } from "react";
import { fetchTrr3Pixel } from "../../../api/inspector";
import { crismPixelAt } from "../../../utils/pdsLabels";
import type { InspectorContext } from "../types";

/**
 * Sensor pixel of a CRISM product at the inspected point.
 *
 * The clicked product already carries pixelLine/pixelSample; other products
 * covering the same point are resolved here (TRR3: DDR lookup on the
 * backend; MTRDR: projection-aware label mapping). Null when the product
 * does not cover the point or cannot be resolved reliably.
 */
export default function usePixelAtPoint(product: InspectorContext): { line?: number; sample?: number } {
  const { instrument, productId, lat, lon, pixelLine, pixelSample } = product;
  const [resolved, setResolved] = useState<{ line: number; sample: number } | null>(null);

  useEffect(() => {
    setResolved(null);
    if (pixelLine !== undefined) return;
    const lookup =
      instrument === "CRISM_TRR3" ? fetchTrr3Pixel(productId, lat, lon)
      : instrument === "CRISM" ? crismPixelAt(productId, lat, lon)
      : null;
    if (!lookup) return;
    let cancelled = false;
    lookup.then((px) => {
      if (!cancelled) setResolved(px);
    });
    return () => {
      cancelled = true;
    };
  }, [instrument, productId, pixelLine, lat, lon]);

  return pixelLine !== undefined
    ? { line: pixelLine, sample: pixelSample }
    : { line: resolved?.line, sample: resolved?.sample };
}
