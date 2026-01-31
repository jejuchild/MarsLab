import { useRef, useState } from "react";
import { fetchHiRISEWindow } from "../../../api/hirise";
import type { HiRISEWindowResponse } from "../../../types/hirise";

export function useHiRISEAnalysis() {
  const cacheRef = useRef<Map<string, HiRISEWindowResponse>>(new Map());

  const [running, setRunning] = useState(false);
  const [hiriseData, setHiriseData] =
    useState<HiRISEWindowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runHiRISEAnalysis(
    productId: string,
    pin: { x: number; y: number } | null,
    halfSize = 32
  ) {
    if (!pin) {
      setError("No pinned pixel.");
      return;
    }

    const { x, y } = pin;
    const key = `${productId}:${x}:${y}:${halfSize}`;

    if (cacheRef.current.has(key)) {
      setHiriseData(cacheRef.current.get(key)!);
      return;
    }

    setRunning(true);
    setError(null);
    setHiriseData(null);

    try {
      const res = await fetchHiRISEWindow(productId, x, y, halfSize);
      cacheRef.current.set(key, res);
      setHiriseData(res);
      console.log("[HiRISE response]", res);
        console.log("[Histogram keys]", res.histogram);

    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setRunning(false);
    }
  }

  return {
    running,
    error,
    hiriseData,
    runHiRISEAnalysis,
  };
}
