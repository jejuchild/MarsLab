import { useRef, useState } from "react";
import { fetchCRISMSpectrum } from "../../../api/crism";

type Spectrum = {
  wavelength_um: number[];
  reflectance: number[];
  meta: { sample: number; line: number };
};

export function useCRISMAnalysis() {
  const cacheRef = useRef<Map<string, Spectrum>>(new Map());
  const [running, setRunning] = useState(false);
  const [spectrum, setSpectrum] = useState<Spectrum | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(productId: string, lat: number, lon: number) {
    const key = `${productId}:${lat}:${lon}`;
    if (cacheRef.current.has(key)) {
      setSpectrum(cacheRef.current.get(key)!);
      return;
    }

    setRunning(true);
    setError(null);

    try {
      const res = await fetchCRISMSpectrum(productId, lat, lon);
      cacheRef.current.set(key, res);
      setSpectrum(res);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setRunning(false);
    }
  }

  return { running, spectrum, error, runCRISMAnalysis: run };
}
