import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import type { InstrumentType, SpectrumData, DustAssessment } from "../types";

/**
 * Fetches CRISM spectrum data for a given pixel coordinate.
 * Extracted from Inspector.tsx for reuse and cleaner separation.
 */
export default function useSpectrumData(
  productId: string | null,
  instrument: InstrumentType | null,
  pixelLine: number | undefined,
  pixelSample: number | undefined,
  lat: number,
  lon: number,
): {
  spectrumData: SpectrumData | null;
  dustAssessment: DustAssessment | null;
  loading: boolean;
} {
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [dustAssessment, setDustAssessment] = useState<DustAssessment | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!productId || (instrument !== "CRISM" && instrument !== "CRISM_TRR3")) {
      setSpectrumData(null);
      setDustAssessment(null);
      return;
    }

    if (pixelLine === undefined || pixelSample === undefined) {
      setSpectrumData(null);
      setDustAssessment(null);
      return;
    }

    let cancelled = false;
    const isTRR3 = instrument === "CRISM_TRR3";

    async function fetchSpectrum() {
      setLoading(true);
      try {
        let url: string;
        if (isTRR3) {
          const obsId = productId!.replace(/_\d{2}$/, "");
          url = `/api/crism-trr3/${obsId}/spectrum`;
        } else {
          url = `/crism/${productId}/spectrum`;
        }

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ line: pixelLine, sample: pixelSample, lat, lon }),
        });

        if (!response.ok) {
          // 404 = spectral data not available for this product (expected)
          if (response.status === 404) {
            setSpectrumData(null);
            setDustAssessment(null);
            return;
          }
          throw new Error(`Failed to fetch spectrum: ${response.status}`);
        }

        const data = await response.json();
        if (cancelled) return;

        setSpectrumData({
          wavelengths: data.wavelengths,
          reflectance: data.reflectance,
          validBands: data.valid_bands,
        });
        setDustAssessment(data.dust_assessment ?? null);
      } catch (e) {
        if (cancelled) return;
        console.error("Failed to fetch spectrum:", e);
        toast.error("Failed to load spectrum data");
        setSpectrumData(null);
        setDustAssessment(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchSpectrum();
    return () => { cancelled = true; };
  }, [productId, instrument, pixelLine, pixelSample, lat, lon]);

  return { spectrumData, dustAssessment, loading };
}
