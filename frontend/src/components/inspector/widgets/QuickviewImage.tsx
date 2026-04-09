import { useState, useEffect } from "react";
import type { InstrumentType } from "../types";

type QuickviewImageProps = {
  productId: string;
  instrument: InstrumentType;
};

function buildCrismUrls(productId: string): string[] {
  const obsId = productId.replace(/_\d{2}$/, "");
  const baseKey = productId
    .replace(/_if\w+$/i, "")
    .replace(/_br\w+$/i, "");

  return [
    `/crism/quickview/${baseKey}.png`,
    `/crism/quickview/${obsId}_VNIR.png`,
    `/crism/browse/${baseKey}_brvnaj_mtr3.png`,
    `/crism/quickview/${baseKey}_brvnaj_mtr3.png`,
  ];
}

function getImageUrl(productId: string, instrument: InstrumentType): string | string[] {
  switch (instrument) {
    case "HIRISE":
      // Try JPG (static) first, then transparent PNG (generated endpoint)
      return [
        `/hirise/quickview/${productId}.jpg`,
        `/hirise/quickview/${productId}.png`,
      ];
    case "HIRISE_DTM":
      return `/hirise_dtm/overlay/${productId}.png`;
    case "CRISM_TRR3": {
      const obsId = productId.replace(/_\d{2}$/, "");
      return `/api/mineral-cnn/quickview/${obsId}`;
    }
    case "CRISM":
      return buildCrismUrls(productId);
    default:
      return `/quickview/${productId}.jpg`;
  }
}

export default function QuickviewImage({ productId, instrument }: QuickviewImageProps) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");
  const [currentUrl, setCurrentUrl] = useState("");

  useEffect(() => {
    setStatus("loading");
    setCurrentUrl("");

    function tryLoadImage(urls: string[], index: number) {
      if (index >= urls.length) {
        setStatus("error");
        return;
      }

      const img = new Image();
      img.onload = () => {
        setCurrentUrl(urls[index] ?? "");
        setStatus("loaded");
      };
      img.onerror = () => {
        tryLoadImage(urls, index + 1);
      };
      img.src = urls[index] ?? "";
    }

    const urlOrUrls = getImageUrl(productId, instrument);

    if (Array.isArray(urlOrUrls)) {
      tryLoadImage(urlOrUrls, 0);
    } else {
      const img = new Image();
      img.onload = () => {
        setCurrentUrl(urlOrUrls);
        setStatus("loaded");
      };
      img.onerror = () => {
        setStatus("error");
      };
      img.src = urlOrUrls;
    }
  }, [productId, instrument]);

  if (status === "loading") {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-border-dark bg-surface-dark">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-600 border-t-blue-400" />
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex h-32 flex-col items-center justify-center gap-1 rounded-lg border border-border-dark bg-surface-dark">
        <span className="material-symbols-outlined text-2xl text-slate-500">
          satellite_alt
        </span>
        <span className="text-xs text-slate-500">No quickview available</span>
      </div>
    );
  }

  return (
    <img
      src={currentUrl}
      alt={`Quickview for ${productId}`}
      className="w-full rounded-lg border border-border-dark object-cover"
    />
  );
}
