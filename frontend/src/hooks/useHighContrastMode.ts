import { useState, useCallback } from "react";
import * as Cesium from "cesium";
import { getInstrumentCesiumColor } from "../config/instrumentRegistry";

const STORAGE_KEY = "marslab_high_contrast";

/** High-contrast hex palette — vivid colors distinguishable on dark Mars terrain */
const HIGH_CONTRAST_HEX: Record<string, string> = {
  crism: "#ff00ff",
  hirise: "#00ff88",
  ctx: "#ffff00",
  sharad: "#00ffff",
  sharad_highres: "#ff8800",
  hirise_dtm: "#ff4488",
  crism_trr3: "#4488ff",
  custom: "#ffffff",
};

function hexToCesiumColor(hex: string): Cesium.Color {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return new Cesium.Color(r, g, b, 1.0);
}

function loadPreference(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

export interface HighContrastColors {
  /** Returns a high-contrast Cesium Color for the given instrument */
  getColor: (instrument: string) => Cesium.Color;
  /** Whether high contrast mode is active */
  isActive: boolean;
  /** Toggle high contrast mode */
  toggle: () => void;
}

export function useHighContrastMode(): HighContrastColors {
  const [isActive, setIsActive] = useState(loadPreference);

  const toggle = useCallback(() => {
    setIsActive((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(STORAGE_KEY, String(next));
      } catch {
        // storage unavailable — ignore
      }
      return next;
    });
  }, []);

  const getColor = useCallback(
    (instrument: string): Cesium.Color => {
      if (isActive) {
        const hex = HIGH_CONTRAST_HEX[instrument.toLowerCase()];
        if (hex) return hexToCesiumColor(hex);
        // Unknown instrument in high-contrast mode → white
        return hexToCesiumColor("#ffffff");
      }

      // Normal mode — delegate to registry
      const { r, g, b } = getInstrumentCesiumColor(instrument);
      return new Cesium.Color(r, g, b, 1.0);
    },
    [isActive],
  );

  return { getColor, isActive, toggle };
}
