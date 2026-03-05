import { useEffect, useState } from "react";
import type { InstrumentType, ProductMetadata } from "../types";

/**
 * Fetches enriched metadata for a product from GeoJSON index files.
 * These are the same index files used by FootprintManager,
 * so this is essentially a cache-first lookup with no extra network cost
 * after initial index load.
 */

type GeoJsonFeature = {
  properties?: Record<string, unknown>;
  geometry?: { coordinates?: number[][][] };
};

type GeoJsonIndex = {
  features?: GeoJsonFeature[];
};

// Module-level caches (shared across hook instances)
const indexCache = new Map<string, GeoJsonIndex>();
const metadataCache = new Map<string, ProductMetadata>();

const INDEX_URLS: Record<string, string> = {
  HIRISE: "/hirise_index.geojson",
  CRISM: "/crism_index.geojson",
  CRISM_TRR3: "/crism_trr3_index.geojson",
  HIRISE_DTM: "/hirise_dtm_index.geojson",
  CTX: "/ctx_index.geojson",
};

async function loadIndex(instrument: InstrumentType): Promise<GeoJsonIndex | null> {
  // Normalize instrument to index key
  const key = instrument === "CRISM_TRR3" ? "CRISM_TRR3" : instrument;
  if (indexCache.has(key)) return indexCache.get(key)!;

  const url = INDEX_URLS[key];
  if (!url) return null;

  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    indexCache.set(key, data);
    return data;
  } catch {
    return null;
  }
}

function extractMetadata(feature: GeoJsonFeature, instrument: InstrumentType): ProductMetadata {
  const p = feature.properties ?? {};
  const meta: ProductMetadata = {};

  // Common fields
  if (typeof p.title === "string") meta.title = p.title;
  if (typeof p.center_latitude === "number") meta.centerLatitude = p.center_latitude;
  if (typeof p.center_longitude === "number") meta.centerLongitude = p.center_longitude;
  if (typeof p.observation_date === "string") meta.observationDate = p.observation_date;
  if (typeof p.date === "string" && !meta.observationDate) meta.observationDate = p.date;
  if (typeof p.solar_incidence === "number") meta.solarIncidence = p.solar_incidence;
  if (typeof p.incidence_angle === "number" && meta.solarIncidence == null) meta.solarIncidence = p.incidence_angle;
  if (typeof p.emission_angle === "number") meta.emissionAngle = p.emission_angle;
  if (typeof p.phase_angle === "number") meta.phaseAngle = p.phase_angle;
  if (typeof p.resolution === "number") meta.resolution = p.resolution;
  if (typeof p.map_scale === "number") meta.mapScale = p.map_scale;
  if (typeof p.image_lines === "number") meta.imageLines = p.image_lines;
  if (typeof p.lines === "number" && meta.imageLines == null) meta.imageLines = p.lines;
  if (typeof p.image_samples === "number") meta.imageSamples = p.image_samples;
  if (typeof p.samples === "number" && meta.imageSamples == null) meta.imageSamples = p.samples;
  if (typeof p.orbit_number === "number") meta.orbitNumber = p.orbit_number;
  if (typeof p.product_type === "string") meta.productType = p.product_type;
  if (typeof p.target_name === "string") meta.targetName = p.target_name;
  if (typeof p.description === "string") meta.description = p.description;

  // CRISM-specific
  if (typeof p.sensor_id === "string") meta.sensorId = p.sensor_id;
  if (typeof p.wavelength_range === "string") meta.wavelengthRange = p.wavelength_range;

  // HiRISE-specific
  if (instrument === "HIRISE") {
    meta.hasColorData = p.red_tif !== undefined && p.red_tif !== "";
    if (typeof p.map_scale === "number") meta.mapScale = p.map_scale;
  }

  return meta;
}

export default function useProductMetadata(
  productId: string | null,
  instrument: InstrumentType | null,
): { metadata: ProductMetadata | null; loading: boolean } {
  const [metadata, setMetadata] = useState<ProductMetadata | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!productId || !instrument) {
      setMetadata(null);
      return;
    }

    // Check cache first
    const cacheKey = `${instrument}:${productId}`;
    if (metadataCache.has(cacheKey)) {
      setMetadata(metadataCache.get(cacheKey)!);
      return;
    }

    let cancelled = false;
    setLoading(true);

    (async () => {
      const index = await loadIndex(instrument);
      if (cancelled) return;

      if (!index?.features) {
        setMetadata(null);
        setLoading(false);
        return;
      }

      // Find the matching feature
      const feature = index.features.find((f) => {
        const pid = f.properties?.product_id;
        return pid === productId;
      });

      if (feature) {
        const meta = extractMetadata(feature, instrument);
        metadataCache.set(cacheKey, meta);
        setMetadata(meta);
      } else {
        setMetadata(null);
      }
      setLoading(false);
    })();

    return () => { cancelled = true; };
  }, [productId, instrument]);

  return { metadata, loading };
}
