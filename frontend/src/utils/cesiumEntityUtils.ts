import * as Cesium from "cesium";

export type InstrumentType = "CRISM" | "HIRISE" | "SHARAD" | "SHARAD_HIGHRES" | "CTX" | "CUSTOM" | "HIRISE_DTM" | "CRISM_TRR3";

const ALL_INSTRUMENTS: InstrumentType[] = ["HIRISE", "CRISM", "CTX", "SHARAD", "SHARAD_HIGHRES", "HIRISE_DTM", "CRISM_TRR3", "CUSTOM"];

/**
 * Find a Cesium entity by product ID across all instrument prefixes.
 * Entity ID format: INSTRUMENT_FP_productId
 */
export function findEntityByProductId(
  viewer: Cesium.Viewer,
  productId: string,
): { entity: Cesium.Entity; instrument: InstrumentType } | null {
  for (const inst of ALL_INSTRUMENTS) {
    const entity = viewer.entities.getById(`${inst}_FP_${productId}`);
    if (entity) return { entity, instrument: inst };
  }
  return null;
}

/**
 * Detect instrument type from product ID string.
 */
export function detectInstrument(productId: string): InstrumentType {
  if (productId.startsWith("ESP_") || productId.startsWith("PSP_")) return "HIRISE";
  if (productId.startsWith("DTE")) return "HIRISE_DTM";
  if (/^(frs|msv|frt|hrl|hrs|arl|atl)[0-9a-f]+_\d{2}$/i.test(productId)) return "CRISM_TRR3";
  if (/^(frt|hrl|hrs|frs)/i.test(productId)) return "CRISM";
  return "CRISM"; // default fallback
}

/**
 * Safely get a string property value from a Cesium entity.
 */
export function getEntityProperty(entity: Cesium.Entity, key: string): string | undefined {
  const props = entity.properties;
  if (!props) return undefined;
  const prop = props[key];
  if (!prop) return undefined;
  const val = typeof prop.getValue === "function" ? prop.getValue(Cesium.JulianDate.now()) : prop;
  return typeof val === "string" ? val : undefined;
}

/**
 * Get instrument type from entity properties.
 */
export function getEntityInstrument(entity: Cesium.Entity): InstrumentType | null {
  const inst = getEntityProperty(entity, "instrument");
  return inst && inst.length > 0 ? inst as InstrumentType : null;
}

/**
 * Get product ID from entity properties.
 */
export function getEntityProductId(entity: Cesium.Entity): string | null {
  return getEntityProperty(entity, "product_id") ?? null;
}
