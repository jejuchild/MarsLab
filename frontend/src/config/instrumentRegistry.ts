/**
 * Instrument Registry - Single source of truth for instrument configuration.
 * Replaces hardcoded instrument-specific logic across the frontend.
 */

export type InstrumentId = 'crism' | 'hirise' | 'sharad';
export type GeometryType = 'Polygon' | 'LineString' | 'Point';

export interface InstrumentConfig {
  /** Unique identifier (lowercase) */
  id: InstrumentId;
  /** Short name for display */
  name: string;
  /** Full display name */
  displayName: string;
  /** GeoJSON geometry type */
  geometryType: GeometryType;
  /** Hex color for map rendering */
  color: string;
  /** Cesium Color components (computed) */
  cesiumColor: { r: number; g: number; b: number };
  /** Regex pattern for product ID matching */
  productIdPattern: RegExp;
  /** Whether this instrument supports spectrum visualization */
  supportsSpectrum: boolean;
  /** Whether this instrument supports RGB composition */
  supportsRgb: boolean;
}

/**
 * Parse hex color to RGB components (0-1 range)
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return { r: 1, g: 1, b: 1 };
  return {
    r: parseInt(result[1], 16) / 255,
    g: parseInt(result[2], 16) / 255,
    b: parseInt(result[3], 16) / 255,
  };
}

/**
 * All registered instruments
 */
export const INSTRUMENTS: Record<InstrumentId, InstrumentConfig> = {
  crism: {
    id: 'crism',
    name: 'CRISM',
    displayName: 'Compact Reconnaissance Imaging Spectrometer for Mars',
    geometryType: 'Polygon',
    color: '#00FFFF',
    cesiumColor: hexToRgb('#00FFFF'),
    productIdPattern: /^(frt|hrl|hrs|frs|arl|atl)[0-9a-f]+/i,
    supportsSpectrum: true,
    supportsRgb: true,
  },
  hirise: {
    id: 'hirise',
    name: 'HiRISE',
    displayName: 'High Resolution Imaging Science Experiment',
    geometryType: 'Polygon',
    color: '#FFFF00',
    cesiumColor: hexToRgb('#FFFF00'),
    productIdPattern: /^(ESP|PSP|TRA)_[0-9]+/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  sharad: {
    id: 'sharad',
    name: 'SHARAD',
    displayName: 'SHAllow RADar',
    geometryType: 'LineString',
    color: '#FFA500',
    cesiumColor: hexToRgb('#FFA500'),
    productIdPattern: /^S_[0-9]+/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
};

/**
 * Get all instrument IDs
 */
export function getInstrumentIds(): InstrumentId[] {
  return Object.keys(INSTRUMENTS) as InstrumentId[];
}

/**
 * Get all instrument configs
 */
export function getAllInstruments(): InstrumentConfig[] {
  return Object.values(INSTRUMENTS);
}

/**
 * Get instrument configuration by ID (case-insensitive)
 */
export function getInstrument(id: string): InstrumentConfig | undefined {
  return INSTRUMENTS[id.toLowerCase() as InstrumentId];
}

/**
 * Detect instrument from product ID using pattern matching
 */
export function detectInstrument(productId: string): InstrumentConfig | undefined {
  for (const config of Object.values(INSTRUMENTS)) {
    if (config.productIdPattern.test(productId)) {
      return config;
    }
  }
  return undefined;
}

/**
 * Get instrument color by ID
 */
export function getInstrumentColor(id: string): string {
  return INSTRUMENTS[id.toLowerCase() as InstrumentId]?.color ?? '#FFFFFF';
}

/**
 * Get Cesium-compatible color components by ID
 */
export function getInstrumentCesiumColor(id: string): { r: number; g: number; b: number } {
  return INSTRUMENTS[id.toLowerCase() as InstrumentId]?.cesiumColor ?? { r: 1, g: 1, b: 1 };
}

/**
 * Check if an instrument supports spectrum visualization
 */
export function supportsSpectrum(id: string): boolean {
  return INSTRUMENTS[id.toLowerCase() as InstrumentId]?.supportsSpectrum ?? false;
}

/**
 * Check if an instrument supports RGB composition
 */
export function supportsRgb(id: string): boolean {
  return INSTRUMENTS[id.toLowerCase() as InstrumentId]?.supportsRgb ?? false;
}

/**
 * Type guard for InstrumentId
 */
export function isInstrumentId(id: string): id is InstrumentId {
  return id.toLowerCase() in INSTRUMENTS;
}
