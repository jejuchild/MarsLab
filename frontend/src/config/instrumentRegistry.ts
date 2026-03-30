/**
 * Instrument Registry - Single source of truth for instrument configuration.
 * Replaces hardcoded instrument-specific logic across the frontend.
 */

export type InstrumentId = 'crism' | 'hirise' | 'sharad' | 'sharad_highres' | 'ctx' | 'ctx_mosaic' | 'hirise_dtm' | 'crism_trr3';
export type GeometryType = 'Polygon' | 'LineString' | 'Point';
export type InstrumentGroupId = 'spectral' | 'imaging' | 'radar';

export interface InstrumentGroup {
  id: InstrumentGroupId;
  displayName: string;
  icon: string;
  instruments: InstrumentId[];
}

export interface InstrumentConfig {
  /** Unique identifier (lowercase) */
  id: InstrumentId;
  /** Short name for display */
  name: string;
  /** Full display name */
  displayName: string;
  /** Sub-label shown in hierarchy tree (e.g. "MTRDR", "Images", "DTM") */
  subLabel: string;
  /** Parent group */
  group: InstrumentGroupId;
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
    r: parseInt(result[1]!, 16) / 255,
    g: parseInt(result[2]!, 16) / 255,
    b: parseInt(result[3]!, 16) / 255,
  };
}

/**
 * Instrument groups for hierarchical display
 */
export const INSTRUMENT_GROUPS: InstrumentGroup[] = [
  { id: 'spectral', displayName: 'Spectrum',  icon: 'science',       instruments: ['crism', 'crism_trr3'] },
  { id: 'imaging',  displayName: 'Imagery',   icon: 'photo_camera',  instruments: ['hirise', 'hirise_dtm', 'ctx', 'ctx_mosaic'] },
  { id: 'radar',    displayName: 'Radar',     icon: 'radar',         instruments: ['sharad', 'sharad_highres'] },
];

/**
 * All registered instruments
 */
export const INSTRUMENTS: Record<InstrumentId, InstrumentConfig> = {
  crism: {
    id: 'crism',
    name: 'CRISM',
    displayName: 'Compact Reconnaissance Imaging Spectrometer for Mars',
    subLabel: 'CRISM MTRDR',
    group: 'spectral',
    geometryType: 'Polygon',
    color: '#00FFFF',
    cesiumColor: hexToRgb('#00FFFF'),
    productIdPattern: /^frt[0-9a-f]+/i,
    supportsSpectrum: true,
    supportsRgb: true,
  },
  hirise: {
    id: 'hirise',
    name: 'HiRISE',
    displayName: 'High Resolution Imaging Science Experiment',
    subLabel: 'HiRISE JP2',
    group: 'imaging',
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
    subLabel: 'SHARAD THM',
    group: 'radar',
    geometryType: 'LineString',
    color: '#FFA500',
    cesiumColor: hexToRgb('#FFA500'),
    productIdPattern: /^S_[0-9]+/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  sharad_highres: {
    id: 'sharad_highres',
    name: 'SHARAD_HIGHRES',
    displayName: 'SHARAD High-Resolution RDR',
    subLabel: 'SHARAD RDR',
    group: 'radar',
    geometryType: 'LineString',
    color: '#FFD700',
    cesiumColor: hexToRgb('#FFD700'),
    productIdPattern: /^R_[0-9]+/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  ctx: {
    id: 'ctx',
    name: 'CTX',
    displayName: 'Context Camera (MRO)',
    subLabel: 'CTX',
    group: 'imaging',
    geometryType: 'Polygon',
    color: '#FF69B4',
    cesiumColor: hexToRgb('#FF69B4'),
    productIdPattern: /.*CTX.*|^b[0-9]+_|^curiosity_ctx|^olympus_mons|^DEM_1m_|^InSight|^JEZ_ctx|^West_Candor/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  hirise_dtm: {
    id: 'hirise_dtm',
    name: 'HIRISE_DTM',
    displayName: 'HiRISE Digital Terrain Model',
    subLabel: 'HiRISE DTM',
    group: 'imaging',
    geometryType: 'Polygon',
    color: '#8B4513',
    cesiumColor: hexToRgb('#8B4513'),
    productIdPattern: /^DTE.*|^DTEEC.*/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  ctx_mosaic: {
    id: 'ctx_mosaic',
    name: 'CTX_MOSAIC',
    displayName: 'CTX Mosaic (Murray Lab 5m)',
    subLabel: 'CTX 5m Mosaic',
    group: 'imaging',
    geometryType: 'Polygon',
    color: '#DAA520',
    cesiumColor: hexToRgb('#DAA520'),
    productIdPattern: /^CTX_MOSAIC_/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  crism_trr3: {
    id: 'crism_trr3',
    name: 'CRISM_TRR3',
    displayName: 'CRISM Targeted Reduced Data Record v3',
    subLabel: 'CRISM TRDR',
    group: 'spectral',
    geometryType: 'Polygon',
    color: '#00CED1',
    cesiumColor: hexToRgb('#00CED1'),
    productIdPattern: /^frt[0-9a-f]+_[0-9]+/i,
    supportsSpectrum: true,
    supportsRgb: true,
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
