// detail-inspector/types.ts

export type InstrumentType = "HIRISE" | "CRISM" | "SHARAD";

export type DetailItem = {
  instrument: InstrumentType;
  productId: string;
  obsId?: string;
  lat: number;
  lon: number;
};
