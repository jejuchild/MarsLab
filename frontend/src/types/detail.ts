export type InstrumentType = "HIRISE" | "CRISM";

export type DetailItem = {
  instrument: InstrumentType;
  productId: string;
  lat: number;
  lon: number;
  tifUrl?: string;
};
