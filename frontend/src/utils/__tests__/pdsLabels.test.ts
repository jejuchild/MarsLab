import { describe, it, expect, vi, afterEach } from "vitest";

const label = (projection: string, id: string) => `
  OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "${projection}"
  MINIMUM_LATITUDE = 18.0
  MAXIMUM_LATITUDE = 19.0
  WESTERNMOST_LONGITUDE = 77.0
  EASTERNMOST_LONGITUDE = 78.0
  END_OBJECT = IMAGE_MAP_PROJECTION
  LINES = 100
  LINE_SAMPLES = 200
  /* ${id} */`;

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

async function load(projection: string) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => ({ ok: true, text: async () => label(projection, url) })));
  return import("../pdsLabels");
}

describe("crismPixelAt", () => {
  it("maps lat/lon into an equirectangular product", async () => {
    const { crismPixelAt } = await load("EQUIRECTANGULAR");
    expect(await crismPixelAt("frtA_07_if166j_mtr3", 18.75, 77.25)).toEqual({ line: 25, sample: 50 });
  });

  it("returns null outside the product", async () => {
    const { crismPixelAt } = await load("EQUIRECTANGULAR");
    expect(await crismPixelAt("frtB_07_if166j_mtr3", 17.5, 77.25)).toBeNull();
  });

  it("refuses polar-stereographic products instead of guessing", async () => {
    const { crismPixelAt } = await load("POLAR STEREOGRAPHIC");
    expect(await crismPixelAt("frtC_07_if166j_mtr3", 18.75, 77.25)).toBeNull();
  });
});
