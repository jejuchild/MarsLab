import { describe, it, expect } from "vitest";
import { boundsFromPolygon } from "../../utils/productBounds";

describe("boundsFromPolygon", () => {
  it("returns a plain box for an ordinary footprint", () => {
    expect(boundsFromPolygon([[77.4, 18.4], [77.5, 18.4], [77.5, 18.5], [77.4, 18.5]]))
      .toEqual({ west: 77.4, east: 77.5, south: 18.4, north: 18.5 });
  });

  it("normalizes 0-360 longitudes", () => {
    const b = boundsFromPolygon([[257.4, 18.4], [257.5, 18.4], [257.5, 18.5], [257.4, 18.5]])!;
    expect(b.west).toBeCloseTo(-102.6);
    expect(b.east).toBeCloseTo(-102.5);
  });

  it("keeps a footprint straddling 0/360 narrow", () => {
    const b = boundsFromPolygon([[359.9, 1], [0.1, 1], [0.1, 2], [359.9, 2]])!;
    expect(b.west).toBeCloseTo(-0.1);
    expect(b.east).toBeCloseTo(0.1);
  });

  it("represents an antimeridian crossing as west > east", () => {
    const b = boundsFromPolygon([[179.9, 1], [-179.9, 1], [-179.9, 2], [179.9, 2]])!;
    expect(b.west).toBeCloseTo(179.9);
    expect(b.east).toBeCloseTo(-179.9);
  });
});
