import { describe, it, expect } from "vitest";
import {
  computeDownsample, nearestTrace, depthFromTwt, twtFromDepth, niceStep, toCsv, haversineM,
} from "../sharad";

describe("computeDownsample", () => {
  it("keeps images within the max width", () => {
    expect(computeDownsample(97438)).toBe(6);
    expect(Math.ceil(97438 / computeDownsample(97438))).toBeLessThanOrEqual(16384);
    expect(computeDownsample(60196)).toBe(4);
    expect(computeDownsample(1000)).toBe(1);
    expect(computeDownsample(0)).toBe(1);
  });
});

describe("depth conversion", () => {
  it("uniform εr: d = c t / (2√εr)", () => {
    const t = 1e-6;
    expect(depthFromTwt(t, 4)).toBeCloseTo((299792458 * t) / 4, 6);
  });
  it("two-layer is continuous at the boundary and invertible", () => {
    const t1 = twtFromDepth(50, 3, 9, 50);
    expect(depthFromTwt(t1, 3, 9, 50)).toBeCloseTo(50, 6);
    const t = twtFromDepth(120, 3, 9, 50);
    expect(depthFromTwt(t, 3, 9, 50)).toBeCloseTo(120, 6);
    // deeper layer is slower → more time per metre
    expect(twtFromDepth(120, 3, 9, 50)).toBeGreaterThan(twtFromDepth(120, 3, 3, 50));
  });
});

describe("nearestTrace", () => {
  it("finds the closest trace", () => {
    const lats = [18.0, 18.2, 18.4, 18.6];
    const lons = [77.4, 77.42, 77.44, 77.46];
    const r = nearestTrace(lats, lons, 18.41, 77.45)!;
    expect(r.index).toBe(2);
    expect(r.distanceM).toBeLessThan(1000);
  });
  it("haversine ~59 km per degree of latitude on Mars", () => {
    expect(haversineM(0, 0, 1, 0) / 1000).toBeCloseTo(59.15, 1);
  });
});

describe("misc", () => {
  it("niceStep", () => {
    expect(niceStep(10, 5)).toBe(2);
    expect(niceStep(73, 5)).toBe(10);
    expect(niceStep(90, 5)).toBe(20);
  });
  it("toCsv quotes", () => {
    expect(toCsv(["a", "b"], [[1, 'x,"y"'], [null, 2]])).toBe('a,b\n1,"x,""y"""\n,2\n');
  });
});

describe("pick export", () => {
  it("computes depth/uncertainty and serialises CSV + GeoJSON", async () => {
    const { completePick, picksToCsv, picksToGeoJSON, PICK_COLUMNS } = await import("../sharad");
    const p = completePick({
      kind: "click", product_id: "R_X", display_trace: 10, full_trace: 60, lat: 18.4, lon: 77.45,
      bin: 250, surface_bin: 150, abs_delay_us: null, epsilon_r1: 4, epsilon_r2: 4, boundary_m: 0, snr_db: null,
    });
    expect(p.depth_m).toBeCloseTo((299792458 * 100 * 37.5e-9) / 4, 3);
    expect(p.depth_uncertainty_m).toBeCloseTo((299792458 * 37.5e-9) / 4, 3);
    expect(p.vertical_resolution_m).toBeCloseTo(7.5, 3);
    const csv = picksToCsv([p]).trim().split("\n");
    expect(csv[0]!.split(",")).toEqual([...PICK_COLUMNS]);
    expect(csv[1]!.startsWith("click,R_X,10,60,18.4,77.45,250,150,100,3.75,")).toBe(true);
    const gj = JSON.parse(picksToGeoJSON([p], { product_id: "R_X" }));
    expect(gj.features[0].geometry.coordinates).toEqual([77.45, 18.4]);
    expect(gj.features[0].properties.depth_m).toBeCloseTo(281.05, 1);
    expect(gj.metadata.lat_convention).toBe("planetocentric");
  });
});
