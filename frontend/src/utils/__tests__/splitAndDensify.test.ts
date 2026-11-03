import { describe, it, expect } from "vitest";
import { splitAndDensify } from "../FootprintManager";

describe("splitAndDensify", () => {
  it("densifies a long two-point track", () => {
    const [part] = splitAndDensify([[77, -40], [77, 40]]);
    expect(part!.length).toBeGreaterThan(100);
    expect(part![0]).toEqual([77, -40]);
    expect(part![part!.length - 1]).toEqual([77, 40]);
  });

  it("splits at the antimeridian instead of wrapping around the planet", () => {
    const parts = splitAndDensify([[179, 0], [-179, 2]]);
    expect(parts).toHaveLength(2);
    expect(parts[0]![parts[0]!.length - 1]![0]).toBe(180);
    expect(parts[1]![0]![0]).toBe(-180);
    const maxGap = Math.max(...parts.flatMap((p) => p.slice(1).map((q, i) => Math.abs(q[0] - p[i]![0]))));
    expect(maxGap).toBeLessThanOrEqual(0.5 + 1e-9);
  });

  it("normalizes 0-360 longitudes", () => {
    const [part] = splitAndDensify([[257.4, 18.4], [257.5, 18.5]]);
    expect(part![0]![0]).toBeCloseTo(-102.6);
  });
});
