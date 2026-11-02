import { describe, it, expect } from "vitest";
import { laneForInstrument, instrumentFor, LANES, LANE_CONFIG } from "../laneConfig";

describe("laneConfig", () => {
  it("maps every product instrument to its lane and variant", () => {
    expect(laneForInstrument("SHARAD_HIGHRES")).toEqual({ lane: "SHARAD", variant: "highres" });
    expect(laneForInstrument("CRISM_TRR3")).toEqual({ lane: "CRISM", variant: "trr3" });
    expect(laneForInstrument("HIRISE_DTM")).toEqual({ lane: "HIRISE", variant: "dtm" });
    expect(laneForInstrument("CTX")).toEqual({ lane: "CTX", variant: "image" });
    expect(laneForInstrument("CUSTOM")).toBeNull();
  });

  it("round-trips lane+variant -> instrument -> lane+variant", () => {
    for (const lane of LANES) {
      for (const v of LANE_CONFIG[lane].variants) {
        const inst = instrumentFor(lane, v.value);
        expect(laneForInstrument(inst)?.lane).toBe(lane);
      }
    }
  });
});
