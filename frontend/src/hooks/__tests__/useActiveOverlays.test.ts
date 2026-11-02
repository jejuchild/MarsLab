import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import useActiveOverlays from "../useActiveOverlays";

describe("useActiveOverlays", () => {
  it("sets, updates opacity and removes overlays", () => {
    const { result } = renderHook(() => useActiveOverlays());
    act(() => result.current.setOverlay("ESP_1", { type: "quickview", opacity: 100 }));
    expect(result.current.activeOverlays.get("ESP_1")).toEqual({ type: "quickview", opacity: 100 });

    act(() => result.current.setOpacity("ESP_1", 40));
    expect(result.current.activeOverlays.get("ESP_1")?.opacity).toBe(40);

    act(() => result.current.setOverlay("ESP_1", null));
    expect(result.current.activeOverlays.has("ESP_1")).toBe(false);
  });

  it("ignores opacity for a product without an overlay", () => {
    const { result } = renderHook(() => useActiveOverlays());
    const before = result.current.activeOverlays;
    act(() => result.current.setOpacity("missing", 50));
    expect(result.current.activeOverlays).toBe(before);
  });

  it("removeMany and clear", () => {
    const { result } = renderHook(() => useActiveOverlays());
    act(() => {
      result.current.setOverlay("A", { type: "quickview", opacity: 100 });
      result.current.setOverlay("B", { type: "highres", opacity: 80 });
      result.current.setOverlay("C", { type: "quickview", opacity: 100 });
    });
    act(() => result.current.removeMany(["A", "C"]));
    expect([...result.current.activeOverlays.keys()]).toEqual(["B"]);
    act(() => result.current.clear());
    expect(result.current.activeOverlays.size).toBe(0);
  });

  it("exposes the latest map through the stable ref", () => {
    const { result } = renderHook(() => useActiveOverlays());
    const ref = result.current.activeOverlaysRef;
    act(() => result.current.setOverlay("X", { type: "quickview", opacity: 100 }));
    expect(ref.current.has("X")).toBe(true);
    expect(result.current.activeOverlaysRef).toBe(ref);
  });
});
