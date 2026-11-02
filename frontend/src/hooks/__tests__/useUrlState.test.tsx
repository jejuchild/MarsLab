import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import useUrlState from "../useUrlState";

function setup(initialUrl: string) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter initialEntries={[initialUrl]}>{children}</MemoryRouter>
  );
  return renderHook(() => ({ url: useUrlState(), location: useLocation() }), { wrapper });
}

describe("useUrlState (one-shot deep links)", () => {
  it("reads shared-link params once and strips them from the URL", () => {
    const { result } = setup("/?lat=18.44&lon=77.45&instruments=hirise,crism&product=ESP_1&base=HRSC");
    expect(result.current.url.urlState).toMatchObject({
      lat: 18.44,
      lon: 77.45,
      instruments: ["hirise", "crism"],
      product: "ESP_1",
      base: "HRSC",
    });
    expect(result.current.location.search).toBe("");
  });

  it("keeps the initial state frozen after the URL is cleaned", () => {
    const { result, rerender } = setup("/?lat=1&lon=2");
    rerender();
    expect(result.current.url.urlState).toMatchObject({ lat: 1, lon: 2 });
  });

  it("maps legacy ?flyTo= to product", () => {
    const { result } = setup("/?flyTo=frt00003452_07&instrument=CRISM_TRR3");
    expect(result.current.url.urlState.product).toBe("frt00003452_07");
    expect(result.current.location.search).toBe("");
  });

  it("does not write to the address bar; share URL reflects recorded view", () => {
    const { result } = setup("/");
    act(() => result.current.url.updateUrl({ lat: 18.4401, lon: 77.4502, instruments: ["hirise"] }));
    expect(result.current.location.search).toBe("");
    const share = new URL(result.current.url.getShareUrl());
    expect(share.searchParams.get("lat")).toBe("18.4401");
    expect(share.searchParams.get("instruments")).toBe("hirise");
    act(() => result.current.url.updateUrl({ instruments: undefined }));
    expect(new URL(result.current.url.getShareUrl()).searchParams.has("instruments")).toBe(false);
  });
});
