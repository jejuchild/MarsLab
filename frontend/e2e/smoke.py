"""
MarsLab end-to-end smoke test (headless Chromium via Python Playwright).

Exercises the core "compare instruments at one location" flow against a
running dev server and exits non-zero on the first failure.

    pip install playwright && python -m playwright install chromium
    BASE_URL=http://localhost:5173 python frontend/e2e/smoke.py [screenshot_dir]

Camera checks need the dev build (window.__marslabViewer); they are skipped on a production build.
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

BASE = os.environ.get("BASE_URL", "http://localhost:5173").rstrip("/")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
JEZERO = (18.44, 77.45)

failures: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        failures.append(label)


async def shot(page, name):
    if OUT:
        await page.screenshot(path=os.path.join(OUT, f"{name}.png"))


async def overlay_texture_std(page) -> float:
    """Luminance spread in the map centre: ~3 for a grey placeholder, >10 for imagery."""
    png = await page.screenshot(clip={"x": 640, "y": 320, "width": 200, "height": 360})
    from io import BytesIO
    from PIL import Image
    import numpy as np
    return float(np.asarray(Image.open(BytesIO(png)).convert("L"), dtype=float).std())


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        await ctx.add_init_script("localStorage.setItem('marslab-tour-completed','true')")
        page = await ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1. Shared link restores once, then the address bar is clean.
        await page.goto(f"{BASE}/?lat={JEZERO[0]}&lon={JEZERO[1]}", wait_until="domcontentloaded")
        await page.wait_for_timeout(12000)
        cam = await page.evaluate(
            "() => { const v = window.__marslabViewer; if (!v) return null;"
            " const c = v.camera.positionCartographic;"
            " return [c.latitude * 57.29578, c.longitude * 57.29578]; }"
        )
        if cam is None:
            print("SKIP shared link camera check (production build has no debug handle)")
        else:
            check(abs(cam[0] - JEZERO[0]) < 0.2 and abs(cam[1] - JEZERO[1]) < 0.2, "shared link flies to its location")
        check("?" not in page.url, "deep-link params are stripped after restore")

        # 2. Load HiRISE footprints and inspect a point.
        bar = page.get_by_role("toolbar", name="Instrument footprints")
        await bar.get_by_role("button", name="HiRISE footprints: not loaded").click()
        await page.wait_for_timeout(10000)
        check(await bar.get_by_role("button", name="HiRISE footprints: shown").count() == 1, "instrument bar loads and shows HiRISE footprints")
        inspector = page.locator('aside[aria-label="Inspector"]')
        for x, y in [(800, 500), (760, 470), (840, 540), (700, 420)]:
            await page.mouse.click(x, y)
            await page.wait_for_timeout(5000)
            if await inspector.count():
                break
        check(await inspector.count() == 1, "clicking a footprint opens the Inspector")
        text = await inspector.inner_text() if await inspector.count() else ""
        check("compare at point" in text.lower() and " cover" in text.lower(), "Compare view lists instruments covering the point")
        cards = await inspector.locator("section[aria-label$='at this point']").count()
        check(cards == 4, f"Compare view shows one card per instrument ({cards})")

        # 3. Auto quickview renders as imagery (not a grey placeholder).
        await page.mouse.move(1150, 950)
        await page.wait_for_timeout(9000)
        await shot(page, "smoke_overlay")
        check(await overlay_texture_std(page) > 10, "quickview overlay shows imagery")

        # 4. Fullscreen viewer: Escape closes only the viewer.
        if await inspector.count():
            await inspector.get_by_role("tab", name="By instrument").click()
            await page.wait_for_timeout(1500)
            check("covers" in (await inspector.inner_text()).lower(), "lane view marks products that cover the point")
            await inspector.get_by_role("button", name="View High-Res Image").first.click()
            await page.wait_for_timeout(3000)
            viewer_open = await page.locator("div.fixed.inset-0").count()
            await page.wait_for_timeout(5000)
            fits = await page.evaluate("""() => {
              const i = document.querySelector('[role=dialog] img');
              if (!i || !i.naturalWidth) return null;
              const b = i.getBoundingClientRect();
              return b.left >= 0 && b.top >= 0 && b.right <= innerWidth && b.bottom <= innerHeight;
            }""")
            check(fits is True, "viewer opens with the whole image fitted on screen")
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1500)
            check(viewer_open == 1 and await page.locator("div.fixed.inset-0").count() == 0, "Escape closes the fullscreen viewer")
            check(await inspector.count() == 1, "Escape in the viewer keeps the Inspector open")

        # 5. Map controls never paint over a page-level modal.
        on_top = await page.evaluate("""() => {
          const btn = document.querySelector('[aria-label="Measure distance, area or profile"]');
          if (!btn) return null;
          const r = btn.getBoundingClientRect();
          const m = document.createElement('div'); m.className = 'fixed inset-0 z-50'; m.id = 'probe-modal';
          document.querySelector('main').appendChild(m);
          const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
          m.remove();
          return hit && hit.id === 'probe-modal';
        }""")
        check(on_top is True, "a z-50 modal covers the map's measure button")

        # 6. Removed routes fall back to the map.
        await page.goto(f"{BASE}/news", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        check(page.url.rstrip("/") == BASE, "removed /news route redirects to the map")

        check(not errors, f"no uncaught page errors ({len(errors)})")
        await browser.close()

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nall smoke checks passed")


asyncio.run(main())
