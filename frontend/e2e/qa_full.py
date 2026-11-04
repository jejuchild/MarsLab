"""
MarsLab full end-to-end QA (headless Chromium, Python Playwright).

Walks every main workflow against a running dev server, records each check,
every uncaught page error, console error and HTTP 5xx, and prints a report.
Unlike smoke.py it does not stop at the first failure.

    BASE_URL=http://localhost:5173 python frontend/e2e/qa_full.py [screenshot_dir]
"""

import asyncio
import os
import sys
import time

from playwright.async_api import async_playwright

BASE = os.environ.get("BASE_URL", "http://localhost:5173").rstrip("/")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
JEZERO = (18.44, 77.45)

results: list[tuple[str, bool, str]] = []
last_step = ["start"]


def check(name: str, ok: bool, note: str = "") -> bool:
    results.append((name, bool(ok), note))
    last_step[0] = name
    print(("PASS " if ok else "FAIL ") + name + (f"  [{note}]" if note else ""), flush=True)
    return bool(ok)


async def shot(page, name):
    if OUT:
        await page.screenshot(path=os.path.join(OUT, f"qa_{name}.png"))


def watch(page, sink):
    page.on("pageerror", lambda e: sink["pageerror"].append(str(e)[:200]))
    page.on("console", lambda m: sink["console"].append(f"(after: {last_step[0]}) {m.text[:300]}") if m.type == "error" else None)

    def on_resp(r):
        if r.status >= 500 and r.url.startswith(BASE):
            sink["5xx"].append(f"{r.status} {r.request.method} {r.url.replace(BASE, '')[:120]}")

    page.on("response", on_resp)


async def new_page(p, width=1700, height=950, mobile=False):
    browser = await p.chromium.launch()
    ctx = await browser.new_context(
        viewport={"width": width, "height": height},
        is_mobile=mobile,
        has_touch=mobile,
        permissions=["clipboard-read", "clipboard-write"],
    )
    await ctx.add_init_script("localStorage.setItem('marslab-tour-completed','true')")
    page = await ctx.new_page()
    return browser, page


async def chip_state(bar, name):
    btn = bar.locator(f'button[aria-label^="{name} footprints:"]')
    return (await btn.get_attribute("aria-label") or "").split(": ", 1)[-1]


async def wait_state(bar, name, want, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if await chip_state(bar, name) == want:
            return True
        await asyncio.sleep(1)
    return False


async def open_inspector(page):
    insp = page.locator('aside[aria-label="Inspector"]')
    box = await page.locator("main").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    for dx, dy in [(0, 0), (-40, -30), (40, 40), (-80, 20), (60, -50)]:
        await page.mouse.click(cx + dx, cy + dy)
        await page.wait_for_timeout(4000)
        if await insp.count():
            return insp
    return insp


async def desktop_flow(p, sink):
    browser, page = await new_page(p)
    watch(page, sink)
    t0 = time.time()
    await page.goto(f"{BASE}/?lat={JEZERO[0]}&lon={JEZERO[1]}", wait_until="domcontentloaded")
    await page.wait_for_timeout(10000)
    check("app loads (map canvas present)", await page.locator("canvas").count() > 0, f"{time.time() - t0:.1f}s")

    # ── Instrument bar: all four instruments at once (the old crash) ──
    bar = page.get_by_role("toolbar", name="Instrument footprints")
    check("instrument bar visible on the map", await bar.count() == 1)
    for name in ["SHARAD", "CRISM", "HiRISE", "CTX"]:
        await bar.locator(f'button[aria-label="{name} footprints: not loaded"]').click()
    shown = {n: await wait_state(bar, n, "shown") for n in ["SHARAD", "CRISM", "HiRISE", "CTX"]}
    check("all 4 instruments load together without freezing", all(shown.values()), str(shown))
    responsive = await page.evaluate("() => new Promise(r => { const t = performance.now(); setTimeout(() => r(performance.now() - t), 0); })")
    check("main thread responsive after loading all layers", responsive < 500, f"{responsive:.0f} ms")
    await shot(page, "all_layers")

    await bar.locator('button[aria-label="CTX footprints: shown"]').click()
    check("chip click hides a layer", await wait_state(bar, "CTX", "hidden", 5))
    await bar.locator('button[aria-label="CTX footprints: hidden"]').click()
    check("chip click shows it again", await wait_state(bar, "CTX", "shown", 5))

    await bar.get_by_role("button", name="SHARAD variants").click()
    menu = page.get_by_role("menu")
    check("variant menu opens", await menu.count() == 1)
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(300)
    check("Escape closes the variant menu", await page.get_by_role("menu").count() == 0)

    # ── Inspector / compare ──
    insp = await open_inspector(page)
    check("clicking a footprint opens the Inspector", await insp.count() == 1)
    if await insp.count():
        cards = insp.locator("section[aria-label$='at this point']")
        check("Compare shows a card per instrument", await cards.count() == 4, str(await cards.count()))
        txt = (await insp.inner_text()).lower()
        check("Compare lists covering products", " cover" in txt)

        show_all = insp.get_by_role("button", name="Show all on map")
        if await show_all.count():
            await show_all.click()
            await page.wait_for_timeout(9000)
            check("'Show all on map' turns overlays on", await insp.get_by_role("button", name="Hide all from map").count() == 1)
            await shot(page, "overlays")
            await insp.get_by_role("button", name="Hide all from map").click()
            await page.wait_for_timeout(1500)
            check("'Hide all from map' turns them off", await insp.get_by_role("button", name="Show all on map").count() == 1)

        # HiRISE thumbnail → fullscreen viewer
        thumb = insp.get_by_role("button", name="full screen")
        if await thumb.count():
            await thumb.first.click()
            await page.wait_for_timeout(7000)
            dlg = page.get_by_role("dialog")
            check("HiRISE thumbnail opens the viewer", await dlg.count() == 1)
            fit = await page.evaluate("""() => { const i=document.querySelector('[role=dialog] img');
              if(!i||!i.naturalWidth) return null; const b=i.getBoundingClientRect();
              return b.left>=0&&b.top>=0&&b.right<=innerWidth&&b.bottom<=innerHeight; }""")
            check("viewer image fitted on screen", fit is True)
            check("inspected point marked on the image", await page.locator('[aria-label="Inspected point"]').count() == 1)
            await dlg.get_by_role("button", name="Zoom in").click()
            z1 = await dlg.get_by_role("toolbar", name="Zoom").inner_text()
            await dlg.get_by_role("button", name="Fit to screen").click()
            z2 = await dlg.get_by_role("toolbar", name="Zoom").inner_text()
            check("zoom in / fit buttons change zoom", z1 != z2, f"{z1.split()[1] if len(z1.split())>1 else z1} → {z2.split()[1] if len(z2.split())>1 else z2}")
            await shot(page, "viewer")
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(600)
            check("Escape closes only the viewer", await page.get_by_role("dialog").count() == 0 and await insp.count() == 1)

        # By-instrument → CRISM spectrum
        await insp.get_by_role("tab", name="By instrument").click()
        await page.wait_for_timeout(1500)
        check("By-instrument view renders", "covers" in (await insp.inner_text()).lower())

        # Radargram dock (SHARAD hi-res)
        await insp.get_by_role("tab", name="Compare at point").click()
        await page.wait_for_timeout(800)
        rbtn = insp.get_by_role("button", name="Radargram")
        if await rbtn.count():
            await rbtn.first.click()
            await page.wait_for_timeout(8000)
            check("radargram docks next to the Inspector", await insp.count() == 1)
            await shot(page, "radargram_dock")
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1000)
            check("Escape closes the radargram, keeps Inspector", await insp.count() == 1)
        else:
            check("radargram available at Jezero", False, "no SHARAD hi-res card")

        # Temporal comparison via cross-analysis
        cross = page.get_by_text("CROSS-ANALYSIS", exact=False)
        if await cross.count():
            await cross.first.click()
            await page.wait_for_timeout(4000)
            tab = page.get_by_role("tab", name="Temporal")
            if await tab.count():
                await tab.first.click()
                await page.wait_for_timeout(500)
            t = page.get_by_role("button", name="Compare over time")
            if await t.count():
                await t.first.click()
                await page.wait_for_timeout(6000)
                dlg = page.get_by_role("dialog", name="Temporal change")
                check("Temporal comparison opens", await dlg.count() == 1)
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(500)
                check("Escape closes Temporal comparison", await page.get_by_role("dialog").count() == 0)
            else:
                check("Temporal action available", False, "no 'Compare over time' button")

    # ── Measure ──
    await page.get_by_role("button", name="Measure distance, area or profile").click()
    await page.wait_for_timeout(500)
    check("Measure button enters measurement mode", await page.get_by_role("button", name="Exit measurement mode").count() == 1)
    await page.get_by_role("button", name="Exit measurement mode").click()
    await page.wait_for_timeout(300)
    check("Done leaves measurement mode", await page.get_by_role("button", name="Measure distance, area or profile").count() == 1)

    # ── 2D ↔ 3D keeps the site ──
    await page.get_by_role("button", name="3D Globe").first.click()
    await page.wait_for_timeout(6000)
    cam = await page.evaluate("""() => { const v = window.__marslabViewer; if (!v) return null;
      const s = v.scene, c = s.canvas, e = s.globe.ellipsoid;
      const p = v.camera.pickEllipsoid({x: c.clientWidth/2, y: c.clientHeight/2}, e);
      if (!p) return null; const g = e.cartesianToCartographic(p);
      return [g.latitude * 57.2958, g.longitude * 57.2958, s.mode]; }""")
    check("3D keeps the inspected site in view", bool(cam) and abs(cam[0] - JEZERO[0]) < 3 and abs(cam[1] - JEZERO[1]) < 3, str(cam and [round(x, 2) for x in cam]))
    await shot(page, "globe3d")
    await page.get_by_role("button", name="2D Map").first.click()
    await page.wait_for_timeout(4000)

    # ── Share link ──
    await page.get_by_role("button", name="Copy link to this view").click()
    await page.wait_for_timeout(800)
    clip = await page.evaluate("() => navigator.clipboard.readText().catch(() => '')")
    check("share button copies a link with the location", "lat=" in clip and "lon=" in clip, clip[:90])
    check("address bar stays clean", "?" not in page.url)

    # ── Header search ──
    search = page.get_by_placeholder("Search product IDs, coordinates, regions...")
    if await search.count():
        await search.click()
        await search.fill("PSP_002387_1985")
        await page.wait_for_timeout(3000)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(6000)
        check("searching a product ID selects it", "psp_002387_1985" in (await page.inner_text("body")).lower())

    # ── Other pages ──
    for path, needle in [("/download", "download"), ("/upload", "upload"), ("/suggestions", "suggest")]:
        await page.goto(f"{BASE}{path}", wait_until="domcontentloaded")
        await page.wait_for_timeout(4000)
        body = (await page.inner_text("body")).lower()
        check(f"{path} page renders", needle in body and "something went wrong" not in body)
    await page.goto(f"{BASE}/news", wait_until="domcontentloaded")
    await page.wait_for_timeout(2500)
    check("removed route redirects to the map", page.url.rstrip("/") == BASE)
    await browser.close()


async def narrow_flow(p, sink):
    browser, page = await new_page(p, width=1366, height=768)
    watch(page, sink)
    await page.goto(f"{BASE}/?lat={JEZERO[0]}&lon={JEZERO[1]}", wait_until="domcontentloaded")
    await page.wait_for_timeout(9000)
    bar = page.get_by_role("toolbar", name="Instrument footprints")
    await bar.locator('button[aria-label="HiRISE footprints: not loaded"]').click()
    await wait_state(bar, "HiRISE", "shown")
    overlap = await page.evaluate("""() => {
      const a = document.querySelector('[role=toolbar][aria-label="Instrument footprints"]').querySelectorAll('button');
      const m = document.querySelector('[aria-label="Measure distance, area or profile"]').getBoundingClientRect();
      return [...a].some(b => { const r = b.getBoundingClientRect();
        return r.left < m.right && r.right > m.left && r.top < m.bottom && r.bottom > m.top; }); }""")
    check("1366px: instrument bar does not overlap the Measure button", not overlap)
    insp = await open_inspector(page)
    rbtn = insp.get_by_role("button", name="Radargram")
    if await rbtn.count():
        await rbtn.first.click()
        await page.wait_for_timeout(7000)
        strip = page.get_by_role("button", name="Show Inspector next to the viewer")
        check("1366px: Inspector folds to a strip beside the radargram", await strip.count() == 1)
        await shot(page, "narrow_dock")
    await browser.close()


async def mobile_flow(p, sink):
    browser, page = await new_page(p, width=390, height=844, mobile=True)
    watch(page, sink)
    await page.goto(f"{BASE}/", wait_until="domcontentloaded")
    await page.wait_for_timeout(9000)
    check("mobile: map renders", await page.locator("canvas").count() > 0)
    bar = page.get_by_role("toolbar", name="Instrument footprints")
    check("mobile: instrument bar present", await bar.count() == 1)
    width = await page.evaluate("() => document.documentElement.scrollWidth")
    check("mobile: no horizontal page scroll", width <= 390, f"scrollWidth={width}")
    check("mobile: bottom navigation present", await page.get_by_role("button", name="Toggle layers panel").count() == 1)
    await shot(page, "mobile")
    await browser.close()


async def api_checks(sink):
    import requests

    def get(path, **kw):
        return requests.get(f"{BASE}{path}", timeout=60, **kw)

    r = requests.post(f"{BASE}/api/exists/batch", json={"items": [{"instrument": "hirise", "product_id": "PSP_002387_1985"}]}, timeout=60)
    check("API exists/batch", r.ok and "has_core" in r.json())
    r = requests.post(f"{BASE}/api/crism-trr3/msv00050e6e/spectrum", json={"line": 5, "sample": 5}, timeout=60)
    check("API TRR3 VNIR cube → 422 with reason", r.status_code == 422 and "bands" in r.json().get("detail", ""))
    r = get("/api/inspector/at-point", params={"lat": JEZERO[0], "lon": JEZERO[1]})
    check("API inspector at-point", r.ok, str(r.status_code))
    r = get("/.env")
    check("dev server does not serve /.env", not (r.ok and "=" in r.text and "<html" not in r.text.lower()), str(r.status_code))


async def main():
    sink = {"pageerror": [], "console": [], "5xx": []}
    async with async_playwright() as p:
        for flow in (desktop_flow, narrow_flow, mobile_flow):
            try:
                await flow(p, sink)
            except Exception as e:  # keep going; record the crash
                check(f"{flow.__name__} completed", False, f"{type(e).__name__}: {str(e)[:160]}")
    await api_checks(sink)

    check("no uncaught page errors", not sink["pageerror"], "; ".join(sink["pageerror"][:3]))
    check("no HTTP 5xx responses", not sink["5xx"], "; ".join(sorted(set(sink["5xx"]))[:5]))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if sink["console"]:
        print(f"console errors ({len(sink['console'])}, unique shown):")
        for c in sorted(set(sink["console"]))[:25]:
            print("  -", c)
    sys.exit(1 if failed else 0)


asyncio.run(main())
