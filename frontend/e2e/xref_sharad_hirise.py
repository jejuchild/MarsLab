"""SHARAD x HiRISE cross-reference e2e (needs the dev build: BASE default http://127.0.0.1:5174)."""
import asyncio, sys
from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5174"
OUT = (sys.argv[2] + "/") if len(sys.argv) > 2 else "/tmp/"
fails = []


def check(ok, name, note=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{note}]" if note else ""), flush=True)
    if not ok:
        fails.append(name)


ENTITY = """(id) => { const v = window.__marslabViewer; if (!v) return null;
  const e = v.entities.values.filter(x => x.id.startsWith(id)); return e.length; }"""
HOVER_POS = """() => { const v = window.__marslabViewer; const e = v.entities.getById('SHARAD_HOVER_CURSOR');
  if (!e) return 'no entity'; const p = e.position.getValue(); if (!p) return null;
  const c = v.scene.globe.ellipsoid.cartesianToCartographic(p); return [c.latitude*57.2958, c.longitude*57.2958]; }"""


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1700, "height": 950})
        await ctx.add_init_script("localStorage.setItem('marslab-tour-completed','true')")
        pg = await ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
        await pg.goto(f"{BASE}/?lat=18.44&lon=77.45", wait_until="domcontentloaded")
        await pg.wait_for_timeout(10000)
        bar = pg.get_by_role("toolbar", name="Instrument footprints")
        await bar.locator('button[aria-label="HiRISE footprints: not loaded"]').click()
        await bar.get_by_role("button", name="SHARAD variants").click()
        await pg.get_by_role("menu").get_by_text("SHARAD RDR").click()
        await pg.keyboard.press("Escape")
        await pg.wait_for_timeout(12000)

        # 1. Click a hi-res track on the map → radargram docked at that point
        track = await pg.evaluate("""async () => {
          const r = await fetch('/api/sharad_highres/track?product_id=R_1205201_001_SS19_700_A&downsample=200');
          const d = await r.json(); const v = window.__marslabViewer; const e = v.scene.globe.ellipsoid;
          let best = null;
          for (let i = 0; i < d.lats.length; i++) {
            const c = e.cartographicToCartesian({longitude: d.lons[i]*Math.PI/180, latitude: d.lats[i]*Math.PI/180, height: 150});
            const s = v.scene.cartesianToCanvasCoordinates(c); if (!s) continue;
            const rect = v.canvas.getBoundingClientRect();
            const x = rect.left + s.x, y = rect.top + s.y;
            if (x > rect.left + 60 && x < rect.right - 60 && y > rect.top + 120 && y < rect.bottom - 60) { best = {x, y, lat: d.lats[i], lon: d.lons[i]}; break; }
          }
          return best; }""")
        check(track is not None, "hi-res track visible on screen")
        radar = pg.get_by_text("SHARAD HIGH-RES", exact=False)
        if track:
            await pg.mouse.click(track["x"], track["y"])
            await pg.wait_for_timeout(12000)
            check(await radar.count() > 0, "clicking a SHARAD track docks the radargram (no modal)")
            check(await pg.get_by_role("dialog").count() == 0, "no modal opened")
        await pg.screenshot(path=OUT + "xref_dock.png")

        # 2. Track highlighted on the map
        n = await pg.evaluate(ENTITY, "SHARAD_ACTIVE_TRACK")
        check((n or 0) >= 1, "radargram ground track drawn on the map", str(n))

        # 3. Hover sync
        canvas = pg.locator("canvas.cursor-grab").first
        bb = await canvas.bounding_box()
        if bb:
            await pg.mouse.move(bb["x"] + bb["width"] * 0.3, bb["y"] + bb["height"] * 0.5)
            await pg.wait_for_timeout(500)
            p1 = await pg.evaluate(HOVER_POS)
            await pg.mouse.move(bb["x"] + bb["width"] * 0.7, bb["y"] + bb["height"] * 0.5)
            await pg.wait_for_timeout(500)
            p2 = await pg.evaluate(HOVER_POS)
            check(isinstance(p1, list) and isinstance(p2, list) and p1 != p2, "radargram hover moves the map cursor", f"{p1} → {p2}")
            await pg.mouse.move(5, 500)
            await pg.wait_for_timeout(400)
            check(await pg.evaluate(HOVER_POS) is None, "map cursor hides when leaving the radargram")

        # 4. Crossing bands on the radargram
        bands = pg.locator('[aria-label="HiRISE crossings"] button')
        await pg.wait_for_timeout(3000)
        nb = await bands.count()
        check(nb > 0, "HiRISE crossing bands shown on the radargram", str(nb))
        if nb:
            # Fit-all so the crossing near Jezero is in view
            await bands.first.click()
            await pg.wait_for_timeout(400)
            items = pg.locator('[aria-label="HiRISE crossings"] [class*="w-56"] button')
            pick = await items.first.inner_text() if await items.count() else ""
            if await items.count():
                await items.first.click()
                await pg.wait_for_timeout(9000)
                ov = await pg.evaluate("(id) => !!window.__marslabViewer.entities.values.find(e => e.id.includes(id) && e.id.includes('OVERLAY'))", pick.strip().split()[-1])
                check(ov, "picking a crossing drapes that HiRISE image on the map", pick.strip())
        await pg.screenshot(path=OUT + "xref_bands.png")

        # 5. HiRISE viewer shows SHARAD lines; clicking one opens its radargram
        await pg.keyboard.press("Escape")  # close radargram
        await pg.wait_for_timeout(1000)
        insp = pg.locator('aside[aria-label="Inspector"]')
        if not await insp.count():
            box = await pg.locator("main").bounding_box()
            await pg.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            await pg.wait_for_timeout(6000)
        thumb = insp.get_by_role("button", name="full screen")
        if await thumb.count():
            await thumb.first.click()
            await pg.wait_for_function("() => { const i = document.querySelector('[role=dialog] img'); return i && i.naturalWidth > 0; }", timeout=40000)
            lines = pg.locator('[role=dialog] svg polyline[role=button]')
            try:
                await lines.first.wait_for(state="attached", timeout=30000)
            except Exception:
                pass
            nl = await lines.count()
            check(nl > 0, "HiRISE viewer draws SHARAD tracks crossing the image", str(nl))
            await pg.screenshot(path=OUT + "xref_viewer.png")
            if nl:
                bbx = await lines.first.bounding_box()
                # click on the line's middle by sampling points of the polyline
                pt = await lines.first.evaluate("""(el) => { const L = el.getTotalLength(); const p = el.getPointAtLength(L/2);
                  const m = el.getScreenCTM(); return {x: p.x*m.a + p.y*m.c + m.e, y: p.x*m.b + p.y*m.d + m.f}; }""")
                await pg.mouse.click(pt["x"], pt["y"])
                await pg.wait_for_timeout(9000)
                check(await pg.get_by_role("dialog").count() == 0 and await radar.count() > 0,
                      "clicking a track line opens its radargram (viewer closes)")
        else:
            check(False, "HiRISE thumbnail available", "no thumbnail")

        check(not errs, "no page errors", "; ".join(errs[:3]))
        await b.close()
    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")


asyncio.run(main())
