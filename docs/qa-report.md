# QA Report — 2026-04-09

## Test Results

| Check | Status | Details |
|-------|--------|---------|
| TypeScript compilation | PASS | Zero errors (`tsc --noEmit`) |
| Vite build | PASS | Builds in ~1m42s, all chunks generated |
| ESLint errors | PASS | 0 errors (was 7, all fixed) |
| ESLint warnings | 77 warnings | Mostly `@typescript-eslint/no-explicit-any` (6) and `react-hooks/exhaustive-deps` (dozens) |
| Unit tests | PASS | 8/8 tests pass (ErrorBoundary suite) |
| `target="_blank"` safety | PASS | All external links use `rel="noopener noreferrer"` |
| XSS (dangerouslySetInnerHTML) | PASS | All 6 usages sanitize via DOMPurify |
| API key exposure | PASS | No secrets found in frontend source |

## Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| Main bundle (index.js) | 583 KB (154 KB gzip) | Moderate — could benefit from more code-splitting |
| Cesium vendor chunk | 5.5 MB (1.4 MB gzip) | Expected for Cesium; already isolated |
| Three.js vendor chunk | 1.1 MB (299 KB gzip) | Expected; lazy-loaded via HiRiseDTM3DViewer |
| Recharts vendor chunk | 387 KB (115 KB gzip) | OK; separated from main bundle |
| CSS | 84 KB + 24 KB (Cesium) | Normal |
| Total dist size | ~5.8 GB (includes Cesium assets) | Normal for Cesium (includes terrain/imagery tiles) |
| Lazy loading | Active | 9 secondary pages + heavy panels lazy-loaded |
| Module preload | Disabled | Correctly prevents 5.4 MB preload |

**Recommendations:**
- Consider dynamic-importing `MapView.tsx` components (582 KB main bundle includes map logic)
- Recharts could be lazy-loaded per-panel instead of bundled

## Security

| Check | Status |
|-------|--------|
| `dangerouslySetInnerHTML` sanitized | PASS (DOMPurify in all 6 files) |
| `eval()` / `document.write()` | PASS (none found) |
| `target="_blank"` with `rel="noopener noreferrer"` | PASS |
| API keys in source | PASS (none exposed) |
| CSP header | N/A (set at server level) |
| SRI on external scripts | N/A (no CDN scripts, all bundled) |

## SEO

| Check | Status |
|-------|--------|
| `<title>` | PASS — "MarsLab" |
| `<meta description>` | PASS — "Mars Surface Exploration & Analysis Platform" |
| `<meta viewport>` | PASS |
| Favicon | PASS — SVG favicon + apple-touch-icon |
| PWA manifest | PASS — `manifest.json` linked |
| Open Graph / Twitter cards | MISSING |
| Canonical URL | MISSING |
| Structured data (JSON-LD) | MISSING |
| Heading hierarchy | PASS — `h1` present on all pages |

## Accessibility (WCAG 2.1 AA)

| Check | Status |
|-------|--------|
| `aria-label` / `role` usage | PARTIAL — 106 occurrences across 25 files |
| `<label>` / `htmlFor` | PARTIAL — 69 occurrences across 25 files |
| Skip navigation link | MISSING |
| High contrast mode | PRESENT — `useHighContrastMode` hook |
| Keyboard shortcuts | PRESENT — `KeyboardShortcuts` component |
| Semantic HTML | GOOD — proper use of `<nav>`, `<main>`, `<header>` |
| Color contrast | Not audited (needs browser-based testing) |

## Applied Fixes (this session)

1. **`AccessibilityPanel.tsx:312`** — Removed unnecessary `String()` wrapper on string value
2. **`MapView.tsx:732`** — Prefixed unused `ctxMosaicOpacity` destructured prop with `_`
3. **`LayerPanel.tsx:51-52`** — Prefixed unused `ctxMosaicOpacity` and `onCtxMosaicOpacityChange` with `_`
4. **`QuickviewImage.tsx:48-63`** — Moved recursive `tryLoadImage` inside `useEffect` to fix self-referencing `useCallback` before declaration
5. **`MastcamLabelPage.tsx:78,268`** — Reordered `draw()` function declarations before `useEffect` calls in both `HiRISEMapCanvas` and `MastcamPreview` components

**Result:** ESLint errors reduced from 7 to 0.

## Remaining Issues (by severity)

### Medium
- **77 ESLint warnings** — Mostly `no-explicit-any` (6 in utils) and missing `exhaustive-deps` in hooks (many). These are non-blocking but should be cleaned up over time.
- **No OG/Twitter card meta tags** — Social sharing will show generic previews
- **No skip-navigation link** — Accessibility gap for keyboard/screen-reader users
- **Test coverage is minimal** — Only 1 test file (ErrorBoundary) for 52+ components

### Low
- **Large main bundle** (583 KB) — Could code-split MapView internals
- **No canonical URL** — Minor SEO concern for a SPA
- **No structured data** — Not critical for a research tool

## Recommendations for Next Steps

1. Add `<meta property="og:*">` and `<meta name="twitter:*">` tags to `index.html`
2. Add a skip-navigation link in `AppShell.tsx`
3. Gradually add tests for critical paths (MapView interactions, API layer, Inspector)
4. Address `exhaustive-deps` warnings in hooks (risk of stale closures)
5. Consider `/web:deploy` for production deployment
