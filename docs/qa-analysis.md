# MarsLab QA Analysis — Post Phase 3

**Date**: 2026-04-09
**Scope**: Code quality, dead code, architecture health after Phases 1-3
**Tag analyzed**: `phase-3-complete` (`b8aff28`)

---

## Executive Summary

1. **One critical bug**: `backend/api/report_orchestrator.py:52` has an unprotected `from .mars_climate import …` for a module deleted in Phase 1. Crashes the app if this file is loaded by anything.
2. **~3,000 LOC of dead frontend code** in 5 orphan components (TimelineNavigator, Subsurface3DViewer, ComparisonMode, ComparisonTray, LandformClassCard) — zero references, safe to delete now.
3. **Phase 2 hooks half-adopted**: only `useMapNavigation` is actually used in MainPage. The other 4 hooks (`useFootprintLayers`, `useInspectorContext`, `useActiveOverlays`, `useCatalogSearch`) are imported as `void` placeholders. MainPage is still 2,133 LOC with 56 useState calls because of this.

---

## Code Metrics

| Area | Files | LOC |
|---|---|---|
| Frontend (`frontend/src/`) | 152 | 47,471 |
| Backend (`backend/`) | — | 87,445 |
| Backend API routers (registered) | 33 | — |
| Backend API .py files (total) | 49 | — |

### Top 10 frontend files

| # | File | LOC | Notes |
|---|---|---|---|
| 1 | `pages/DataDownloadPage.tsx` | 3,068 | Acceptable — complex download UI |
| 2 | `pages/MainPage.tsx` | **2,133** | Hotspot — Phase 2 hooks half-adopted |
| 3 | `components/Inspector.tsx` | 2,102 | **Legacy fallback** (only `?v=legacy`) |
| 4 | `components/SharadHiresInspector.tsx` | 2,008 | Used by lane "Open inspector" |
| 5 | `hooks/useMapViewer.ts` | 1,308 | Cesium init — fine for its scope |
| 6 | `hooks/useOverlays.ts` | 1,271 | Cesium overlay rendering — fine |
| 7 | `components/MeasurementTools.tsx` | 1,195 | Used |
| 8 | `components/MapView.tsx` | 1,045 | Used |
| 9 | `components/HiRiseDTM3DViewer.tsx` | 1,014 | Lazy-loaded |
| 10 | `api/search.ts` | 979 | Used |

### Top 5 backend files

| # | File | LOC | Notes |
|---|---|---|---|
| 1 | `api/agent_tasks.py` | 4,487 | MARVIS tools |
| 2 | `api/agent_orchestrator.py` | 3,360 | MARVIS chat orchestrator |
| 3 | `api/report_orchestrator.py` | 1,963 | **⚠ broken import** |
| 4 | `api/ode_client.py` | 1,729 | ODE search helper |
| 5 | `app.py` | 1,479 | FastAPI bootstrap |

### Backend data dir (5.8 GB)

| Dir | Size | Status |
|---|---|---|
| `hirise_landforms/` | 2.7 GB | ✅ Active |
| `swim/` | 986 MB | ⚠️ **Orphaned** (Phase 1 cut) |
| `themis_irbtr_arcadia/` | 958 MB | ❓ Verify |
| `mola_derived/` | 190 MB | ✅ Active (terrain_router) |
| `thermal_pinn/` | 125 MB | ⚠️ **Orphaned** (Phase 1 cut) |
| `tes/`, `tes_thermal_inertia.npy` | 221 MB | ✅ Active |
| `rag_vectordb/` | 44 MB | ✅ Active (MARVIS) |

→ **~1.1 GB safely removable** (swim + thermal_pinn).

---

## 🔴 Critical Findings

### C1. Broken import in `report_orchestrator.py`
**File**: `backend/api/report_orchestrator.py:52`

```python
from .mars_climate import climate_analysis_for_region
```

`mars_climate.py` was deleted in Phase 1. The import is at module top — **app crashes on startup if anything imports this file**. Currently it's NOT imported by `app.py` (we removed `report_router`), so the bug is dormant. But any future code that imports `report_orchestrator` will hit it.

**Fix**: Either delete `report_orchestrator.py` entirely (it served the cut Report feature) or stub the import.

**Effort**: 5 minutes.

---

### C2. 5 orphaned frontend components (~3,000 LOC dead)

| File | LOC | Imports |
|---|---|---|
| `components/TimelineNavigator.tsx` | 928 | 0 |
| `components/Subsurface3DViewer.tsx` | 847 | 0 |
| `components/ComparisonMode.tsx` | ? | 0 |
| `components/ComparisonTray.tsx` | ? | 0 |
| `components/LandformClassCard.tsx` | ? | 0 |

These were referenced from features cut in Phase 1 but the files survived. Zero static or lazy imports across the codebase.

**Fix**: `rm` + verify build.

**Effort**: 15 minutes including verification.

---

## 🟠 High-Priority Findings

### H1. MainPage.tsx — Phase 2 hooks not adopted

**Current state**:
- 56 `useState` calls
- 58 `useRef` / `useMemo` / `useCallback` calls
- 9-variant `analysisMode` enum (some entries are stale Phase 1 leftovers)
- Imports 4 of the 5 Phase 2 hooks as `void` (placeholders, not actually called)

**Adoption matrix**:

| Phase 2 hook | Status in MainPage | Blocker |
|---|---|---|
| `useMapNavigation` | ✅ adopted | — |
| `useFootprintLayers` | ❌ `void` | 5 useState calls (`instrumentVisibility`, `loadFootprintsTrigger`, `highResOnly`, `footprintsLoading`, `footprintCounts`) entangled with overlay logic |
| `useInspectorContext` | ❌ `void` | `selected` + `recentProducts` state ownership unclear (MainPage sets it from many places) |
| `useActiveOverlays` | ❌ `void` | Map<productId, overlay> state has callback dependencies on multiple effects |
| `useCatalogSearch` | ❌ `void` (used in TopBar instead) | Not needed in MainPage — should remove the `void` import |

**Fix path** (incremental):
1. Remove `useCatalogSearch` from MainPage `void` imports (it's only used by TopBar/SearchBar) — 5 min
2. Adopt `useInspectorContext` (medium risk, ~2 hours)
3. Adopt `useFootprintLayers` (highest value, ~3 hours)
4. Adopt `useActiveOverlays` (last, ~2 hours)

**Effort total**: 1 day to bring MainPage to ~1,500 LOC.

---

### H2. Backend orphaned data (~1.1 GB)

```bash
rm -rf backend/data/swim          # 986 MB
rm -rf backend/data/thermal_pinn  # 125 MB
```

**Verify first**: `grep -r "data/swim\|data/thermal_pinn" backend/ --include="*.py"` returns nothing.

**Effort**: 5 minutes.

---

### H3. Inspector2 lanes are "thin" — `Inspector.tsx` (2,102 LOC) still needed

Current lane bodies show only:
- Quickview thumbnail
- Variant toggle
- Product picker
- "Open inspector →" button → falls back to legacy `Inspector.tsx`

**Missing from each lane** (still requires legacy):
| Lane | Missing |
|---|---|
| HiRISE | Pixel statistics, landform classification, DTM 3D viewer launcher |
| CRISM | Spectrum plot, RGB band picker, dust assessment, mineral CNN result |
| SHARAD | Radargram viewer, depth picker, regolith thickness, attenuation |
| CTX | Mostly OK (just imagery) |

**Implication**: Inspector.tsx cannot be deleted without inlining ~6 sub-panels into Inspector2 lanes. Each migration is a real piece of work.

**Recommended path**: Don't delete Inspector.tsx yet. Pick the most-used lane (CRISM or HiRISE) and migrate one sub-panel as a proof-of-concept. Defer full deletion to a separate iteration.

**Effort**: ~1 day per sub-panel migration.

---

## 🟡 Medium-Priority Findings

### M1. `/api/inspector/at-point` antimeridian bug

**File**: `backend/api/inspector_router.py:139` `_bbox_intersects()`

The fast-path bbox check has a comment admitting longitude wrap is not handled. Features whose bbox crosses ±180° (rare but exists in polar SHARAD tracks) can produce false negatives.

**Impact**: Low — affects only a small number of products near the antimeridian. Not a crash.

**Fix**: 30 minutes — implement wrap-aware longitude distance check.

---

### M2. `/api/inspector/at-point` cache thread safety

The endpoint reads `_geojson_cache` from `app.py` via lazy import. Under FastAPI's async model, multiple concurrent requests can read the dict simultaneously. The cache is only written at startup and via `refresh_geojson_cache()` (rare). Risk is low but not zero.

**Fix**: Wrap `_geojson_cache` access in a `threading.RLock` (~15 min) or document that the cache is read-only post-startup.

---

### M3. `analysisMode` enum has stale variants

`MainPage.tsx:245` declares:
```typescript
type AnalysisMode = "slope" | "hirise_dtm_3d" | "line" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column" | null;
```

Most of these (`regolith`, `stratigraphy`, `attenuation`, `mineral_sequence`, `strat_column`) reference features that were partially cut. The render branches in MainPage still exist but the LayerPanel "Analysis Tools" section was reduced — these modes can still be triggered from elsewhere (Inspector, CraterDetect callbacks) so they're not fully dead.

**Recommendation**: Audit each remaining variant. If any have no entry point, remove. Document the survivors.

**Effort**: 1 hour audit + cleanup.

---

### M4. No tests for `/api/inspector/at-point`

The Phase 3 endpoint has no smoke tests. Recommend 5 cases:
- Equator point
- Antimeridian wrap
- Out-of-bounds radius (should 422)
- Empty result
- Concurrent requests

**Effort**: 1 hour.

---

## 🔵 Low-Priority Findings

### L1. 7 `eslint-disable @typescript-eslint/no-explicit-any` directives
All in utility files (FootprintManager, dtmHover, overlapFilter, perfMonitor, RegolithPanel). Acceptable — these handle GeoJSON / browser-global / Recharts payload shapes that are awkward to type strictly. Document why and move on.

### L2. 48 backend endpoints without `response_model`
Bulk update opportunity. No correctness impact, but client-side validation gets weaker.

### L3. ~345 backend functions lacking return type hints
~53% of functions. Mechanical bulk fix opportunity for IDE/type-check quality.

### L4. Backend `slowapi` rate limiting is inconsistent
Applied to a few endpoints but not most. Decide on a global policy.

### L5. `useUrlState.ts` — `flyTo` legacy param compat
`useUrlState.ts:80-86` still maps legacy `flyTo` → `product`. Consider whether enough time has passed to drop it.

---

## Recommended Action Plan

### Quick wins (today, ~30 min total)
1. **Fix `report_orchestrator.py:52`** broken import (5 min) — or delete the file entirely
2. **Delete 5 orphan frontend components** (15 min)
3. **Remove unused `void useCatalogSearch` import** from MainPage (2 min)
4. **`rm -rf backend/data/{swim,thermal_pinn}`** after confirming no usage (10 min)

### Medium effort (1-2 days)
5. **Adopt useInspectorContext + useFootprintLayers** in MainPage → cuts ~300 LOC, removes `void` placeholders
6. **Audit `analysisMode` enum** — remove variants with no entry point
7. **Fix antimeridian bug** in `/api/inspector/at-point`
8. **Add 5 smoke tests** for `/api/inspector/at-point`

### Bigger lifts (multi-day)
9. **Migrate first lane sub-panel** (e.g. CRISM Spectrum into CrismLane) — proof-of-concept for Inspector.tsx removal
10. **Bulk add `response_model`** to backend endpoints (~4 hours mechanical work)

---

## Snapshot Summary

| Health indicator | Status |
|---|---|
| TypeScript (`tsc -b`) | ✅ 0 errors |
| ESLint | ✅ 0 errors / 0 warnings |
| Tests | ✅ 8/8 frontend |
| Build | ✅ vite build success |
| Backend `import app` | ✅ OK |
| Critical broken imports | ⚠️ 1 (dormant) |
| Dead code (frontend) | ⚠️ ~3,000 LOC in 5 files |
| Dead code (backend) | ⚠️ ~1.1 GB in 2 data dirs |
| Phase 2 hook adoption | 🟡 1 of 5 actually used |
| Inspector2 lane completeness | 🟡 thin — falls back to legacy |
| `/api/inspector/at-point` quality | 🟢 good, minor bugs |

**Overall verdict**: Refactoring achieved its main goals. Core flows work, build is green, no warnings. Cleanup work remaining is mostly **mechanical** (delete dead files, finish hook adoption). The biggest open question is **how aggressively to push Inspector2 lane sub-panels** — this is real product work, not refactoring.
