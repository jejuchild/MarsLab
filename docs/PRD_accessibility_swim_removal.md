# PRD: Accessibility Algorithm — SWIM 완전 제거

## Overview

현재 Ice Accessibility 알고리즘은 4개 sub-score로 구성:
- `ice_presence` (35%) — **SWIM consistency + landform bonus** ← SWIM 의존
- `ice_depth` (25%) — **SWIM depth products + TES TI + landform** ← SWIM 의존
- `excavation` (20%) — TES TI + DCI + slope ✅
- `landing` (20%) — MOLA elevation + slope + TRI ✅

**문제**: SWIM은 남이 가공한 데이터. 온보드에 쓸 수 없음.
**결정**: Ice Presence / Ice Depth sub-score 삭제. SWIM 완전 제거.

## New Architecture

```
Landform Model (HiRISE+MOLA) → 얼음 존재 여부 판단 (LDA/LVF/CCF)
Accessibility (TES+MOLA only) → 착륙·굴착 가능성 판단
Fusion                        → 둘 합쳐서 최종 ice prospecting score
```

### New Sub-scores (2개만)

| Sub-score | Weight | Inputs | 변경사항 |
|-----------|--------|--------|----------|
| `excavation` | 0.55 | TES TI + DCI + slope | 비중 ↑ |
| `landing` | 0.45 | MOLA elevation + slope + TRI | 비중 ↑ |

> Ice Presence는 landform model이 담당. Accessibility는 "갈 수 있는가 + 팔 수 있는가"만 판단.

### Removed

- `ice_presence` sub-score 삭제
- `ice_depth` sub-score 삭제
- SWIM GeoTIFF 로딩 전부 삭제 (consistency, 0-1m, 1-5m, 5m+)
- SWIM 관련 상수 삭제 (LANDFORM_BONUS, LANDFORM_DEPTH_WEIGHTS, LANDFORM_DEPTH_PRIOR)
- `swim_common.geotiff_loader` import 제거 (accessibility 파이프라인에서만)
- `swim_consistency` layer tile 제거

### Kept As-Is

- TES Thermal Inertia (.npy) loading
- MOLA GeoTIFF loading (elevation, slope, TRI)
- `compute_excavation()` logic
- `compute_landing()` logic
- `tile_renderer.py` (색상 맵 동일)
- `geotiff_loader.py` (MOLA 로딩용)

## File Changes

### 1. `backend/analysis/accessibility/algorithm.py`
- Remove: `LANDFORM_BONUS`, `LANDFORM_DEPTH_WEIGHTS`, `LANDFORM_DEPTH_PRIOR`
- Remove: `compute_ice_presence()`, `compute_ice_depth()`
- Remove: `swim_*` params from `compute_accessibility()` and `compute_accessibility_grid()`
- Update: `DEFAULT_WEIGHTS = {"excavation": 0.55, "landing": 0.45}`
- Update: `AccessibilityResult` — remove `ice_presence`, `ice_depth` fields
- Update: `layers_total` = 6 → 4 (TES TI, elevation, slope, TRI)
- Update: `inputs` dict — remove swim fields
- Simplify: `compute_accessibility_grid()` — remove swim/landform grid params

### 2. `backend/analysis/accessibility/pipeline.py`
- Remove: `from analysis.swim_common.geotiff_loader import ...`
- Remove: `self._swim_*` fields (4개)
- Remove: SWIM loading in `_ensure_loaded()`
- Remove: SWIM sampling in `query_point()`
- Update: `_extract_layers()` — return only TES TI + MOLA layers
- Update: `compute_accessibility()` call — no swim params

### 3. `backend/api/accessibility_router.py`
- Remove: `w_ice`, `w_depth` query params
- Remove: SWIM layer status from `/layers` endpoint
- Update: `/weights` descriptions — 2개만
- Update: explain prompt — no SWIM references
- Update: fallback explanation — no SWIM references

### 4. `hirise-api/routers/analyze.py`
- Remove: `ice_presence`, `ice_depth` from sub_scores dict
- Remove: `swim_*` from inputs dict
- Remove: SWIM sensor rows from HTML report
- Remove: Ice Presence / Ice Depth bars from HTML report

### 5. `hirise-api/core/explainer.py`
- Remove: `Ice Presence`, `Ice Depth` from sub-scores prompt
- Remove: `SWIM Consistency` from sensor data prompt

## API Changes

### `/api/accessibility/score` Response (before → after)
```json
// BEFORE
{
  "score": 0.42,
  "ice_presence": 0.65,
  "ice_depth": 0.35,
  "excavation": 0.48,
  "landing": 0.55,
  "inputs": { "swim_consistency": 0.3, "swim_0_1m": ..., ... }
}

// AFTER
{
  "score": 0.51,
  "excavation": 0.48,
  "landing": 0.55,
  "inputs": { "thermal_inertia": 245.0, "elevation": -2100, "slope": 3.2, "tri": 42.5 }
}
```

### `/api/accessibility/weights` Response (after)
```json
{
  "weights": { "excavation": 0.55, "landing": 0.45 },
  "description": {
    "excavation": "How easy to dig (thermal inertia + dust cover + slope)",
    "landing": "How safe for landing/traversal (elevation + slope + roughness)"
  }
}
```

## Non-Goals
- swim_common 패키지 자체는 건들지 않음 (다른 SWIM 모듈들이 씀)
- scoring_methodology.py의 normalize_swim은 별도 모듈이라 건들지 않음
- Frontend SWIM tile layers는 별도 작업
