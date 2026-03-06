# PRD: CRISM Mineral Classification + ISRU Accessibility 5-Score

## Overview

HiRISE landform classification에 **CRISM mineral classification**을 추가하고,
Ice Accessibility를 **ISRU Accessibility**로 확장 (2→5 sub-scores).

### 현재 구조
```
HiRISE Landform (LDA/LVF/CCF/OTHER)
+ Accessibility (excavation + landing) = 2 sub-scores
```

### 새 구조
```
HiRISE Landform (LDA/LVF/CCF/OTHER)
+ CRISM Mineral Classification (25 classes → 물관련 점수)
+ ISRU Accessibility (5 sub-scores)
```

## 1. CRISM Mineral Classification 통합

### 파이프라인 흐름
```
HiRISE product_id → lat/lon 추출
    ↓
ODE REST API로 근처 CRISM 관측 검색 (반경 ~50km)
    ↓
캐시 확인 (mineral_cnn_results/{obs_id}/)
  ├─ 있으면 → 바로 로드
  └─ 없으면 → acquire_and_classify() 실행 (다운로드+JCAT+CNN)
    ↓
mineral_map, confidence_map 로드
    ↓
광물별 통계 + 물관련 점수 계산
```

### CRISM 관측 검색 (ODE API)
```
GET https://oderest.rsl.wustl.edu/live2?
    target=mars&ihid=mro&iid=crism&pt=TRDR
    &westernlon={lon-0.5}&easternlon={lon+0.5}
    &minlat={lat-0.5}&maxlat={lat+0.5}
    &output=json
```
- HiRISE 좌표 기준 ±0.5° 범위 검색
- L-sensor TRR3 우선 (IR bands for CNN)
- 여러 관측 중 가장 가까운 것 선택

### 결과 표시
- CRISM quickview (원본) + mineral map (분류) side-by-side
- Legend: 감지된 광물별 색상 + 이름 + 비율
- HiRISE heatmap과 동일 레이아웃

## 2. ISRU Accessibility (5 Sub-scores)

### New Structure

| # | Sub-score | Weight | Source | 점수 기준 |
|---|-----------|--------|--------|----------|
| 1 | **Ice-Related Landform** | 0.25 | HiRISE 분류 | LDA→1.0, LVF→0.8, CCF→0.6, OTHER→0 × confidence |
| 2 | **Water-Related Mineral** | 0.20 | CRISM CNN | poly-sulfate→1.0, smectite/mono-sulfate→0.5, unrelated→0 × confidence |
| 3 | **Surface Ice Signal** | 0.15 | CRISM CNN | H2O Ice 분류 있으면 → high score, 없으면 → 0 |
| 4 | **Excavation** | 0.20 | TES TI + slope | 기존 그대로 |
| 5 | **Landing** | 0.20 | MOLA elev + slope + TRI | 기존 그대로 |

### Sub-score 1: Ice-Related Landform (HiRISE)

```python
LANDFORM_SCORE = {
    "LDA": 1.0,   # Lobate Debris Apron — strongest ice indicator
    "LVF": 0.8,   # Lineated Valley Fill
    "CCF": 0.6,   # Concentric Crater Fill
    "OTHER": 0.0,
    "Uncertain": 0.0,
}
# score = LANDFORM_SCORE[class] × confidence
```

### Sub-score 2: Water-Related Mineral (CRISM)

광물 분류 결과에서 가장 높은 water-related signal 사용.

```python
# Tier 1 — 강한 수화 증거 (score 1.0)
TIER_1 = {19}  # Polyhydrated sulfate

# Tier 2 — 중간 수화 증거 (score 0.6)
TIER_2 = {26, 3, 16}  # Monohydrated sulfate, Gypsum, Bassanite

# Tier 3 — 약한 수화 증거 (score 0.4)
TIER_3 = {6, 7, 14, 18, 15, 31, 38, 23, 27}
# Fe/Mg/Al smectite, Kaolinite, Chlorite, Chlorite-smectite, Illite, Hydrated silica

# Tier 0 — 점수 없음
TIER_0 = {100, 1, 8, 9, 10, 11, 12, 17, 25, 29, 4}
# Water-unrelated, CO2 Ice, Prehnite, Jarosite, Serpentine, Alunite, etc.
```

계산:
- CRISM 관측의 전체 픽셀 중 각 tier 비율 계산
- `score = max(tier1_frac × 1.0, tier2_frac × 0.6, tier3_frac × 0.4)`
- confidence 가중: 해당 광물 평균 confidence 곱함

### Sub-score 3: Surface Ice Signal (CRISM)

```python
# H2O Ice (class 2) 감지 여부
h2o_ice_fraction = (mineral_map == 2).sum() / valid_pixels
if h2o_ice_fraction > 0.01:  # 1% 이상이면 signal
    score = min(1.0, h2o_ice_fraction * 10)  # 10%면 만점
    score *= avg_confidence_of_h2o_ice_pixels
else:
    score = 0.0
```

### Sub-score 4 & 5: Excavation + Landing (기존 그대로)

변경 없음.

## 3. HTML 리포트 변경

### 현재
```
┌──────────────────┐ ┌──────────────────┐
│ Landform         │ │ Ice Accessibility│
│ Classification   │ │ (2 sub-scores)   │
└──────────────────┘ └──────────────────┘
┌──────────────────────────────────────────┐
│ 🧠 Analysis (LLM explanation)            │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ 🗺️ Original + HiRISE Heatmap             │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ 📡 Sensor Data                            │
└──────────────────────────────────────────┘
```

### 새 레이아웃
```
┌──────────────────┐ ┌──────────────────┐
│ Landform         │ │ ISRU             │
│ Classification   │ │ Accessibility    │
│ (HiRISE)         │ │ (5 sub-scores)   │
└──────────────────┘ └──────────────────┘
┌──────────────────┐ ┌──────────────────┐
│ Mineral          │ │ CRISM Mineral    │
│ Classification   │ │ Statistics       │
│ (CRISM)          │ │ (top minerals)   │
└──────────────────┘ └──────────────────┘
┌──────────────────────────────────────────┐
│ 🧠 Combined Analysis (LLM explanation)   │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ 🗺️ HiRISE: Original + Heatmap + Legend   │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ 🔬 CRISM: Quickview + Mineral Map + Legend│
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ 📡 Sensor Data                            │
└──────────────────────────────────────────┘
```

### Legend (HiRISE + CRISM)

**HiRISE Legend:**
```
🟦 LDA (Lobate Debris Apron)
🟩 LVF (Lineated Valley Fill)
🟨 CCF (Concentric Crater Fill)
⬜ OTHER
⬛ Uncertain
```

**CRISM Legend (감지된 광물만 표시):**
```
🔵 H2O Ice
🟣 Polyhydrated Sulfate
🟪 Monohydrated Sulfate
🟡 Gypsum
🟤 Fe Smectite
🟢 Mg Smectite
⚫ Water-unrelated
... (detected minerals only)
```

색상은 기존 `_MINERAL_COLORS` dict 그대로 사용.

## 4. File Changes

### New Files

| File | Purpose |
|------|---------|
| `hirise-api/core/crism_bridge.py` | CRISM 관측 검색 + 분류 결과 로드/실행 |

### Modified Files

| File | Changes |
|------|---------|
| `backend/analysis/accessibility/algorithm.py` | 5 sub-scores, ISRU weights, landform/mineral/ice scoring |
| `backend/analysis/accessibility/pipeline.py` | query_point에 landform + CRISM 결과 통합 |
| `backend/api/accessibility_router.py` | 5 weight params, descriptions 업데이트 |
| `hirise-api/routers/analyze.py` | CRISM 파이프라인 추가, HTML 리포트 확장, legend |
| `hirise-api/core/explainer.py` | CRISM 광물 정보 포함 prompt 업데이트 |
| `hirise-api/core/accessibility.py` | query에 CRISM 결과 전달 |

### NOT Modified (이미 완성)

| File | Reason |
|------|--------|
| `backend/api/mineral_cnn/pipeline.py` | 이미 완성된 CNN 파이프라인 그대로 사용 |
| `backend/api/mineral_cnn/model.py` | 이미 학습된 25-class 모델 그대로 사용 |
| `backend/api/mineral_cnn/acquire.py` | 이미 완성된 ODE 검색+다운로드 그대로 사용 |
| `backend/api/mineral_cnn/constants.py` | CLASS_NAME, _MINERAL_COLORS 그대로 사용 |

## 5. API Response Changes

### `/api/accessibility/score` (before → after)
```json
// BEFORE
{
  "score": 0.79, "excavation": 0.73, "landing": 0.86,
  "weights": {"excavation": 0.55, "landing": 0.45}
}

// AFTER
{
  "score": 0.62,
  "ice_landform": 0.85,
  "water_mineral": 0.45,
  "surface_ice": 0.0,
  "excavation": 0.73,
  "landing": 0.86,
  "weights": {
    "ice_landform": 0.25, "water_mineral": 0.20,
    "surface_ice": 0.15, "excavation": 0.20, "landing": 0.20
  },
  "crism_obs_id": "frt00009e0b_07",
  "crism_minerals": {"H2O Ice": 2.3, "Al smectite 2": 15.1, ...}
}
```

### HiRISE API HTML Report
- "Ice Accessibility" → "ISRU Accessibility"
- 5개 sub-score bars
- CRISM section 추가 (quickview + mineral map + legend)
- HiRISE heatmap에도 legend 추가
- Footer: "MarsLandformNet v4b-FiLM + MineralCNN v7 + ISRU Accessibility"

## 6. Execution Plan

### Phase 1: crism_bridge.py (CRISM 관측 검색 + 결과 로드)
1. ODE API로 lat/lon 기반 CRISM 검색
2. 캐시 확인 → 없으면 acquire_and_classify
3. mineral_map + confidence_map에서 water-related scores 계산

### Phase 2: algorithm.py 5-subscore 재설계
1. `compute_ice_landform()` — HiRISE 분류 기반
2. `compute_water_mineral()` — CRISM 광물 tier 기반
3. `compute_surface_ice()` — CRISM H2O Ice 감지 기반
4. `compute_excavation()` — 기존 유지
5. `compute_landing()` — 기존 유지
6. `DEFAULT_WEIGHTS` 5개로 업데이트

### Phase 3: pipeline.py + router 업데이트
1. query_point에 landform + crism_result 인자 추가
2. API params 5개 weight로 확장

### Phase 4: hirise-api analyze.py 리포트 확장
1. CRISM classification 단계 추가
2. CRISM quickview + mineral map 이미지 serving
3. HiRISE/CRISM legend HTML 생성
4. "Ice Accessibility" → "ISRU Accessibility"

### Phase 5: explainer.py LLM prompt 업데이트
1. CRISM 광물 정보 포함
2. 5개 sub-score 설명

## 7. Constraints

- mineral_cnn 코드 건들지 않음 (이미 완성)
- CRISM 다운로드+분류는 시간이 걸림 — analyze endpoint에서 timeout 고려
- CRISM 관측이 없는 지역도 있음 — 없으면 3개 sub-score(landform + excavation + landing)만 사용
- confidence threshold: CNN default 0.95 그대로 유지
