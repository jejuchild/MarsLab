# MarsLab 정체성 & 리팩토링 계획

**작성일**: 2026-04-09
**Git 세이프티**: `pre-refactor-2026-04-09` tag, `legacy-full` branch

---

## 정체성 (한 줄)

> **"MarsLab은 'Jezero crater'처럼 장소 이름을 검색하거나 좌표를 입력하면, 그 지점의 SHARAD·CRISM·HiRISE·CTX 데이터를 한 화면에서 즉시 비교·측정·해석할 수 있게 해주는, Mars 연구자와 학생을 위한 데스크탑 도구다. AI는 사이드킥으로 옆에서 도와준다."**

---

## 타겟 사용자

| 우선순위 | 사용자 | 특징 |
|---|---|---|
| 1차 | Mars 연구자 | 행성과학 PhD/포닥/교수, CRISM·SHARAD·HiRISE 데이터 일상적으로 다룸 |
| 2차 | 학생/입문자 | 대학원생, 행성과학 수업 듣는 학부생 |

**제약**: 전문가 수준 데이터를 다루되, **학생도 5분 내 쓸 수 있는 UX**여야 함.

---

## 핵심 사용자 플로우 (5분 안에)

1. **장소 검색** (1순위) — "Jezero crater" 같은 지명 입력 → 자동 fly-to
2. **좌표 입력** (2순위) — lat/lon 직접 입력
3. **지도 클릭** (3순위) — 탐색하다가 원하는 지점 클릭
4. → 해당 지점의 SHARAD/CRISM/HiRISE/CTX 데이터가 Inspector에 즉시 표시
5. → 필요한 instrument lane에서 심화 분석

---

## 5가지 설계 원칙

1. **장소/좌표가 1순위** — 카탈로그 브라우징은 보조
2. **SHARAD/CRISM/HiRISE/CTX가 주인공** — 그 외 instrument는 보조 또는 cut
3. **통합 비교가 핵심** — 같은 좌표의 데이터가 한 화면에, 따로 패널 ❌
4. **해석까지 OK, 단 instrument에 직결된 것만** — 광물 분류/서브서피스 해석 OK, 로버 경로 계획 ❌
5. **AI는 사이드킥** — 메인 UI는 데이터, AI는 floating chatbot (MARVIS)

---

## 아키텍처: 4-Lane Inspector

```
┌─────────────────────────────────────────────────────────┐
│  TopBar: [ 🔍 Search "Jezero crater" or lat,lon ]       │
├─────────┬───────────────────────────────────┬───────────┤
│         │                                   │           │
│ Layer   │         Map (Cesium)              │ Inspector │
│ Panel   │                                   │  ┌─────┐  │
│         │                                   │  │SHARAD│ │
│ ☑ SHARAD│         [ Clicked point ]         │  ├─────┤  │
│ ☑ CRISM │                                   │  │CRISM│  │
│ ☑ HiRISE│                                   │  ├─────┤  │
│ ☑ CTX   │                                   │  │HiRISE│ │
│         │                                   │  ├─────┤  │
│         │                                   │  │ CTX │  │
│         │                                   │  ├─────┤  │
│         │                                   │  │Cross│  │
│         │                                   │  └─────┘  │
│         │                                   │           │
└─────────┴───────────────────────────────────┴───────────┘
                                                💬 MARVIS
```

### 4 Lanes (Inspector 내부 탭)

#### 🔴 SHARAD Lane (서브서피스)
- Radargram viewer (`SharadHiresInspector`)
- 3D subsurface (`Subsurface3DViewer`)
- Regolith thickness (통합)
- Attenuation analysis (통합)
- SWIM ice detection (SHARAD 결과)
- Variants: [Standard RDR | Hi-res]

#### 🟡 CRISM Lane (분광·광물)
- Spectrum plot (`CRISMSpectrumTab`)
- Band ratio / RGB composite (`CRISMBandsTab`)
- Spectral comparison (`SpectralComparison`)
- Band math (`BandRatioCalculator`)
- Mineral classification (TRR3 + CNN)
- Variants: [Standard | TRR3]

#### 🟢 HiRISE Lane (고해상도 이미지·지형)
- High-res image viewer (`HiResImageViewer`)
- DTM 3D viewer (`HiRiseDTM3DViewer`)
- Slope analysis (`SlopeAnalysis`)
- Line profile (`LineProfile`)
- Pixel stats (`HiRISEPixelTab`)
- Landform classification (`HiriseLandformPanel`)
- Crater detection (`CraterDetectPanel`)
- Variants: [Image | DTM]

#### 🔵 CTX Lane (광역 컨텍스트)
- CTX mosaic viewer
- 광역 컨텍스트 이미지

#### ⚪ Cross (Inspector 하단 "통합 해석" 섹션)
- Stratigraphy (`StratigraphyPanel` + `StratColumnPanel`)
- Mineral sequence (`MineralSequencePanel`)
- Temporal comparison (`TemporalComparison`)
- Measurement tools (거리/면적/단면)
- Field notes

---

## Scope 결정

### ❌ CUT (정체성 밖)

| 항목 | 이유 |
|---|---|
| `DailyDiscussionsPage` `/discussions` | 게시판 — 데이터 도구 아님 |
| `MastcamPanoPage` `/mastcam` | 4 lane 외, 지도와 분리됨 |
| `MastcamLabelPage` `/mastcam-label` | 동상 |
| `PathfinderPanel` (40KB) | 로버 경로 = 미션 플래너 영역 |
| `AgenticPanel` (65KB) | AI는 사이드킥으로 격하 |
| `GuidedWorkflows` (36KB) | 단순 UX면 가이드 불필요 |
| `ReportPanel` (52KB) | "빠른 분석"과 반대 |
| `RegionDashboard` (32KB) | "한 좌표" 동선과 충돌 |
| `RegionStatsPanel` (18KB) | "한 좌표" 동선과 충돌 (폴리곤 통계) |
| `SwimIcePanel` + 6 SWIM methods (`SwimMethodLayer`, `IceHub`, `IceConsistencyLegend`) | 연구 level 도구, "빠르게" 원칙과 충돌 |
| `AccessibilityPanel` | 미션 플래너용 |
| `AiAnalysisPanel` | MARVIS chat로 일원화 |
| `backend/agent/` legacy | CLAUDE.md에서 "old" |
| Backend: `accessibility_router`, `pathfinder_router`, `thermal_pinn`, `neural_climate_router`, `mars_climate.py`, `multi_report_router`, `report_router`, `agentic_router`, `swim_router`, `swim_ice_router` | 프론트 cut에 따라 백엔드도 제거 |
| 모바일 `BottomSheet` 복잡 분기 | 데스크탑 우선으로 단순화 |

### ✅ KEEP (정체성 밖이지만 남김 — 사용자 요청)

| 항목 | 비고 |
|---|---|
| `MarsNewsPage` `/news` (**Research 통합**) | 단일 `/news` 라우트 안에 News 탭 + Research 탭 |
| `FeatureSuggestionsPage` `/suggestions` | 사용자 요청 |
| `EasterEggs.tsx` (Curiosity, Olympus, Watney, Terraform), `SpaceGame.tsx` | 사용자 요청 — SearchBar에서 직접 trigger 유지 |

### ✅ KEEP (정체성 직결)

#### Instrument 관련
- 4 lane 모든 컴포넌트 (위 4-Lane 섹션 참조)
- `Inspector` + `InspectorPanel` + `InspectorHeader`
- `MapView` (Cesium)
- `LayerPanel` — **4개 instrument로 간소화** (SHARAD/CRISM/HiRISE/CTX)
- `FootprintManager`

#### 데이터 인프라
- `DataDownloadPage` `/download`
- `DataUploadPage` `/upload`
- `CustomDatasets` hook + API
- 관련 backend routers: `search_router`, `crism_router`, `ctx_tile_router`, `sharad_highres_router`, `terrain_router`, `hirise_landforms_router`, `swim_router`, `stratigraphy_router`, `mineral_sequence_router`, `epsilon_router`, `attenuation_router`, `regolith_router`, `crism_spectral`, `mineral_cnn`, `marvis_chat`

#### 공용 도구
- `MeasurementTools`
- `FieldNoteModal` + `useFieldNotes`
- `SpectralComparison`
- `TemporalComparison`
- `CopilotFab` (MARVIS chat)
- `CommandPalette`
- `KeyboardShortcuts`
- `OnboardingTour`

### 🔄 REORGANIZE (유지하되 재배치)

| 현재 | 새 구조 | 이유 |
|---|---|---|
| `MainPage.tsx` 2,444줄, 161 hooks | 5개 custom hook + lane별 컴포넌트 | 모놀리스 분해 |
| `analysisMode` union type (16+ modes) | 4 lane × 탭 구조 | "어느 instrument인가"가 1차 분류 |
| `LayerPanel` 7개 instrument 토글 | 4개 (SHARAD/CRISM/HiRISE/CTX) | 정체성 강화 |
| `TopBar` 검색 (product_id, 좌표, 자연어, 이스터에그 multi-mode) | 장소/좌표 중심 + 이스터에그는 trigger만 유지 | 검색 단순화 |
| 모바일 `BottomSheet` 양방향 | 모바일은 read-only fallback | 데스크탑 우선 결정 |

---

## 예상 효과 (수치)

| 지표 | Before | After (목표) |
|---|---|---|
| 분석 패널 | 17개 + 5개 (Agentic 등) | 4 lane + Cross 섹션 |
| `MainPage.tsx` | 2,444줄 | ~800줄 |
| `analysisMode` union | 16+ modes | 4 lanes (SHARAD/CRISM/HiRISE/CTX) |
| Instrument 토글 | 7개 | 4개 |
| 백엔드 라우터 | 50개 | 약 30개 |
| 프론트 코드 | ~40,000줄 | ~25,000줄 (목표) |
| 사용자 진입 장벽 | 높음 (길을 잃음) | 낮음 (5분 flow 명확) |

---

## 롤백 방법

```bash
# 현재 상태로 완전 롤백
git reset --hard pre-refactor-2026-04-09

# 또는 legacy 브랜치로 전환 (비파괴적)
git checkout legacy-full

# 특정 파일만 legacy에서 복원
git checkout legacy-full -- <file-path>
```

---

## 다음 단계

1. **`/sc:design`** — 4-lane 아키텍처의 상세 설계
   - 새 `Inspector` 컴포넌트 트리
   - State 분해 (5개 custom hook)
   - Router 재설계
   - Data flow 다이어그램
2. **`/sc:workflow`** — 단계별 리팩토링 작업 분해
   - Cut 작업 (삭제할 파일 목록 + 영향도)
   - Rename/Move 작업
   - 새 구조 구현 순서
3. **`/sc:implement`** — 실제 구현
4. **`/web:qa`** — 리팩토링 후 품질 검증
5. **`/web:deploy`** — 배포
