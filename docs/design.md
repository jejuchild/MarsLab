# MarsLab 4-Lane Inspector — 상세 설계

**작성일**: 2026-04-09
**참조**: `docs/identity.md` (정체성 + Cut/Keep 결정)
**롤백 안전망**: `pre-refactor-2026-04-09` tag, `legacy-full` branch

---

## 0. 설계 원칙 (identity.md에서 도출)

| # | 원칙 | 설계 함의 |
|---|---|---|
| 1 | 카탈로그 검색 1순위, 좌표 2순위, 클릭 3순위 | `SearchBar` 재설계, `TopBar` 단순화 |
| 2 | SHARAD/CRISM/HiRISE/CTX가 주인공 | 4 lane 구조, 기타 instrument는 variant 또는 cut |
| 3 | 통합 비교 | Inspector 하나에 4 lane tab + Cross 섹션 |
| 4 | 해석까지 OK (instrument 직결) | Lane 내부에 "분석 서브패널" — Stratigraphy 등 통합만 Cross |
| 5 | AI는 사이드킥 | `CopilotFab` (MARVIS) 하나만, 다른 AI 패널 cut |
| + | 데스크탑 우선 | `BottomSheet` 복잡 분기 단순화 |

---

## 1. 최상위 아키텍처

```
┌────────────────────────────────────────────────────────────────────┐
│  TopBar                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 🔍 Search: "Jezero crater" | lat,lon | product_id           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  [Undo] [Redo]     MarsLab     [News] [Research] [Download]        │
├──────────┬──────────────────────────────────────┬──────────────────┤
│          │                                      │                  │
│          │                                      │  ┌────────────┐  │
│ Layer    │                                      │  │ Inspector  │  │
│ Panel    │                                      │  ├────────────┤  │
│          │         Map (Cesium)                 │  │ 📍 18.4°N  │  │
│ Instrum. │                                      │  │    77.7°E  │  │
│ ☑ SHARAD │                                      │  ├─Lane Tabs──┤  │
│ ☑ CRISM  │        [ Clicked point ]             │  │ 🔴 SHARAD  │  │
│ ☑ HiRISE │                                      │  │ 🟡 CRISM   │  │
│ ☑ CTX    │                                      │  │ 🟢 HiRISE  │  │
│          │                                      │  │ 🔵 CTX     │  │
│ Map Tools│                                      │  ├─Active Lane┤  │
│ • Slope  │                                      │  │            │  │
│ • Line   │                                      │  │ (lane UI)  │  │
│ • Measure│                                      │  │            │  │
│          │                                      │  ├─Cross──────┤  │
│ Fieldnote│                                      │  │ Strati.    │  │
│          │                                      │  │ Min. Seq.  │  │
│          │                                      │  │ Temporal   │  │
│          │                                      │  └────────────┘  │
└──────────┴──────────────────────────────────────┴──────────────────┘
                                                               💬 MARVIS
```

**핵심 영역 3개**: LayerPanel (좌) — Map (중) — Inspector (우).
모바일에서는 Map만 풀스크린, LayerPanel/Inspector는 drawer로.

---

## 2. 데이터 흐름 (Critical Paths)

### 2.1 카탈로그 검색 플로우 ("Jezero crater")

```
User: types "jezero" in SearchBar
  │
  ▼
useCatalogSearch.match("jezero")
  │ (checks frontend/src/data/mars_catalog.json — extracted from CommandPalette)
  ▼
Match: { name: "Jezero Crater", lat: 18.41, lon: 77.69, zoom: 10km }
  │
  ▼
useMapNavigation.flyTo(18.41, 77.69, 10km)
  │ (Cesium camera fly animation)
  ▼
On fly complete:
  useInspectorContext.setPoint(18.41, 77.69)
  │
  ▼
GET /api/inspector/at-point?lat=18.41&lon=77.69&radius_km=5
  │ (new backend aggregator)
  ▼
Response: {
  SHARAD:  [{productId: "s_00123_000", ...}, ...],
  CRISM:   [{productId: "frt0000abcd_07_if166", ...}, ...],
  HIRISE:  [{productId: "ESP_042345_1985", ...}, ...],
  CTX:     [{productId: "B01_009892_1985", ...}, ...],
}
  │
  ▼
Inspector opens with context={mode:"point", lat, lon, nearbyProducts}
  │
  ▼
Default lane selection: lane with most products (or last-used lane)
  │
  ▼
Lane renders: shows first product's data + product picker (if N>1)
```

### 2.2 지도 클릭 플로우 (특정 footprint 클릭)

```
User: clicks SHARAD footprint
  │
  ▼
MapView.onFeatureClick(feature)
  │
  ▼
useInspectorContext.setProduct(productId, instrument, lat, lon)
  │
  ▼
Inspector opens with context={mode:"product", productId, primaryInstrument:"SHARAD", lat, lon}
  │
  ▼
Active lane = SHARAD (instrument of clicked product)
  │
  ├──→ Parallel: GET /api/inspector/at-point?lat&lon  (populate other lanes)
  │
  ▼
Lane renders immediately (product mode), other lanes fill in as they return
```

### 2.3 Lane 내부 데이터 흐름 (CRISM 예시)

```
CrismLane: activeProductId = "frt0000abcd_07_if166"
  │
  ├──→ GET /crism/spectrum/{productId}        → CRISMSpectrumTab
  ├──→ GET /api/crism-trr3/bands/{productId}  → CRISMBandsTab (TRR3 variant)
  ├──→ GET /api/mineral-cnn/predict/{productId} → MineralPanel
  └──→ GET /crism/quickview/{productId}.png   → Quickview thumbnail
```

---

## 3. 컴포넌트 트리 (After)

```
App.tsx
├── ErrorBoundary
├── BrowserRouter
│   ├── MainPage (/)
│   │   └── AppShell [desktop | mobile-drawer]
│   │       ├── TopBar
│   │       │   ├── SearchBar                ★ NEW: catalog-first
│   │       │   ├── UndoRedoButtons
│   │       │   └── NavLinks                 (News, Research, Download)
│   │       │
│   │       ├── LayerPanel                   ★ SIMPLIFIED
│   │       │   ├── InstrumentToggles        (4 only: SHARAD/CRISM/HiRISE/CTX)
│   │       │   ├── MapToolsSection          (Slope, Line, Measure)
│   │       │   ├── ViewModeSection          (2D/3D, base layer)
│   │       │   ├── IceHub                   (SWIM — reduced)
│   │       │   └── FieldNotesSection
│   │       │
│   │       ├── MapView                      (minor changes)
│   │       │   ├── CesiumViewer
│   │       │   ├── FootprintRenderer        (via FootprintManager)
│   │       │   ├── OverlayRenderer          (quickviews, DTMs)
│   │       │   ├── DTMHoverReadout
│   │       │   ├── ScaleBar / ZoomGuide
│   │       │   └── MapClickHandler          ★ NEW: point-mode emit
│   │       │
│   │       ├── Inspector                    ★ RESTRUCTURED
│   │       │   ├── InspectorHeader          (lat/lon, product title, close)
│   │       │   ├── LaneTabs                 ★ NEW
│   │       │   │   ├── SharadLane           ★ NEW (wraps radargram, regolith, attenuation, ice)
│   │       │   │   ├── CrismLane            ★ NEW (wraps spectrum, bands, mineral CNN)
│   │       │   │   ├── HiriseLane           ★ NEW (wraps image, DTM, slope, landform, crater)
│   │       │   │   └── CtxLane              ★ NEW (wraps mosaic, context image)
│   │       │   ├── CrossSection             ★ NEW (collapsible bottom)
│   │       │   │   ├── StratigraphyTool
│   │       │   │   ├── MineralSequenceTool
│   │       │   │   ├── TemporalCompareTool
│   │       │   │   └── SpectralCompareTool
│   │       │   └── InspectorActionBar       (download, save, share)
│   │       │
│   │       ├── CopilotFab                   (MARVIS chat — unchanged)
│   │       ├── CommandPalette               (Cmd+K — trimmed commands)
│   │       ├── KeyboardShortcuts
│   │       ├── OnboardingTour
│   │       └── EasterEggs                   (kept per user request, isolated)
│   │
│   ├── DataDownloadPage (/download)         (unchanged, internal cleanup later)
│   ├── DataUploadPage (/upload)             (unchanged)
│   ├── MarsNewsPage (/news)                 ★ MERGED: News + Research in tabs
│   └── FeatureSuggestionsPage (/suggestions)(kept)
```

### 삭제되는 컴포넌트 파일 (identity.md Cut 리스트 참조)

```
frontend/src/pages/
├── DailyDiscussionsPage.tsx           [DELETE]
├── MastcamPanoPage.tsx                [DELETE]
├── MastcamLabelPage.tsx               [DELETE]
└── MarsResearchPage.tsx               [MERGE into MarsNewsPage as Research tab] ← Q6

frontend/src/components/
├── PathfinderPanel.tsx                [DELETE]
├── AgenticPanel.tsx                   [DELETE]
├── ReportPanel.tsx                    [DELETE]
├── GuidedWorkflows.tsx                [DELETE]
├── RegionDashboard.tsx                [DELETE]
├── RegionStatsPanel.tsx               [DELETE] ← Q1 answered: cut
├── AccessibilityPanel.tsx             [DELETE]
├── AccessibilityExplainTooltip.tsx    [DELETE]
├── AiAnalysisPanel.tsx                [DELETE]
├── SwimIcePanel.tsx                   [DELETE] ← Q2 answered: cut all SWIM
├── SwimMethodLayer.tsx                [DELETE]
├── IceConsistencyLegend.tsx           [DELETE]
└── CraterDetectPanel.tsx              [KEEP — HiRISE lane]

frontend/src/hooks/
├── useRoverSimulation.ts              [DELETE]
└── usePathfinderOverlay.ts            [DELETE]
```

### 신규 파일

```
frontend/src/
├── components/inspector/lanes/
│   ├── SharadLane.tsx                 ★ NEW
│   ├── CrismLane.tsx                  ★ NEW
│   ├── HiriseLane.tsx                 ★ NEW
│   ├── CtxLane.tsx                    ★ NEW
│   └── LaneTabs.tsx                   ★ NEW
├── components/inspector/cross/
│   └── CrossSection.tsx               ★ NEW
├── components/search/
│   └── SearchBar.tsx                  ★ NEW (catalog-first)
├── hooks/
│   ├── useInspectorContext.ts         ★ NEW
│   ├── useCatalogSearch.ts            ★ NEW
│   └── useFootprintLayers.ts          ★ NEW (extracted from useFootprints)
└── data/
    └── mars_catalog.json              ★ NEW (extracted from CommandPalette)

backend/api/
└── inspector_router.py                ★ NEW (at-point aggregator)
```

---

## 4. State 분해 (5 Custom Hooks)

현재 `MainPage.tsx`의 40+ state 변수를 5개 hook으로 분해:

### 4.1 `useInspectorContext`
Inspector의 핵심 상태 — 어느 지점/product를 보고 있는지.

```typescript
type InspectorContext =
  | { mode: "point"; lat: number; lon: number;
      nearbyProducts: Record<Lane, ProductRef[]>;
      activeProductByLane: Partial<Record<Lane, string>>;
    }
  | { mode: "product"; productId: string; primaryLane: Lane;
      lat: number; lon: number; title?: string;
      nearbyProducts: Record<Lane, ProductRef[]>; // filled async
      activeProductByLane: Partial<Record<Lane, string>>;
    }
  | null;

type Lane = "SHARAD" | "CRISM" | "HIRISE" | "CTX";
type ProductRef = { productId: string; title?: string; lat?: number; lon?: number };

interface UseInspectorContextReturn {
  context: InspectorContext;
  activeLane: Lane;
  setPoint: (lat: number, lon: number) => void;           // → API query
  setProduct: (p: ProductRef, lane: Lane) => void;
  setActiveLane: (lane: Lane) => void;
  setActiveProductInLane: (lane: Lane, productId: string) => void;
  close: () => void;
  recentProducts: ProductRef[];  // last 5 (moved from MainPage)
}
```

**State 흡수**: `selected`, `recentProducts`, `hiRiseDTM3DPoint`, `terrainPoint`, `guidedLocation`.

---

### 4.2 `useMapNavigation`
지도 카메라, base layer, view mode, fly-to.

```typescript
interface UseMapNavigationReturn {
  baseLayer: BaseLayerType;                // "MOLA" | "HRSC" | ...
  setBaseLayer: (l: BaseLayerType) => void;
  mapMode: "2D" | "3D";
  setMapMode: (m: "2D" | "3D") => void;
  viewBounds: BoundingBox | null;
  setViewBounds: (b: BoundingBox | null) => void;
  cameraViewport: ViewportRect | null;
  flyTo: (lat: number, lon: number, zoomKm?: number) => Promise<void>;
  flyToProductId: string | null;
  setFlyToProductId: (id: string | null) => void;
}
```

**State 흡수**: `baseLayer`, `mapMode`, `viewBounds`, `cameraViewportRef`, `viewBoundSelectionMode`, `flyToProductId`.

---

### 4.3 `useFootprintLayers`
Instrument footprint 표시/로딩 — 4 instrument로 정리.

```typescript
interface UseFootprintLayersReturn {
  visibility: Record<Lane, boolean>;        // 4개만 (이전 7개 → 4개)
  variants: {
    sharad: "standard" | "highres";         // 이전 SHARAD_HIGHRES를 variant로
    crism: "standard" | "trr3";             // 이전 CRISM_TRR3를 variant로
    hirise: "image" | "dtm";                // 이전 HIRISE_DTM을 variant로
  };
  counts: Record<Lane, number>;
  loading: Record<Lane, boolean>;
  toggle: (lane: Lane, v: boolean) => void;
  setVariant: <K extends keyof Variants>(key: K, v: Variants[K]) => void;
  loadFootprints: (lane: Lane) => Promise<void>;
  manager: FootprintManager;
}
```

**State 흡수**: `instrumentVisibility`, `footprintLoadTrigger`, `loadingFootprints`, `footprintCounts`, `highResOnly`, 각종 variants (SHARAD_HIGHRES, CRISM_TRR3, HIRISE_DTM).

---

### 4.4 `useOverlays`
지도에 표시되는 product overlay (quickview, DTM, CTX mosaic).

```typescript
type Overlay = {
  productId: string;
  lane: Lane;
  kind: "quickview" | "dtm" | "mosaic";
  opacity: number;
  zIndex: number;
};

interface UseOverlaysReturn {
  overlays: Map<string, Overlay>;           // keyed by productId
  add: (o: Overlay) => void;
  remove: (productId: string) => void;
  setOpacity: (productId: string, v: number) => void;
  bringToFront: (productId: string) => void;
  clear: () => void;
}
```

**State 흡수**: `activeOverlays`, `bringToFrontId`, 각종 opacity state.

---

### 4.5 `useCatalogSearch`
검색바 로직 — 카탈로그 → 좌표 → product_id → easter egg.

```typescript
type SearchResult =
  | { type: "catalog"; name: string; lat: number; lon: number; zoomKm: number }
  | { type: "coordinate"; lat: number; lon: number }
  | { type: "product"; productId: string; instrument: Lane }
  | { type: "easter_egg"; eggId: string }
  | { type: "none" };

interface UseCatalogSearchReturn {
  query: string;
  setQuery: (q: string) => void;
  suggestions: SearchResult[];               // live suggestions
  parse: (q: string) => SearchResult;        // priority: catalog → coord → product → egg
  execute: (result: SearchResult) => void;   // triggers flyTo + setPoint etc.
}
```

**자료**: `frontend/src/data/mars_catalog.json` — 현재 `CommandPalette.tsx`에 있는 13+ 장소를 추출 + 확장.

---

### MainPage.tsx의 Before → After

| Before | After |
|---|---|
| 2,444 줄 | ~800 줄 (목표) |
| 161 hooks | ~20 hooks |
| 40+ state 변수 | 5개 custom hook + 약간의 UI state |
| `analysisMode` union 16+ | 제거됨 (4 lane + Cross) |
| Prop drilling 지옥 | Context/hook 기반 |

---

## 5. Backend API 추가/정리

### 5.1 신규 Endpoint: `GET /api/inspector/at-point`

**목적**: 한 좌표 근처의 4 instrument products를 **한 번의 호출**로 가져옴.

**Request**:
```
GET /api/inspector/at-point?lat=18.41&lon=77.69&radius_km=5
```

**Response**:
```json
{
  "lat": 18.41,
  "lon": 77.69,
  "radius_km": 5,
  "lanes": {
    "SHARAD": [
      { "product_id": "s_00123456_000", "title": "SHARAD ...", "lat": 18.40, "lon": 77.70, "variant": "standard" },
      { "product_id": "shr_00099_001", "variant": "highres" }
    ],
    "CRISM": [
      { "product_id": "frt0000abcd_07_if166", "variant": "standard" },
      { "product_id": "frt0000abcd_07_trr3", "variant": "trr3" }
    ],
    "HIRISE": [
      { "product_id": "ESP_042345_1985", "variant": "image" },
      { "product_id": "DTEEC_042345_1985", "variant": "dtm" }
    ],
    "CTX": [
      { "product_id": "B01_009892_1985", "variant": "image" }
    ]
  },
  "counts": { "SHARAD": 2, "CRISM": 2, "HIRISE": 2, "CTX": 1 }
}
```

**구현**: 기존 `search_router.py`의 spatial filter 로직을 재사용. 4개 instrument에 대해 `asyncio.gather()`로 병렬 처리.

**위치**: `backend/api/inspector_router.py` (신규).

### 5.2 삭제되는 Backend Routers

```
backend/api/
├── agentic_router.py              [DELETE]
├── report_router.py               [DELETE]
├── multi_report_router.py         [DELETE]
├── pathfinder_router.py           [DELETE]
├── accessibility_router.py        [DELETE]
├── neural_climate_router.py       [DELETE]
├── mars_climate.py                [DELETE]
├── mastcam_router.py              [DELETE]
├── mastcam_label_router.py        [DELETE]
├── mastcam_spice_router.py        [DELETE]
├── discussions_router.py          [DELETE]
├── swim_router.py                 [DELETE] ← Q2 answered
├── swim_ice_router.py             [DELETE]
└── scoring_methodology.py         [REVIEW — may delete]

backend/analysis/
├── thermal_pinn/                  [DELETE]
├── pinns_interior/                [DELETE]
├── neural_climate/                [DELETE]
├── pathfinder/                    [DELETE]
└── swim_*/                        [DELETE] ← all 6 SWIM method modules
                                    (neutron, thermal, surface, dielectric, geomorphic, fusion)

backend/agent/                     [DELETE — legacy framework]
```

### 5.3 유지되는 Backend Routers (정리 대상)

```
backend/api/
├── search_router.py               (core)
├── point_search.py                (helpers)
├── footprints_router.py           (instrument footprints)
├── proximity_router.py
├── crism_router.py                (CRISM data)
├── crism_trr3                     (CRISM TRR3 variant)
├── ctx_tile_router.py             (CTX tiles/mosaic)
├── sharad_highres_router.py
├── terrain_router.py              (slope, line profile, DEM)
├── terrain_features.py
├── hirise_landforms_router.py
├── stratigraphy_router.py         (cross-instrument)
├── mineral_sequence_router.py     (cross-instrument)
├── strat_column_router.py
├── epsilon_router.py              (SHARAD subsurface)
├── attenuation_router.py          (SHARAD)
├── regolith_router.py             (SHARAD)
├── swim_router.py                 (reduced — fusion only, not all 5 methods)
├── swim_ice_router.py
├── fieldnotes_router.py
├── custom_router.py
├── mars_news_router.py            (kept per user)
├── mars_research_router.py        (kept per user)
├── suggestions_router.py          (kept per user)
├── rag_router.py                  (MARVIS)
├── marvis_chat.py                 (MARVIS)
└── inspector_router.py            ★ NEW
```

**결과**: 50 routers → 약 30 routers.

---

## 6. LayerPanel 재구성

### 6.1 Before (현재)

```
LayerPanel
├── ProductsHub                 (current active products)
├── FootprintSection
│   ├── ☐ CRISM
│   ├── ☐ HIRISE
│   ├── ☐ SHARAD
│   ├── ☐ SHARAD_HIGHRES        ← variant
│   ├── ☐ CTX
│   ├── ☐ HIRISE_DTM            ← variant
│   └── ☐ CRISM_TRR3            ← variant
├── FieldNotesSection
├── IceHub                       (SWIM 6 methods)
├── AnalysisTools                (Slope, Line, 3D, AI, Region, Crater, Regolith, ...)
├── ViewModeSection              (2D/3D, base layer)
├── NavigationSection
└── OverlapFilter
```

### 6.2 After (간소화)

```
LayerPanel
├── Instruments (4 only)        ★ SIMPLIFIED
│   ├── ☐ 🔴 SHARAD
│   │      variant: [Standard | Hi-res]
│   ├── ☐ 🟡 CRISM
│   │      variant: [Standard | TRR3]
│   ├── ☐ 🟢 HiRISE
│   │      variant: [Image | DTM]
│   └── ☐ 🔵 CTX
├── MapTools                    ★ MOVED (from AnalysisTools, reduced)
│   ├── ▸ Slope
│   ├── ▸ Line Profile
│   └── ▸ Measure
├── Overlays
│   └── ▸ Grid
├── ViewMode                    (2D/3D, base layer)
├── FieldNotes
└── Bookmarks
```

**삭제된 섹션**: `ProductsHub` (Inspector가 대체), `AnalysisTools` 대부분, `IceHub` 6 methods → `Ice (fusion)`, `OverlapFilter`, `NavigationSection` (Command Palette로 이동).

---

## 7. Search UX 상세 스펙

### 7.1 입력 우선순위 (`useCatalogSearch.parse`)

1. **이스터에그** — "game", "terraform", "watney" 등 하드코드 키워드 → 해당 trigger
2. **카탈로그 이름** (`mars_catalog.json`) — fuzzy match: "jezero", "jez", "Jezero Crater" 모두 match
3. **좌표 파싱** — `18.41, 77.69` / `18.41N 77.69E` / `-23.98 -33.30` 패턴 인식
4. **Product ID** — 형식 인식 (`ESP_`, `frt`, `s_`, `B01_`)
5. **Fallback**: AI search (기존 `search_router` 호출)

### 7.2 카탈로그 데이터

`frontend/src/data/mars_catalog.json`:

```json
[
  {
    "id": "jezero",
    "name": "Jezero Crater",
    "lat": 18.41,
    "lon": 77.69,
    "zoom_km": 10,
    "keywords": ["mars 2020", "perseverance", "landing site", "delta"],
    "description": "Perseverance landing site (ancient lake delta)"
  },
  {
    "id": "gale",
    "name": "Gale Crater",
    "lat": -5.34,
    "lon": 137.65,
    "zoom_km": 15,
    "keywords": ["curiosity", "mount sharp"],
    "description": "Curiosity landing site"
  }
  // ... 13+ entries extracted from CommandPalette.tsx
]
```

### 7.3 SearchBar UX

```
┌─────────────────────────────────────────────┐
│ 🔍 Search: |                                │ ← placeholder
└─────────────────────────────────────────────┘
      │ (user types "jez")
      ▼
┌─────────────────────────────────────────────┐
│ 🔍 jez                                       │
├─────────────────────────────────────────────┤
│ 📍 Jezero Crater (18.41°N, 77.69°E)          │ ← Top result
│    Perseverance landing site                  │
├─────────────────────────────────────────────┤
│ 🔬 Jezero delta (CRISM frt0000...)           │ ← secondary (product search)
└─────────────────────────────────────────────┘
      │ (user presses Enter or clicks)
      ▼
  flyTo + inspector point mode
```

### 7.4 URL State Spec

**Shareable URLs**:
```
/                                              # empty
/?lat=18.41&lon=77.69                          # point mode
/?lat=18.41&lon=77.69&zoom=10&lane=SHARAD      # + active lane
/?product=ESP_042345_1985&lane=HIRISE          # product mode
/?place=jezero                                 # catalog shortcut
```

`useUrlState`를 확장 — `mode=assistant` 같은 레거시 파라미터는 제거 (crash-loop 원인이었음).

---

## 8. Cross 섹션 스펙

Inspector 하단 collapsible section. **기본은 접힌 상태, 사용자가 수동 펼침** (Q3 answered: B).
- 항상 Inspector 하단에 collapsed header로 표시 (`▸ Cross-Analysis (N available)`)
- 사용할 수 있는 도구 개수를 헤더에 뱃지로 표시 (ex: 2+ lane 데이터 있을 때 "2 tools available")
- 클릭하면 확장

도구들:
| 도구 | 입력 | 출력 |
|---|---|---|
| **Stratigraphy** | HiRISE DTM product + crater polygon | 지층 단면 + 나이 추정 |
| **Mineral Sequence** | CRISM TRR3 + HiRISE context | 광물 시퀀스 해석 |
| **Temporal Compare** | 2+ products of same area, different epochs | Before/after diff |
| **Spectral Compare** | 2+ CRISM spectra | Overlay plot |

**중요**: 기존 `StratigraphyPanel`, `MineralSequencePanel`, `TemporalComparison`, `SpectralComparison` 컴포넌트는 **재활용**. Inspector Cross 섹션이 이들을 모듈처럼 로드.

---

## 9. 모바일 전략 (데스크탑 우선)

| 영역 | 데스크탑 | 모바일 |
|---|---|---|
| LayerPanel | 상시 (320px) | Drawer (슬라이드) |
| MapView | 중앙 flex | 풀스크린 |
| Inspector | 상시 (420px) | Drawer (슬라이드) |
| Cross 섹션 | 인스펙터 하단 상시 | 추가 tab으로 flatten |
| SearchBar | TopBar | TopBar (폭 축소) |
| Lane tabs | 가로 4 tab | 가로 스와이프 가능 |

**단순화**: 기존 `BottomSheet`의 양방향 drawer 로직 제거. 한 번에 하나의 drawer만 (LayerPanel OR Inspector).

---

## 10. 마이그레이션 계획 (3 Phase)

### Phase 1 — Cut (삭제) ⚠️ 가장 안전
**목표**: 프론트 ~5,000 lines, 백엔드 ~15 routers 제거
**리스크**: 낮음 (삭제 대상이 다른 모듈과 의존성 적음)

1. Cut 리스트의 프론트 파일 삭제
2. `MainPage.tsx`에서 이들 import / usage 제거
3. Cut 리스트의 백엔드 라우터 삭제
4. `backend/app.py`에서 라우터 등록 제거
5. `CommandPalette` 중 cut된 기능 명령어 제거
6. **검증**: `npm run build`, `/web:qa`
7. **커밋**: `refactor: cut out-of-identity features (phase 1)`

### Phase 2 — Extract Hooks 🔧 중간 리스크
**목표**: `MainPage.tsx` 40+ state 변수를 5개 hook으로 분해
**리스크**: 중간 (기존 동작 유지하며 리팩토링)

1. `useInspectorContext` 생성 (빈 구현 + 기존 state 이동)
2. `useMapNavigation` 생성
3. `useFootprintLayers` 생성 (기존 `useFootprints` 확장/교체)
4. `useOverlays` 리팩토링 (이미 존재)
5. `useCatalogSearch` 생성 + `mars_catalog.json` 추출
6. `MainPage.tsx` 재조립 — hook 호출 + prop 전달만
7. **검증**: manual smoke test + `/web:qa`
8. **커밋**: `refactor: decompose MainPage into 5 custom hooks (phase 2)`

### Phase 3 — New Inspector 🏗️ 가장 큰 변경
**목표**: 4-lane Inspector 구조로 교체
**리스크**: 높음 (UI 동작 대거 변경)

1. Backend: `/api/inspector/at-point` 신규 endpoint 구현
2. Frontend: `Inspector2.tsx` + `LaneTabs.tsx` + 4 lane 컴포넌트 생성 (기존 `Inspector.tsx` 와 parallel)
3. 기존 `Inspector.tsx` 로직을 4 lane으로 분배
4. `MainPage.tsx`에서 `Inspector2`로 교체 (feature flag `?v=new` behind)
5. `CrossSection.tsx` 생성 — 기존 Stratigraphy/MineralSequence 컴포넌트 wrap
6. `LayerPanel` 간소화 (4 instrument + MapTools + 축소된 섹션)
7. `SearchBar.tsx` 신규 + `TopBar` 재설계
8. URL state 정리 (`useUrlState` — `mode=X` 레거시 제거)
9. Feature flag 제거, `Inspector.tsx` 삭제
10. **검증**: full `/web:qa` run
11. **커밋**: `refactor: introduce 4-lane Inspector architecture (phase 3)`

각 phase 후 **git commit + tag** 권장:
- `phase-1-cut-complete`
- `phase-2-hooks-complete`
- `phase-3-new-inspector-complete`

---

## 11. 리스크 & 완화책

| 리스크 | 영향 | 완화책 |
|---|---|---|
| Phase 3에서 UI 회귀 발생 | 사용성 저하 | Phase 3 feature flag 사용, legacy-full 브랜치로 비교 |
| 카탈로그 검색이 기존 검색 기능보다 약함 | UX 후퇴 | Phase 2에 fallback to existing search 유지 |
| Cross 섹션 도구들이 lane state에 의존하는데 lane 추가/제거 시 깨짐 | 기능 손실 | Cross 도구를 Inspector 상태만 구독하게 격리 |
| Backend `at-point` endpoint가 느림 (4 instrument 병렬 query) | 응답 지연 | asyncio.gather + 캐싱 + 빠른 spatial index 사용 |
| `useFootprints` 확장이 기존 동작 깨뜨림 | 지도 footprint 안 뜸 | Phase 2에서 기존 hook 유지, 신규 hook parallel로 구현 |
| 삭제한 파일이 다른 곳에서 import 되어 build 실패 | Phase 1 중단 | TypeScript 체크 + grep으로 사전 스캔 |
| `CommandPalette` 명령어 중 cut된 기능 남아서 런타임 에러 | 사용자 혼란 | Phase 1에서 명령어도 함께 정리 |
| 모바일 `BottomSheet` 단순화가 기존 모바일 사용자 불편 | UX 후퇴 | 데스크탑 우선이 identity 결정사항, 모바일은 read-only fallback |

---

## 12. 성능 목표

| 지표 | 현재 | Phase 3 목표 |
|---|---|---|
| MainPage bundle | 583 KB (154 KB gzip) | 400 KB (110 KB gzip) |
| Inspector 초기 렌더 | N/A (product 단위) | <300ms after point click |
| `/api/inspector/at-point` 응답 | N/A (신규) | <500ms (p95) |
| Time to interactive (cold load) | 3-5s | <3s |
| MainPage.tsx 줄 수 | 2,444 | <800 |

---

## 13. Open Questions — 해결됨 ✅

모든 5개 질문 확정됨 (사용자 답변 2026-04-09):

| # | 질문 | 답변 |
|---|---|---|
| Q1 | `RegionStatsPanel` 위치 | **CUT** — 정체성 밖 (한 좌표 ≠ 영역) |
| Q2 | `SwimIcePanel` + 6 SWIM methods | **CUT all** — 연구 level, "빠르게" 원칙과 충돌 |
| Q3 | Cross 섹션 표시 | **B** — 기본 접힌 상태, 수동 펼침 |
| Q4 | Easter egg trigger | **A** — SearchBar에서 직접 유지 ("game", "watney" 등) |
| Q5 | Phase 1 배포 | **A** — 커밋만, Phase 3 완료 후 배포 |
| Q6 | News / Research 처리 | **Merge** — `/news` 하나에 News 탭 + Research 탭 |

---

## 14. 다음 단계

1. **사용자 검토** — 이 설계 문서 + Open Questions 답변
2. **`/sc:workflow`** — 이 설계를 단계별 구현 태스크로 분해 (파일 단위)
3. **`/sc:implement`** — Phase 1부터 실제 구현 시작

---

## 부록 A: 4 Lane의 컴포넌트 재사용 맵

| 새 컴포넌트 | 재사용하는 기존 컴포넌트 |
|---|---|
| `SharadLane` | `SharadHiresInspector`, `Subsurface3DViewer`, `RegolithPanel` (통합), `AttenuationPanel` (통합), `SHARADPopupOverlay` |
| `CrismLane` | `CRISMSpectrumTab`, `CRISMBandsTab`, `SpectralComparison` (basic), `BandRatioCalculator`, `TRR3MineralSection` |
| `HiriseLane` | `HiResImageViewer`, `HiRiseDTM3DViewer`, `SlopeAnalysis`, `SlopeAnalysis3DTab`, `LineProfile`, `HiRISEPixelTab`, `HiriseLandformPanel`, `CraterDetectPanel` |
| `CtxLane` | CTX mosaic logic (기존 `ctx_tile_router`에서) |
| `CrossSection` | `StratigraphyPanel`, `StratColumnPanel`, `MineralSequencePanel`, `TemporalComparison`, `SpectralComparison` (advanced) |

대부분의 기존 분석 컴포넌트는 **삭제되지 않고 새 구조에 재배치**됩니다. Cut 되는 것은 identity 밖 컴포넌트 (Pathfinder, Agentic, Report 등)뿐입니다.

---

**END of design.md**
