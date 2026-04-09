# MarsLab 리팩토링 — 구현 워크플로우

**작성일**: 2026-04-09
**참조**: `docs/identity.md`, `docs/design.md`
**롤백**: `git reset --hard pre-refactor-2026-04-09` 또는 `git checkout legacy-full`

---

## 0. 실행 원칙

1. **한 Phase씩 완료 → 커밋 + tag** — 각 phase 후 `git tag phase-N-*-complete`
2. **각 Phase 끝에 validation 통과해야 다음 phase로** — build + lint + test
3. **Phase 1, 2는 배포 X** — Phase 3 완료 후 한 번만 배포 (사용자 결정)
4. **의존 순서 엄격 준수** — 리프 파일 삭제 → 중간 의존 정리 → 루트 수정
5. **두 서버 항상 재시작 안 함** — validation 시에만, 각 phase 끝에

---

# 📦 Phase 1: CUT (정체성 밖 기능 제거)

**목표**: 프론트 약 5,000+ 줄, 백엔드 15+ 라우터 제거
**리스크**: 낮음 (cut 대상은 다른 모듈과 결합도 낮음)
**예상 소요**: 2~3시간 (에이전트 병렬 실행 시)

## 1.A. Frontend 컴포넌트 & 훅 파일 삭제

각 파일을 rm으로 삭제 (git으로 추적됨). **삭제 순서 중요** — 리프부터.

### 1.A.1 Leaf 컴포넌트 (다른 컴포넌트가 import 안 함)
```bash
rm frontend/src/components/PathfinderPanel.tsx
rm frontend/src/components/AgenticPanel.tsx
rm frontend/src/components/ReportPanel.tsx
rm frontend/src/components/GuidedWorkflows.tsx
rm frontend/src/components/RegionDashboard.tsx
rm frontend/src/components/RegionStatsPanel.tsx
rm frontend/src/components/AccessibilityPanel.tsx
rm frontend/src/components/AccessibilityExplainTooltip.tsx
rm frontend/src/components/AiAnalysisPanel.tsx
rm frontend/src/components/SwimIcePanel.tsx
rm frontend/src/components/SwimMethodLayer.tsx
rm frontend/src/components/IceConsistencyLegend.tsx
```

### 1.A.2 관련 훅 & API
```bash
rm frontend/src/hooks/useRoverSimulation.ts
rm frontend/src/hooks/usePathfinderOverlay.ts
rm frontend/src/api/pathfinder.ts
rm frontend/src/api/accessibility.ts
# swim_ice.ts, workflow.ts는 아래에서 개별 결정
```

### 1.A.3 페이지
```bash
rm frontend/src/pages/DailyDiscussionsPage.tsx
rm frontend/src/pages/MastcamPanoPage.tsx
rm frontend/src/pages/MastcamLabelPage.tsx
# MarsResearchPage.tsx는 1.D에서 merge 후 삭제
```

### 1.A.4 LayerPanel sub-section 파일
```bash
rm frontend/src/components/layerpanel/sections/IceHub.tsx
rm frontend/src/components/layerpanel/sections/OverlapFilter.tsx  # cut
# AnalysisTools.tsx는 아래 수정 (일부만 cut)
```

## 1.B. Frontend 참조 제거 (기존 파일 수정)

삭제한 파일을 import하는 곳 모두 정리.

### 1.B.1 `frontend/src/App.tsx` — Route 제거
- [ ] `/discussions` route 제거
- [ ] `/mastcam` route 제거
- [ ] `/mastcam-label` route 제거
- [ ] `/research` route 제거 (Phase 1.D에서 /news로 merge)
- [ ] 관련 lazy import 삭제

### 1.B.2 `frontend/src/pages/MainPage.tsx` — 대수술 ⚠️
현재 2,444줄. Phase 1 후에도 대략 1,800~2,000줄 남을 것.

삭제할 항목:
- [ ] 모든 cut된 panel import 문
- [ ] `analysisMode` union type에서 cut된 모드 제거:
  - 제거: `"ai_analysis" | "agentic" | "report" | "guided" | "region_stats" | "pathfinder"`
  - 유지: `"slope" | "hirise_dtm_3d" | "line" | "crater_detect" | "regolith" | "stratigraphy" | "attenuation" | "mineral_sequence" | "strat_column"`
- [ ] Pathfinder state block (lines 244~255):
  - `pathfinderStart`, `pathfinderGoal`, `pathfinderRoute`, `simPlaying`, `simSpeed`, `simCameraFollow`, `simSeekTo`, `simProgress`, `simTelemetry`, `simComplete`
- [ ] Guided workflow state:
  - `guidedLocation`
- [ ] Cut panel rendering blocks (대략 lines 1850~2010에 있는 conditional rendering)
- [ ] RegionDashboard, RegionStatsPanel 관련 state/props
- [ ] SWIM 관련 state: ice consistency, ice method, SWIM overlays
- [ ] Accessibility 관련: `accessibilityOpacity`, fusion overlay state
- [ ] AiAnalysisPanel 관련 state
- [ ] `addRecentProduct`가 cut panel로 trigger되는 부분

### 1.B.3 `frontend/src/components/layerpanel/LayerPanel.tsx`
- [ ] Cut된 props 제거 (RegionDashboard, Pathfinder, etc.)
- [ ] `IceHub` import 제거 + `<IceHub>` 렌더 제거
- [ ] `OverlapFilter` import 제거 + 렌더 제거
- [ ] Cut된 analysis tool 버튼들 제거

### 1.B.4 `frontend/src/components/layerpanel/sections/AnalysisTools.tsx`
이 파일은 **일부만 cut** — 유지할 도구들을 남기고 나머지 삭제:
- [ ] 유지: Slope, Line Profile, HiRISE DTM 3D, Crater Detect, Regolith, Stratigraphy, Attenuation, Mineral Sequence, Strat Column
- [ ] 제거: AI Analysis, Agentic, Report, Guided Workflows, Pathfinder, Region Dashboard, Region Stats

### 1.B.5 `frontend/src/components/layerpanel/types.ts`
- [ ] Cut된 analysis mode type 제거

### 1.B.6 `frontend/src/components/CommandPalette.tsx`
이스터에그는 유지 (사용자 결정 Q4=A). Cut 대상만 제거:
- [ ] `agent/*`, `report/*`, `pathfinder/*`, `guided/*`, `region-stats/*` 액션 제거
- [ ] `swim-*` 액션 제거
- [ ] `accessibility-*` 액션 제거
- [ ] Mastcam 관련 액션 제거
- [ ] `/discussions`, `/research` 네비게이션 액션 제거 (news는 유지)

### 1.B.7 `frontend/src/hooks/useMapContext.ts`
- [ ] Cut된 panel 관련 context 값 제거

### 1.B.8 `frontend/src/hooks/usePanelManager.ts`
- [ ] Cut된 panel identifier 제거

### 1.B.9 Backend API client 파일들
- [ ] `frontend/src/api/search.ts` — Agentic/Report 관련 함수 제거
- [ ] `frontend/src/api/workflow.ts` — GuidedWorkflow 전용이면 삭제, 아니면 유지

## 1.C. Backend 삭제

### 1.C.1 Router 파일 삭제
```bash
rm backend/api/agentic_router.py
rm backend/api/report_router.py
rm backend/api/multi_report_router.py
rm backend/api/sharad_report_router.py  # report의 일부였음
rm backend/api/pathfinder_router.py
rm backend/api/accessibility_router.py
rm backend/api/neural_climate_router.py
rm backend/api/mars_climate.py
rm backend/api/mastcam_router.py
rm backend/api/mastcam_label_router.py
rm backend/api/mastcam_spice_router.py
rm backend/api/discussions_router.py
rm backend/api/swim_router.py
rm backend/api/swim_ice_router.py
rm backend/api/workflow_router.py     # GuidedWorkflows backend
rm backend/api/region_scores_router.py  # RegionDashboard backend
rm backend/api/ice_evidence_router.py   # ice/* (SWIM 관련이면 cut)
rm backend/api/smart_search_router.py   # Llama smart search (Agentic 의존)
```

### 1.C.2 Analysis 모듈 삭제
```bash
rm -rf backend/analysis/thermal_pinn
rm -rf backend/analysis/pinns_interior
rm -rf backend/analysis/neural_climate
rm -rf backend/analysis/pathfinder
rm -rf backend/analysis/swim_neutron
rm -rf backend/analysis/swim_thermal
rm -rf backend/analysis/swim_surface
rm -rf backend/analysis/swim_dielectric
rm -rf backend/analysis/swim_geomorphic
rm -rf backend/analysis/swim_fusion
rm -rf backend/analysis/ice_evidence   # SWIM 기반
```

### 1.C.3 Legacy 폴더
```bash
rm -rf backend/agent/
rm backend/data/agent_sessions.json  # legacy sessions
```

### 1.C.4 Data 파일 (선택적)
```bash
# 확인 후 삭제 (디스크 절약):
# backend/data/swim/*
# backend/data/hirise_landforms/* (유지 — HiRISE lane에서 사용)
# backend/data/tes_thermal_inertia.npy (accessibility 전용이면 삭제)
```

## 1.D. `backend/app.py` 라우터 등록 제거

`/disk1/cspark/MarsLab/backend/app.py` 의 line 556~688 주변에서 다음 `app.include_router(...)` 제거:
- [ ] `agentic_router` (line 595)
- [ ] `smart_search_router` (596)
- [ ] `report_router` (597)
- [ ] `sharad_report_router` (600)
- [ ] `multi_report_router` (601)
- [ ] `region_scores_router` (602)
- [ ] `ice_evidence_router` (608)
- [ ] `workflow_router` (617)
- [ ] `discussions_router` (644)
- [ ] `swim_router` (647)
- [ ] `swim_ice_router` (650)
- [ ] `mars_research_router` (659) — **유지** (news와 함께)
- [ ] `accessibility_router` (662)
- [ ] `pathfinder_router` (668)
- [ ] `mastcam_router` (674)
- [ ] `mastcam_label_router` (677)
- [ ] `mastcam_spice_router` (680)
- [ ] `neural_climate_router` (683)
- [ ] `pinns_router` (685)

그리고 각 import 문도 제거.

## 1.E. MarsResearchPage → MarsNewsPage 병합

Q6 결정: 단일 `/news` 안에 News 탭 + Research 탭.

### 1.E.1 `MarsNewsPage.tsx` 수정
- [ ] 상단에 탭 컨트롤 추가 (`News | Research`)
- [ ] 기본 탭: News
- [ ] Research 탭 구현: `MarsResearchPage.tsx`의 내용을 함수화해서 import
- [ ] URL state: `?tab=research` 지원 (useSearchParams)
- [ ] `/api/mars-research` 호출 로직 추가

### 1.E.2 `MarsResearchPage.tsx` 삭제
```bash
rm frontend/src/pages/MarsResearchPage.tsx
```

### 1.E.3 `App.tsx`에서 `/research` 라우트 제거
- [ ] `/research` 제거
- [ ] 필요시 `/research` → `/news?tab=research` redirect 추가

### 1.E.4 관련 참조 정리
- [ ] `TopBar.tsx`의 News/Research 링크 통합
- [ ] `Footer.tsx`의 링크 업데이트
- [ ] `CommandPalette.tsx`의 research 명령어 → `/news?tab=research`로

## 1.F. Phase 1 검증

```bash
# TypeScript
cd /disk1/cspark/MarsLab/frontend && npx tsc --noEmit
# → 0 errors 기대

# ESLint
npx eslint .
# → 0 errors, 0 warnings 유지 (현재 깨끗한 상태)

# Tests
npx vitest run
# → 8/8 pass 유지

# Build
npx vite build
# → ✓ built 기대

# Backend import test
cd /disk1/cspark/MarsLab/backend
python -c "import app; print('OK')"
# → "OK" 출력 기대 (라우터 import 에러 없어야)

# Manual smoke (optional — 서버 재시작 생략)
# - 현재 서버 동작 중이면 백엔드는 reload로 자동 적용됨
# - 프론트는 Vite HMR이 react tree 에러 유발할 수 있으니, 문제 시에만 재시작
```

## 1.G. Phase 1 커밋

```bash
cd /disk1/cspark/MarsLab
git add -A
git commit -m "refactor(phase-1): cut out-of-identity features

Removed ~5,000 frontend LOC and 17 backend routers.

Cut: Pathfinder, Agentic, Report, GuidedWorkflows, RegionDashboard,
RegionStatsPanel, AccessibilityPanel, AiAnalysisPanel, SwimIcePanel,
SwimMethodLayer, IceConsistencyLegend, 6 SWIM analysis modules,
thermal_pinn, neural_climate, pathfinder backend, legacy agent/.

Pages removed: DailyDiscussions, MastcamPano, MastcamLabel.
MarsResearchPage merged into MarsNewsPage as Research tab.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git tag phase-1-cut-complete
```

## 1.H. Phase 1 완료 기준 체크리스트

- [ ] 모든 cut 대상 파일 삭제됨
- [ ] TypeScript 0 errors
- [ ] ESLint 0 errors/warnings
- [ ] Tests 8/8 pass
- [ ] Frontend build 성공
- [ ] Backend `python -c "import app"` 성공
- [ ] `git tag phase-1-cut-complete` 생성됨
- [ ] **배포 안 함** (사용자 결정 Q5=A)

---

# 🔧 Phase 2: EXTRACT HOOKS (MainPage 모놀리스 분해)

**목표**: `MainPage.tsx` 2,000줄 → 1,000줄, 5개 custom hook으로 분해
**리스크**: 중간 (기존 동작 유지하며 리팩토링)
**예상 소요**: 4~6시간

## 2.A. 준비

### 2.A.1 Hook 디렉토리 생성
```bash
mkdir -p frontend/src/hooks/inspector
mkdir -p frontend/src/data
```

### 2.A.2 카탈로그 데이터 추출
- [ ] `frontend/src/data/mars_catalog.json` 생성
- [ ] `CommandPalette.tsx`의 13+ 지명 액션에서 데이터 추출:
  - Jezero, Gale, Valles Marineris, Olympus Mons, Hellas, Syrtis Major, Nili Fossae, Mawrth Vallis, South Pole, North Pole, Eberswalde, Columbia Hills, Meridiani
- [ ] 각 엔트리 구조: `{ id, name, lat, lon, zoom_km, keywords, description }`

## 2.B. `useInspectorContext` (가장 중요, 가장 먼저)

### 2.B.1 파일 생성
`frontend/src/hooks/inspector/useInspectorContext.ts`

```typescript
// Types
export type Lane = "SHARAD" | "CRISM" | "HIRISE" | "CTX";
export type ProductRef = { productId: string; title?: string; lat?: number; lon?: number };
export type InspectorContext =
  | { mode: "point"; lat: number; lon: number;
      nearbyProducts: Record<Lane, ProductRef[]>;
      activeProductByLane: Partial<Record<Lane, string>>; }
  | { mode: "product"; productId: string; primaryLane: Lane;
      lat: number; lon: number; title?: string;
      nearbyProducts: Record<Lane, ProductRef[]>;
      activeProductByLane: Partial<Record<Lane, string>>; }
  | null;

// Hook
export function useInspectorContext() {
  // state
  // setPoint, setProduct, setActiveLane, setActiveProductInLane, close
  // recentProducts (last 5)
  // return { context, activeLane, setPoint, setProduct, ..., recentProducts }
}
```

### 2.B.2 MainPage에서 적용
- [ ] 기존 `selected`, `recentProducts`, `hiRiseDTM3DPoint`, `terrainPoint` state 제거
- [ ] `useInspectorContext()` 호출
- [ ] 기존 `setSelected` 호출 모두 `setProduct` / `setPoint`로 교체
- [ ] prop drilling: Inspector, MapView에 `context` 전달

### 2.B.3 검증
```bash
npx tsc --noEmit
# → 0 errors
# Manual: footprint 클릭하면 Inspector 열림, lat/lon 표시됨
```

## 2.C. `useMapNavigation`

### 2.C.1 파일 생성
`frontend/src/hooks/useMapNavigation.ts`

흡수 대상 state:
- `baseLayer`, `mapMode`, `viewBounds`, `cameraViewportRef`
- `viewBoundSelectionMode`, `flyToProductId`, `bringToFrontId`

Exported API:
- `baseLayer, setBaseLayer`
- `mapMode, setMapMode`
- `viewBounds, setViewBounds`
- `cameraViewport` (ref-backed)
- `flyTo(lat, lon, zoomKm?)` — Promise
- `flyToProductId, setFlyToProductId`

### 2.C.2 MainPage에서 적용
- [ ] 기존 state 제거
- [ ] `useMapNavigation()` 호출
- [ ] `MapView` prop 재구성

### 2.C.3 검증: 2D/3D 토글, base layer 변경 동작

## 2.D. `useFootprintLayers` (기존 `useFootprints` 래핑/확장)

### 2.D.1 파일 생성
`frontend/src/hooks/useFootprintLayers.ts`

- 7 instrument → 4 instrument로 정규화
- Variant 개념 도입:
  ```typescript
  variants: {
    sharad: "standard" | "highres",
    crism: "standard" | "trr3",
    hirise: "image" | "dtm",
  }
  ```
- 기존 `useFootprints`를 내부에서 호출 (점진적 마이그레이션)

### 2.D.2 MainPage에서 적용
- [ ] `instrumentVisibility` state 제거
- [ ] `showSharadHighres`, `showCrismTrr3`, `showHiRISEDTM` → variants로 이동
- [ ] LayerPanel의 footprint section 간소화

### 2.D.3 검증: 4 instrument 토글 모두 동작, variant 전환 동작

## 2.E. `useOverlays` (기존 훅 리팩토링)

이미 `frontend/src/hooks/useOverlays.ts` 존재. 개선만.

- [ ] state를 `Map<productId, Overlay>` 통일
- [ ] `Overlay` type에 `kind`, `zIndex`, `lane` 추가
- [ ] `bringToFront` 구현 (현재 `bringToFrontId` state 대체)
- [ ] MainPage의 `activeOverlays`, `bringToFrontId` state 제거

## 2.F. `useCatalogSearch`

### 2.F.1 파일 생성
`frontend/src/hooks/useCatalogSearch.ts`

```typescript
type SearchResult =
  | { type: "catalog"; name: string; lat: number; lon: number; zoomKm: number }
  | { type: "coordinate"; lat: number; lon: number }
  | { type: "product"; productId: string; instrument: Lane }
  | { type: "easter_egg"; eggId: string }
  | { type: "none" };

export function useCatalogSearch() {
  // load mars_catalog.json
  // parse(query): SearchResult
  //   1. easter egg (game, watney, terraform)
  //   2. catalog fuzzy match
  //   3. coordinate regex
  //   4. product_id regex (ESP_, frt, s_, B01_)
  //   5. none
  // suggestions(query): SearchResult[]
}
```

### 2.F.2 아직 UI 교체 X
Phase 2에서는 hook만 만들고, Phase 3에서 `SearchBar` 컴포넌트와 연결.
기존 `TopBar` 검색은 그대로 유지.

## 2.G. MainPage 재조립

### 2.G.1 목표
`MainPage.tsx` 구조:
```typescript
export default function MainPage() {
  // 5 custom hooks
  const inspector = useInspectorContext();
  const mapNav = useMapNavigation();
  const footprints = useFootprintLayers();
  const overlays = useOverlays();
  const search = useCatalogSearch();

  // UI-only state (~10개 남음)
  const [mobilePanel, setMobilePanel] = useState(...);
  const [showMeasurementTools, setShowMeasurementTools] = useState(...);
  // ...

  // Composition only — no business logic
  return (
    <AppShell ...>
      <TopBar search={search} ... />
      <LayerPanel footprints={footprints} ... />
      <MapView mapNav={mapNav} overlays={overlays} inspector={inspector} ... />
      <Inspector inspector={inspector} ... />
      <CopilotFab />
      ...
    </AppShell>
  );
}
```

목표: 800~1,000줄.

### 2.G.2 Checklist
- [ ] 모든 business logic이 hook 안에 있음
- [ ] MainPage는 composition + UI routing만
- [ ] analysisMode union 유지 (Phase 3에서 lane으로 교체)

## 2.H. Phase 2 검증

```bash
cd /disk1/cspark/MarsLab/frontend
npx tsc --noEmit                         # 0 errors
npx eslint .                              # 0 errors
npx vitest run                            # 8/8 pass
npx vite build                            # ✓ built

# Manual smoke:
# - 지도 로드
# - 4 instrument footprint 토글
# - 각 instrument의 footprint 클릭 → Inspector 열림
# - 2D/3D 전환
# - Base layer 전환
# - CommandPalette "Fly to Jezero" 작동
# - Field notes 생성 / 로드
# - MARVIS chat 작동
```

## 2.I. Phase 2 커밋

```bash
git add -A
git commit -m "refactor(phase-2): decompose MainPage into 5 custom hooks

Introduced:
- useInspectorContext: inspector state (point/product/lanes)
- useMapNavigation: camera, base layer, 2D/3D, fly-to
- useFootprintLayers: 4-instrument visibility + variants
- useOverlays: active map overlays with z-order
- useCatalogSearch: search parsing (catalog → coord → product → egg)

MainPage.tsx reduced from ~1,800 to ~900 lines.
Business logic moved into hooks; MainPage now handles composition only.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git tag phase-2-hooks-complete
```

---

# 🏗️ Phase 3: NEW INSPECTOR (4-Lane 구조 교체)

**목표**: 새로운 Inspector 컴포넌트 트리로 교체, UX 재설계
**리스크**: 높음 (UI 동작 대거 변경)
**예상 소요**: 1~2일

## 3.A. Backend: `/api/inspector/at-point` 신규 endpoint

### 3.A.1 파일 생성
`backend/api/inspector_router.py`

```python
from fastapi import APIRouter, Query
import asyncio

router = APIRouter(prefix="/api/inspector", tags=["inspector"])

@router.get("/at-point")
async def at_point(lat: float = Query(...), lon: float = Query(...),
                   radius_km: float = Query(5.0)):
    # Parallel query 4 instruments using existing search_router logic
    results = await asyncio.gather(
        query_sharad(lat, lon, radius_km),
        query_crism(lat, lon, radius_km),
        query_hirise(lat, lon, radius_km),
        query_ctx(lat, lon, radius_km),
    )
    return {
        "lat": lat, "lon": lon, "radius_km": radius_km,
        "lanes": {
            "SHARAD": results[0],
            "CRISM": results[1],
            "HIRISE": results[2],
            "CTX": results[3],
        },
        "counts": {k: len(v) for k, v in zip(["SHARAD","CRISM","HIRISE","CTX"], results)}
    }
```

### 3.A.2 구현 전략
- `backend/api/point_search.py`의 기존 spatial 로직 재사용
- GeoJSON index (`_geojson_cache`)를 in-memory lookup
- Variant (SHARAD_HIGHRES → SHARAD highres, CRISM_TRR3 → CRISM trr3) 올바르게 tagging

### 3.A.3 등록
- [ ] `backend/app.py` 에 `app.include_router(inspector_router)` 추가

### 3.A.4 검증
```bash
curl "http://localhost:8000/api/inspector/at-point?lat=18.41&lon=77.69&radius_km=5"
# → JSON with lanes + counts
```

## 3.B. Frontend: API client

### 3.B.1 파일 생성
`frontend/src/api/inspector.ts`

```typescript
export type AtPointResponse = {
  lat: number; lon: number; radius_km: number;
  lanes: Record<Lane, ProductRef[]>;
  counts: Record<Lane, number>;
};

export async function fetchAtPoint(lat: number, lon: number, radiusKm = 5):
  Promise<AtPointResponse> {
  const res = await fetch(`/api/inspector/at-point?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`);
  return res.json();
}
```

### 3.B.2 `useInspectorContext` 업데이트
- [ ] `setPoint(lat, lon)` 내부에서 `fetchAtPoint` 호출
- [ ] 응답을 `nearbyProducts`로 저장
- [ ] Default `activeLane` = products가 가장 많은 lane

## 3.C. Lane 컴포넌트 (병렬 구현 가능)

### 3.C.1 `frontend/src/components/inspector/lanes/SharadLane.tsx`
```typescript
export function SharadLane({ inspector, activeProductId, onProductSelect }: LaneProps) {
  const [variant, setVariant] = useState<"standard" | "highres">("standard");
  const [subPanel, setSubPanel] = useState<"radargram" | "3d" | "regolith" | "attenuation">("radargram");

  return (
    <div>
      <VariantToggle value={variant} onChange={setVariant} />
      <ProductPicker products={inspector.context.nearbyProducts.SHARAD} ... />
      <SubPanelTabs value={subPanel} onChange={setSubPanel} />
      {subPanel === "radargram" && <SharadHiresInspector productId={activeProductId} ... />}
      {subPanel === "3d" && <Subsurface3DViewer productId={activeProductId} ... />}
      {subPanel === "regolith" && <RegolithToolEmbedded productId={activeProductId} ... />}
      {subPanel === "attenuation" && <AttenuationToolEmbedded productId={activeProductId} ... />}
    </div>
  );
}
```

### 3.C.2 `CrismLane.tsx`
- Variant: Standard | TRR3
- Sub-panels: Spectrum | Bands (RGB) | Mineral CNN | Band Ratio

### 3.C.3 `HiriseLane.tsx`
- Variant: Image | DTM
- Sub-panels: Quickview | 3D DTM | Slope | Line Profile | Landform | Crater

### 3.C.4 `CtxLane.tsx`
- Sub-panels: Mosaic | Single image

### 3.C.5 `LaneTabs.tsx`
- 4개 탭 헤더 (SHARAD/CRISM/HiRISE/CTX)
- 각 탭에 product count 뱃지
- 비활성 lane은 dim
- Active lane 컴포넌트 렌더

## 3.D. `CrossSection.tsx`

```typescript
export function CrossSection({ inspector }: { inspector: InspectorContext }) {
  const [expanded, setExpanded] = useState(false); // Q3=B: default collapsed
  const [tool, setTool] = useState<"stratigraphy" | "mineral_seq" | "temporal" | "spectral">("stratigraphy");

  const availableTools = computeAvailable(inspector);

  return (
    <div>
      <button onClick={() => setExpanded(v => !v)}>
        {expanded ? "▾" : "▸"} Cross-Analysis ({availableTools.length} available)
      </button>
      {expanded && (
        <>
          <ToolTabs tools={availableTools} active={tool} onChange={setTool} />
          {tool === "stratigraphy" && <StratigraphyPanel ... />}
          {tool === "mineral_seq" && <MineralSequencePanel ... />}
          {tool === "temporal" && <TemporalComparison ... />}
          {tool === "spectral" && <SpectralComparison ... />}
        </>
      )}
    </div>
  );
}
```

기존 panel 컴포넌트는 그대로 재활용.

## 3.E. `Inspector2.tsx` (feature flag 보호)

### 3.E.1 파일 생성
`frontend/src/components/Inspector2.tsx`

```typescript
export function Inspector2({ inspector }: { inspector: UseInspectorContextReturn }) {
  const { context, activeLane, setActiveLane } = inspector;
  if (!context) return null;

  return (
    <aside className="w-[420px] ...">
      <InspectorHeader context={context} />
      <LaneTabs
        lanes={["SHARAD","CRISM","HIRISE","CTX"]}
        active={activeLane}
        counts={context.nearbyProducts}
        onChange={setActiveLane}
      />
      <div className="flex-1 overflow-auto">
        {activeLane === "SHARAD" && <SharadLane ... />}
        {activeLane === "CRISM" && <CrismLane ... />}
        {activeLane === "HIRISE" && <HiriseLane ... />}
        {activeLane === "CTX" && <CtxLane ... />}
      </div>
      <CrossSection inspector={context} />
      <InspectorActionBar context={context} />
    </aside>
  );
}
```

### 3.E.2 MainPage에 feature flag 도입
```typescript
const useNewInspector = new URLSearchParams(location.search).get("v") === "new";
// ...
{useNewInspector
  ? <Inspector2 inspector={inspectorHook} />
  : <Inspector /* legacy */ ... />}
```

### 3.E.3 테스트
- [ ] `http://localhost:5173/?v=new` 로 접속
- [ ] 4 lane tab 동작
- [ ] 각 lane의 sub-panel 동작
- [ ] Cross section 펼침/접기 동작
- [ ] Point mode vs Product mode 전환

## 3.F. `SearchBar.tsx`

### 3.F.1 파일 생성
`frontend/src/components/search/SearchBar.tsx`

```typescript
export function SearchBar({ search, mapNav, inspector }: Props) {
  const [query, setQuery] = useState("");
  const suggestions = useMemo(() => search.suggestions(query), [query]);

  const handleSubmit = () => {
    const result = search.parse(query);
    switch (result.type) {
      case "catalog":
        mapNav.flyTo(result.lat, result.lon, result.zoomKm);
        inspector.setPoint(result.lat, result.lon);
        break;
      case "coordinate":
        mapNav.flyTo(result.lat, result.lon);
        inspector.setPoint(result.lat, result.lon);
        break;
      case "product":
        inspector.setProduct({productId: result.productId, ...}, result.instrument);
        break;
      case "easter_egg":
        triggerEasterEgg(result.eggId);
        break;
    }
  };

  return (
    <div>
      <input value={query} onChange={...} onKeyDown={e => e.key === "Enter" && handleSubmit()} />
      <SuggestionsDropdown items={suggestions} onSelect={...} />
    </div>
  );
}
```

### 3.F.2 TopBar에 통합
- [ ] `TopBar.tsx`에서 기존 검색 로직 제거
- [ ] `<SearchBar />` 컴포넌트 삽입

## 3.G. LayerPanel 간소화

### 3.G.1 섹션 교체
- [ ] `FootprintSection` → 4 instrument 토글 + variant 선택
- [ ] `AnalysisTools` → `MapTools` (Slope, Line, Measure만)
- [ ] `ProductsHub` 섹션 제거 (Inspector가 대체)
- [ ] `NavigationSection` 섹션 제거 (CommandPalette로 이동)

### 3.G.2 목표 구조 (design.md 6.2 참조)
```
LayerPanel
├── Instruments (4)
├── MapTools (Slope, Line, Measure)
├── Overlays (Grid)
├── ViewMode (2D/3D, base layer)
├── FieldNotes
└── Bookmarks
```

## 3.H. URL state 정리

### 3.H.1 `useUrlState.ts` 수정
- [ ] 레거시 `mode=X` 파라미터 제거 (crash-loop 원인이었음)
- [ ] 신규 파라미터: `lat`, `lon`, `zoom`, `lane`, `product`, `place`
- [ ] Shareable URL 테스트

### 3.H.2 예시 URL
```
/?place=jezero
/?lat=18.41&lon=77.69&lane=SHARAD
/?product=ESP_042345_1985&lane=HIRISE
```

## 3.I. Legacy 제거

### 3.I.1 기존 Inspector 삭제
- [ ] Feature flag 제거 (`?v=new` → 기본값)
- [ ] `frontend/src/components/Inspector.tsx` 삭제
- [ ] `frontend/src/components/inspector/InspectorPanel.tsx` 삭제 (만약 `Inspector2`에 흡수)
- [ ] `Inspector2.tsx` → `Inspector.tsx`로 rename

### 3.I.2 잔여 `analysisMode` 제거
- [ ] MainPage에서 `analysisMode` union type 완전 제거
- [ ] 관련 state, handler 삭제

## 3.J. Phase 3 검증 (full QA)

### 3.J.1 자동 검증
```bash
cd /disk1/cspark/MarsLab/frontend
npx tsc --noEmit                         # 0 errors
npx eslint .                              # 0 errors
npx vitest run                            # 8/8 pass
npx vite build                            # ✓ built

# Backend
cd /disk1/cspark/MarsLab/backend
python -c "import app; print('OK')"
curl "http://localhost:8000/api/inspector/at-point?lat=18.41&lon=77.69"
```

### 3.J.2 수동 End-to-End 시나리오
1. **홈 로드** — 에러 없이 Mars 지도 로드
2. **카탈로그 검색** — "jezero" 입력 → Jezero Crater suggestion → Enter → fly-to + Inspector 열림
3. **4 lane 확인** — 각 탭 클릭 → 해당 instrument 데이터 로드
4. **Lane variant** — SHARAD standard/hi-res 전환
5. **Sub-panel** — HiRISE lane에서 Image/DTM/Slope/LineProfile 전환
6. **Cross section** — 접혀 있음 → 클릭하면 펼침 → Stratigraphy 동작
7. **좌표 검색** — "18.4, 77.7" 입력 → 같은 동작
8. **Product 검색** — "ESP_042345_1985" 입력 → HiRISE lane 자동 선택
9. **Easter egg** — "game" 검색 → SpaceGame 열림
10. **MARVIS chat** — floating button 클릭 → chat 동작
11. **2D/3D 전환** — LayerPanel에서 전환
12. **Base layer 전환** — MOLA → HRSC
13. **Field note** — 지도 클릭 → 노트 생성 → 목록 확인
14. **Download page** — `/download` 네비게이션 → 기본 동작
15. **News + Research** — `/news` → News 탭, Research 탭 전환
16. **Undo/Redo** — camera 이동 후 Cmd+Z
17. **Command Palette** — Cmd+K → 명령어 팔레트 열림
18. **Keyboard shortcuts** — `?` → 단축키 help
19. **모바일 뷰** — 개발자 도구에서 모바일 크기 → drawer 동작 (read-only 수준)
20. **URL 공유** — `/?place=jezero&lane=SHARAD` 붙여넣기 → 같은 상태 복원

### 3.J.3 `/web:qa` 풀 실행
```bash
# 설정 후 실행 (Playwright 등)
# docs/qa-report.md 업데이트
```

## 3.K. Phase 3 커밋 & 배포

```bash
git add -A
git commit -m "refactor(phase-3): introduce 4-lane Inspector architecture

Complete restructure of Inspector into 4 instrument lanes
(SHARAD, CRISM, HiRISE, CTX) with a collapsible Cross-Analysis section.

New components:
- Inspector2 → Inspector (replaces legacy)
- LaneTabs + SharadLane, CrismLane, HiriseLane, CtxLane
- CrossSection (Stratigraphy, MineralSequence, Temporal, Spectral)
- SearchBar (catalog-first)

Backend:
- POST /api/inspector/at-point — parallel 4-instrument aggregator

UX improvements:
- Catalog search first: 'jezero' → fly-to + Inspector
- LayerPanel reduced to 4 instrument toggles
- MainPage.tsx reduced from 900 to ~600 lines
- Shareable URLs via lat/lon/lane/product/place params
- Legacy mode=X URL params removed (crash-loop fix)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git tag phase-3-new-inspector-complete
git push origin responsive-web
git push origin phase-1-cut-complete phase-2-hooks-complete phase-3-new-inspector-complete
```

### 3.K.1 서버 재시작
```bash
# Backend
lsof -ti :8000 | xargs kill 2>/dev/null
cd /disk1/cspark/MarsLab/backend
nohup python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload > /tmp/marslab_backend.log 2>&1 &

# Frontend
lsof -ti :5173 | xargs kill 2>/dev/null
cd /disk1/cspark/MarsLab/frontend
nohup npx vite --host 0.0.0.0 --port 5173 > /tmp/marslab_frontend.log 2>&1 &

# Verify
sleep 20
curl -s -o /dev/null -w "Backend: %{http_code}\n" http://localhost:8000/api/health || curl -s -o /dev/null -w "Backend: %{http_code}\n" http://localhost:8000/
curl -s -o /dev/null -w "Frontend: %{http_code}\n" http://localhost:5173/
```

---

# 🛡️ 리스크 대응 (phase 별 롤백)

## Phase 1 롤백
```bash
# Phase 1 중 문제 발생:
git reset --hard pre-refactor-2026-04-09
# 또는
git checkout legacy-full
```

## Phase 2 롤백
```bash
# Phase 2 중 문제 발생:
git reset --hard phase-1-cut-complete
```

## Phase 3 롤백
```bash
# Phase 3 중 문제 발생:
git reset --hard phase-2-hooks-complete
```

## Production 배포 후 긴급 롤백
```bash
# Phase 3 배포 후 사용자 불만:
git checkout phase-2-hooks-complete
# 서버 재시작
# 또는 legacy-full 브랜치로 긴급 복구
git checkout legacy-full
```

---

# 📊 예상 효과 (Phase 별 누적)

| 지표 | 현재 | After P1 | After P2 | After P3 (목표) |
|---|---|---|---|---|
| Frontend LOC | ~40,000 | ~33,000 | ~32,000 | ~25,000 |
| Backend routers | 50 | ~35 | ~35 | ~30 |
| `MainPage.tsx` 줄 수 | 2,444 | ~1,800 | ~900 | ~600 |
| 분석 패널 | 22 | 10 | 10 | 4 lane + Cross |
| Instrument 토글 | 7 | 7 | 4 | 4 |
| `analysisMode` modes | 16+ | 9 | 9 | 0 (lane으로 대체) |
| Main bundle (gzip) | 154 KB | ~130 KB | ~130 KB | ~110 KB |
| ESLint errors/warnings | 0/0 | 0/0 | 0/0 | 0/0 |

---

# 📝 완료 기준 (DONE Definition)

## Phase 1
- [ ] 모든 cut 파일 삭제
- [ ] TypeScript/ESLint/Tests clean
- [ ] Backend import OK
- [ ] News/Research merged
- [ ] tag `phase-1-cut-complete`

## Phase 2
- [ ] 5개 hook 파일 존재 + 사용됨
- [ ] `MainPage.tsx` <1,000 줄
- [ ] `mars_catalog.json` 존재
- [ ] TypeScript/ESLint/Tests clean
- [ ] Manual smoke 통과
- [ ] tag `phase-2-hooks-complete`

## Phase 3
- [ ] `/api/inspector/at-point` endpoint 응답
- [ ] 4 lane 컴포넌트 + LaneTabs 동작
- [ ] CrossSection 동작
- [ ] SearchBar 카탈로그 검색 동작
- [ ] LayerPanel 간소화 적용
- [ ] Legacy Inspector 삭제
- [ ] URL state 정리 (mode=X 제거)
- [ ] `/web:qa` 20개 시나리오 통과
- [ ] tag `phase-3-new-inspector-complete`
- [ ] Production 서버 재시작 + health check 통과
- [ ] GitHub push 완료

---

# 🚀 실행 순서 요약

```
1. Phase 1 (Cut)          → 2~3시간 → commit + tag
2. Phase 2 (Extract)      → 4~6시간 → commit + tag
3. Phase 3 (New Inspector)→ 1~2일  → commit + tag + deploy
```

각 Phase 후:
1. `npx tsc --noEmit` 0 errors
2. `npx eslint .` 0 errors/warnings
3. `npx vitest run` 8/8 pass
4. `npx vite build` success
5. Backend `python -c "import app"` OK
6. Manual smoke test
7. Git commit + tag
8. (Phase 3에서만) Deploy + GitHub push

---

**END of workflow.md**
