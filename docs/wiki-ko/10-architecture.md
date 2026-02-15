# 시스템 아키텍처

---

## 전체 구조

```
┌─────────────────────────────────────────────────┐
│                    브라우저                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │LayerPanel│  │  MapView  │  │   Inspector   │  │
│  │  (좌측)   │  │ (Cesium) │  │    (우측)      │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       └──────────┬───┴───────────────┘           │
│              MainPage (상태 허브)                   │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/JSON
┌──────────────────┴──────────────────────────────┐
│              FastAPI 백엔드 (:8000)                │
│  ┌──────────────────────────────────────┐       │
│  │           app.py (라이프사이클)         │       │
│  │  - GeoJSON 인덱스 캐시                 │       │
│  │  - aiohttp 연결 풀                    │       │
│  │  - 인덱스 자동 복구                    │       │
│  └──────┬───────────────────────────────┘       │
│         │                                        │
│  ┌──────┴──────────────────────────────┐        │
│  │           API 라우터                   │        │
│  │  search_router   footprints_router  │        │
│  │  ai_search       proximity_router   │        │
│  │  terrain_router  sharad_highres_*   │        │
│  │  custom_router   fieldnotes_router  │        │
│  │  crism/router    hirise_pixel       │        │
│  │  index_repair    ai_analysis_router │        │
│  └──────┬───────────────────────────────┘        │
│         │                                        │
│  ┌──────┴──────────┐  ┌────────────────┐        │
│  │  로컬 데이터      │  │  외부 API       │        │
│  │  - index.geojson │  │  - NASA ODE    │        │
│  │  - .img/.tif/.dat│  │  - Gemini AI   │        │
│  │  - score_stats   │  │                │        │
│  └─────────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────┘
```

---

## 프론트엔드 아키텍처

### 상태 관리: MainPage 허브 패턴

`MainPage.tsx`가 전체 앱 상태를 관리하고, 자식 컴포넌트에 props로 전달합니다.

**주요 상태 (~40개 useState)**:

| 카테고리 | 상태 예시 |
|----------|----------|
| 지도 설정 | viewMode, baseLayer, viewBounds, showGrid |
| 장비 표시 | showCRISM, showHIRISE, showSHARAD, ... |
| 선택 제품 | selected, inspectedProductId |
| 분석 모드 | analysisMode, terrainPoint, aiAnalysisPin |
| 오버레이 | activeOverlays (Map), bringToFrontId |
| 필터 | overlapResult, iceScoreFilter, filteredProductIds |
| 필드 노트 | mapFieldNotes, showFieldNotesOnMap |

### 성능 최적화

| 기법 | 설명 |
|------|------|
| `React.memo()` | MapView, Inspector, LayerPanel 등 대형 컴포넌트 감싸기 |
| `useRef` 안정화 | activeOverlaysRef 등으로 콜백 의존성 안정화 |
| `useMemo` | inspectedProductId, mapFieldNotesForView 메모이제이션 |
| 코드 분할 | Three.js, Recharts, DataDownloadPage → `React.lazy()` |
| 쓰로틀링 | MapView drillPick 50ms 쓰로틀 (최대 20 picks/sec) |
| 아이콘 캐시 | 필드 노트 아이콘 장비별 캐싱 |

### 주요 컴포넌트

```
MainPage
├── LayerPanel (좌측)
│   ├── ViewMode 섹션
│   ├── BaseMap 섹션
│   ├── Footprints 섹션
│   ├── Analysis Tools 섹션
│   ├── Overlap Filter 섹션
│   ├── Ice Score Filter 섹션
│   ├── Field Notes 섹션
│   ├── Displayed Products 섹션
│   └── Active Overlays 섹션
│
├── MapView (중앙)
│   └── Cesium.js Viewer
│       ├── Footprint Entities
│       ├── Overlay Entities
│       ├── Field Note Markers
│       └── Coordinate Grid
│
└── Inspector / Analysis Panel (우측)
    ├── Inspector (제품별)
    ├── SlopeAnalysis
    ├── Slope3DViewer (lazy)
    ├── LineProfile
    ├── AiAnalysisPanel
    └── SharadHiresInspector
```

### FootprintManager

Cesium 엔티티와 GeoJSON feature를 매핑하는 유틸리티 클래스:

- `loadFootprints(instrument)` — 서버에서 footprint 로드, Cesium 엔티티 생성
- `getFeatures(instrument)` — 장비별 GeoJSON feature 목록 반환
- `setVisible(instrument, bool)` — 장비별 가시성 일괄 토글
- **요청 중복 방지**: 동일 장비에 대한 동시 요청을 하나로 합침 (in-flight dedup)

### 엔티티 ID 규칙

| 패턴 | 용도 |
|------|------|
| `{INSTRUMENT}_FP_{productId}` | Footprint 폴리곤/라인 |
| `{INSTRUMENT}_LBL_{productId}` | 라벨 텍스트 |
| `QUICKVIEW_OVERLAY_{productId}` | 오버레이 이미지 |

### 가시성(Visibility) 패턴

여러 조건이 AND로 결합되어 각 엔티티의 `show` 속성을 제어:

```
entity.show = instrumentVisible
           && !overlapFiltered
           && !iceScoreFiltered
           && inViewBounds
```

`applyInstrumentVisibility` 헬퍼 + `overlapResultRef` 패턴으로 여러 useEffect 간 충돌 방지.

---

## 백엔드 아키텍처

### FastAPI Lifespan

서버 시작/종료 시 자동 실행:

```python
@asynccontextmanager
async def lifespan(app):
    # Startup
    app.state.http_session = aiohttp.ClientSession()   # ODE 연결 풀
    _preload_indices_parallel()                         # GeoJSON 병렬 로드
    asyncio.create_task(_background_index_repair(app))  # 인덱스 자동 복구

    yield

    # Shutdown
    await app.state.http_session.close()
```

### 캐싱 전략

| 캐시 | 유형 | 설명 |
|------|------|------|
| GeoJSON 인덱스 | 메모리 (startup) | 6개 index.geojson → dict + bytes + gzip 사전 압축 |
| ODE 응답 | TTLCache(256, 300s) | ODE API 응답 5분 캐시 |
| SHARAD PNG | LRUCache(5) | 라다그램 이미지 |
| MOLA 타일 | LRUCache(30) | 지형 데이터 |
| DTM 파일 | LRUCache(10) | rasterio 핸들 (해제 관리) |
| 타일 | lru_cache(8192) | 렌더링된 PNG 타일 |

### GZip 압축

- `GZipMiddleware(minimum_size=1000)` — 1KB 이상 JSON 응답 자동 압축
- GeoJSON 인덱스: 사전 압축된 gzip bytes를 직접 전송 (per-request 압축 회피)
- Cache-Control: 24시간 (86400초)

### 인덱스 자동 복구 시스템

서버 시작 시 백그라운드 태스크로 실행:

1. 각 장비의 데이터 디렉토리 스캔
2. 디스크에 있지만 index.geojson에 없는 "고아" 제품 탐지
3. ODE API로 footprint 좌표 조회 (실패 시 Point(0,0) 폴백)
4. index.geojson에 추가 → 인메모리 캐시 갱신

---

## 데이터 흐름

### Footprint 표시 흐름

```
사용자: CRISM 토글 ON
  → MainPage: showCRISM = true
  → MapView: useEffect 감지
  → FootprintManager.loadFootprints("CRISM")
    → GET /api/footprints?instrument=CRISM&bbox=...
    → 서버: 로컬 인덱스에서 bbox 필터링
    → GeoJSON 반환
  → Cesium 엔티티 생성 (폴리곤 + 라벨)
  → onFootprintsLoaded(count)
  → LayerPanel: footprintCounts 업데이트
```

### 오버레이 활성화 흐름

```
사용자: Inspector에서 Quickview 버튼 클릭
  → MainPage: activeOverlays.set(productId, {type: "quickview", opacity: 0.7})
  → MapView: useEffect 감지
  → Cesium: Entity rectangle의 material을 이미지 URL로 설정
  → 이미지 로드: GET /crism/quickview/{product_id}.png
  → 지도에 반투명 이미지 표시
```

### AI 검색 흐름

```
사용자: "Jezero crater 근처 CRISM 데이터" 입력
  → POST /api/ai/gemini/preview
    → Gemini: 자연어 파싱
    → 결과: {region: "jezero_crater", instruments: ["CRISM"]}
  → POST /api/ai/gemini/execute
    → region → bbox 변환 (mars_regions.py 룩업)
    → ODE spatial 검색 (bbox 기반)
    → 결과 반환
  → DataDownloadPage: 결과 목록 표시
```

---

## 외부 서비스 연동

### NASA ODE (Orbital Data Explorer)

- Base URL: `https://oderest.rsl.wustl.edu/live2`
- 사용처: 제품 검색, 공간 검색, footprint 조회, 번들 해석
- 주의: 경도는 0-360 체계 → 프론트엔드의 -180/180으로 변환 필요

### Google Gemini

- 사용처: AI 검색 (쿼리 파싱), AI 분석 (evidence 기반 추론)
- API 키: 환경 변수 또는 `~/.gemini/settings.json`

### aria2c

- 사용처: 대용량 파일 병렬 다운로드
- 선택 사항: 미설치 시 aiohttp 폴백
