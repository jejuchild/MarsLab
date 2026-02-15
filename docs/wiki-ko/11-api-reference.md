# API 레퍼런스

MarsLab 백엔드 REST API 전체 문서입니다. 기본 URL: `http://localhost:8000`

---

## 인덱스 & 메타데이터

### GeoJSON 인덱스

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/crism_index.geojson` | CRISM footprint 인덱스 |
| `GET` | `/hirise_index.geojson` | HiRISE footprint 인덱스 |
| `GET` | `/sharad_index.geojson` | SHARAD footprint 인덱스 |
| `GET` | `/hirise_dtm_index.geojson` | HiRISE DTM footprint 인덱스 |

응답: GeoJSON FeatureCollection. 24시간 캐시. GZip 압축.

### 메타데이터

| Method | Path | 파라미터 | 설명 |
|--------|------|----------|------|
| `GET` | `/meta/{name}` | name: 제품명 | HiRISE GeoTIFF 메타데이터 (width, height) |
| `GET` | `/world_meta` | — | 전체 타일 범위, 줌 레벨 설정 |

---

## 타일 & 이미지

### 지도 타일

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/world_tiles/{name}/{z}/{x}/{y}.png` | XYZ 타일 (MOLA/HRSC) |

### 오버레이 이미지

| Method | Path | 파라미터 | 설명 |
|--------|------|----------|------|
| `GET` | `/hirise/overlay/{product_id}.png` | max_size (기본 2048) | HiRISE 투명 오버레이 |
| `GET` | `/hirise/quickview/{product_id}.png` | — | HiRISE 퀵뷰 (투명 배경) |
| `GET` | `/crism/quickview/{product_id}.png` | — | CRISM 퀵뷰 (투명 배경) |
| `GET` | `/sharad/quickview/{product_id}.jpg` | — | SHARAD 라다그램 썸네일 |
| `GET` | `/hirise_dtm/overlay/{product_id}.png` | max_size (기본 2048) | DTM 오버레이 (hillshade 폴백) |

---

## Footprint API

### 뷰포트 기반 Footprint 조회

```
GET /api/footprints
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `instrument` | string | O | CRISM, HIRISE, SHARAD, SHARAD_HIGHRES, CTX, HIRISE_DTM |
| `bbox` | string | O | minLon,minLat,maxLon,maxLat |
| `lod` | string | X | none, point, poly (기본: 자동) |
| `limit` | int | X | 최대 결과 수 (기본 2000, 최대 5000) |
| `camera_height_km` | float | X | LOD 자동 결정용 카메라 높이 |

**응답**: GeoJSON FeatureCollection + 메타데이터
```json
{
  "type": "FeatureCollection",
  "features": [...],
  "metadata": {
    "returned": 142,
    "total_estimate": 142,
    "truncated": false,
    "lod_enforced": "poly"
  }
}
```

### Footprint 통계

```
GET /api/footprints/stats?instrument=CRISM
```

---

## 검색 API

### 제품 ID 검색 (Typeahead)

```
GET /api/search?q={query}&instrument={inst}&limit={n}
```

| 파라미터 | 설명 |
|----------|------|
| `q` | 검색 문자열 (부분 매칭) |
| `instrument` | 장비 필터 (선택) |
| `limit` | 최대 결과 수 (1-50) |

### 로컬 검색

```
GET /api/search/local?q={query}&limit={n}
```

모든 로컬 인덱스 + 화성 지역명을 검색합니다.

### 공간 검색

```
GET /api/search/spatial
```

| 파라미터 | 설명 |
|----------|------|
| `minlat`, `maxlat` | 위도 범위 |
| `westernlon`, `easternlon` | 경도 범위 |
| `instrument` | 장비 필터 |
| `limit` | 최대 결과 수 |

### 좌표 검색

```
GET /api/search/point?lat={lat}&lon={lon}&radius={deg}
```

모든 장비를 한번에 검색. 결과에 거리(km) 포함.

---

## AI 검색 API

### Gemini 상태 확인

```
GET /api/ai/gemini/status
```

### Gemini 프리뷰 (파싱만)

```
POST /api/ai/gemini/preview
Content-Type: application/json

{
  "query": "Jezero crater 근처 CRISM 데이터",
  "max_results": 20
}
```

### Gemini 실행 (파싱 + 검색)

```
POST /api/ai/gemini/execute
Content-Type: application/json

{
  "query": "Jezero crater 근처 CRISM 데이터",
  "max_results": 20,
  "plan": { ... }  // 선택: preview 결과 재사용
}
```

### 화성 지역 목록

```
GET /api/ai/regions
```

~55개 화성 지역의 bbox, 중심 좌표, 설명, 태그 반환.

---

## 근접 검색 API

### 제품 근접 검색

```
GET /api/proximity/search
```

| 파라미터 | 설명 |
|----------|------|
| `product_id` | 기준 제품 ID |
| `instrument` | 기준 장비 |
| `target_instruments` | 대상 장비 (콤마 구분) |
| `mode` | overlap 또는 nearest |
| `limit` | 최대 결과 수 |

### 화성 지역

```
GET /api/proximity/regions           # 전체 목록
GET /api/proximity/regions/{id}      # 특정 지역 상세
```

---

## 다운로드 API

### 다운로드 시작

```
POST /api/download
Content-Type: application/json

{
  "product_id": "frt0001fd76_07",
  "instrument": "CRISM",
  "lat": 22.5,
  "lon": 74.2
}
```

### 다운로드 관리

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/download` | 모든 작업 나열 |
| `GET` | `/api/download/{task_id}` | 특정 작업 상태 |
| `DELETE` | `/api/download/{task_id}` | 특정 작업 취소 |
| `DELETE` | `/api/download` | 전체 취소 |
| `DELETE` | `/api/download/history` | 이력 삭제 |

### 로컬 존재 확인

```
GET /api/exists/{instrument}/{product_id}
```

응답:
```json
{
  "exists": true,
  "has_core": true,
  "has_header": true,
  "has_wavelength": true,
  "has_browse": true,
  "missing_files": [],
  "existing_files": ["img", "lbl", "hdr", "tab", "png"]
}
```

---

## 지형 분석 API

### Slope 통계

```
GET /terrain/slope_stats?lat={lat}&lon={lon}&radius_m={m}
```

응답:
```json
{
  "mean_slope": 2.3,
  "max_slope": 8.1,
  "elevation_m": -4120,
  "count": 156,
  "distribution": { "0_3": 78, "3_5": 12, "5_plus": 10 },
  "safety": "FAVORABLE"
}
```

### 고도 프로파일

```
GET /terrain/line_profile?start_lat=...&start_lon=...&end_lat=...&end_lon=...&num_samples=100
```

### DEM 패치 (3D 시각화용)

```
GET /terrain/dem_patch?lat={lat}&lon={lon}&radius_m={m}&grid_size=128
```

### HiRISE DTM 패치

```
GET /terrain/hirise_dtm_patch?product_id={id}&lat={lat}&lon={lon}&radius_m={m}&grid_size=128
```

### DTM 고도 그리드

```
GET /terrain/hirise_dtm_elevation_grid?product_id={id}&max_size=256
```

---

## DTM 고도 조회

```
GET /hirise_dtm/elevation/{product_id}?lat={lat}&lon={lon}&radius=0.01
```

응답:
```json
{
  "elevation_m": -3890.5,
  "patch_stats": {
    "min_m": -3920.1,
    "max_m": -3850.2,
    "mean_m": -3885.3,
    "std_m": 12.4
  }
}
```

---

## SHARAD High-Res API

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/sharad_highres/index` | 가용 제품 목록 |
| `GET` | `/api/sharad_highres/radargram/{id}` | 라다그램 이미지 (PNG) |
| `GET` | `/api/sharad_highres/geometry/{id}` | 궤도 지오메트리 |
| `GET` | `/api/sharad_highres/surface/{id}` | 표면 picking |

---

## Ice/Hydration 필터 API

```
GET /api/filter/ice?min_score=0.3&min_percent=5.0
GET /api/filter/hyd?min_score=0.3&min_percent=5.0
GET /api/score/stats
```

---

## CRISM 전문 API

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/crism/{product_id}` | 제품 정보 |
| `POST` | `/crism/{product_id}/spectrum` | 픽셀 스펙트럼 (line, sample) |
| `GET` | `/crism/{product_id}/rgb` | RGB 합성 이미지 |

---

## 필드 노트 API

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/fieldnotes` | 전체 노트 조회 |
| `POST` | `/api/fieldnotes` | 노트 생성 |
| `PUT` | `/api/fieldnotes/{id}` | 노트 수정 |
| `DELETE` | `/api/fieldnotes/{id}` | 노트 삭제 |

---

## Custom Data API

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/custom/validate` | GeoTIFF 검증 |
| `POST` | `/api/custom/upload` | GeoTIFF 업로드 |
| `GET` | `/api/custom/datasets` | 데이터셋 목록 |
| `GET` | `/api/custom/{id}/overlay.png` | 오버레이 이미지 |
| `DELETE` | `/api/custom/{id}` | 데이터셋 삭제 |

---

## AI 분석 API

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/ai_analysis/gather_evidence` | 증거 수집 (반경 내 데이터) |
| `POST` | `/api/ai_analysis/ask` | Gemini에 질문 (증거 + 질문) |

---

## 인덱스 복구 API

```
POST /api/repair-index
```

응답:
```json
{
  "total_added": 86,
  "instruments": {
    "sharad_highres": { "scanned": 53, "orphaned": 1, "added": 1, "failed": 0 },
    "crism": { "scanned": 86, "orphaned": 85, "added": 85, "failed": 0 },
    "hirise": { "scanned": 0, "orphaned": 0, "added": 0, "failed": 0 },
    "sharad": { "scanned": 1, "orphaned": 0, "added": 0, "failed": 0 },
    "hirise_dtm": { "scanned": 20, "orphaned": 0, "added": 0, "failed": 0 }
  }
}
```
