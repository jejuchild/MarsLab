# 필드 노트 (Field Notes)

연구 과정에서 발견한 내용이나 관찰 기록을 제품에 연결하여 저장하는 메모 시스템입니다.

---

## 필드 노트 작성

### 방법 1: Inspector에서 작성
1. 지도에서 제품(footprint) 클릭 → Inspector 열기
2. 하단 **Add Field Note** 버튼 클릭
3. 메모 다이얼로그에서 내용 입력

### 방법 2: 좌측 패널에서 관리
1. 좌측 패널 **Field Notes** 섹션 열기
2. 기존 노트 목록 확인 및 편집

---

## 메모 구조

| 필드 | 설명 |
|------|------|
| Product ID | 연결된 제품 식별자 |
| Instrument | 관측 장비 |
| Latitude | 위도 |
| Longitude | 경도 |
| Memo | 메모 텍스트 (자유 형식) |
| Tags | 태그 목록 (콤마로 구분) |
| Created At | 작성 시각 |

---

## 태그 시스템

태그를 활용하여 메모를 분류하고 필터링할 수 있습니다.

**태그 예시**:
- `ice` — 얼음 관련 관찰
- `mineral` — 광물 발견
- `landing-site` — 착륙지 후보
- `anomaly` — 이상 징후
- `follow-up` — 추가 조사 필요

**태그 필터링**:
- 좌측 패널에서 특정 태그 선택 시 해당 태그의 노트만 표시
- 여러 태그 동시 선택 가능

---

## 지도 위 마커 표시

- **Show on Map** 토글로 필드 노트를 지도 위에 핀 마커로 표시
- 각 장비별 고유 색상 아이콘
  - 아이콘은 장비별로 캐싱되어 재사용 (성능 최적화)
- 마커 클릭 시 해당 제품의 Inspector 열기 및 위치 이동

---

## CRUD 작업

| 작업 | 방법 |
|------|------|
| **생성** | Inspector → Add Field Note → 메모 입력 → Save |
| **조회** | 좌측 패널 Field Notes 섹션 또는 Inspector → View Field Note |
| **수정** | 좌측 패널에서 노트 선택 → 편집 |
| **삭제** | 좌측 패널에서 노트 선택 → 삭제 |

---

## API

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/fieldnotes` | 모든 필드 노트 조회 |
| `POST` | `/api/fieldnotes` | 새 노트 생성 |
| `PUT` | `/api/fieldnotes/{id}` | 노트 수정 |
| `DELETE` | `/api/fieldnotes/{id}` | 노트 삭제 |

데이터는 `backend/data/field_notes.json`에 JSON 파일로 저장됩니다.
