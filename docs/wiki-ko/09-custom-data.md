# 사용자 데이터 업로드 (Custom Data)

사용자가 직접 GeoTIFF 파일을 업로드하여 화성 지도 위에 오버레이로 표시할 수 있습니다.

---

## 지원 형식

- **GeoTIFF (.tif, .tiff)**: 화성 좌표계(Mars IAU 등) 또는 일반 좌표계

---

## 업로드 워크플로우

### 1. 파일 선택
- Data Download 페이지 또는 메인 페이지에서 업로드 UI 접근
- GeoTIFF 파일 드래그 앤 드롭 또는 파일 선택

### 2. 검증 (Validation)
서버가 자동으로 검증하는 항목:

| 항목 | 설명 |
|------|------|
| CRS | 좌표 참조 시스템 유효성 |
| Bounds | 지리적 범위 (경위도) |
| Size | 파일 크기 |
| Bands | 밴드 수 |
| Data Type | 픽셀 데이터 타입 |

### 3. 업로드 & 오버레이 생성
- 검증 통과 시 서버에 저장
- 투명 PNG 오버레이 자동 생성
- 지도에 즉시 표시 가능

---

## Inspector 표시

업로드된 데이터셋을 선택하면 Inspector에 표시되는 정보:

- Dataset 이름 (파일명 또는 사용자 지정)
- CRS 정보
- 해상도 (m/pixel)
- 밴드 수
- 지리적 범위 (서/동/남/북 경위도)

---

## 관리

| 작업 | 설명 |
|------|------|
| 목록 조회 | 업로드된 모든 데이터셋 나열 |
| 오버레이 표시 | 지도 위에 반투명 이미지로 표시 |
| 삭제 | 서버에서 파일 및 오버레이 삭제 |

---

## API

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/custom/validate` | 파일 검증만 (업로드 X) |
| `POST` | `/api/custom/upload` | 파일 업로드 + 오버레이 생성 |
| `GET` | `/api/custom/datasets` | 데이터셋 목록 |
| `GET` | `/api/custom/{id}/overlay.png` | 오버레이 이미지 |
| `DELETE` | `/api/custom/{id}` | 데이터셋 삭제 |

데이터는 `backend/custom_data/` 디렉토리에 저장됩니다.
