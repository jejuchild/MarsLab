# 지도 인터페이스

MarsLab의 핵심 UI는 Cesium.js 기반 인터랙티브 화성 지도입니다.

---

## 뷰 모드

### 3D Globe (기본)
- 화성 전체를 지구본 형태로 표시
- 마우스 드래그로 회전, 스크롤로 줌
- 우클릭 드래그로 틸트(기울기) 조절

### 2D 평면 지도
- 좌측 패널 **View Mode** 섹션에서 전환
- 메르카토르 투영의 평면 지도
- 넓은 영역을 한눈에 볼 때 유용

---

## 베이스맵

좌측 패널 **Base Map** 섹션에서 선택:

| 베이스맵 | 설명 |
|----------|------|
| **MOLA ColorShade** | Mars Orbiter Laser Altimeter 고도 데이터 기반 색상 음영 지도. 기본값. |
| **Mars Express HRSC** | High Resolution Stereo Camera 촬영 이미지 기반 지도. 더 사실적인 표면 텍스처. |

---

## Fly To (위치 이동)

특정 좌표로 카메라를 이동하는 기능:

- **좌표 입력**: 위도, 경도를 입력하고 Fly To 버튼 클릭
- **제품 클릭**: Displayed Products 목록에서 항목 클릭 시 해당 위치로 이동
- **URL 딥링크**: `?flyTo=PRODUCT_ID` 파라미터로 직접 이동 가능

---

## View Bounds (뷰 영역 제한)

지도에서 특정 영역만 표시하도록 제한:

- 좌측 패널에서 **View Bounds** 토글 활성화
- Min/Max Latitude, West/East Longitude 입력
- 설정된 범위 밖의 footprint는 숨김 처리
- 넓은 지역에서 관심 영역만 집중 분석할 때 유용

---

## 좌표 그리드

- **Coordinate Grid** 토글로 위도/경도 격자선 표시
- 10도 간격으로 격자선 렌더링
- 위도/경도 레이블 표시

---

## 마우스 인터랙션

### 호버 (마우스 이동)
- 화면 하단에 현재 커서 위치의 **위도/경도** 실시간 표시
- Footprint 위에 호버 시 **하이라이트** 효과
- HiRISE DTM 영역 위에 호버 시 **실시간 고도값** 표시

### 클릭
- **Footprint 클릭** → 우측 Inspector 패널에 해당 제품 정보 표시
- **지형 클릭** (분석 모드 활성 시) → Slope 분석 / 3D 시각화 / Line Profile 시작
- **SHARAD 트랙 클릭** → 라다그램 퀵뷰 팝업 표시

### 줌 레벨에 따른 LOD (Level of Detail)
지도의 카메라 높이에 따라 footprint 표시 방식이 자동 조절됩니다:

| 카메라 높이 | LOD | 표시 방식 |
|------------|-----|----------|
| > 15,000 km | none | footprint 숨김 (성능 보호) |
| 5,000~15,000 km | point | 중심점만 표시 |
| < 5,000 km | poly | 전체 폴리곤/라인 표시 |

---

## Footprint 레이어

6종 관측 장비의 footprint를 지도 위에 겹쳐 표시합니다. 각 장비마다 고유한 색상이 배정됩니다.

- 좌측 패널 **Footprints** 섹션에서 각 장비별 토글
- **Load** 버튼: 현재 뷰 영역의 footprint를 서버에서 로드
- 로드된 제품 수가 표시됨 (예: `CRISM (142)`)
- 최대 2,000개까지 표시 (초과 시 `truncated` 표시)

자세한 내용은 [관측 장비](03-instruments.md) 페이지 참조.

---

## 오버레이

선택한 제품의 이미지를 지도 위에 직접 겹쳐 표시하는 기능:

- Inspector에서 오버레이 버튼 클릭으로 활성화
- 7가지 오버레이 타입 지원 (Quickview, High-Res, Browse HYD/ICE/IC2, Score Ice/Hyd)
- 좌측 패널 **Active Overlays** 섹션에서 관리
- 개별 투명도(Opacity) 조절 가능 (0~100%)
- **Clear All** 버튼으로 모든 오버레이 일괄 제거

---

## 필드 노트 마커

지도 위에 사용자가 작성한 필드 노트를 핀 형태로 표시:

- **Show on Map** 토글로 표시/숨김
- 장비별 아이콘 색상 구분
- 마커 클릭 시 해당 제품의 Inspector 열기

자세한 내용은 [필드 노트](08-field-notes.md) 페이지 참조.
