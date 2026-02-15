# 제품 인스펙터 (Inspector)

지도에서 footprint를 클릭하면 우측에 Inspector 패널이 열립니다. 선택한 제품의 상세 정보를 확인하고, 오버레이를 활성화하며, 분석 기능을 사용할 수 있습니다.

---

## 공통 기능

모든 장비의 제품에 공유되는 기능:

### 메타데이터
- Product ID
- 위도/경도 좌표
- Quickview 미리보기 이미지 (있는 경우)

### 오버레이 컨트롤 (Footer)

Inspector 하단에 위치한 오버레이 버튼:

| 버튼 | 색상 | 설명 | 대상 장비 |
|------|------|------|-----------|
| Quickview | 초록 | 축소된 미리보기 이미지 | 전체 |
| High-Res | 보라 | 원본 해상도 이미지 | HiRISE (다운로드 필요) |
| Browse HYD | 자홍 | 수화물 분포도 | CRISM |
| Browse ICE | 파랑 | 얼음 분포도 | CRISM |
| Browse IC2 | 청록 | CO2 얼음 분포도 | CRISM |
| Score Ice | 하늘색 | 얼음 점수 히트맵 | CRISM |
| Score Hyd | 분홍 | 수화물 점수 히트맵 | CRISM |

- 활성화된 오버레이는 지도 위에 반투명하게 표시됨
- **Opacity 슬라이더**: 투명도 0~100% 조절

### 추가 액션

- **Add Field Note**: 현재 제품에 대한 메모 작성
- **View Field Note**: 이미 작성된 메모 조회
- **Find Related Products**: Data Download 페이지로 이동하여 관련 제품 검색
- **Export Statistics**: 통계 내보내기

---

## HiRISE Inspector

### Pixel 탭

HiRISE 고해상도 이미지의 픽셀 수준 분석:

1. 지도에서 HiRISE 오버레이가 활성화된 영역 클릭
2. 클릭 지점 주변의 DN(Digital Number) 값 분석

**설정**:
- **Window Size**: 분석 윈도우 크기 선택 (3x3, 5x5, 7x7, 9x9, 11x11, 15x15, 21x21)

**출력**:
- Mean (평균)
- Median (중앙값)
- StdDev (표준편차)
- Min / Max (최솟값 / 최댓값)
- DN 히스토그램 (막대 그래프)

---

## CRISM Inspector

### Spectrum 탭

클릭 지점의 분광 프로파일(스펙트럼) 표시:

1. CRISM footprint 클릭으로 제품 선택
2. 오버레이 활성화 후 관심 지점 클릭
3. 해당 픽셀의 **파장(wavelength) vs 반사율(reflectance)** 그래프 표시

- X축: 파장 (μm), 약 1.0~4.0 μm 범위
- Y축: I/F 반사율
- 광물 식별의 핵심 도구

### Bands 탭

RGB 밴드 합성 이미지 생성:

**파장 슬라이더**:
- R(빨강) 파장: 1.0~4.0 μm
- G(초록) 파장: 1.0~4.0 μm
- B(파랑) 파장: 1.0~4.0 μm

**프리셋**:

| 프리셋 | R | G | B | 용도 |
|--------|---|---|---|------|
| True Color | 0.6 μm | 0.53 μm | 0.44 μm | 자연색 |
| Mineralogy | 2.5 μm | 1.5 μm | 1.08 μm | 광물 분류 |
| Inverted | 1.08 μm | 1.5 μm | 2.5 μm | 반전 광물 |
| Hydration | 1.9 μm | 1.5 μm | 1.2 μm | 수화물 탐지 |

- **Apply RGB Changes** 버튼으로 합성 이미지 적용
- 합성 결과는 오버레이로 지도에 표시됨

---

## HiRISE DTM Inspector

### 메타데이터
- Product ID, 좌표
- DTM 파일명, Orthoimage 파일명 (있는 경우)

### 3D View
- **Show 3D View** 버튼 클릭 → Three.js 기반 3D 지형 뷰어 실행
- DTM 고도 데이터를 3D 메시로 렌더링
- 마우스로 회전/줌/팬 조작
- 색상화된 고도 표현 (낮은 곳: 파란색, 높은 곳: 빨간색)

---

## SHARAD High-Res Inspector

### 라다그램 표시
- SHARAD High-Res 제품의 라다그램 이미지 표시
- 전력(power) 기반 이미지: 밝을수록 강한 반사 신호
- 지하 층 구조, 얼음 분포 등 분석 가능

### 궤도 지오메트리
- 관측 트랙을 따른 위도/경도/고도 정보
- 시작점~끝점 좌표 표시

---

## Custom Dataset Inspector

사용자가 업로드한 GeoTIFF 데이터의 메타데이터:

- Dataset 이름
- CRS (좌표 참조 시스템)
- 해상도 (m/pixel)
- 밴드 수
- Geographic bounds (서/동/남/북 경위도)
