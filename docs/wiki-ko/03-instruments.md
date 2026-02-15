# 관측 장비 (Instruments)

MarsLab은 Mars Reconnaissance Orbiter(MRO) 탑재 6종 관측 장비의 데이터를 지원합니다.

---

## 장비 요약

| 장비 | 색상 | Footprint 형태 | 주요 데이터 |
|------|------|----------------|-------------|
| CRISM | 청록 `#00FFFF` | 폴리곤 | 적외선 분광 이미지 |
| HiRISE | 노랑 `#FFFF00` | 폴리곤 | 초고해상도 표면 이미지 |
| SHARAD | 주황 `#FFA500` | 라인 | 지하 레이더 프로파일 |
| SHARAD High-Res | 금색 `#FFD700` | 라인 | 고해상도 레이더 프로파일 |
| CTX | 분홍 `#FF69B4` | 폴리곤 | 중해상도 컨텍스트 이미지 |
| HiRISE DTM | 갈색 `#8B4513` | 폴리곤 | 수치 지형 모델 (고도) |

---

## CRISM (Compact Reconnaissance Imaging Spectrometer for Mars)

**개요**: 가시광~적외선 영역(0.4~4.0 μm)의 분광 이미지를 촬영하는 장비. 광물 조성 분석에 핵심적.

**데이터 유형**:
- **MTRDR** (Map-projected Targeted Reduced Data Record): 지도 투영된 처리 데이터
- **TRDR** (Targeted Reduced Data Record): 미투영 처리 데이터

**지원 기능**:
- Footprint 지도 표시 (폴리곤)
- **Quickview 오버레이**: 표면 이미지 빠른 미리보기
- **Browse 오버레이**: HYD(수화물), ICE(얼음), IC2(이산화탄소 얼음) 광물 분포도
- **Score 오버레이**: Ice/Hydration 점수 히트맵
- **스펙트럼 분석**: 클릭 지점의 파장별 반사율 그래프
- **RGB 밴드 합성**: 3개 파장을 R/G/B에 배정하여 합성 이미지 생성
  - 프리셋: True Color, Mineralogy, Inverted, Hydration

**Footprint 색상**: 청록 (`#00FFFF`)
**제품 ID 패턴**: `frt0001fd76_07_if166j_mtr3`

---

## HiRISE (High Resolution Imaging Science Experiment)

**개요**: 화성 표면을 최대 25cm/px 해상도로 촬영하는 초고해상도 카메라.

**데이터 유형**:
- **RDRV11** (Reduced Data Record): RED 채널 고해상도 이미지 (.JP2 → .tif 변환)

**지원 기능**:
- Footprint 지도 표시 (폴리곤)
- **Quickview 오버레이**: 축소된 미리보기 이미지
- **High-Res 오버레이**: 원본 해상도 투명 오버레이
- **픽셀 통계**: 클릭 지점 주변의 DN(Digital Number) 분석
  - 윈도우 크기 선택 (3x3 ~ 21x21)
  - Mean, Median, StdDev, Min, Max 통계
  - 히스토그램

**Footprint 색상**: 노랑 (`#FFFF00`)
**제품 ID 패턴**: `ESP_024943_2345_RED`

---

## SHARAD (Shallow Radar)

**개요**: 15~25 MHz 주파수의 지하투과 레이더. 화성 표면 아래 수백 미터~수 킬로미터의 지하 구조를 탐사.

**데이터 유형**:
- **USRDRV2** (US Reduced Data Record): 처리된 라다그램 이미지

**지원 기능**:
- Footprint 지도 표시 (**LineString** — 궤도 트랙 형태)
- **Quickview 팝업**: 트랙 클릭 시 라다그램 썸네일 표시
- 지하 구조 시각화

**Footprint 색상**: 주황 (`#FFA500`)
**제품 ID 패턴**: `S_00172401_THM`

---

## SHARAD High-Res (고해상도 SHARAD RDR)

**개요**: SHARAD의 고해상도 처리 버전. 더 정밀한 지하 프로파일 데이터.

**데이터 유형**:
- **RDR** (Reduced Data Record): PDS3 바이너리 형식 (.dat + .lbl)

**지원 기능**:
- Footprint 지도 표시 (**LineString** — 상세 궤도 좌표)
- **라다그램 이미지 생성**: .dat 파일에서 실시간 파워 이미지 렌더링
- **궤도 지오메트리**: 트랙을 따른 위도/경도/고도 정보
- **Surface Picking**: 표면 반사 신호 자동 검출
- **Cluttergram**: 지형 산란 시뮬레이션 이미지 (비교 분석용)
- **3D 지하 시각화**: Three.js 기반 3D 라다그램 뷰어

**Footprint 색상**: 금색 (`#FFD700`)
**제품 ID 패턴**: `R_3578901_001_SS19_700_A`

---

## CTX (Context Camera)

**개요**: 약 6m/px 해상도로 넓은 영역을 촬영하는 컨텍스트 카메라. HiRISE 관측 대상 선정에 활용.

**데이터 유형**:
- 원격 타일 서비스 기반 (로컬 다운로드 불필요)

**지원 기능**:
- Footprint 지도 표시 (폴리곤)
- 타일 오버레이 표시 (외부 타일 서버 연동)

**Footprint 색상**: 분홍 (`#FF69B4`)
**제품 ID 패턴**: `b21_017819_2025_xn_22n048w`

---

## HiRISE DTM (Digital Terrain Model)

**개요**: HiRISE 스테레오 이미지 쌍에서 생성된 수치 지형 모델. 고해상도 고도 데이터 제공.

**데이터 유형**:
- **DTM**: .IMG 형식 고도 그리드 (1m/px 급 해상도)
- **Orthoimage**: .JP2 형식 정사보정 이미지

**지원 기능**:
- Footprint 지도 표시 (폴리곤)
- **오버레이**: 정사보정 이미지 또는 hillshade 표시
- **3D 지형 뷰어**: Three.js 기반 DTM 3D 렌더링
- **고도 쿼리**: 마우스 호버 시 실시간 고도값 표시
- **Slope 분석**: DTM 기반 정밀 경사도 분석

**Footprint 색상**: 갈색 (`#8B4513`)
**제품 ID 패턴**: `DTEEC_060706_2195_060416_2195_A01`

> DTM 제품 ID는 `DTEEC_`, `DTEED_` 등 다양한 접두사를 가집니다. 모두 `DTE`로 시작합니다.

---

## Footprint 로드 방식

1. 좌측 패널 **Footprints** 섹션에서 원하는 장비 토글
2. 지도를 원하는 영역으로 이동
3. 현재 뷰포트 범위의 footprint가 자동 로드됨
4. 로드된 제품 수가 표시됨 (최대 2,000개, 초과 시 `truncated`)

### 뷰포트 기반 로딩
- 지도 이동/줌 시 해당 영역의 footprint를 서버에서 가져옴
- **LOD(Level of Detail)** 시스템으로 줌 레벨에 따라 표시 방식 자동 조절
- 멀리서 보면 중심점만, 가까이 가면 전체 폴리곤/라인 표시
