# 7. SPICE, XYZ Products, and Co-Registration Methodology

## 7.1 SPICE System

### Primary References
- **Acton, C.H. (1996). "Ancillary data services of NASA's Navigation and Ancillary Information Facility." Planet. Space Sci., 44(1), 65-70.** DOI: 10.1016/0032-0633(95)00107-7
- **Acton, C.H., Bachman, N., Semenov, B., Wright, E. (2018). "A look towards the future in the handling of space science mission geometry." Planet. Space Sci., 150, 9-12.** DOI: 10.1016/j.pss.2017.02.013

### Kernel Types

| Kernel | 확장자 | 내용 |
|---|---|---|
| **SPK** | .bsp | 위치/속도 (Chebyshev 다항식 또는 이산 상태) |
| **CK** | .bc | 자세 (quaternion + 각속도), SCLK 인덱싱 |
| **IK** | .ti | 기기 FOV, boresight, 픽셀-각도 매핑 |
| **FK** | .tf | 프레임 정의 및 프레임 트리 (고정/시변 관계) |
| **LSK** | .tls | Leapseconds (UTC→ET 변환) |
| **PCK** | .tpc/.bpc | 천체 상수 (반경, 자전축, GM) |
| **SCLK** | .tsc | 우주선 시계→ET 매핑 |
| **DSK** | .bds | 상세 형상 모델 (삼각 메시) |
| **MK** | .tm | 메타커널 (로드 목록) |

### 좌표 변환 메커니즘
- **Frame tree** 구성: 방향 그래프로 연결된 참조 프레임들
- A→B 변환: R(A→B) = R(B→root)^T x R(A→root) (회전 합성)
- 위치: SPK 세그먼트 체이닝 (공통 중심체 경유)
- 핵심 루틴: `PXFORM` (3x3 회전), `SPKEZ`/`SPKEZR` (위치+속도), `FURNSH` (커널 로드)

## 7.2 Mars 2020 프레임 체계

### 프레임 계층
```
IAU_MARS (body-fixed)
  └─ M2020_SITE_nnn (local level, 지표 고정)
       └─ M2020_LOCAL_LEVEL_nnn
            └─ M2020_ROVER_NAV_FRAME (로버 본체 고정)
                 └─ M2020_RSM_HEAD_FRAME (마스트 헤드)
                      └─ M2020_MASTCAM-Z_LEFT / RIGHT
```

### 프레임 정의
| 프레임 | 정의 | 원점 |
|---|---|---|
| **SITE_FRAME** | 국소 수준면, +X=N, +Y=E, +Z=Down | 사이트 선언 시 로버 위치 |
| **ROVER_NAV_FRAME** | 로버 본체 고정, +X=전진, +Y=우현, +Z=Down | 로버 중앙 바닥 |
| **RSM_HEAD_FRAME** | 마스트 헤드, pan/tilt에 따라 변동 | RSM 회전 중심 |
| **IAU_MARS** | Mars 본체 고정, Airy-0 기준 경선 | Mars 질량 중심 |

### 변환 체인: 기기 → IAU_MARS
```
Mastcam-Z pixel → Camera Frame (IK/CAHVOR 역투영)
    → RSM_HEAD_FRAME (FK: 카메라→마스트 고정 회전)
    → ROVER_NAV_FRAME (CK: RSM pan/tilt)
    → SITE_FRAME (CK: 로버 자세)
    → IAU_MARS (FK/SPK: 사이트→Mars 변환)
```

## 7.3 SPICE 정확도 (Mars 2020)

| 항목 | 정확도 |
|---|---|
| 착륙 위치 (EDL 후 HiRISE 확인) | ~1-2 m |
| 사이트 내 상대 위치 (VO) | ~0.1 m |
| 절대 위치 (PLACES, 궤도 tie-point 후) | **~1-3 m** |
| RSM pointing (CK) | ~0.05 deg (~1 mrad) |
| VO 누적 드리프트 | ~0.1-0.5% 주행거리 |

## 7.4 XYZ Product 생성 파이프라인

### References
- **Di, K., Li, R. (2004). "CAHVOR camera model and its photogrammetric conversion for planetary applications." JGR, 109, E04004.** DOI: 10.1029/2003JE002199
- **Gennery, D.B. (2006). "Generalized Camera Calibration Including Fish-Eye Lenses." IJCV, 68(3), 239-266.** DOI: 10.1007/s11263-006-5168-1

### 파이프라인
1. Mastcam-Z 스테레오 쌍 촬영 (24.4cm baseline, 매칭 줌)
2. OPGS correlator: 다중해상도 area-based 밀집 스테레오 매칭 → disparity map
3. CAHVOR 모델로 삼각측량 → 3D 점
4. 3-band IMG (X, Y, Z), 32-bit float, BSQ
5. 좌표계: **SITE_FRAME** (`COORDINATE_SYSTEM_NAME = "SITE_FRAME"`)

### CAHVOR 카메라 모델
- **C** (3): 투영 중심 위치
- **A** (3, unit): 카메라 축 (촬영 방향)
- **H** (3): 수평 정보 (초점거리, 이미지 중심, 픽셀 스케일)
- **V** (3): 수직 정보
- **O** (3, unit): 광학축 (렌즈 편심 고려)
- **R** (3): 방사 왜곡 계수

순방향: P → (h,v) = ((P-C)·H / (P-C)·A, (P-C)·V / (P-C)·A) + 왜곡 보정
역방향: (h,v) → 왜곡 제거 → 시선 벡터 → 스테레오 삼각측량

### XYZ 정밀도 vs 거리 (110mm 줌 기준)

| 거리 | Range 정밀도 | Lateral 정밀도 |
|---|---|---|
| 2 m | ~1 cm | ~0.5 cm |
| 5 m | ~5 cm | ~1 cm |
| 10 m | ~20 cm | ~3 cm |
| 25 m | ~1-2 m | ~10 cm |
| 50 m | ~5-10 m | ~25 cm |
| >100 m | 신뢰 불가 | lateral만 가용 |

**기본 관계**: sigma_range ≈ (range^2 x sigma_disparity) / (focal_length x baseline)
- sigma_disparity ≈ 0.25-0.5 pixels

## 7.5 로버 위치결정 (Localization)

### References
- **Li, R., et al. (2004). "Rover Localization and Landing-Site Mapping Technology for the 2003 Mars Exploration Rover Mission." PE&RS, 70(1), 77-90.** DOI: 10.14358/PERS.70.1.77
- **Li, R., et al. (2006). "Spirit rover localization and topographic mapping at the landing site of Gusev crater, Mars." JGR, 111, E02S06.** DOI: 10.1029/2005JE002483

### 계층적 위치결정
1. **Wheel odometry**: 2-10% 오차 (지형 의존)
2. **Visual odometry (VO)**: Navcam 스테레오, 특징점 추적, 6-DOF 추정 → **0.1-0.5%** 드리프트
3. **PLACES** (Parker et al.): HiRISE tie-point 기반 bundle adjustment → **절대 1-3 m**

## 7.6 선행 정합 연구

### Tao & Muller (2016), Icarus 280, DOI: 10.1016/j.icarus.2016.06.017
- MER/MSL Navcam ortho mosaic → HiRISE 정합
- NCC/MI 기반 멀티스케일 매칭
- 정합 정확도: **1-2 HiRISE pixels (25-50 cm)**
- 한계: nadir vs oblique 시점 차이, 해상도 불일치, 조명 차이

### Di et al. (2008), ISPRS J., DOI: 10.1016/j.isprsjprs.2007.07.007
- MER Pancam XYZ → CAHVOR → IAU_MARS → 정사영 → HiRISE 특징 매칭
- 정합 정확도: **1-5 m**

### Paar et al. (2023), Earth Space Sci., DOI: 10.1029/2022EA002532
- PRoViP/PRo3D: Mastcam-Z 다중 스테레오 bundle adjustment
- 밀집 3D 점군 + 텍스처 메시
- 다중 baseline → 원거리 정밀도 향상
- 궤도 정합은 수행하지 않음 (3D만)

### 정합 정확도 비교

| 연구 | 카메라 | 궤도 기준 | 방법 | 정확도 |
|---|---|---|---|---|
| Tao & Muller (2016) | Navcam | HiRISE 25cm | NCC ortho vs ortho | 25-50 cm |
| Di et al. (2008) | MER Pancam | HiRISE 25cm | 정사영 + feature | 1-5 m |
| Paar et al. (2023) | Mastcam-Z | N/A | 다중스테레오 BA | cm급 (국소) |
| PLACES (Parker) | Navcam | HiRISE 25cm | 수동+반자동 tie-point | 1-3 m |

## 7.7 화성 좌표계

### Reference
- **Archinal, B.A., et al. (2018). "Report of the IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015." CMDA, 130, 22.** DOI: 10.1007/s10569-017-9805-5

### IAU_MARS
- 원점: Mars 질량 중심
- +Z: 북극 자전축
- 경선 0: Airy-0 크레이터
- 자전 모델: W = 176.049 + 350.891982443 deg/day x d (J2000 기준)

### Areocentric vs Areographic
| | Areocentric (planetocentric) | Areographic (planetographic) |
|---|---|---|
| 정의 | 중심→표면점 직선과 적도면의 각 | 표면 법선과 적도면의 각 |
| 관계 | tan(phi_g) = tan(phi_c) / (1-f)^2 | 편평한 천체에서 항상 |phi_g| >= |phi_c| |
| 경도 | East-positive (0-360) | West-positive (0-360) |
| **HiRISE** | **Planetocentric, East-positive** | |

### Mars 참조 타원체 (IAU 2015)
| 파라미터 | 값 |
|---|---|
| 적도 반경 (a) | 3396.19 km |
| 극 반경 (c) | 3376.20 km |
| 편평률 (f) | 0.005886 |
| 평균 반경 | 3389.50 km |

## 7.8 완전한 변환 체인: Mastcam-Z 픽셀 → HiRISE 픽셀

```
Mastcam-Z pixel (h, v)
    ↓ CAHVOR 역투영 + 스테레오 삼각측량
XYZ in SITE_FRAME (meters)
    ↓ SPICE: SITE→ROVER(CK) → IAU_MARS(SPK/FK)
IAU_MARS body-fixed XYZ (km)
    ↓ RECGEO/RECLAT: Cartesian → areocentric (lon, lat, alt)
Areocentric (lon, lat)
    ↓ HiRISE Equirectangular 투영
    ↓ sample = (lon - lon_0) x R x cos(lat_0) / scale
    ↓ line   = (lat_0 - lat) x R / scale
HiRISE RDR pixel (sample, line)
```

## 7.9 오차 예산 (Error Budget)

| 오차 원인 | 크기 | HiRISE 픽셀 영향 |
|---|---|---|
| Mastcam-Z XYZ (5m 거리) | ~5cm range, ~1cm lateral | <1 px |
| Mastcam-Z XYZ (50m 거리) | ~5-10m range, ~25cm lateral | 1-2 px |
| RSM pointing (CK) | ~1 mrad | 5m: 5mm, 50m: 5cm → <1 px |
| **로버 위치결정 (PLACES)** | **1-3 m** | **4-12 px ← 지배적 오차** |
| HiRISE 정사보정 | 0.1-1 px (2.5-25 cm) | 0.1-1 px |
| IAU_MARS 실현 | ~수 m (체계적) | 동일 PCK 사용 시 상쇄 |

**핵심 결론**: 정합 정확도의 병목은 **로버 절대 위치결정 (1-3m = 4-12 HiRISE 픽셀)**. 근거리(<10m) XYZ와 pointing 오차는 서브픽셀. Feature-based 정합으로 추가 개선 필요.
