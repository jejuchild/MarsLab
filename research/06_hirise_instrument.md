# 6. HiRISE Instrument - Detailed Technical Review

## 6.1 Primary Reference

**McEwen, A.S., Eliason, E.M., Bergstrom, J.W., et al. (2007). "Mars Reconnaissance Orbiter's High Resolution Imaging Science Experiment (HiRISE)." JGR, 112, E05S02.**
- DOI: 10.1029/2005JE002605

## 6.2 Telescope Design

- Aperture: **0.5 m** (행성 탐사 사상 최대 구경)
- Focal length: **12 m** (3-mirror anastigmat, folded optical path)
- f-number: **f/24**
- All-reflective TMA 설계, 복합재 구조로 열안정성 확보
- RED 파장에서 회절한계 성능, ~1 urad 각해상도

## 6.3 Focal Plane / Detector

- **14 CCDs** 총:
  - RED0-RED9 (10개): 스태거 배열, 전체 RED 스워스 구성. 인접 CCD 간 ~48px 중첩
  - BG12, BG13 (2개): RED 스워스 중앙
  - IR10, IR11 (2개): RED 스워스 중앙
- CCD: Teledyne 커스텀, **2048 pixels cross-track** per CCD
- Pixel size: **12 um x 12 um**
- TDI: 8/32/64/128 라인 설정 가능 (지표 이동에 맞춰 적분, SNR 향상)
- 14-bit A/D, 선택적 LUT 압축 (8-bit)
- 각 CCD 2채널 리드아웃 (ch0: 1024px, ch1: 1024px)
- Along-track: pushbroom 방식, 촬영 시간에 따라 20,000-100,000+ 라인

## 6.4 Spectral Channels

| Channel | 파장 범위 | 중심파장 (approx) | CCDs |
|---|---|---|---|
| **RED** | 570-830 nm | ~700 nm | RED0-RED9 (10) |
| **BG** | 400-600 nm | ~500 nm | BG12, BG13 (2) |
| **NIR** | 800-1000 nm | ~900 nm | IR10, IR11 (2) |

RED가 광대역 → SNR 최대화.

## 6.5 공간 해상도

| 고도 | RED | BG/NIR (2x2 binning) |
|---|---|---|
| 300 km (nominal) | ~0.30 m/px | ~0.60 m/px |
| 250 km (periapsis) | **~0.25 m/px** | ~0.50 m/px |

통상 인용 "25 cm/pixel"은 periapsis 기준.

## 6.6 Swath Width

| Channel | 폭 (300km 기준) | Pixels cross-track |
|---|---|---|
| RED | **~6 km** | ~20,000 |
| BG/NIR | ~1.2 km | ~4,000 (binned) |

## 6.7 Typical Image Size

- RED: ~20,000 x 40,000+ pixels (10+ GB uncompressed)
- 개별 CCD: 2048 x N pixels
- Full RED mosaic: 수 GB~수십 GB

## 6.8 SNR

- 설계 목표: SNR > 100:1
- 128 TDI, RED: SNR > 200:1 (잘 조명된 지표)
- SNR ∝ sqrt(TDI lines)
- BG/NIR: RED보다 낮은 SNR (좁은 대역폭, 낮은 QE)

## 6.9 기하 정확도

- Pointing knowledge: ~0.3 mrad (3sigma) → ~100m 지상 위치 불확실성 (300km)
- 제어점 보정 후: **절대 1-5 m** (MOLA 연결)
- **Jitter**: 반작용 휠/태양전지판 → along-track 기하 왜곡. 스태거 CCD로 감지/보정 가능
  - Mattson et al. (2009) HiJACK 보정 기법
- 광학 왜곡: ISIS 파이프라인에서 잘 보정됨

## 6.10 Radiometric Calibration

**Delamere, W.A., Tornabene, L.L., McEwen, A.S., et al. (2010). "Color imaging of Mars by the High Resolution Imaging Science Experiment (HiRISE)." Icarus, 205, 38-52.**
- DOI: 10.1016/j.icarus.2009.03.012

### hical 파이프라인 (ISIS)
1. Dark current subtraction (마스크 픽셀 활용)
2. A/D offset correction
3. Flat-field correction (비행 전 + 비행 중)
4. Gain normalization (채널간, CCD간)
5. DN → I/F 변환: I/F = (pi x radiance) / (solar_irradiance / dist^2)
6. 절대 방사 정확도: ~20% (탑재 보정 타깃 없음, 비행 전 측정 + 교차 보정 의존)

### Known Issues
1. **CCD seam artifacts**: 인접 CCD 경계에서 미세 밝기 불일치 → 모자이크에서 수직 줄무늬
2. **Channel offset**: CCD당 2채널 리드아웃 → 1024px 경계에서 밝기 단차
3. **Jitter**: 기하왜곡 + TDI 적분 불일치 → 번짐/해상도 저하
4. **Detector aging**: 미션 수명 (2005~) 동안 노이즈 증가, hot pixels, CTE 저하 (RED4 ch1 특히)

## 6.11 DTM Production

### Key Reference
**Kirk, R.L., Howington-Kraus, E., Rosiek, M.R., et al. (2008). "Ultrahigh resolution topographic mapping of Mars with MRO HiRISE stereo images." JGR, 113, E00A24.**
- DOI: 10.1029/2007JE003000

### Pipeline
1. 스테레오 쌍 촬영 (수렴각 15-25 deg)
2. hical 방사 보정 (ISIS)
3. Bundle adjustment (ISIS jigsaw) + MOLA 수직 제어
4. 스테레오 매칭 + DTM 추출:
   - **SOCET SET/GXP** (USGS, 반자동, 수동 QA → 최고 품질, PDS DTEEC 대부분)
   - **Ames Stereo Pipeline (ASP)** (NASA, 오픈소스, 완전 자동)
     - Beyer et al. (2018), Earth Space Sci. 5, DOI: 10.1029/2018EA000409

### DTM 사양
- 수평 posting: **~1 m** (0.5-2 m 범위)
- 수직 정밀도 (상대): **~0.1-0.3 m** (양호 조건)
- 수직 정확도 (절대): ~1-2 m (MOLA 제어 후)
- 수직 정밀도 ≈ GSD / parallax-to-height ratio

### DTEEC Product
- Digital Terrain Elevation Equivalent Corrected
- PDS3 label + GeoTIFF 또는 ISIS cube
- Equirectangular 투영 (저위도), Polar Stereographic (고위도)
- 고도: Mars areoid (MOLA 기반) 기준 meters
- ID 패턴: `DTEEC_######_####_######_####`
- Archive: PDS Geosciences Node + HiRISE website (UofA)

## 6.12 Orthoimage Products

| 프로덕트 | 설명 |
|---|---|
| EDR | 미보정 개별 CCD |
| RDR | 보정, 맵투영 개별 CCD (`PSP_######_####_RED#`) |
| RED Mosaic (NOMAP) | 10 RED CCD 모자이크, 카메라 좌표 |
| RED Mosaic (MAP) | 맵투영 RED 모자이크 |
| COLOR Mosaic | RED+BG+NIR 합성, IRB false-color |
| **ORTHO** | DTM 기반 정사보정 (`_ORTHO`) → **최고 기하 정확도** |

### 투영/좌표계
- Equirectangular (|lat| < 65 deg)
- Mars IAU 2000 (Re=3396.19km, Rp=3376.20km)
- Planetocentric lat, East-positive lon (0-360)
- Pixel scale: 0.25 m/px (RED), 0.50 m/px (BG/NIR)

### Archive
- PDS Imaging Node: https://pds-imaging.jpl.nasa.gov/
- HiRISE website: https://www.uahirise.org/
- PDS Geosciences Node: DTEEC products
- ODE: https://ode.rsl.wustl.edu/

## 6.13 Jezero Crater Coverage

- 착륙지 선정 (2015-2018) 중 집중 촬영 → 착륙 전 **50+개** HiRISE 관측
- 착륙 후 로버 운영 지원 추가 촬영 → 총 **100+개** 추정
- 착륙 타원 + 서쪽 삼각주 고밀도 스테레오 커버리지
- 우리 프로젝트 DTM: `DTEEC_048842_1985_048908_1985_U01` (388MB, 1m posting, Jezero 전체 커버)

## 6.14 HiRISE Super-Resolution 선행 연구

**Tao, Y., and Muller, J.-P. (2016). "A novel method for surface exploration: Super-resolution restoration of Mars repeat-pass orbital imagery." Planet. Space Sci., 121, 103-114.**
- DOI: 10.1016/j.pss.2015.11.010

### GPT-SRR (Got-Point Triangulation Super-Resolution Restoration)
1. 동일 영역 다중 HiRISE 영상 (N>=3) 입력
2. Sub-pixel 정합
3. 측광 삼각측량 → 각 픽셀 정밀 3D 위치 결정
4. 다중 sub-pixel shifted 관측 융합 → 고해상도 복원
5. **달성 해상도: 2-5x 향상** (0.25 m → ~0.05-0.12 m)
6. 검증: 알려진 지물 비교, PSF sharpening, edge profile, degradation test

**우리 연구와의 차이**: Tao & Muller는 동일 센서 다중패스. 우리는 **cross-platform** (ground-truth 로버 영상 → 궤도 SR). 근본적으로 다른 장점: 실제 cm급 ground truth 보유.

## 6.15 타 화성 궤도 카메라 비교

| 파라미터 | HiRISE | CTX | HRSC | CaSSIS |
|---|---|---|---|---|
| 해상도 | **0.25 m/px** | 6 m/px | 12.5-25 m/px | 4.5 m/px |
| Swath | 6 km | 30 km | 60 km | 8 km |
| 컬러 | 3-band (1.2km) | Pan | 4+nadir | 4-band |
| 스테레오 | 타깃 쌍 | 가능 | Along-track | Along-track |
| 커버리지 | 3-5% | ~99% | ~90% | 확장 중 |
| DTM posting | 1 m | 18-20 m | 50-100 m | 8-10 m |

Refs:
- CTX: Malin et al. (2007), JGR 112, E05S04, DOI: 10.1029/2006JE002808
- HRSC: Jaumann et al. (2007), PSS 55, DOI: 10.1016/j.pss.2006.12.003
- CaSSIS: Thomas et al. (2017), SSR 212, DOI: 10.1007/s11214-017-0421-1

## 6.16 왜 HiRISE가 SR 최적 대상인가

1. **최고 네이티브 해상도**: 0.25 m/px, 타 궤도 카메라의 15-25x
2. **Scale bridging**: Mastcam-Z (5-25m 거리, 0.5-5 cm/px) vs HiRISE (25 cm/px) → **~5-50x gap**, 특히 x4 SR (25cm→6cm)은 현실적
3. **기하 충실도**: 잘 특성화된 광학, 광범위한 보정
4. **DTM 가용**: 1m posting DTM으로 정밀 3D 정합 가능
5. **Jezero 커버리지 밀도**: 다수 중첩 관측
6. **분광 호환**: RED (570-830nm) ↔ Mastcam-Z L 필터 (528-800nm) 상당 중첩
