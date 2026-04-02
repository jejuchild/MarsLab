# 5. Mastcam-Z Instrument - Detailed Technical Review

## 5.1 Primary Reference

**Bell, J.F., III, Maki, J.N., Mehall, G.L., et al. (2021). "The Mars 2020 Perseverance Rover Mast Camera Zoom (Mastcam-Z) Multispectral, Stereoscopic Imaging Investigation." Space Science Reviews, 217, 24.**
- DOI: 10.1007/s11214-020-00755-x

## 5.2 Camera Optics

- Zoom: 26-110 mm 연속 가변 (4:1 줌비), 양안 동일
- f-number: f/6.7 (26mm) ~ f/9.5 (110mm)
- 양안 광학적으로 동일 설계

## 5.3 Detector

- Kodak KAI-2020 interline-transfer CCD (MSL Mastcam과 동일 heritage)
- 1600 x 1200 pixels (1.92 MP)
- Pixel size: 7.4 um x 7.4 um
- Bayer RGB on-chip filter

## 5.4 FOV / IFOV

| 줌 | FOV | IFOV |
|---|---|---|
| 26 mm | 25.6 x 19.2 deg | ~0.280 mrad/pixel |
| 110 mm | 6.2 x 4.6 deg | ~0.067 mrad/pixel |

## 5.5 Stereo

- Stereo baseline: **24.4 cm** (center-to-center)
- RSM 높이: 지표면 위 ~2.0 m
- Toe-in angle: 2.3 deg (수렴점 ~3m)
- **MSL 대비 핵심 개선**: 양안 매칭된 줌에서 스테레오 가능 (MSL은 34mm vs 100mm 고정이라 FOV 불일치)

## 5.6 공간 해상도

| 거리 | 26 mm | 110 mm |
|---|---|---|
| 2 m | ~0.56 mm/px | **~0.13 mm/px** |
| 5 m | ~1.4 mm/px | ~0.34 mm/px |
| 100 m | ~28 mm/px | ~6.7 mm/px |
| 1 km | ~280 mm/px | ~67 mm/px |

## 5.7 Stereo 정밀도

| 거리 | Range precision (1-sigma) |
|---|---|
| 2 m | ~1-2 mm |
| 5 m | ~5-10 mm |
| 10 m | ~20-40 mm |
| 100 m | ~2-4 m |

**SR 연구 시사점**: 근거리(2-5m) cm급 정밀, 원거리(>50m) 급격히 저하. 정사영 품질이 거리에 의존.

## 5.8 Filter Wavelengths (11 unique geological filters)

### Left Camera

| 위치 | 중심파장 (nm) | FWHM (nm) | 비고 |
|---|---|---|---|
| L0 | ~440, 554, 640 | Wide (Bayer) | Broadband RGB |
| L1 | 800 | 18 | |
| L2 | 754 | 18 | 양안 공유 파장 |
| L3 | 677 | 22 | |
| L4 | 605 | 18 | |
| L5 | 528 | 22 | |
| L6 | 442 | 24 | |
| L7 | Broadband+ND | - | Solar imaging |

### Right Camera

| 위치 | 중심파장 (nm) | FWHM (nm) | 비고 |
|---|---|---|---|
| R0 | ~440, 554, 640 | Wide (Bayer) | Broadband RGB |
| R1 | 1013 | 26 | |
| R2 | 978 | 22 | |
| R3 | 937 | 22 | |
| R4 | 866 | 22 | |
| R5 | 836 | 22 | |
| R6 | 754 | 18 | 양안 공유 파장 |
| R7 | Broadband+ND | - | Solar imaging |

442-1013 nm 범위: Fe 광물 흡수대 (800-1000nm Fe2+/Fe3+, 530nm Fe3+ charge transfer) 커버.

## 5.9 Calibration

### Pre-flight (Hayes et al., 2021, Earth Space Sci., DOI: 10.1029/2020EA001516)
- MSSS 시설에서 종합 지상 보정
- Flat field, dark current, read noise, gain, linearity, spectral throughput, geometric distortion, MTF, stray light
- Flat field 잔차 < 1%
- 방사 정확도: 절대 < 5%, 상대(밴드간) < 1-2%

### In-flight (Kinch et al., 2020, Space Sci. Rev. 216, 141, DOI: 10.1007/s11214-020-00774-8)
- 로버 데크의 8패치 보정 타깃 + 중앙 gnomon
- AluWhite98 (백색 ~98% 반사율), Lucideon 컬러 패치, Carbon Black (암색)
- Gnomon 그림자로 직달/산란 조명 비율 추정
- DN → I/F 변환 계수 도출

### Dust monitoring (Kinch et al., 2023, Earth Space Sci. 10, DOI: 10.1029/2022EA002590)
- 보정 타깃 먼지 퇴적 모니터링 및 보정

## 5.10 Data Products (PDS4)

| Product | Level | Description |
|---|---|---|
| EDR | Raw | 미보정 DN 값 |
| RAD | Calibrated | 방사 보정 radiance (W/m2/sr/nm) |
| RAS | Calibrated | RAD 11bit→8bit stretch |
| IOF | Calibrated | I/F (radiance factor) |
| IOL | Calibrated | IOF + linearized Bayer |
| **XYZ** | Derived | **3D 점군 (Site frame, meters)** |
| RNE | Derived | Range + normal error map |
| **UVW** | Derived | **Surface normal vectors** |
| MXY | Derived | Stereo disparity map |
| SLP/SMG | Derived | Slope / slope magnitude |

## 5.11 Camera Model: CAHVOR

- JPL 개발 카메라 모델 (MER부터 사용)
- C (projection center), A (axis), H (horizontal), V (vertical), O (optical axis), R (radial distortion)
- 확장형 CAHVORE: fisheye/광각 렌즈용 entrance pupil 추가
- Mastcam-Z: 대부분 줌 설정에서 CAHVOR 사용
- 비행 중 기하 보정으로 업데이트

**Ref**: Gennery, 2006, Int. J. Computer Vision 68(3), DOI: 10.1007/s11263-006-5168-1

## 5.12 MSL Mastcam vs Mastcam-Z 비교

| 파라미터 | MSL Mastcam | Mastcam-Z |
|---|---|---|
| 줌 | 고정 (L: 34mm, R: 100mm) | 26-110mm 연속 가변 |
| 스테레오 | 비대칭 FOV (불일치) | **양안 동일 FOV (모든 줌)** |
| Baseline | 24.2 cm | 24.4 cm |
| 검출기 | KAI-2020 CCD | 동일 |
| 필터 | M-34: 7개, M-100: 7개 | L: 6개, R: 6개 + 공유 754nm |

핵심 개선: **매칭된 스테레오** + **줌 가변** → 동일 타깃 멀티스케일 촬영 가능

## 5.13 SR 연구와의 관련성

1. **알려진 PSF/MTF**: Hayes et al. 2021의 모든 줌 설정별 MTF 특성화 → deconvolution/learning SR의 ground truth OTF
2. **멀티스케일 자연 페어**: 줌 가변으로 같은 장면을 26mm/110mm로 촬영 → 자연적 LR/HR 학습 쌍 (MSL에서는 불가)
3. **스테레오 기하 제약**: XYZ로 거리별 PSF 스케일링, 시점간 기하 재투영 가능
4. **방사 일관성**: IOF 1-2% 상대 정확도 → 다중필터/다중시간 관측 간 광도 일관성 활용 가능
5. **최적 SR 거리**: 5-50m (110mm에서 0.34-6.7 mm/px) → 서브픽셀 구조 미분해, SR 효과 최대

## 5.14 PDS Archive

- URL: https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/
- Stereo bundle: mars2020_mastcamz_ops_stereo
- Calibrated bundle: mars2020_mastcamz_ops_calibrated
- 6개월 proprietary period (실제로는 더 빨리 공개)
- 수만 장 영상 (2024 기준)
