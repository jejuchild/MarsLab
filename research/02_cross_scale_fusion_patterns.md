# 2. Cross-Scale Data Fusion Patterns - Literature Review

다른 학문 분야에서 orbital/wide-area (저해상도) + ground-level/close-up (고해상도) 데이터를 융합한 연구 패턴 정리.

## Pattern 1: Label Transfer (근접 → 광역 확장)

| 분야 | 논문 | 핵심 |
|---|---|---|
| 화성 지질 | Stack et al., 2016, Icarus | Curiosity Mastcam 지질 분류 → HiRISE에서 안 보이는 퇴적구조 식별. 로버 분류를 궤도 확장 |
| 농업 | Gao et al., 2020, Remote Sensing | 드론 개별 식물 병해 감지 → Sentinel-2 전역 조기 탐지 학습 |
| 산림 | LiDAR Fusion Review, 2024, Current Forestry Reports | 드론 LiDAR + 위성 GEDI → 바이오매스 추정 오차 20-40% 감소 |
| 도시 | MIT Treepedia | Street View 가로수 segmentation → 위성 NDVI 교차검증 |

## Pattern 2: Attribute Enrichment (각 뷰의 고유 속성 투영)

| 분야 | 논문 | 핵심 |
|---|---|---|
| 도시 | Ye et al., CVPR 2024 (SG-BEV) | Street View(파사드) + 위성(지붕) → 건물 속성 segmentation, mIOU +10% |
| 도시 교통 | Workman & Jacobs, CVPR 2020 | 위성 도로 기하 + 교통카메라 → 정적 위성영상만으로 동적 교통속도 예측 |
| 지질 | Buckley et al., 2013, Computers & Geosciences | 지상 하이퍼스펙트럴 + LiDAR → "hypercloud" → 광물 수준 분류를 위성 범위로 |
| 의학 | Amunts et al., 2022, Science Advances | 조직학 + 7T MRI → MRI 신호→세포타입 lookup table |

## Pattern 3: Cross-View Synthesis (시점 간 변환)

| 분야 | 논문 | 핵심 |
|---|---|---|
| CV | Regmi & Borji, CVPR 2018 | 최초 aerial↔street-view 양방향 GAN 합성 |
| CV | Li et al., CVPR 2024 (Sat2Scene) | 위성 1장 → diffusion → 걸어다닐 수 있는 3D street-level 장면 |
| CV | Xu et al., CVPR 2025 (GroundScape) | 위성 → 시간적 일관 ground-level 비디오 합성 |
| CV | 3DGS cross-view, 2025 | 3D Gaussian Splatting으로 street→pseudo-aerial 합성 |

## Pattern 4: Subsurface/Hidden Feature Inference

| 분야 | 논문 | 핵심 |
|---|---|---|
| 고고학 | Agapiou et al., 2017, Geosciences | GPR + 지표 분광 → 위성 반사율 회귀 → 위성으로 지하 유적 탐지 (r=0.70) |
| 고고학 | Orengo et al., PNAS 2020 | 지상 조사 + 다중시기 위성 → 23,000km2에서 14,000개 미발견 고분 탐지 |
| 의학 | PMC 2015 | 폐 조직 염증(조직학) → MRI 공간 매핑 → 비침습 MRI 진단 |

## Pattern 5: Temporal Bridging

| 분야 | 논문 | 핵심 |
|---|---|---|
| 재난 | Natural Hazards, 2025 | SNS 현장 사진(시간) + 위성 침수범위(공간) → 구조 우선순위 자동화 |
| 농업 | Moretti et al., Sensors 2022 | 드론(핵심 시기) + Sentinel 시계열 → 위성 NDVI 대기보정 앵커 |

## Medical Imaging Analogy (가장 정확한 비유)

| 의학 | 화성 |
|---|---|
| MRI (mm급, 전체 뇌) | HiRISE (1m/px, 전체 크레이터) |
| 조직학 (um급, 슬라이스) | Mastcam XYZ (cm급, 프레임별) |
| 조직학→MRI 정합 | Mastcam→HiRISE SPICE 정합 |
| MRI 신호 → 세포타입 lookup | HiRISE 텍스처 → 암상/퇴적구조 lookup |
| 비침습 진단 확장 | 로버 미방문 영역 지질 해석 확장 |

Key papers:
- Bhattacharjee et al., 2019, Nature Communications (CLARITY microscopy→MRI registration)
- Amunts et al., 2022, Science Advances (histology→MRI 3D concordance maps)
- Nature, 2025 (probabilistic histological atlas for MRI segmentation)
