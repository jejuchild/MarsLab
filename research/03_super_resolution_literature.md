# 3. Super-Resolution & Cross-Modal Image Translation - Literature Review

## 3.1 SAR ↔ Optical Translation (pix2pix 계열)

| 방법 | 연도 | 핵심 |
|---|---|---|
| pix2pix SAR-to-optical (IEEE 2021) | 2021 | Sentinel-1 SAR → Sentinel-2 광학. conditional GAN |
| Diffusion 기반 (Frontiers in Neuroscience 2024) | 2024 | SAR을 condition으로 diffusion → 광학 영상 생성 |
| Schrodinger Bridge (Scientific Reports 2024) | 2024 | Unpaired SAR→optical. CycleGAN 대비 SSIM +64%, PSNR +7.5% |
| ISPRS Survey (2025) | 2025 | "Generative models for SAR-optical image translation: A systematic review" |

## 3.2 Cross-Resolution Fusion (Drone ↔ Satellite)

| 방법 | 연도 | 핵심 |
|---|---|---|
| Res-PGGAN (Han & Du, Drones 2024) | 2024 | 드론 고해상 → 위성 SR 학습. SSIM 0.97, PSNR 44.80 |
| UAS-Guided SR (arXiv 2025) | 2025 | 드론으로 Sentinel-2 SR 가이드. 바이오매스 추정 18% 향상 |

## 3.3 Reference-based Super-Resolution (RefSR) -- 가장 유망

우리 시나리오: 고해상 참조(Mastcam ortho) 텍스처를 저해상(HiRISE)에 전이

| 방법 | 연도 | 학회 | 핵심 |
|---|---|---|---|
| **SRNTT** (Zhang et al.) | 2019 | CVPR | 참조 이미지에서 텍스처 패치 매칭 → LR에 전이 |
| **TTSR** (Yang et al.) | 2020 | CVPR | Transformer 기반. LR=query, Ref=key. 교차 스케일 |
| **DATSR** (Cao et al.) | 2022 | ECCV | Deformable attention. 정합 불완전해도 robust |
| **RRSR** | 2022 | ECCV | 양방향 feature exchange (LR↔Ref) |
| **RASR** | 2025 | arXiv | 자동 참조 이미지 검색 → retrieval-augmented SR |

**General RefSR pipeline**:
1. Search: Ref 이미지에서 상관 컨텐츠 탐색
2. Align: 매칭된 패턴을 LR 피처와 정합
3. Fuse: 정합된 Ref 피처를 LR 피처에 융합
4. Reconstruct: HR 출력 복원

GitHub reference list: https://github.com/ahmadmughees/Awesome-RefSR

## 3.4 Mars Orbital Super-Resolution (기존 연구)

| 논문 | 연도 | 핵심 |
|---|---|---|
| **Tao & Muller**, Planet. Space Sci. | 2016 | GPT-SRR: HiRISE 다중패스 4-8장 → 25cm→5cm. Navcam으로 검증 |
| **Tao & Muller**, ISPRS Archives | 2016 | HiRISE SRR vs Navcam 정량 비교 |
| **Tao et al.**, Remote Sensing | 2021 | TGO CaSSIS 컬러 영상 single-image SR |

## 3.5 Mars Rover-Orbital Co-registration (직접 관련)

| 논문 | 연도 | 핵심 |
|---|---|---|
| **Tao & Muller**, Icarus | 2016 | Navcam ortho mosaic → HiRISE co-registration. 60cm 정합 정확도 |
| **Paar et al.**, Earth & Space Sci. | 2023 | PRoViP/PRo3D: Mastcam-Z 스테레오 DTM+텍스처 메시, 궤도 데이터 융합 |

## 3.6 우리 케이스에의 적용

```
HiRISE (1m/px, 전체 traverse) + Mastcam ortho patches (cm급, sparse)
                    |
         Reference-based Super-Resolution (DATSR 등)
                    |
       HiRISE SR (10-25cm급?) traverse 전체
```

**가능성 근거**:
1. Mastcam ortho가 RefSR의 "참조 이미지" 역할을 완벽히 함
2. SPICE 지리참조로 학습 데이터 페어 자동 생성 가능
3. 50+ sol 다양한 지형 커버
4. 화성 지표 정적 → 멀티템포럴 매칭 유효

**난제**:
- 조명 차이: HiRISE (궤도, 오후) vs Mastcam (지표, 다양한 시각)
- 시점 차이: nadir vs oblique 정사영 → 오클루전, 그림자
- 커버리지 불균형: traverse 주변 좁은 띠
- x10 SR은 aggressive → x4 (1m→25cm) 시작 권장
