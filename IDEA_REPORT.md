# Idea Discovery Report

**Direction**: Mastcam-Z와 HiRISE를 이용한 연구
**Date**: 2026-04-09
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine

## Executive Summary

**추천 아이디어: MarsRefSR — 화성 최초의 Cross-Sensor Reference-Based Super-Resolution 벤치마크 + Baseline 모델**

Mastcam-Z(6.25cm/px)를 참조 이미지로, HiRISE(25cm/px)를 저해상도 입력으로 사용하여 4배 초해상도를 달성하는 벤치마크 데이터셋과 DTM-aware RefSR 모델을 제시한다. SPICE 기반 서브픽셀 정합 파이프라인은 이미 47 sols에 대해 구축 완료되어 있으며, 이는 세계적으로 유일한 화성 cross-sensor SR 데이터셋이다. 기존 SOTA SR 모델 대비 성능 평가와 지질학적 보존 분석을 포함한다.

## Literature Landscape

### A. Cross-Sensor Reference-Based SR (Remote Sensing)
- **CRefDiff (2025)** — Diffusion 기반 RefSR SOTA, NAIP↔Sentinel-2, >8x gap 처리
- **RRSGAN** — GAN 기반 원격탐사 RefSR
- **LMR Dataset** — 다중 참조 이미지 SR 데이터셋
- **SEN2NAIP** — Sentinel-2↔NAIP 4x SR 데이터셋 (2,851 pairs)
- **BreizhSR** — Sentinel-2↔SPOT-6 시계열 MISR

### B. Mars Image Deep Learning
- **RSTSRN** — Swin Transformer 기반 화성 단일 이미지 SR
- **CRISM SR** — 화성 하이퍼스펙트럴 SR
- **MCTED (2025)** — CTX → DEM ML 데이터셋
- **MADNet 2.0** — 단일 이미지 DTM 추정
- **MarsRetrieval (2026)** — VLM 기반 화성 검색 벤치마크
- **Martian World Models (2025)** — 스테레오 → 3D + 비디오 합성

### C. Mars Terrain Classification
- **DepthFormer (2025)** — 깊이 강화 시맨틱 세그멘테이션
- **TerSeg (2025)** — CNN+Swin Transformer 하이브리드
- **MarsSeg** — 다중 스케일 화성 표면 세그멘테이션
- **AI4Mars** — 350K 이미지, Soil/Bedrock/Sand/Big Rock

### Key Gaps
1. Mastcam-Z ↔ HiRISE cross-sensor SR 논문/데이터셋 없음
2. SPICE 기반 서브픽셀 정합을 SR에 활용한 연구 없음
3. DTM 기하학을 condition으로 활용한 SR 없음
4. 다중 프레임 집적 기반 화성 SR 없음

## Ranked Ideas

### 1. MarsRefSR + MarsOrtho-Benchmark — RECOMMENDED

**결합 아이디어: 벤치마크 데이터셋 + DTM-aware RefSR 모델**

- **문제**: HiRISE 25cm/px → 6.25cm/px (x4 SR), Mastcam-Z를 Reference로 활용
- **데이터**: 47 sols, ~1,100+ SPICE-정합 LR-HR 페어
- **모델**: DTM-conditioned Reference-Based SR
- **Novelty**: CONFIRMED — 가장 가까운 연구는 SEN2NAIP(지구)이며, 화성 cross-sensor SR은 최초
- **Reviewer Score**: 7/10
- **약점**: 데이터 규모(47 sols), SPICE 정합 오차, 시간 차이로 인한 표면 변화
- **대응**: Self-supervised pretrain, 변화 마스킹, 정합 오차 분석
- **다음 단계**: /run-experiment → E1-E7 실험 실행

### 2. DTM-Guided 3D-Aware SR — BACKUP

- **문제**: DTM 표면 법선/경사를 SR condition으로 활용
- **차별점**: 기존 SR은 2D만 사용, 3D geometry 활용 최초
- **Novelty**: CONFIRMED
- **위치**: #1의 ablation 또는 독립 후속 논문

### 3. Cross-Resolution Mars Terrain Classification — BACKUP

- **문제**: Mastcam-Z fine-grained 지형 → HiRISE 스케일로 transfer
- **차별점**: 로버 데이터로 학습, 궤도 스케일에서 예측
- **Novelty**: CONFIRMED
- **위치**: 별도 논문 (Planetary Science journal)

## Eliminated Ideas

| 아이디어 | 탈락 단계 | 이유 |
|---|---|---|
| #3 Multi-Frame Aggregation SR | Phase 2 | 독창성 낮음, classical MFSR과 차이 부족 |
| #8 Self-Supervised Mastcam-Z SR | Phase 2 | 줌 쌍 데이터 수집 불확실 |
| #9 VLM-Guided Analysis | Phase 2 | SR이 아닌 응용, 주 방향과 상이 |
| #10 Mars NeRF | Phase 2 | Martian World Models와 중복 |
| #7 Temporal Change Detection | Phase 2 | 시간차 중첩 데이터 불충분 |

## Proposed Paper Structure

```
Title: MarsRefSR: A Benchmark and Baseline for Reference-Based
       Super-Resolution of Mars Orbital Imagery Using Rover Cameras

1. Introduction
   - 화성 표면 이미징의 해상도 한계
   - Rover-orbital cross-sensor 기회
   - Contributions: dataset + model + benchmark

2. Related Work
   - Single-image SR for remote sensing
   - Reference-based SR (CRefDiff, MASA-SR, RRDB+Ref)
   - Mars image processing (RSTSRN, MADNet, MCTED)

3. MarsOrtho Dataset
   - SPICE-based coregistration pipeline
   - Data statistics: 47 sols, resolution, coverage
   - Quality analysis: registration accuracy, temporal gap effects
   - Train/val/test split strategy

4. Method: MarsRefSR
   - Architecture: RRDB backbone + DTM condition branch + Ref attention
   - DTM-guided attention: surface normal, slope as spatial priors
   - Training: pretrain on SEN2NAIP, fine-tune on MarsOrtho

5. Experiments
   - E1: SISR baselines (EDSR, SwinIR, HAT, Real-ESRGAN)
   - E2: RefSR baselines (CRefDiff, MASA-SR)
   - E3: MarsRefSR (proposed)
   - E4: Ablation (DTM guidance, SPICE vs feature matching)
   - E5: Cross-sol generalization
   - E6: Multi-frame vs single-frame
   - E7: Geological preservation metrics

6. Analysis
   - Geological feature preservation
   - Failure cases and limitations
   - Impact of registration accuracy

7. Conclusion + Dataset Release
   - Public benchmark release plan
   - Future work: more sols, Curiosity Mastcam extension
```

## Experiment Plan

| ID | Experiment | GPU Hours | Purpose | Priority |
|---|---|---|---|---|
| E1 | SISR baselines (EDSR, SwinIR, HAT, Real-ESRGAN) | 4h | Lower bound | MUST |
| E2 | RefSR baselines (CRefDiff, MASA-SR, RRDB+Ref) | 8h | Upper bound | MUST |
| E3 | MarsRefSR (DTM-conditioned RefSR) | 12h | Core result | MUST |
| E4 | Ablation: DTM on/off, SPICE vs feature-match | 8h | Contribution analysis | MUST |
| E5 | Cross-sol generalization (40 train / 7 test) | 4h | Generalization | MUST |
| E6 | Multi-frame aggregation comparison | 4h | MFSR vs RefSR | SHOULD |
| E7 | Geological metrics (edge, texture fidelity) | 2h | Scientific validation | MUST |
| **Total** | | **~42h** | | |

### First 3 Runs to Launch
1. E1: SISR baselines — 결과가 다른 모든 실험의 기준선
2. E3: MarsRefSR — 핵심 모델, 가장 오래 걸림
3. E5: Cross-sol generalization — reviewer가 반드시 물어볼 질문

## Venue Considerations

| Venue | Fit | Notes |
|---|---|---|
| CVPR/ECCV Workshop (Earth Vision) | HIGH | Cross-sensor SR + benchmark |
| IEEE TGRS / ISPRS | HIGH | 원격탐사 전문 저널 |
| Icarus / PSJ | HIGH | Planetary Science 저널, 과학적 기여 강조 |
| NeurIPS Datasets & Benchmarks | MEDIUM | 벤치마크 트랙, 데이터 규모가 관건 |

## Next Steps

- [ ] 데이터셋 정리: MarsOrtho benchmark format 표준화
- [ ] /run-experiment: E1-E3 실험 실행
- [ ] /auto-review-loop: 논문 초안 반복 개선
- [ ] /paper-write: 최종 논문 작성
- [ ] GitHub/HuggingFace에 데이터셋 + 코드 공개
