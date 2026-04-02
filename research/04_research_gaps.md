# 4. Research Gaps Analysis

## Gap Map (2025년 기준)

| # | 연구 주제 | 상태 | 가장 가까운 기존 연구 |
|---|---|---|---|
| 1 | Rover 라벨 → 궤도 분류기 학습 | **GAP** | DoMars16k (궤도영상 자체 라벨, 로버 미사용) |
| 2 | Mastcam orthoimage (DTM 정사영) | **GAP** | Di et al. 2008 (MER Pancam만) |
| 3 | 로버 참조 궤도 초해상화 (RefSR) | **완전 GAP** | 행성과학 선례 0건 |
| 4 | 화성 cross-view synthesis | **완전 GAP** | 지구만 (Regmi & Borji 2018) |
| 5 | XYZ surface normal → 광도보정 | **GAP** | 광도 연구 있으나 XYZ normal 활용 없음 |
| 6 | 궤도 텍스처 → 퇴적구조 예측 | **완전 GAP** | 선례 0건 |
| 7 | 멀티sol XYZ 3D 변화탐지 | **GAP (3D)** | Baker et al. 2018 (2D만) |
| 8 | 자동 strike/dip 추출 | **EXISTS (반자동)** | Stein et al. 2018 |
| 9 | Mastcam-CRISM 픽셀급 분광 융합 | **GAP (픽셀급)** | 정성적 비교만 (Wellington 2017) |
| 10 | 로버+궤도 DSM 융합 | **GAP (융합)** | 각각 따로 존재 |

## Gap 강도 분류

### 완전 빈 자리 (선례 0):
- **#3** 로버 참조 궤도 SR
- **#4** 화성 cross-view synthesis
- **#6** 궤도 텍스처 → 지표 퇴적구조 예측

### 인프라는 있는데 아무도 안 한 것:
- **#1** 라벨 트랜스퍼
- **#2** Mastcam orthoimage
- **#5** XYZ normal 광도보정

### 2D는 있는데 3D 확장 안 된 것:
- **#7** 3D 변화탐지
- **#10** 멀티해상도 DSM 융합

## 논문 구조와의 매핑

연구 파이프라인 #2 → #1 → #5 → #3:
1. Mastcam orthoimage 생성 (Phase 1)
2. Label transfer (Phase 2)
3. XYZ normal 광도보정 (전처리)
4. RefSR 초해상화 (핵심 기여)

## Key Citations for Gap Justification

- Wagstaff et al., 2018 - "Deep Mars: CNN Classification" (로버 영상 분류, 궤도 전이 없음)
- Wilhelm et al., 2020 - "DoMars16k" (HiRISE 자체 라벨, 로버 ground truth 미사용)
- Di et al., 2008 - MER Pancam orthoimage (Mastcam 아님)
- Baker et al., 2018 - Bagnold Dunes 2D 변화탐지 (3D 아님)
- Stein et al., 2018 - Mastcam 반자동 strike/dip (완전 자동 아님)
- Wellington et al., 2017 - Mastcam-CRISM 정성 비교 (픽셀급 융합 아님)
