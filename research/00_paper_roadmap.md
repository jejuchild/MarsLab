# Paper Roadmap: Ground-Truth Guided Super-Resolution of HiRISE Using Mastcam-Z Orthoimages

## Research Flow

```
Step 1: Perseverance Mission 조사          → 01_perseverance_mission.md     ✅
Step 2: Mastcam-Z 상세 조사               → 05_mastcamz_instrument.md     ✅
Step 3: HiRISE 상세 조사                  → 06_hirise_instrument.md        ✅
Step 4: SPICE + XYZ + 정합 방법론 조사     → 07_spice_xyz_methodology.md    ✅
Step 5: 선행 로버(Curiosity) 시도 조사     → (TODO)
Step 6: 정합 구현 및 검증                  → (진행 중 - run_batch.py)
Step 7: 지구 SR 논문 조사                  → 03_super_resolution_literature.md  ✅ (초안)
Step 8: 전처리/보정 방법론                  → (TODO)
Step 9: Mastcam 적용 (ortho + SR)          → (TODO)
```

## 논문 구조 매핑

```
조사 단계                          → 논문 섹션
─────────────────────────────────────────────
Step 1. Perseverance 미션          → 1. Introduction (mission context)
Step 2. Mastcam-Z 상세             → 2.1 Instruments
Step 3. HiRISE 상세               → 2.2 Orbital data
Step 4. SPICE + XYZ + 정합        → 3. Co-registration methodology
Step 5. 선행 로버 시도들           → 2.3 Related work
Step 6. 정합 수행                  → 4. Implementation & validation
Step 7. 지구 SR 논문              → 2.4 Super-resolution background
Step 8. 전처리/보정                → 5. Data preparation
Step 9. Mastcam 적용              → 6. Experiments & results
```

## 연구 파일 구조

```
research/
  00_paper_roadmap.md              # 이 파일 (전체 로드맵)
  01_perseverance_mission.md       # Step 1 결과
  02_cross_scale_fusion_patterns.md # 타 분야 cross-scale fusion 패턴
  03_super_resolution_literature.md # SR 관련 문헌
  04_research_gaps.md              # 연구 갭 분석
  05_mastcamz_instrument.md        # (TODO) Step 2
  06_hirise_instrument.md          # (TODO) Step 3
  07_spice_xyz_methodology.md      # (TODO) Step 4
  08_prior_rover_work.md           # (TODO) Step 5
  09_preprocessing_methods.md      # (TODO) Step 8
```
