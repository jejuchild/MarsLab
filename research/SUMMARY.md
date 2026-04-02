# Mastcam-Z → HiRISE Super-Resolution: 통합 연구 요약

논문 제목(안): **Ground-Truth Guided Super-Resolution of HiRISE Orbital Imagery Using Mastcam-Z Stereo Orthoimages**

---

## 1. Perseverance 미션 (01)

**미션**: Mars 2020, Jezero Crater 착륙 (2021.02.18). 과학 목표: 거주가능성, 생체서명, 샘플 캐싱, 유인 준비.

**Jezero**: 고대 호수+삼각주 시스템. 탄산염/점토 분광 탐지 (Goudge 2015). Gilbert형 삼각주 확인 (Mangold 2021, Science). 크레이터 바닥은 예상과 달리 화성암 (Farley 2022, Science) — ground-truth가 궤도 해석을 뒤집은 대표 사례.

**핵심 인용**: Farley 2020 (SSR, 미션), Goudge 2015/2018 (Jezero), Mangold 2021 (Science, 삼각주), Farley 2022 (Science, 화성암)

---

## 2. Mastcam-Z 기기 (05)

| 사양 | 값 |
|---|---|
| 줌 | 26-110mm 연속 (4:1) |
| 검출기 | KAI-2020 CCD, 1600x1200, 7.4um |
| 스테레오 baseline | 24.4 cm |
| 필터 | 11개 (442-1013nm), L/R 공유 754nm |
| IFOV | 0.280 mrad (26mm) ~ 0.067 mrad (110mm) |

**해상도 (110mm 줌)**:
- 2m: 0.13 mm/px | 5m: 0.34 mm/px | 100m: 6.7 mm/px

**스테레오 정밀도**: 2m→1cm, 5m→5cm, 10m→20cm, 50m→5-10m

**MSL 대비 핵심 개선**: 양안 동일 줌으로 매칭 스테레오 가능 (MSL은 34/100mm 고정 → FOV 불일치)

**SR 관련**: 줌 가변 → 자연적 LR/HR 학습 쌍. UVW 프로덕트(surface normal) PDS 제공. 최적 SR 거리 5-50m.

**핵심 인용**: Bell 2021 (SSR, 기기), Hayes 2021 (E&SS, 보정), Kinch 2020 (SSR, 보정타깃)

---

## 3. HiRISE 기기 (06)

| 사양 | 값 |
|---|---|
| 구경 | 0.5m, f/24, 12m 초점거리 |
| 검출기 | 14 CCD (10 RED + 2 BG + 2 NIR), TDI |
| RED 해상도 | 0.25 m/px (periapsis) |
| RED 스워스 | ~6 km (~20,000 px) |
| RED 파장 | 570-830 nm |
| DTM | 1 m posting, 수직 0.1-0.3m 정밀도 |

**Mastcam-Z (5m, 110mm) 대비**: 0.34mm vs 250mm = **~735배** 해상도 차이

**SR 목표**: x4 (25cm→6cm) 현실적, x10 (25cm→2.5cm) aggressive

**분광 호환**: HiRISE RED (570-830nm) ↔ Mastcam-Z L필터 (528-800nm) 상당 중첩

**핵심 인용**: McEwen 2007 (JGR, 기기), Kirk 2008 (JGR, DTM), Delamere 2010 (Icarus, 보정)

---

## 4. SPICE + XYZ + 정합 방법론 (07)

### 변환 체인
```
Mastcam-Z pixel (h,v)
  → CAHVOR 역투영 + 스테레오 삼각측량
  → XYZ in SITE_FRAME (meters)
  → SPICE: SITE → ROVER(CK) → IAU_MARS(SPK/FK)
  → areocentric (lon, lat)
  → HiRISE Equirectangular 투영
  → HiRISE pixel (sample, line)
```

### 오차 예산 (Error Budget)

| 오차 원인 | 크기 | HiRISE px |
|---|---|---|
| XYZ (5m) | ~5cm | <1 |
| XYZ (50m) | ~25cm lateral | 1-2 |
| RSM pointing | ~1 mrad | <1 |
| **로버 위치결정** | **1-3 m** | **4-12 ← 지배적** |
| HiRISE 정사보정 | 2.5-25 cm | 0.1-1 |

**결론**: 병목은 로버 절대 위치 (1-3m). Feature-based 정합으로 서브픽셀까지 개선 필요.

**핵심 인용**: Acton 1996/2018 (PSS, SPICE), Di & Li 2004 (JGR, CAHVOR), Gennery 2006 (IJCV)

---

## 5. 선행 로버-궤도 정합 연구 (08)

### 진화: 3개 시대
1. **Era 1 (Viking/Pathfinder)**: 궤도 >10m/px → 융합 불가
2. **Era 2 (MER/MSL + HiRISE, 2004-2015)**: 정합/비교 가능. DTM nesting. 정성적 비교
3. **Era 3 (현재)**: 정량적 비교 시작 — **but SR 시도 0건**

### Stack et al. 2016 (Icarus) — 가장 중요한 선행 연구
- HiRISE vs Mastcam/MAHLI 지질도 체계적 비교
- HiRISE로 **안 보이는 것**: 50cm 미만 박층, 입자 크기, 퇴적구조, 속성 조직
- 궤도 지질도의 **~30% 접촉면 부정확**
- → 우리 SR의 직접적 동기

### 남은 GAP 5개
1. 영상 수준 SR 시도 없음 (정합/비교만)
2. 분광/텍스처 transfer learning 없음
3. 학습된 cross-scale 매핑 없음 (행성과학 0건)
4. 체계적 co-registered 학습 데이터셋 없음
5. 서브픽셀 조성 분해 없음

---

## 6. Cross-Scale Fusion 타 분야 패턴 (02)

| 패턴 | 대표 연구 | 우리 적용 |
|---|---|---|
| Label Transfer | Stack 2016, Gao 2020 (농업) | Mastcam 암상 → HiRISE 분류기 |
| Attribute Enrichment | SG-BEV (CVPR24), Amunts 2022 (의학) | Mastcam 텍스처 → HiRISE 투영 |
| Cross-View Synthesis | Sat2Scene (CVPR24) | HiRISE↔Mastcam 뷰 합성 |
| Subsurface Inference | Agapiou 2017 (고고학) | 퇴적구조 → 궤도 텍스처 연결 |

**가장 정확한 비유**: 의학 조직학↔MRI (Amunts 2022, Science Advances)

---

## 7. Super-Resolution 문헌 (03)

### Reference-based SR (RefSR) — 우리 핵심 방법론

| 방법 | 연도 | 학회 | 핵심 |
|---|---|---|---|
| SRNTT | 2019 | CVPR | 참조 텍스처 매칭→전이 |
| TTSR | 2020 | CVPR | Transformer, LR=query |
| **DATSR** | **2022** | **ECCV** | **Deformable attention, 정합 불완전해도 robust** |
| RASR | 2025 | arXiv | 자동 참조 검색 |

### 화성 궤도 SR 선행
- Tao & Muller 2016: HiRISE 다중패스 → 25cm→5cm (같은 센서)
- **우리**: cross-platform ground-truth (Mastcam) → 궤도 SR (**완전 GAP**)

---

## 8. Research Gaps (04)

| # | 주제 | 상태 |
|---|---|---|
| 1 | Rover 라벨→궤도 분류 | GAP |
| 2 | Mastcam orthoimage | GAP |
| **3** | **로버 참조 궤도 SR** | **완전 GAP (0건)** |
| 4 | 화성 cross-view synthesis | 완전 GAP |
| 5 | XYZ normal 광도보정 | GAP |
| 8 | 자동 strike/dip | EXISTS (반자동, Stein 2018) |

---

## 9. 전처리/보정 방법론 (09)

### 파이프라인 요약

```
1. 방사 정규화
   Mastcam-Z IOF + HiRISE I/F → 분광 밴드 합성 (L필터→RED 응답 합성)
   → PIF 기반 선형 정규화 (밝은 암석, 어두운 모래를 앵커로)

2. 광도 보정
   XYZ-derived surface normal → (i, e, g) 계산
   → Minnaert 모델 (k=0.5-0.8) 또는 Hapke 모델로 공통 기하(i=30,e=0,g=30)에 정규화

3. 대기 보정
   HiRISE: 그림자 영역에서 path radiance 추정 → 차감
   Mastcam-Z: IOF 프로덕트 이미 보정됨 (tau 측정 활용)

4. 그림자/오클루전
   DTM + SPICE 태양위치 → ray-tracing 그림자 탐지
   >30% 그림자 패치 학습 제외. 다중sol로 hole filling

5. 기하 정합 정밀화
   SPICE-only (4-12 px 오차) → RIFT 특징매칭 (<1 px)
   → DATSR deformable attention (잔여 서브픽셀 정합)

6. 도메인 적응
   합성 사전학습 + 실제 fine-tuning
   RealESRGAN식 2차 열화 증강 (화성 특화 파라미터)
```

### 핵심 인용
- Claverie 2018 (HLS 프레임워크), Fernando 2022 (Hapke+대기), Mustard 2021 (HiRISE 광도보정)
- DATSR: Cao 2022 (ECCV), Li 2019 (RIFT 조명불변 매칭)

---

## 10. 논문 구조 매핑

```
조사                        → 논문 섹션
────────────────────────────────────────
Perseverance 미션           → 1. Introduction
Mastcam-Z / HiRISE 기기     → 2.1-2.2 Data & Instruments
SPICE + 정합 방법론          → 3. Co-registration Methodology
선행 로버 시도들             → 2.3 Related Work
Cross-scale / SR 문헌       → 2.4 SR Background
정합 구현+검증               → 4. Implementation & Validation
전처리/보정                  → 5. Data Preparation
Mastcam ortho + SR 적용     → 6. Experiments & Results
```

---

## 11. 핵심 논문 TOP 15 (반드시 인용)

| # | 논문 | 연도 | 역할 |
|---|---|---|---|
| 1 | Bell et al., SSR 217 | 2021 | Mastcam-Z 기기 |
| 2 | McEwen et al., JGR 112 | 2007 | HiRISE 기기 |
| 3 | Stack et al., Icarus 280 | 2016 | 궤도-지표 갭 정량화 |
| 4 | Tao & Muller, Icarus 280 | 2016 | Navcam ortho→HiRISE 정합 |
| 5 | Tao & Muller, PSS 121 | 2016 | HiRISE multi-pass SR |
| 6 | Farley et al., Science 377 | 2022 | Jezero 지질 (ground-truth 반전) |
| 7 | Mangold et al., Science 374 | 2021 | Jezero 삼각주 확인 |
| 8 | Cao et al., ECCV | 2022 | DATSR (RefSR 모델) |
| 9 | Acton et al., PSS 150 | 2018 | SPICE |
| 10 | Kirk et al., JGR 113 | 2008 | HiRISE DTM |
| 11 | Di & Li, JGR 109 | 2004 | CAHVOR 카메라 모델 |
| 12 | Hayes et al., E&SS | 2021 | Mastcam-Z 보정 |
| 13 | Paar et al., E&SS | 2023 | PRoViP 3D (Mastcam-Z) |
| 14 | Claverie et al., RSE 219 | 2018 | HLS 방사 정규화 |
| 15 | Zhang et al., CVPR | 2019 | SRNTT (RefSR 기반) |
