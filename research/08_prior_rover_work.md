# 8. Prior Rover-to-Orbital Cross-Scale Work - Literature Review

## 8.1 MER (Spirit/Opportunity) Pancam

### Stereo DTM & Localization

**Li, R., Di, K., et al. (2006). "Rover localization and landing-site mapping technology for the 2003 Mars Exploration Rover mission." PE&RS, 72(10), 1129-1140.**
- DOI: 10.14358/PERS.72.10.1129
- Pancam/Navcam 스테레오 DTM + 궤도 tie-point bundle adjustment
- 로버 위치 정확도 ~1-2 m (궤도 기준)

**Di, K., Xu, F., Li, R., et al. (2008). "Photogrammetric Processing of Rover Imagery of the 2003 Mars Exploration Rover Mission." ISPRS JPRS, 63(2), 181-201.**
- DOI: 10.1016/j.isprsjprs.2007.07.007
- 체계적 MER 스테레오 파이프라인: 보정→에피폴라→매칭→3D 점군→DTM
- Pancam DTM: 근거리 ~1 mm/px, 10m ~1 cm/px
- Pancam 정사영 → HRSC/MOLA DTM 위에 nesting → 다중해상도 지형 프로덕트

### 궤도 Tie-Point

**Parker, T.J., Golombek, M.P., et al. (2010). "MER Landing Site Mapping and Localization with HiRISE." 41st LPSC, Abs. #2535.**
- HiRISE 25cm에서 로버 궤적, 하드웨어 식별 → 서브미터 절대 위치결정
- 이산 지물 기반 ground-to-orbital 대응 확립

### 분광

**Bell, J.F. III, et al. (2004). "Pancam Multispectral Imaging Results from the Spirit Rover at Gusev Crater." Science, 305(5685), 800-806.**
- DOI: 10.1126/science.1100175
- 13-filter (432-1009nm) 다중분광: 감람석 현무암, 먼지 코팅, 변질 링드 식별
- 궤도 분광(TES, THEMIS)에서는 공간 혼합으로 불가
- **Sub-pixel heterogeneity 문제 직접 입증** → SR 동기

## 8.2 MSL Curiosity Mastcam

### 궤도-지표 비교 (Landmark Study)

**Stack, K.M., Grotzinger, J.P., Lamb, M.P., et al. (2016). "Comparing orbiter and rover image-based mapping of an ancient sedimentary environment, Aeolis Palus, Gale crater, Mars." Icarus, 280, 3-21.**
- DOI: 10.1016/j.icarus.2016.02.024
- **가장 직접적으로 관련된 선행 연구**
- HiRISE vs Mastcam/MAHLI 독립 지질도 체계적 비교
- **HiRISE로 볼 수 있는 것**: 주요 층서 접촉면, 광역 암상 구분 (역암 vs 사암 vs 이암)
- **HiRISE로 볼 수 없는 것**: (1) 50cm 미만 박층, (2) 입자 크기, (3) 퇴적구조 (사층리, 엽리), (4) 속성 조직 (결핵, 맥, 단괴), (5) 미묘한 색상/조성 변화
- 궤도 지질도의 **~30% 접촉면이 부정확 또는 누락**
- **논문 활용**: 궤도-지표 정보 갭을 정량적으로 입증한 유일한 연구. 우리 SR의 동기 부여 핵심 인용

### 궤도-지표 분광

**Fraeman, A.A., Arvidson, R.E., et al. (2013). "A hematite-bearing layer in Gale crater, Mars: Mapping and implications for past aqueous conditions." Geology, 41(10), 1103-1106.**
- DOI: 10.1130/G34613.1
- CRISM 궤도 적철석 탐지 → Curiosity로 확인
- 궤도로 광물 탐지 가능, but 구체적 층서 위치/광물 공생 관계는 로버만 가능

**Fraeman, A.A., Ehlmann, B.L., et al. (2016). "The stratigraphy and evolution of lower Mount Sharp from spectral, morphological, and thermophysical orbital data sets." JGR Planets, 121(9), 1713-1736.**
- DOI: 10.1002/2016JE005095
- CRISM+HiRISE+THEMIS 궤도 통합 → 주요 단위 경계 예측 → 로버로 대부분 확인
- but 상세 내부 층서/속성 overprint는 지표에서만 분해

### Mastcam 스테레오 DTM & 구조지질

**Caravaca, G., Le Mouélic, S., Mangold, N., et al. (2020). "3D Digital Outcrop Model Reconstruction of the Kimberley Outcrop (Gale Crater, Mars)." PSS, 182, 104808.**
- DOI: 10.1016/j.pss.2019.104808
- SfM으로 Mastcam 시퀀스 → cm급 3D 노두 모델 → VR 통합
- 층두께, 횡적 연속성, 절단 관계 정량 측정 → 궤도에서 불가능한 정보

**Stein, N.T., Quinn, D.P., Grotzinger, J.P., et al. (2020). "Regional structural orientation of the upper Murray formation from Mastcam stereo imagery." JGR Planets, 125(6), e2019JE006298.**
- DOI: 10.1029/2019JE006298
- Mastcam 스테레오 DTM → Vera Rubin Ridge 전역 bedding strike/dip 추출
- 미묘한 경사 변화 (2-8도) → HiRISE DTM (1m posting)에서는 불가
- **Mastcam 스테레오 = 정량적 구조지질 도구** 입증

### 풍성 변화 탐지

**Baker, M.M., Lapotre, M.G.A., et al. (2018). "The Bagnold Dunes in southern summer: Active sediment transport on Mars observed by the Curiosity rover." GRL, 45(17), 8853-8863.**
- DOI: 10.1029/2018GL079040
- 반복 Mastcam/Navcam → Bagnold Dune 리플 이동 (~1-3 cm/Earth-year) 탐지
- HiRISE 반복 촬영 탐지 한계(~25cm) 이하 → 지표에서만 가능한 시간 변화

**Bridges, N.T., Sullivan, R., et al. (2012). "Planet-wide sand motion on Mars." Geology, 40(1), 31-34.**
- DOI: 10.1130/G32373.1
- HiRISE 반복 촬영 풍성 변화: 최소 탐지 ~픽셀 규모 (~25cm)
- 로버 수준(cm) 대비 수십 배 낮은 감도

## 8.3 Mars Pathfinder IMP

**Golombek, M.P., et al. (1999). "Overview of the Mars Pathfinder Mission." JGR Planets, 104(E4), 8523-8553.**
- DOI: 10.1029/98JE02554
- IMP 스테레오 파노라마로 Ares Vallis 지형 특성화
- 암석 크기-빈도 분포 → Viking Orbiter 열관성 기반 예측과 비교
- 궤도 해상도(~20-50 m/px) 제한으로 영상 수준 융합 불가

**Smith, P.H., et al. (1997). "The Imager for Mars Pathfinder experiment." JGR Planets, 102(E2), 4003-4025.**
- DOI: 10.1029/96JE03568
- 최초 화성 로버 cm급 스테레오 DTM. 궤도-지표 영상 융합 시도 없음

## 8.4 Viking Landers

**Binder, A.B., Arvidson, R.E., et al. (1977). "The geology of the Viking Lander 1 site." JGR, 82(28), 4439-4451.**
- DOI: 10.1029/JS082i028p04439
- 궤도 해상도 ~50-100 m/px → 정성적 비교만 가능
- "융합 불가능" 시대의 기준선

## 8.5 달 탐사 (Chang'e, Yutu, Apollo)

**Liu, Z., Di, K., et al. (2020). "Landing site topographic mapping and rover localization for Chang'e-4 mission." Science China Info. Sci., 63, 170301.**
- DOI: 10.1007/s11432-019-2796-1
- Yutu-2 스테레오 DTM → LROC NAC (0.5 m/px) DTM 정합
- Coarse-to-fine 특징 매칭 + ICP 정밀화
- 화성 외 가장 직접적인 ground-to-orbital 3D 융합 사례
- 분광/텍스처 SR로 확장되지 않음

**Wu, B., Li, F., et al. (2014). "Topographic modeling and analysis of the landing site of Chang'e-3 on the Moon." EPSL, 405, 257-273.**
- DOI: 10.1016/j.epsl.2014.09.002
- Chang'e-3/Yutu 스테레오 DTM + LROC NAC DTM 통합 → mm~km 다중해상도 지형

**Di, K., Liu, Z., et al. (2019). "Chang'e-4 lander localization based on multi-source data." J. Remote Sensing, 23(1), 177-184.**
- DOI: 10.11834/jrs.20199014
- 하강 카메라 + 표면 파노라마 → LROC NAC 매칭 → ~10m 위치결정

## 8.6 진화 요약: 3개 시대

### Era 1: 융합 불가 (Viking/Pathfinder, 1976-1997)
- 궤도 해상도 >10 m/px → 지표-궤도 대응 불가
- 각 데이터셋 독립 해석

### Era 2: 정합 & 비교 (MER/MSL + HiRISE, 2004-2015)
- HiRISE 25 cm/px → 최초로 지물 수준 대응 가능
- 다중스케일 DTM nesting
- 정성적 지질 비교 (Stack et al. 2016)
- 궤도 예측→지표 확인 패러다임 확립

### Era 3: 현재 (2016-present)
- 정량적 비교 시작 (Stack et al. 2016)
- 로버 3D photogrammetry 고도화
- **BUT: cross-scale learned enhancement 없음** ← 우리 연구 진입점

## 8.7 남아있는 핵심 GAP

1. **정량적 영상 수준 SR 시도 없음**: 모든 선행 연구는 정합 또는 해석 비교. 지표 영상으로 궤도 해상도 향상 시도 0건
2. **분광/텍스처 transfer learning 없음**: Pancam 13-filter, Mastcam-Z 다중분광의 풍부한 정보가 궤도 데이터 향상에 활용된 적 없음
3. **학습된 cross-scale 매핑 없음**: ML로 지표↔궤도 해상도 관계 학습 → 행성과학 0건 (지구에서는 일상)
4. **체계적 co-registered 학습 데이터셋 없음**: 기하 정합은 성숙했으나, SR 학습용 방사 교차보정된 페어 데이터셋은 존재하지 않음
5. **서브픽셀 조성 분해 없음**: 궤도 혼합 픽셀을 지표 ground truth로 분해 시도 없음

## 8.8 우리 연구의 위치

**행성과학 사상 최초**: co-located ground-level 다중분광 영상 (Mastcam-Z) → 궤도 영상 (HiRISE) SR 학습 데이터로 활용

**핵심 enabler**:
1. Mastcam-Z 향상된 역량 (줌, 다중분광, 매칭 스테레오)
2. Jezero 밀집 HiRISE 커버리지
3. DL SR 방법론 발전 (RefSR: SRNTT, TTSR, DATSR)
4. 20년간 축적된 기하 정합 파이프라인 (Li, Di et al.)

**Stack et al. (2016)과의 관계**: Stack은 궤도-지표 정보 갭을 정량화. 우리는 그 갭을 computational하게 줄이는 최초 시도.
