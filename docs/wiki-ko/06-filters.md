# 필터링 (Filters)

MarsLab은 2가지 필터 시스템으로 관심 있는 제품만 선택적으로 표시할 수 있습니다.

---

## Multi-Instrument Overlap Filter (다중 장비 교집합 필터)

여러 장비의 관측 영역이 겹치는 제품만 표시합니다. 동일 지역을 여러 장비로 관측한 경우를 찾을 때 유용합니다.

### 활용 예시
- CRISM 분광 데이터와 HiRISE 고해상도 이미지가 겹치는 영역 탐색
- SHARAD 지하 레이더 트랙이 HiRISE DTM 영역을 지나가는 경우 탐색

### 사용법
1. 좌측 패널 **Multi-Instrument Overlap Filter** 섹션 열기
2. 교집합을 계산할 장비 2개 이상 체크
3. 필터가 자동으로 계산됨
4. 다른 장비와 겹치는 제품만 지도에 표시, 나머지는 숨김

### 통계 표시
각 장비별로 통과한 제품 수 표시:
```
CRISM: 5 / 142 통과
HiRISE: 3 / 89 통과
SHARAD: 2 / 50 통과
```

### 기술 동작
- 각 제품의 bounding box를 계산
- 공간 그리드 인덱스(5도 셀)로 후보군 빠르게 추출
- LineString(SHARAD)은 Liang-Barsky 알고리즘으로 정밀한 선분-사각형 교차 검사
- Polygon(CRISM/HiRISE)은 bounding box 교차로 판정
- 안티메리디안(180도선) 교차도 처리

### 필터 조건
제품 P가 통과하려면:
> P의 장비가 아닌 **다른 장비** 중 하나 이상에서, P와 공간적으로 겹치는 제품이 1개 이상 존재해야 함

---

## Ice Score Filter (얼음 점수 필터)

CRISM 제품의 Ice Score(얼음 점수)를 기준으로 필터링합니다.

### Ice Score란?
CRISM 분광 데이터에서 얼음 관련 흡수 대역의 강도를 수치화한 점수입니다.
- 높을수록 얼음 존재 가능성이 높음
- 각 관측 영역 내 픽셀별로 점수가 계산됨

### 사용법
1. 좌측 패널 **Ice Score Filter** 섹션 열기
2. **Min Ice Score** 슬라이더 조절 (0.05 ~ 1.5)
   - 기본값: 0.3
   - 각 픽셀의 최소 얼음 점수 임계값
3. **Min % of Pixels** 슬라이더 조절 (0% ~ 50%)
   - 기본값: 5%
   - 관측 영역 내 임계값을 초과하는 픽셀의 최소 비율
4. 필터 결과 자동 적용: 기준 충족 CRISM 제품만 표시

### 결과 표시
```
12 / 142 CRISM products passing
min_score: 0.30, min_percent: 5.0%
```

### 기술 동작
- 백엔드의 `crism_score/score_stats.json`에 사전 계산된 통계 활용
- 15개 사전 정의 임계값(0.05~1.5) 중 가장 가까운 값으로 조회
- 임계값 이상 픽셀 수 / 전체 유효 픽셀 수 = 퍼센트
- 퍼센트 ≥ Min % → 통과

### Hydration Score Filter
- 동일한 메커니즘으로 수화물(Hydration) 점수 필터링도 지원
- API: `GET /api/filter/hyd?min_score=0.3&min_percent=5`

---

## 필터 조합

두 필터는 **AND 조건**으로 조합됩니다:

1. Overlap Filter가 활성화되면, 다른 장비와 겹치는 제품만 남김
2. Ice Score Filter가 활성화되면, 점수 기준을 충족하는 CRISM 제품만 남김
3. 두 필터 모두 활성화하면, **두 조건 모두** 충족하는 제품만 표시

이를 통해 "다른 장비와 겹치면서 동시에 얼음 점수가 높은 CRISM 관측"을 정확히 찾아낼 수 있습니다.
