# 검색 & 다운로드 (Search & Download)

상단 네비게이션에서 **Data Download** 페이지로 이동하면 제품 검색 및 다운로드 관리를 할 수 있습니다.

---

## 검색 모드

### 1. ID 검색 (기본)

제품 ID로 직접 검색합니다.

- 검색창에 제품 ID 전체 또는 일부 입력
- **자동완성(Typeahead)**: 입력과 동시에 후보 목록 표시
- NASA ODE API + 로컬 인덱스를 모두 검색
- 예: `frt0001fd76` → CRISM 제품 매칭
- 예: `ESP_024943` → HiRISE 제품 매칭

### 2. Spatial 검색 (공간 범위)

좌표 범위(bounding box)로 검색합니다.

| 파라미터 | 설명 |
|----------|------|
| Min Latitude | 남쪽 위도 |
| Max Latitude | 북쪽 위도 |
| West Longitude | 서쪽 경도 |
| East Longitude | 동쪽 경도 |
| Instrument | 장비 선택 (전체 또는 특정 장비) |

- NASA ODE API를 통해 해당 범위 내 제품 검색
- HIRISE_DTM, CTX, SHARAD_HIGHRES는 로컬 인덱스 기반 검색

### 3. Point 검색 (좌표+반경)

특정 좌표 주변 반경으로 검색합니다.

| 파라미터 | 설명 |
|----------|------|
| Latitude | 중심 위도 |
| Longitude | 중심 경도 |
| Radius | 검색 반경 (도) |

- 모든 장비를 한번에 검색
- 결과에 **거리(km)** 포함
- SHARAD 트랙은 point-to-line 거리 계산

### 4. AI 검색 (자연어)

Google Gemini를 활용한 자연어 기반 검색입니다.

**사용법**:
1. 검색 모드를 **AI Search** 로 전환
2. 자연어 질문 입력
3. Preview → Execute 2단계로 진행

**예시 쿼리**:
- "Jezero crater 근처 CRISM 데이터"
- "화성 적도 부근에서 얼음 흔적이 있는 지역"
- "Valles Marineris를 가로지르는 SHARAD 트랙"

**동작 원리**:
1. Gemini가 쿼리를 구조화된 필터로 파싱
   - 지역명 → bounding box 변환 (~55개 화성 지명 DB)
   - 장비 필터, 공간 조건(intersects/within/nearest), 분포 패턴
2. 필터 기반으로 ODE API + 로컬 인덱스 검색 실행
3. 결과 반환

**Cross-Instrument 검색**:
- "HiRISE DTM과 겹치는 SHARAD 트랙" 같은 교차 검색 지원
- 참조 장비의 인덱스 로드 → 결합 bbox 계산 → 대상 장비 검색 → 공간 교차 필터링

### 5. Product 기반 검색

메인 페이지 Inspector의 **Find Related Products** 버튼에서 자동으로 전환됩니다.

- 선택한 제품의 ID와 장비 정보가 미리 입력됨
- 해당 제품 주변의 다른 장비 제품을 탐색

---

## 검색 결과

검색 결과 목록에 표시되는 정보:

| 항목 | 설명 |
|------|------|
| Product ID | 제품 식별자 |
| Instrument | 관측 장비명 |
| Lat/Lon | 중심 좌표 |
| Local | 로컬 다운로드 여부 (체크 아이콘) |

- 결과 클릭 → 상세 정보 패널 확장
- **Download** 버튼으로 다운로드 시작

---

## 다운로드 시스템

### Downloads 탭

활성 다운로드 작업을 관리하는 탭:

| 기능 | 설명 |
|------|------|
| 진행률 바 | 파일별 다운로드 진행률 (%) |
| 속도 | 현재 다운로드 속도 (MB/s) |
| Cancel | 다운로드 취소 |
| Clear History | 완료/실패 작업 이력 삭제 |

### 장비별 다운로드 내용

| 장비 | 다운로드 파일 |
|------|--------------|
| CRISM | .img (데이터), .lbl (레이블), .hdr (헤더), .tab (파장표), .png (브라우즈) |
| HiRISE | .JP2 (이미지) → .tif (자동 변환), .lbl (레이블) |
| SHARAD | .jpg (THM 퀵뷰), .lbl (레이블) |
| SHARAD HR | .dat (RDR 바이너리), .lbl (레이블), cluttergram (있는 경우) |

### aria2 다운로드 가속

aria2c가 설치되어 있으면 자동으로 활용됩니다:
- **다중 연결 병렬 다운로드**: 대용량 파일 고속 다운로드
- **중단 후 재개**: 네트워크 끊김 시 이어받기
- **백그라운드 실행**: 페이지를 닫아도 서버에서 계속 다운로드

aria2c가 없으면 aiohttp 기반 순차 다운로드로 자동 전환됩니다.

### 인덱스 자동 업데이트

다운로드 완료 시:
1. 제품을 해당 장비의 `index.geojson`에 자동 추가
2. ODE API에서 footprint 좌표 조회
3. 인덱스 갱신 → 지도에서 바로 표시 가능

### 인덱스 자동 복구

서버 시작 시 자동으로 수행:
- 모든 데이터 디렉토리를 스캔
- 다운로드는 완료되었지만 인덱스에 누락된 제품 탐지
- ODE API로 footprint 조회 후 인덱스에 자동 추가
- 수동 실행: `POST /api/repair-index`
