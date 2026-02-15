# FAQ / 문제 해결

---

## 일반

### Q: 지도가 표시되지 않아요
- Cesium.js는 WebGL이 필요합니다. 브라우저에서 하드웨어 가속이 활성화되어 있는지 확인하세요.
- Chrome/Edge 권장. Firefox에서는 일부 렌더링 이슈가 있을 수 있습니다.

### Q: 프론트엔드/백엔드 연결이 안 돼요
- 백엔드가 `http://localhost:8000`에서 실행 중인지 확인
- CORS 설정은 기본적으로 모든 origin을 허용합니다 (`allow_origins=["*"]`)
- 프론트엔드의 API base URL이 올바른지 확인

---

## Footprint

### Q: Footprint가 지도에 표시되지 않아요
- 좌측 패널에서 해당 장비가 **토글 ON** 상태인지 확인
- 줌 레벨이 너무 높으면 LOD에 의해 숨겨질 수 있습니다 (카메라 높이 > 15,000km)
- 해당 장비의 index.geojson에 데이터가 있는지 확인

### Q: "truncated" 표시가 나와요
- 한 번에 최대 2,000개까지만 표시됩니다
- 지도를 줌인하여 더 좁은 영역을 보면 해결됩니다

### Q: SHARAD 트랙이 이상하게 보여요
- SHARAD의 궤도 트랙은 안티메리디안(180도선)을 교차할 수 있습니다
- 이러한 세그먼트는 자동으로 필터링됩니다 (경도 점프 > 180도)

---

## 다운로드

### Q: 다운로드가 느려요
- aria2c가 설치되어 있으면 자동으로 병렬 다운로드합니다
- `aria2c --version`으로 설치 여부 확인
- 미설치 시: `sudo apt install aria2` (Ubuntu) 또는 `brew install aria2` (macOS)

### Q: 다운로드한 제품이 검색에 안 나와요
- 다운로드 완료 후 index.geojson이 자동 업데이트됩니다
- ODE API 응답 실패로 인덱스 업데이트가 실패할 수 있습니다
- 해결: 서버 재시작 (자동 복구) 또는 `POST /api/repair-index` 호출

### Q: HiRISE JP2 파일이 다운로드되는데 TIF로 변환이 안 돼요
- GDAL이 필요합니다: `gdal_translate --version`으로 확인
- JP2 지원을 위해 OpenJPEG 드라이버가 포함된 GDAL 빌드 필요

---

## 분석 도구

### Q: Slope Analysis가 "데이터 없음"이라고 나와요
- MOLA DEM 파일(`Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif`)이 프로젝트 루트에 있는지 확인
- 파일 크기: 약 4.6GB

### Q: AI Analysis가 작동하지 않아요
- Gemini API 키가 설정되어 있는지 확인
- `GET /api/ai/gemini/status`로 API 상태 확인
- API 일일 할당량 초과 시 오류 발생 가능

### Q: 3D 뷰어가 느려요
- Three.js 기반 3D 렌더링은 GPU 가속이 필요합니다
- 분석 반경을 줄이면 메시 크기가 작아져 성능 향상

---

## 필터

### Q: Overlap Filter에서 SHARAD과 DTM 교차가 거의 없어요
- DTM footprint는 매우 작습니다 (~0.1-0.2도)
- SHARAD LineString 트랙은 좁습니다
- Liang-Barsky 알고리즘으로 정밀 교차 검사를 하기 때문에 bbox 겹침만으로는 통과하지 않습니다
- 이는 정상 동작입니다

### Q: Ice Score Filter가 작동하지 않아요
- `backend/crism_score/score_stats.json` 파일이 존재하는지 확인
- 이 파일은 CRISM 데이터의 사전 계산된 점수 통계입니다
- 누락 시 `backend/scripts/generate_score_maps.py`로 생성 가능

---

## 데이터

### Q: 경도가 이상한 값이에요 (예: 270도)
- NASA ODE는 0-360도 경도 체계를 사용합니다
- MarsLab은 내부적으로 -180/180 체계로 변환합니다
- `normalizeLonForMap()` 함수가 이 변환을 처리합니다

### Q: CRISM 제품 ID가 복잡해요
- CRISM 제품 ID 예: `frt0001fd76_07_if166j_mtr3`
- `frt0001fd76` = 관측 ID
- `07` = 버전
- `if166j` = 데이터 타입 (I/F 반사율)
- `mtr3` = 처리 수준 (MTRDR)
- `extractCrismObsId()`로 관측 ID만 추출 가능

### Q: DTM 제품 ID 패턴이 다양해요
- `DTEEC_`, `DTEED_`, `DTEEP_` 등 다양한 접두사
- 모두 `DTE`로 시작 → `startsWith("DTE")`로 매칭
- `startsWith("DTEEC_")`만 사용하면 다른 유형이 누락됩니다

---

## 성능

### Q: 서버 시작이 느려요
- GeoJSON 인덱스 병렬 로드: ~2초
- 인덱스 자동 복구: 백그라운드 태스크로 실행 (시작을 차단하지 않음)
- ODE API 응답이 느릴 경우 복구 시간이 길어질 수 있음

### Q: 지도가 버벅거려요
- 표시된 footprint 수가 2,000개 이상이면 성능 저하
- 줌인하여 표시 영역 줄이기
- 불필요한 장비 레이어 끄기
- drillPick은 50ms 쓰로틀링 적용됨
