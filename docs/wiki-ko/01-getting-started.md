# 시작하기

## 요구사항

| 항목 | 버전 |
|------|------|
| Node.js | 18+ |
| Python | 3.10+ |
| aria2c | 1.36+ (선택, 다운로드 가속) |
| GDAL | 3.0+ (HiRISE JP2→TIF 변환) |

---

## 백엔드 설정

```bash
# 1. 리포지토리 클론
git clone <repo-url> MarsLab
cd MarsLab/backend

# 2. Python 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 데이터 디렉토리 생성 (최초 1회)
mkdir -p crism_data hirise_data sharad_data sharad_highres sharad_highres_data
mkdir -p hirise_dtm_data ctx_data custom_data crism_score
mkdir -p crism_quickview hirise_quickview sharad_quickview crism_browse

# 5. 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

서버가 `http://localhost:8000`에서 시작됩니다.

---

## 프론트엔드 설정

```bash
cd MarsLab/frontend

# 1. 의존성 설치
npm install

# 2. 개발 서버 실행
npm run dev -- --host 0.0.0.0
```

브라우저에서 `http://localhost:5173` 접속합니다.

---

## 환경 변수

### Gemini API (AI 검색/분석용)

AI 검색과 AI 분석 기능을 사용하려면 Google Gemini API 키가 필요합니다.

```bash
# backend 디렉토리에서
export GEMINI_API_KEY="your-api-key-here"
```

또는 `~/.gemini/settings.json`에 저장:
```json
{
  "apiKey": "your-api-key-here"
}
```

---

## MOLA DEM 데이터 (지형 분석용)

Slope Analysis, Line Profile 등 지형 분석 기능을 사용하려면 MOLA 전역 DEM 파일이 필요합니다:

- 파일명: `Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif`
- 위치: 프로젝트 루트 디렉토리
- 용량: 약 4.6GB

---

## 첫 실행 확인

1. 백엔드 서버 시작 (`http://localhost:8000`)
2. 프론트엔드 서버 시작 (`http://localhost:5173`)
3. 브라우저에서 3D 화성 지구본이 표시되는지 확인
4. 좌측 패널에서 Footprint 토글 → 지도에 데이터 표시 확인

---

## 서버 시작 시 자동 처리

백엔드 서버가 시작되면 자동으로 수행되는 작업:

1. **GeoJSON 인덱스 병렬 로드** — 6개 instrument의 index.geojson을 메모리에 캐싱 (gzip 압축 포함)
2. **인덱스 자동 복구** — 다운로드 완료되었지만 index.geojson에 누락된 제품을 자동으로 감지하여 추가
3. **aiohttp 세션 생성** — ODE API 연결 풀링을 위한 공유 HTTP 세션

---

## 프로젝트 디렉토리 구조

```
MarsLab/
├── frontend/                  # React + Vite 프론트엔드
│   ├── src/
│   │   ├── pages/             # MainPage, DataDownloadPage
│   │   ├── components/        # MapView, Inspector, LayerPanel 등
│   │   ├── config/            # instrumentRegistry
│   │   └── utils/             # FootprintManager, overlapFilter
│   └── package.json
│
├── backend/                   # Python FastAPI 백엔드
│   ├── app.py                 # 메인 서버, 라이프사이클, 엔드포인트
│   ├── api/                   # 라우터 모듈
│   │   ├── search_router.py   # 검색 & 다운로드
│   │   ├── footprints_router.py
│   │   ├── ai_search.py       # AI 자연어 검색
│   │   ├── proximity_router.py
│   │   ├── terrain_router.py  # 지형 분석
│   │   ├── index_repair.py    # 인덱스 자동 복구
│   │   └── ...
│   ├── crism_data/            # CRISM 다운로드 데이터
│   ├── hirise_data/           # HiRISE 다운로드 데이터
│   ├── sharad_highres_data/   # SHARAD HR 인덱스
│   ├── hirise_dtm_data/       # HiRISE DTM 데이터
│   └── requirements.txt
│
└── Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif  # MOLA 전역 DEM
```
