# MarsLab

> 화성 탐사 데이터를 통합 분석하는 웹 기반 지리정보 플랫폼

MarsLab은 NASA의 Mars Reconnaissance Orbiter(MRO) 탐사선이 수집한 다양한 원격탐사 데이터를 하나의 인터페이스에서 시각화하고 분석할 수 있는 도구입니다. Cesium.js 기반 3D 화성 지구본 위에 6종의 관측 장비 데이터를 겹쳐 표시하고, 지형 분석, AI 기반 검색, 데이터 다운로드까지 지원합니다.

---

## 주요 기능

- **3D/2D 화성 지도** — Cesium.js 기반 인터랙티브 지구본 + 2D 평면 모드
- **6종 관측 장비 데이터** — CRISM, HiRISE, SHARAD, SHARAD High-Res, CTX, HiRISE DTM
- **제품 인스펙터** — 스펙트럼 분석(CRISM), 픽셀 통계(HiRISE), 3D 지형(DTM)
- **지형 분석 도구** — Slope 분석, 3D Slope 시각화, 고도 프로파일, AI 분석
- **다중 필터** — Multi-Instrument Overlap Filter, Ice/Hydration Score Filter
- **AI 검색** — 자연어로 화성 데이터 검색 (Google Gemini 연동)
- **데이터 다운로드** — aria2 기반 병렬 다운로드, 진행률 추적, 재개 지원
- **필드 노트** — 태그 기반 메모 시스템, 지도 위 마커 표시
- **사용자 데이터 업로드** — GeoTIFF 업로드 및 오버레이 표시

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 프론트엔드 | React 18 + TypeScript + Vite |
| 3D 지도 | Cesium.js (화성 지구본) |
| 3D 시각화 | Three.js (지형, 라다그램) |
| 차트 | Recharts (고도 프로파일) |
| 백엔드 | Python FastAPI |
| 다운로드 | aria2c (병렬 HTTP 다운로드) |
| AI | Google Gemini API |
| 데이터 소스 | NASA ODE (Orbital Data Explorer) |

---

## 목차

| # | 페이지 | 설명 |
|---|--------|------|
| 01 | [시작하기](01-getting-started.md) | 설치, 설정, 첫 실행 |
| 02 | [지도 인터페이스](02-map-interface.md) | 3D/2D 모드, 베이스맵, 좌표 그리드 |
| 03 | [관측 장비 (Instruments)](03-instruments.md) | 6종 instrument 상세 설명 |
| 04 | [제품 인스펙터](04-inspector.md) | 제품 상세 정보, 오버레이, 스펙트럼 |
| 05 | [분석 도구](05-analysis-tools.md) | Slope, 3D, Line Profile, AI Analysis |
| 06 | [필터링](06-filters.md) | Overlap Filter, Ice Score Filter |
| 07 | [검색 & 다운로드](07-search-download.md) | 5가지 검색 모드, 다운로드 관리 |
| 08 | [필드 노트](08-field-notes.md) | 메모 시스템 |
| 09 | [사용자 데이터](09-custom-data.md) | GeoTIFF 업로드 |
| 10 | [시스템 아키텍처](10-architecture.md) | 프론트엔드/백엔드 구조, 데이터 흐름 |
| 11 | [API 레퍼런스](11-api-reference.md) | 전체 REST API 문서 |
| 12 | [FAQ](12-faq.md) | 자주 묻는 질문, 문제 해결 |
