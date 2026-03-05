# MarsLab — CS × 화성 연구 플랫폼 전체 작업 요약

> 작성일: 2026-03-05
> 프로젝트: MarsLab (컴퓨터과학 × 화성 과학 융합 플랫폼)

---

## 개요

"컴퓨터과학을 화성 연구에 적용"이라는 주제로, **즉시 구현 가능한 CS×Mars 연구 기회 3가지**를 선정하여 모두 구현·검증 완료하였습니다.

| # | 모듈 | 핵심 기술 | 상태 |
|---|---|---|---|
| 1 | Mars GCM Neural Emulator | Deep Learning (MLP) | ✅ 완료 |
| 2 | 화성 과학 RAG 시스템 | NLP + Vector DB | ✅ 완료 |
| 3 | PINNs 화성 내부 구조 역산 | Physics-Informed NN | ✅ 완료 |

---

## 1. Mars GCM Neural Emulator (신경망 기후 에뮬레이터)

### 목적
기존 물리 기반 화성 대기순환 모델(GCM)을 딥러닝으로 대체하는 서로게이트 모델. 위도·경도·태양경도(Ls)·고도를 입력하면 기온·기압·바람·먼지 등 7개 기후 변수를 **수 밀리초**만에 예측합니다.

### 아키텍처
- **모델**: 4-layer ResidualBlock MLP (**534,535 파라미터**)
- **입력 6차원**: `lat_norm`, `sin(lon)`, `cos(lon)`, `sin(Ls)`, `cos(Ls)`, `elevation_norm`
- **출력 7차원**: `T_mean`, `T_max`, `T_min`, `pressure`, `dust_tau`, `wind`, `frost_probability`
- **학습 데이터**: 47,952 샘플 (위도 5°, 경도 10°, Ls 10° 그리드 × 고도 변화)
- **최적화**: Adam + Cosine Annealing + Early Stopping (patience 20)

### 예측 예시 (적도, Ls=180°)
| 변수 | 예측값 |
|---|---|
| 평균기온 | 215.8 K (-57.3°C) |
| 최고기온 | 254.8 K |
| 최저기온 | 174.5 K |
| 기압 | 636 Pa |
| 먼지 광학두께 | 0.536 |
| 풍속 | 6.05 m/s |
| 서리 확률 | 9.2% |

### 파일 구조
```
backend/neural_climate/
├── __init__.py
├── model.py              # MarsClimateEmulator (ResidualBlock MLP)
├── dataset.py            # 학습 데이터 생성 (파라메트릭 모델 기반)
├── trainer.py            # 학습 루프 (early stopping, cosine annealing)
├── predictor.py          # 추론 API + 파라메트릭 모델 비교
├── climate_router.py     # FastAPI 엔드포인트
├── checkpoints/
│   ├── best_model.pt     # 학습된 모델 가중치
│   └── norm_stats.npz    # 출력 정규화 통계
└── data/
    └── mars_climate_dataset.npz  # 47,952 학습 샘플
```

### 테스트
- `tests/test_neural_climate.py`: **52개 테스트 케이스** — 전부 통과

---

## 2. 화성 과학 RAG 시스템 (Retrieval-Augmented Generation)

### 목적
화성 과학 문서를 벡터 DB에 저장하고, 자연어 질의 시 관련 문서를 검색해 답변을 생성하는 시스템. 연구자가 "HiRISE 해상도가 뭐야?"라고 물으면 즉시 관련 문서 조각을 찾아줍니다.

### 아키텍처
- **임베딩 모델**: SentenceTransformer `all-MiniLM-L6-v2` (384차원)
- **벡터 DB**: ChromaDB (코사인 유사도 검색)
- **청킹**: 512자 청크, 64자 오버랩
- **리랭킹**: 화성 도메인 부스트 용어 적용

### 적재된 데이터 (총 3,819 벡터)
| 소스 | 내용 | 청크 수 |
|---|---|---|
| 내장 지식 (12문서) | 미션, 장비, 지질, 기후, 우주생물학, AI 응용 | ~43 |
| knowledge/ 마크다운 (15파일) | HiRISE, CRISM, SHARAD, MOLA, CTX, THEMIS, Mars2020, 광물학, 얼음, 유인탐사, Arcadia Planitia 등 | ~300+ |
| 에이전트 보고서 | 일일 토론, 요약, 비평, 증거팩 | ~3,400+ |

### 검색 예시 ("What is HiRISE camera resolution?")
| 순위 | 유사도 | 내용 |
|---|---|---|
| 1위 | 0.734 | HiRISE 개요 문서 |
| 2위 | 0.636 | HiRISE 사양 — "Resolution: 25-50 cm/pixel" |
| 3위 | 0.553 | HiRISE 상세 스펙 |

### 파일 구조
```
backend/rag/
├── __init__.py
├── chunker.py         # 텍스트 청킹 (오버랩 지원)
├── embedder.py        # SentenceTransformer 임베딩
├── vector_store.py    # ChromaDB + 인메모리 폴백
├── retriever.py       # 유사도 검색 + 컨텍스트 포맷
├── generator.py       # LLM 기반 답변 생성
├── ingestion.py       # 텍스트/파일/디렉토리 적재 파이프라인
├── mars_knowledge.py  # 내장 큐레이션 지식 (12문서)
└── rag_router.py      # FastAPI 엔드포인트

backend/data/rag_vectordb/   # ChromaDB 영구 저장소 (3,819 벡터)
knowledge/                    # 15개 큐레이션 마크다운 문서
```

### 테스트
- `tests/test_rag.py`: **20개 테스트 케이스** — 전부 통과

---

## 3. PINNs 화성 내부 구조 역산 (Physics-Informed Neural Networks)

### 목적
InSight 착륙선의 SEIS 지진계 데이터를 기반으로, 화성 내부의 P파 속도 프로파일 V_p(r)을 물리 법칙에 구속받는 신경망으로 학습하는 역문제 풀이 시스템.

### 아키텍처
- **모델**: **12,673 파라미터** Physics-Informed Neural Network
- **물리 제약**: 지진파 전파 방정식 + 경계조건 (지표면/핵 경계)
- **손실 함수**: `L = λ_data × L_data + λ_physics × L_physics + λ_bc × L_bc`
- **참조 모델**: InSight SEIS 실측 기반 (지각/맨틀/핵 3층 구조)

### 화성 내부 참조 모델
| 층 | 깊이 범위 | 참조 Vp | 역할 |
|---|---|---|---|
| 지각 (Crust) | 0–50 km | ~3.5–5.0 km/s | 제약조건 |
| 맨틀 (Mantle) | 50–1,560 km | ~6.5–8.0 km/s | 역산 대상 |
| 핵 (Core) | 1,560–1,700+ km | ~4.5–5.5 km/s | 경계조건 |

### 파일 구조
```
backend/pinns_interior/
├── __init__.py
├── mars_model.py      # 화성 내부 참조 모델 (InSight SEIS)
├── forward.py         # 1D 레이 트레이싱 (주시곡선 계산)
├── pinn_model.py      # PINNs 신경망 (V_p(r) 학습)
├── trainer.py         # 물리+데이터+경계 다중 손실 학습
├── predictor.py       # 추론 + 참조모델 비교
├── pinns_router.py    # FastAPI 엔드포인트
└── checkpoints/
    └── best_pinn_model.pt  # 학습된 PINN 가중치
```

### 테스트
- `tests/test_pinns_interior.py`: **12개 테스트 케이스** — 전부 통과

---

## API 통합

세 모듈 모두 FastAPI 라우터로 `backend/app.py`에 통합되어 있습니다.

| 라우터 | 엔드포인트 | 기능 |
|---|---|---|
| Neural Climate | `/api/climate/neural/*` | 신경망 기후 예측, 학습 상태, 파라미터 비교 |
| RAG | `/api/rag/*` | 문서 검색, 질의응답, 컬렉션 관리 |
| PINNs | `/api/pinns/*` | 내부 구조 예측, 프로파일, 참조모델 비교, 학습 |

---

## 전체 수치 요약

| 항목 | 값 |
|---|---|
| 신규 소스 파일 | 22개 |
| 테스트 케이스 | 84개 (전부 통과) |
| 테스트 코드 | 982줄 |
| 학습된 모델 | 2개 (534K + 12.7K 파라미터) |
| RAG 벡터 | 3,819개 |
| 지식 문서 | 15개 마크다운 + 12개 내장 |
| API 엔드포인트 | 3개 라우터 그룹 |

---

## 향후 과제

1. **외부 논문 크롤링**: arXiv, NASA ADS에서 화성 과학 논문 수집 → RAG 적재
2. **NASA PDS 기술문서**: 장비 매뉴얼, 데이터 카탈로그 문서 수집
3. **Neural Climate 고도화**: 더 많은 에폭 학습, 실제 GCM 출력 데이터로 교체
4. **PINNs 확장**: S파 속도, 밀도 프로파일까지 역산 확장
5. **RAG 생성기**: 실제 LLM 연결하여 답변 생성 파이프라인 완성
