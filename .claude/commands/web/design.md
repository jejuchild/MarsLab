# Web — Design

홈페이지 디자인 시스템을 결정합니다. 컬러, 타이포그래피, 레이아웃, 컴포넌트 스타일.

## Prerequisites
- `docs/homepage-PRD.md`가 있으면 먼저 읽으세요
- 없어도 진행 가능 — 사용자에게 디자인 방향을 직접 물어보세요

## Instructions

### Step 1: 디자인 방향 탐색
`/ui-ux-pro-max` 스킬의 데이터베이스를 활용하여 사용자와 함께 결정:

**컬러 팔레트** — 산업/브랜드에 맞는 팔레트 2-3개 제안, 사용자 선택
**타이포그래피** — 헤딩/본문 폰트 페어링 2-3개 제안
**레이아웃 스타일** — 50+ 스타일 중 후보 제시:
  - Editorial (매거진풍, 대담한 타이포)
  - Corporate (깔끔, 신뢰감)
  - Minimal (여백 중심)
  - Brutalist (raw, 개성 강한)
  - 기타

**핵심 원칙**: "AI가 만든 것 같은" 패턴을 피하세요:
- 보라색 그라데이션 + 흰 배경
- Inter/Roboto 디폴트
- 천편일률적 카드 그리드
- 의미 없는 추상 일러스트

### Step 2: 디자인 시스템 정의
결정사항을 `docs/design-system.md`로 정리:

```markdown
# Design System

## Colors
- Primary: #___
- Secondary: #___
- Accent: #___
- Background: #___
- Text: #___

## Typography
- Heading: [폰트명] / [weight scale]
- Body: [폰트명] / [size scale]

## Spacing
- Base unit: _px
- Section padding: _
- Component gap: _

## Components
- Buttons: [스타일 설명]
- Cards: [스타일 설명]
- Navigation: [스타일 설명]

## Responsive
- Mobile: < 768px
- Tablet: 768-1024px
- Desktop: > 1024px
```

### Step 3: 비주얼 에셋 (선택)
필요 시:
- `/document-skills:canvas-design` — 히어로 이미지, 배너, 포스터 제작
- `/figma:figma-generate-design` — Figma에 디자인 푸시 (Figma 사용 시)

### 다음 단계
사용자에게 `/user:web/build`를 안내하세요.

$ARGUMENTS - 디자인 키워드 (예: "다크모드 미니멀", "화이트 에디토리얼", "에너지 산업 느낌")
