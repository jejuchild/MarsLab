# Web — Build

홈페이지를 실제로 구현합니다. 디자인 시스템을 코드로 변환.

## Prerequisites
- `docs/design-system.md` — 있으면 반드시 읽고 적용
- `docs/homepage-PRD.md` — 있으면 페이지 구조 참조
- 둘 다 없으면 사용자에게 방향을 물어보세요

## Instructions

### Step 1: 현황 파악
프로젝트 루트를 스캔하세요:
- 기존 HTML/CSS/JS 파일
- package.json (프레임워크 사용 여부)
- 배포 설정 (vercel.json, netlify.toml 등)
- 이미지/에셋 폴더

### Step 2: 구현 계획
페이지/컴포넌트 간 의존성을 파악하고 구현 순서 결정:
1. 공통 요소 먼저 (header, footer, nav, 글로벌 CSS)
2. 메인 페이지 (index/home)
3. 서브 페이지 (우선순위순)
4. 인터랙티브 기능 (폼, 애니메이션, API 연동)

### Step 3: 코딩
**`/document-skills:frontend-design` 스킬을 핵심으로 활용하세요.** 이 스킬은:
- "AI스러운" 디자인 패턴을 명시적으로 피함
- 독창적이고 기억에 남는 UI를 생성
- 프로덕션급 코드 품질

코딩 시 체크리스트:
- [ ] design-system.md의 컬러/폰트/간격 일관 적용
- [ ] 반응형 (모바일 우선)
- [ ] 시맨틱 HTML (nav, main, article, section, footer)
- [ ] 접근성 기본 (alt text, aria-label, heading 계층, skip nav)
- [ ] SEO 기본 (title, meta description, og tags, canonical, favicon)
- [ ] rel="noopener noreferrer" on target="_blank"
- [ ] 이미지 lazy loading (below-fold)

### Step 4: 기능 구현
백엔드 연동이 필요한 기능 (문의폼, CMS, 뉴스피드 등):
- `/sc:implement`으로 end-to-end 구현

### 다음 단계
사용자에게 `/user:web/qa`를 안내하세요.

$ARGUMENTS - 구현 대상 (예: "index.html만", "전체 사이트", "nav 컴포넌트", "문의폼 추가")
