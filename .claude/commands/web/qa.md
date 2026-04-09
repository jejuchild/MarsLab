# Web — QA

홈페이지 품질 검증. 테스트 → 분석 → 정리 → 개선 → 리포트.

## Instructions

프로젝트의 HTML/CSS/JS 파일을 자동 탐색하여 전수 검사합니다.
$ARGUMENTS로 특정 단계만 실행 가능 (예: "step1만", "성능만").

### Step 1: UI 자동 테스트
`/document-skills:webapp-testing` (Playwright) 활용.

로컬 서버를 띄우고 (`python3 -m http.server` 또는 프로젝트의 dev server):
- 모든 페이지 로드 확인 (200 OK)
- 콘솔 에러 체크
- 네비게이션 링크 동작 검증
- 반응형 스크린샷 (375px / 768px / 1440px)
- 폼 동작 테스트
- 다국어 토글 (해당 시)
- 깨진 이미지 체크 (lazy-load 이미지는 스크롤 후 확인)

스크린샷은 `docs/qa_screenshots/`에 저장.

### Step 2: 코드 품질 분석
`/sc:analyze` 활용.

**성능**:
- 렌더 블로킹 리소스 (async/defer 누락)
- 이미지 최적화 (WebP, 압축, lazy-load)
- 폰트 로딩 (font-display, preload)
- CSS/JS 파일 크기
- CDN 의존성

**보안**:
- CSP 헤더 존재 여부
- 외부 스크립트 SRI (Subresource Integrity)
- XSS 위험 (innerHTML, eval, document.write)
- API 키 노출
- target="_blank" rel="noopener noreferrer"

**SEO**:
- title, meta description, og tags, twitter cards
- canonical URL 정확성
- heading 계층 (h1 → h2 → h3)
- 구조화 데이터 (JSON-LD)
- favicon
- alt text

**접근성 (WCAG 2.1 AA)**:
- skip navigation
- aria-label on interactive elements
- 색상 명도대비
- 시맨틱 HTML
- 폼 라벨

### Step 3: 코드 정리
- 사용하지 않는 CSS/JS 식별
- 중복 코드 (header/footer 복붙 등)
- 불필요한 파일 (.gitignore 대상)
- 프로젝트 구조 개선점

### Step 4: 개선 적용
`/sc:improve` + `/document-skills:simplify` 활용:
- 발견된 이슈 중 자동 수정 가능한 것 즉시 적용
- 수정 후 Step 1 재테스트로 검증

### Step 5: QA 리포트
결과를 `docs/qa-report.md`로 저장:

```markdown
# QA Report — [날짜]

## 테스트 결과
## 성능/보안/SEO/접근성 점수
## 적용된 수정사항
## 잔여 이슈 (심각도별)
## 추후 개선 권장사항
```

### 다음 단계
사용자에게 `/user:web/deploy`를 안내하세요.

$ARGUMENTS - 포커스 (예: "성능만", "접근성만", "전체", "step1만")
