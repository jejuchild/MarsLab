# Web — Deploy

홈페이지 배포. QA 확인 → 빌드 → 커밋 → 배포 → 프로덕션 검증.

## Instructions

### Step 1: 배포 전 확인
- `docs/qa-report.md`가 있으면 읽고 미해결 critical 이슈 확인
- Critical 이슈가 있으면 사용자에게 경고하고 수정 권고

### Step 2: 배포 체크리스트
자동 검사:
- [ ] .gitignore에 민감 파일 포함 (.env, credentials, API keys)
- [ ] 불필요한 파일이 커밋 대상에 없는지 (node_modules, .DS_Store, *.py 등)
- [ ] meta/OG 태그 프로덕션 URL로 설정
- [ ] 이미지 최적화 완료
- [ ] 배포 설정 파일 확인 (vercel.json, netlify.toml 등)

### Step 3: Git 커밋 & 푸시
- `git status`로 변경사항 확인
- 사용자에게 커밋 메시지 제안
- 사용자 확인 후 커밋 & 푸시
- **반드시 사용자에게 어느 remote/branch에 푸시할지 확인**

### Step 4: 배포
플랫폼별 대응:
- **Vercel**: push 시 자동 배포 또는 `vercel --prod`
- **Netlify**: push 시 자동 배포 또는 `netlify deploy --prod`
- **GitHub Pages**: gh-pages branch 또는 Actions
- **기타**: 사용자에게 배포 방법 확인

### Step 5: 프로덕션 검증
배포 완료 후:
- 라이브 URL 접속 확인
- 주요 페이지 로드 테스트
- OG 태그 미리보기 확인 (가능 시)

$ARGUMENTS - 배포 대상 (예: "preview만", "production", "vercel")
