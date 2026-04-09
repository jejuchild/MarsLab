#!/usr/bin/env python3
"""Print the Colab cell content to copy-paste."""

COLAB_CELL = '''
#@title 🔌 SSH 서버 시작 (이 셀만 실행하면 됨)
#@markdown 실행 후 출력되는 SSH 명령어를 서버에서 실행하세요.

# 1. cloudflared 설치 + SSH 서버 시작
!pip install colab-ssh -q 2>/dev/null

# 방법 A: cloudflared (토큰 불필요, 추천)
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="marsrefsr2026")

# 출력 예시:
# ssh -o ProxyCommand="cloudflared access ssh --hostname %h" root@XXXX.trycloudflare.com
# 이 명령어를 서버 터미널에 붙여넣으면 됩니다!
'''

COLAB_CELL_NGROK = '''
#@title 🔌 SSH 서버 시작 (ngrok 방식)
#@markdown ngrok 토큰: https://dashboard.ngrok.com/get-started/your-authtoken

NGROK_TOKEN = ""  #@param {type:"string"}

!pip install colab-ssh -q 2>/dev/null
from colab_ssh import launch_ssh
launch_ssh(NGROK_TOKEN, password="marsrefsr2026")
'''

print("=" * 60)
print("아래 내용을 Colab 노트북에 복사해서 실행하세요")
print("=" * 60)
print(COLAB_CELL)
print("=" * 60)
print()
print("SSH 접속 후 이 서버에서:")
print("  1. ssh -o ProxyCommand=... root@XXXX.trycloudflare.com")
print("  2. Password: marsrefsr2026")
print("  3. GPU 확인: nvidia-smi")
print("  4. 작업 시작!")
