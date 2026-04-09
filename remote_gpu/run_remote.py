#!/usr/bin/env python3
"""Remote GPU runner — submit experiments to Kaggle or Colab from local server.

Usage:
    # Kaggle (batch job)
    python remote_gpu/run_remote.py kaggle push          # submit to Kaggle GPU
    python remote_gpu/run_remote.py kaggle status         # check status
    python remote_gpu/run_remote.py kaggle pull            # download results

    # Colab (interactive SSH)
    python remote_gpu/run_remote.py colab setup           # print Colab notebook setup
    python remote_gpu/run_remote.py colab connect HOST PORT  # SSH into Colab
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
KAGGLE_DIR = BASE_DIR / "kaggle"
RESULTS_DIR = BASE_DIR.parent / "results"


def kaggle_push():
    """Submit training script to Kaggle GPU."""
    meta_path = KAGGLE_DIR / "kernel-metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    if "INSERT_YOUR_KAGGLE_USERNAME" in meta["id"]:
        # Try to get username from kaggle config
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            with open(kaggle_json) as f:
                creds = json.load(f)
            username = creds.get("username", "")
            if username:
                meta["id"] = f"{username}/marsrefsr-experiment"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"  Updated kernel ID: {meta['id']}")
        else:
            print("ERROR: Kaggle not configured.")
            print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
            print("  2. Save kaggle.json to ~/.kaggle/kaggle.json")
            print("  3. chmod 600 ~/.kaggle/kaggle.json")
            return

    print(f"Pushing to Kaggle: {meta['id']}")
    print(f"  GPU: {meta.get('enable_gpu', False)}")
    print(f"  Script: {meta['code_file']}")

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    else:
        print("\nSubmitted! Check status with:")
        print(f"  python remote_gpu/run_remote.py kaggle status")


def kaggle_status():
    """Check kernel execution status."""
    meta_path = KAGGLE_DIR / "kernel-metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    result = subprocess.run(
        ["kaggle", "kernels", "status", meta["id"]],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")


def kaggle_pull():
    """Download results from completed kernel."""
    meta_path = KAGGLE_DIR / "kernel-metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", meta["id"], "-p", str(RESULTS_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    else:
        # Check for results.json
        results_file = RESULTS_DIR / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                res = json.load(f)
            print(f"\n{'='*50}")
            print(f"Experiment: {res.get('experiment', '?')}")
            print(f"Model: {res.get('model', '?')}")
            print(f"GPU: {res.get('gpu', '?')}")
            print(f"Best PSNR: {res.get('best_psnr', '?'):.2f}")
            print(f"Time: {res.get('training_time_min', '?')} min")
            print(f"{'='*50}")
        print(f"\nResults saved to: {RESULTS_DIR}")


def colab_setup():
    """Print Colab notebook cell to enable SSH access."""
    print("""
╔══════════════════════════════════════════════════════════╗
║  Google Colab SSH 세팅                                   ║
╚══════════════════════════════════════════════════════════╝

[Step 1] 폰이나 노트북에서 Colab 열기
  https://colab.research.google.com

[Step 2] Runtime → Change runtime type → GPU

[Step 3] 이 코드 1개만 셀에 붙여넣고 실행:
────────────────────────────────────────────
!pip install colab-ssh -q
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="marsrefsr2026")
────────────────────────────────────────────

[Step 4] 출력에서 호스트명 복사 (예: abc-xyz.trycloudflare.com)

[Step 5] 이 서버에서 실행:
  python remote_gpu/run_remote.py colab connect abc-xyz.trycloudflare.com

접속 후 자동 세팅:
  python remote_gpu/run_remote.py colab init abc-xyz.trycloudflare.com

데이터 전송:
  python remote_gpu/run_remote.py colab sync abc-xyz.trycloudflare.com

학습 실행 (서버에서 원격 실행):
  python remote_gpu/run_remote.py colab run abc-xyz.trycloudflare.com
""")


def _ssh_cmd(hostname, remote_cmd=None):
    """Build SSH command for cloudflared proxy."""
    base = f'ssh -o ProxyCommand="cloudflared access ssh --hostname %h" -o StrictHostKeyChecking=no root@{hostname}'
    if remote_cmd:
        return f"{base} '{remote_cmd}'"
    return base


def _scp_cmd(hostname, local_path, remote_path):
    """Build SCP command for cloudflared proxy."""
    return (
        f'scp -o ProxyCommand="cloudflared access ssh --hostname %h" '
        f'-o StrictHostKeyChecking=no {local_path} root@{hostname}:{remote_path}'
    )


def colab_connect(hostname, port=None):
    """SSH into Colab instance."""
    cmd = _ssh_cmd(hostname)
    print(f"Connecting to Colab GPU...")
    print(f"Password: marsrefsr2026")
    print()
    subprocess.run(cmd, shell=True)


def colab_init(hostname):
    """Initialize Colab environment (install packages, check GPU)."""
    init_script = BASE_DIR / "colab" / "colab_init.sh"
    print("Colab 환경 초기화 중...")
    cmd = f'{_ssh_cmd(hostname)} "bash -s" < {init_script}'
    subprocess.run(cmd, shell=True)


def colab_sync_data(hostname):
    """Sync training script + data to Colab."""
    # 1. Send training script
    train_script = BASE_DIR / "kaggle" / "train.py"
    print(f"학습 스크립트 전송: train.py")
    subprocess.run(_scp_cmd(hostname, str(train_script), "~/marsrefsr/train.py"), shell=True)

    # 2. Send data (ortho patches)
    data_dir = BASE_DIR.parent / "coregister_data" / "output"
    if not data_dir.exists():
        print(f"  데이터 디렉토리 없음: {data_dir}")
        print(f"  학습 스크립트만 전송됨 (합성 데이터로 테스트 가능)")
        return

    print(f"데이터 동기화: {data_dir} → remote:~/marsrefsr/data/")
    cmd = (
        f'rsync -avz --progress '
        f'-e \'ssh -o ProxyCommand="cloudflared access ssh --hostname %h" -o StrictHostKeyChecking=no\' '
        f'{data_dir}/ root@{hostname}:~/marsrefsr/data/'
    )
    subprocess.run(cmd, shell=True)


def colab_run(hostname):
    """Run training on Colab GPU remotely."""
    print("Colab GPU에서 학습 실행 중...")
    cmd = _ssh_cmd(hostname, "cd ~/marsrefsr && nohup python train.py > train.log 2>&1 & echo $!")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    pid = result.stdout.strip()
    print(f"  PID: {pid}")
    print(f"  로그 확인: python remote_gpu/run_remote.py colab log {hostname}")
    print(f"  결과 다운로드: python remote_gpu/run_remote.py colab pull {hostname}")


def colab_log(hostname):
    """Tail training log from Colab."""
    cmd = _ssh_cmd(hostname, "tail -30 ~/marsrefsr/train.log 2>/dev/null || echo 'No log yet'")
    subprocess.run(cmd, shell=True)


def colab_pull_results(hostname):
    """Download results from Colab."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"결과 다운로드 → {RESULTS_DIR}")
    cmd = (
        f'scp -o ProxyCommand="cloudflared access ssh --hostname %h" '
        f'-o StrictHostKeyChecking=no '
        f'root@{hostname}:~/marsrefsr/results.json {RESULTS_DIR}/ 2>/dev/null; '
        f'scp -o ProxyCommand="cloudflared access ssh --hostname %h" '
        f'-o StrictHostKeyChecking=no '
        f'root@{hostname}:~/marsrefsr/best_model.pth {RESULTS_DIR}/ 2>/dev/null'
    )
    subprocess.run(cmd, shell=True)

    results_file = RESULTS_DIR / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            res = json.load(f)
        print(f"\n{'='*50}")
        print(f"Experiment: {res.get('experiment', '?')}")
        print(f"Best PSNR: {res.get('best_psnr', 0):.2f}")
        print(f"Time: {res.get('training_time_min', '?')} min")
        print(f"{'='*50}")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python remote_gpu/run_remote.py kaggle push|status|pull")
        print("  python remote_gpu/run_remote.py colab setup                    # 세팅 안내")
        print("  python remote_gpu/run_remote.py colab connect HOST             # SSH 접속")
        print("  python remote_gpu/run_remote.py colab init HOST                # 환경 초기화")
        print("  python remote_gpu/run_remote.py colab sync HOST                # 데이터 전송")
        print("  python remote_gpu/run_remote.py colab run HOST                 # 학습 시작")
        print("  python remote_gpu/run_remote.py colab log HOST                 # 로그 확인")
        print("  python remote_gpu/run_remote.py colab pull HOST                # 결과 다운로드")
        return

    platform = sys.argv[1]
    action = sys.argv[2]
    host = sys.argv[3] if len(sys.argv) > 3 else None

    if platform == "kaggle":
        {"push": kaggle_push, "status": kaggle_status, "pull": kaggle_pull}.get(
            action, lambda: print(f"Unknown: {action}"))()

    elif platform == "colab":
        if action == "setup":
            colab_setup()
        elif host is None and action != "setup":
            print(f"Usage: python run_remote.py colab {action} HOSTNAME")
        elif action == "connect":
            colab_connect(host)
        elif action == "init":
            colab_init(host)
        elif action == "sync":
            colab_sync_data(host)
        elif action == "run":
            colab_run(host)
        elif action == "log":
            colab_log(host)
        elif action == "pull":
            colab_pull_results(host)
        else:
            print(f"Unknown: {action}")
    else:
        print(f"Unknown platform: {platform}")


if __name__ == "__main__":
    main()
