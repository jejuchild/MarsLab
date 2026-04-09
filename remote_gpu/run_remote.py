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
║  Google Colab SSH Setup                                  ║
╚══════════════════════════════════════════════════════════╝

1. Open Google Colab (https://colab.research.google.com)
2. Runtime → Change runtime type → GPU (T4)
3. Run this cell in Colab:

────────────────────────────────────────────
# Cell 1: Install SSH + expose via cloudflared
!pip install colab-ssh -q
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="marsrefsr2026")
────────────────────────────────────────────

4. Colab will print a connection command like:
   ssh -o ProxyCommand="cloudflared access ssh --hostname %h" root@XXXX.trycloudflare.com

5. From THIS server, run:
   python remote_gpu/run_remote.py colab connect HOSTNAME

Alternative (ngrok):
────────────────────────────────────────────
# Cell 1: SSH via ngrok
!pip install colab-ssh -q
from colab_ssh import launch_ssh
launch_ssh("YOUR_NGROK_TOKEN", password="marsrefsr2026")
────────────────────────────────────────────

6. Get ngrok token from: https://dashboard.ngrok.com/get-started/your-authtoken
""")


def colab_connect(hostname, port=None):
    """SSH into Colab instance."""
    if port:
        cmd = f"ssh -p {port} root@{hostname}"
    else:
        # cloudflared proxy
        cmd = f'ssh -o ProxyCommand="cloudflared access ssh --hostname %h" root@{hostname}'

    print(f"Connecting: {cmd}")
    print("Password: marsrefsr2026")
    print()
    subprocess.run(cmd, shell=True)


def colab_sync_data(hostname):
    """Sync MarsOrtho data to Colab instance."""
    data_dir = BASE_DIR.parent / "coregister_data" / "output"
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return

    print(f"Syncing data to Colab: {data_dir} → remote:~/marsrefsr/data/")
    cmd = (
        f'rsync -avz --progress -e \'ssh -o ProxyCommand="cloudflared access ssh --hostname %h"\' '
        f'{data_dir}/ root@{hostname}:~/marsrefsr/data/'
    )
    subprocess.run(cmd, shell=True)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python remote_gpu/run_remote.py kaggle push|status|pull")
        print("  python remote_gpu/run_remote.py colab setup|connect|sync")
        return

    platform = sys.argv[1]
    action = sys.argv[2]

    if platform == "kaggle":
        if action == "push":
            kaggle_push()
        elif action == "status":
            kaggle_status()
        elif action == "pull":
            kaggle_pull()
        else:
            print(f"Unknown action: {action}")

    elif platform == "colab":
        if action == "setup":
            colab_setup()
        elif action == "connect":
            host = sys.argv[3] if len(sys.argv) > 3 else None
            port = sys.argv[4] if len(sys.argv) > 4 else None
            if not host:
                print("Usage: python run_remote.py colab connect HOSTNAME [PORT]")
                return
            colab_connect(host, port)
        elif action == "sync":
            host = sys.argv[3] if len(sys.argv) > 3 else None
            if not host:
                print("Usage: python run_remote.py colab sync HOSTNAME")
                return
            colab_sync_data(host)
        else:
            print(f"Unknown action: {action}")
    else:
        print(f"Unknown platform: {platform}")


if __name__ == "__main__":
    main()
