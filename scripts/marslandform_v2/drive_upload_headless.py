#!/usr/bin/env python3
"""Headless Google Drive upload — two-phase OAuth for servers without browsers.

Phase 1 (--gen-url):  Prints auth URL for user to visit in browser.
Phase 2 (--code CODE): Exchanges auth code for token, uploads files.

Usage:
    # Step 1: Get the auth URL
    python drive_upload_headless.py --gen-url \
        --client-id YOUR_ID --client-secret YOUR_SECRET

    # Step 2: After visiting URL and copying code from redirect URL
    python drive_upload_headless.py --code "4/0AX4X..." \
        --client-id YOUR_ID --client-secret YOUR_SECRET
"""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import requests as req

ROOT = Path("/disk1/cspark/MarsLab")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
REDIRECT_URI = "http://localhost:1"  # Won't load, but code appears in URL bar

FILES_TO_UPLOAD = [
    ROOT / "Data/HiRISE/v2_output/tiles/mars_tiles.tar.gz",
    ROOT / "scripts/marslandform_v2/colab_ssl_training.ipynb",
]

TOKEN_CACHE = ROOT / "scripts/marslandform_v2/.drive_token.json"


def generate_auth_url(client_id: str) -> str:
    """Generate OAuth2 authorization URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def exchange_code(code: str, client_id: str, client_secret: str) -> dict:
    """Exchange authorization code for access token."""
    resp = req.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    })
    if resp.status_code != 200:
        print(f"Token exchange failed: {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    token_data = resp.json()
    # Cache token
    TOKEN_CACHE.write_text(json.dumps(token_data))
    print("✓ Token obtained and cached!")
    return token_data


def upload_file_resumable(access_token: str, file_path: Path, folder_id: str = None) -> dict:
    """Upload file to Google Drive using resumable upload."""
    size = file_path.stat().st_size
    size_mb = size / (1024 * 1024)
    print(f"\nUploading {file_path.name} ({size_mb:.1f} MB)...")

    # Step 1: Initiate resumable upload
    metadata = {"name": file_path.name}
    if folder_id:
        metadata["parents"] = [folder_id]

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(size),
    }
    resp = req.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers=headers,
        json=metadata,
    )
    if resp.status_code not in (200, 308):
        print(f"  Failed to initiate upload: {resp.status_code} {resp.text}")
        sys.exit(1)

    upload_url = resp.headers["Location"]

    # Step 2: Upload in chunks
    chunk_size = 50 * 1024 * 1024  # 50MB
    uploaded = 0

    with open(file_path, "rb") as f:
        while uploaded < size:
            chunk = f.read(chunk_size)
            end = min(uploaded + len(chunk), size)
            content_range = f"bytes {uploaded}-{end - 1}/{size}"

            resp = req.put(
                upload_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Range": content_range,
                },
                data=chunk,
            )

            if resp.status_code == 200 or resp.status_code == 201:
                # Upload complete
                result = resp.json()
                print(f"  ✓ {file_path.name} uploaded! ID: {result.get('id')}")
                return result
            elif resp.status_code == 308:
                # Chunk accepted, continue
                uploaded = end
                pct = int(uploaded / size * 100)
                print(f"  {pct}% uploaded ({uploaded / (1024*1024):.0f} / {size_mb:.0f} MB)", end="\r")
            else:
                print(f"\n  Upload failed at {uploaded}: {resp.status_code} {resp.text}")
                sys.exit(1)

    return {}


def extract_code_from_url(url_or_code: str) -> str:
    """Extract auth code from redirect URL or bare code."""
    if url_or_code.startswith("http"):
        parsed = urlparse(url_or_code)
        params = parse_qs(parsed.query)
        if "code" in params:
            return params["code"][0]
        print(f"ERROR: No 'code' parameter found in URL: {url_or_code}")
        sys.exit(1)
    return url_or_code  # Assume it's a bare code


def main():
    parser = argparse.ArgumentParser(description="Headless Google Drive upload")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--gen-url", action="store_true", help="Phase 1: Generate auth URL")
    parser.add_argument("--code", type=str, help="Phase 2: Auth code or redirect URL")
    args = parser.parse_args()

    if args.gen_url:
        url = generate_auth_url(args.client_id)
        print("=" * 70)
        print("STEP 1: Visit this URL in your browser and authorize:")
        print()
        print(url)
        print()
        print("STEP 2: After authorizing, your browser will try to redirect to")
        print("http://localhost:1/... which will FAIL TO LOAD. That's OK!")
        print()
        print("STEP 3: Copy the FULL URL from your browser's address bar.")
        print("It looks like: http://localhost:1/?code=4/0AXXXX...&scope=...")
        print()
        print("STEP 4: Paste that URL (or just the code) back here.")
        print("=" * 70)
        return

    if args.code:
        code = extract_code_from_url(args.code)
        print(f"Auth code: {code[:20]}...")

        # Check files exist
        for f in FILES_TO_UPLOAD:
            if not f.exists():
                print(f"ERROR: File not found: {f}")
                sys.exit(1)
            print(f"Found: {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

        # Exchange code for token
        token_data = exchange_code(code, args.client_id, args.client_secret)
        access_token = token_data["access_token"]

        # Upload files
        for f in FILES_TO_UPLOAD:
            upload_file_resumable(access_token, f)

        print("\n" + "=" * 60)
        print("✓ All files uploaded to Google Drive!")
        print("Next steps:")
        print("1. Open colab_ssl_training.ipynb in Google Colab")
        print("2. Select GPU runtime: Runtime → Change runtime type → T4 GPU")
        print("3. Run all cells (Runtime → Run all)")
        print("4. Training takes ~2-3 hours on T4 GPU")
        print("5. LoRA weights saved to Drive at: marslandform_lora_weights/")
        print("=" * 60)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
