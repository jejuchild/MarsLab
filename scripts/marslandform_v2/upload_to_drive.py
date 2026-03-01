#!/usr/bin/env python3
"""Upload mars_tiles.tar.gz and colab notebook to Google Drive via OAuth.

Usage:
    python upload_to_drive.py --client-id YOUR_ID --client-secret YOUR_SECRET

Or set environment variables:
    export GOOGLE_CLIENT_ID=...
    export GOOGLE_CLIENT_SECRET=...
    python upload_to_drive.py
"""
import argparse
import os
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path("/disk1/cspark/MarsLab")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

FILES_TO_UPLOAD = [
    ROOT / "Data/HiRISE/v2_output/tiles/mars_tiles.tar.gz",
    ROOT / "scripts/marslandform_v2/colab_ssl_training.ipynb",
]


def authenticate(client_id: str, client_secret: str) -> Credentials:
    """OAuth2 flow — opens browser or prints URL for auth code."""
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    # Try local server first, fall back to console
    try:
        creds = flow.run_local_server(port=0, open_browser=False)
    except Exception:
        creds = flow.run_console()
    return creds


def upload_file(service, file_path: Path, folder_id: str = None) -> str:
    """Upload a file to Google Drive root (or specified folder). Returns file ID."""
    metadata = {"name": file_path.name}
    if folder_id:
        metadata["parents"] = [folder_id]

    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Uploading {file_path.name} ({size_mb:.1f} MB)...")

    media = MediaFileUpload(
        str(file_path),
        resumable=True,
        chunksize=50 * 1024 * 1024,  # 50MB chunks
    )
    request = service.files().create(body=metadata, media_body=media, fields="id,name,webViewLink")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  {pct}% uploaded...", end="\r")

    file_id = response.get("id")
    link = response.get("webViewLink", "")
    print(f"  ✓ {file_path.name} uploaded! ID: {file_id}")
    if link:
        print(f"    Link: {link}")
    return file_id


def main():
    parser = argparse.ArgumentParser(description="Upload files to Google Drive for Colab SSL training")
    parser.add_argument("--client-id", default=os.environ.get("GOOGLE_CLIENT_ID"), help="OAuth client ID")
    parser.add_argument("--client-secret", default=os.environ.get("GOOGLE_CLIENT_SECRET"), help="OAuth client secret")
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        print("ERROR: Provide --client-id and --client-secret (or set GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET)")
        sys.exit(1)

    # Check files exist
    for f in FILES_TO_UPLOAD:
        if not f.exists():
            print(f"ERROR: File not found: {f}")
            sys.exit(1)
        print(f"Found: {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

    # Authenticate
    print("\n--- Authenticating with Google Drive ---")
    creds = authenticate(args.client_id, args.client_secret)
    service = build("drive", "v3", credentials=creds)
    print("✓ Authenticated!\n")

    # Upload files
    for f in FILES_TO_UPLOAD:
        upload_file(service, f)

    print("\n" + "=" * 60)
    print("All files uploaded!")
    print("Next steps:")
    print("1. Open colab_ssl_training.ipynb in Google Colab")
    print("2. Select GPU runtime: Runtime → Change runtime type → T4 GPU")
    print("3. Run all cells (Runtime → Run all)")
    print("4. Training takes ~2-3 hours on T4 GPU")
    print("5. LoRA weights will be saved to Drive at: marslandform_lora_weights/")
    print("=" * 60)


if __name__ == "__main__":
    main()
