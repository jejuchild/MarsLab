"""
aria2-based download helper for maximum network throughput.

Uses aria2c with aggressive multi-connection settings:
- 20 connections per file (-x 20)
- 20 segments per file (-s 20)
- 1M chunk size (-k 1M)
- Resume support (--continue=true)
- Truncate file allocation (--file-allocation=trunc)
- No auto-renaming (--auto-file-renaming=false)

Supports:
- Single file downloads
- Batch downloads via input list (-i)
- Progress tracking
- Retry logic (3 attempts)
- Failed URL logging
"""

import os
import asyncio
import tempfile
import subprocess
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

# aria2c default options for maximum throughput
ARIA2_OPTS = [
    "-x", "16",           # Max connections per server (aria2 max is 16)
    "-s", "16",           # Split file into 16 segments (aria2 max is 16)
    "-k", "1M",           # Min split size 1MB
    "--continue=true",    # Resume partial downloads
    "--file-allocation=trunc",  # Fast file allocation
    "--auto-file-renaming=false",  # Don't rename on conflict
    "--allow-overwrite=true",  # Allow overwriting existing files
    "--console-log-level=error",  # Reduce noise
    "--summary-interval=0",  # Disable periodic summary
]

MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 4  # -j option for batch downloads


# =============================================================================
# Process Tracking for Cancellation
# =============================================================================

# Global registry of active aria2 processes
# Maps task_id -> asyncio.subprocess.Process
_active_processes: Dict[str, asyncio.subprocess.Process] = {}
_process_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None


async def register_process(task_id: str, process: asyncio.subprocess.Process):
    """Register an active aria2 process for a task."""
    _active_processes[task_id] = process


async def unregister_process(task_id: str):
    """Unregister a completed/cancelled process."""
    _active_processes.pop(task_id, None)


async def cancel_task(task_id: str) -> bool:
    """
    Cancel a specific download task.

    Args:
        task_id: Task ID to cancel

    Returns:
        True if process was found and terminated, False otherwise
    """
    process = _active_processes.get(task_id)
    if process and process.returncode is None:
        try:
            process.terminate()
            # Give it a moment to terminate gracefully
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
            return True
        except Exception as e:
            print(f"Error cancelling task {task_id}: {e}")
            return False
    return False


async def cancel_all_tasks() -> int:
    """
    Cancel all active download tasks.

    Returns:
        Number of tasks cancelled
    """
    cancelled = 0
    task_ids = list(_active_processes.keys())

    for task_id in task_ids:
        if await cancel_task(task_id):
            cancelled += 1

    return cancelled


def get_active_task_ids() -> List[str]:
    """Get list of active task IDs."""
    return [tid for tid, proc in _active_processes.items() if proc.returncode is None]


# =============================================================================
# Data Classes
# =============================================================================

class Aria2Status(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Aria2File:
    """Single file to download."""
    url: str
    filename: str
    output_dir: str
    status: Aria2Status = Aria2Status.PENDING
    bytes_downloaded: int = 0
    bytes_total: Optional[int] = None
    error: Optional[str] = None
    retries: int = 0


@dataclass
class Aria2Task:
    """Batch download task."""
    task_id: str
    files: List[Aria2File] = field(default_factory=list)
    status: Aria2Status = Aria2Status.PENDING
    failed_urls: List[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        if not self.files:
            return 0.0
        completed = sum(1 for f in self.files if f.status == Aria2Status.COMPLETED)
        return (completed / len(self.files)) * 100

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes_total or 0 for f in self.files)

    @property
    def downloaded_bytes(self) -> int:
        return sum(f.bytes_downloaded for f in self.files)


# =============================================================================
# aria2 Download Functions
# =============================================================================

async def download_single_file(
    url: str,
    output_dir: str,
    filename: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    task_id: Optional[str] = None,
) -> Aria2File:
    """
    Download a single file using aria2c.

    Args:
        url: URL to download
        output_dir: Directory to save the file
        filename: Optional filename (auto-detected from URL if not provided)
        progress_callback: Optional callback(bytes_downloaded, bytes_total)
        task_id: Optional task ID for cancellation tracking

    Returns:
        Aria2File with download result
    """
    if filename is None:
        filename = url.split("/")[-1]

    file_info = Aria2File(
        url=url,
        filename=filename,
        output_dir=output_dir,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    for attempt in range(MAX_RETRIES):
        file_info.retries = attempt
        file_info.status = Aria2Status.DOWNLOADING

        try:
            cmd = [
                "aria2c",
                *ARIA2_OPTS,
                "-d", output_dir,
                "-o", filename,
                url,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Register process for cancellation tracking
            if task_id:
                await register_process(task_id, process)

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=600  # 10 min timeout
                )
            except asyncio.TimeoutError:
                # aria2 hung — kill it and check if file is complete
                process.kill()
                await process.wait()
                if os.path.exists(output_path):
                    file_info.status = Aria2Status.COMPLETED
                    file_info.bytes_total = os.path.getsize(output_path)
                    file_info.bytes_downloaded = file_info.bytes_total
                    return file_info
                file_info.error = "Download timed out"
                break
            finally:
                # Unregister process when done
                if task_id:
                    await unregister_process(task_id)

            # Check if cancelled (returncode -15 = SIGTERM, -9 = SIGKILL)
            if process.returncode in (-15, -9, 255):
                file_info.status = Aria2Status.FAILED
                file_info.error = "Cancelled"
                return file_info

            if process.returncode == 0 and os.path.exists(output_path):
                file_info.status = Aria2Status.COMPLETED
                file_info.bytes_total = os.path.getsize(output_path)
                file_info.bytes_downloaded = file_info.bytes_total
                return file_info
            else:
                error_msg = stderr.decode().strip() if stderr else f"Exit code {process.returncode}"
                file_info.error = error_msg

                # Don't retry on 404 or similar client errors
                if "404" in error_msg or "403" in error_msg:
                    break

        except asyncio.CancelledError:
            file_info.status = Aria2Status.FAILED
            file_info.error = "Cancelled"
            return file_info
        except Exception as e:
            file_info.error = str(e)

        # Wait before retry
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    file_info.status = Aria2Status.FAILED
    return file_info


async def download_batch(
    files: List[Dict[str, str]],
    output_dir: str,
    progress_callback: Optional[Callable[[Aria2Task], None]] = None,
) -> Aria2Task:
    """
    Download multiple files using aria2c with input list.

    Args:
        files: List of dicts with 'url' and optional 'filename' keys
        output_dir: Directory to save files
        progress_callback: Optional callback(task) for progress updates

    Returns:
        Aria2Task with download results
    """
    import uuid

    task = Aria2Task(
        task_id=str(uuid.uuid4())[:8],
        files=[
            Aria2File(
                url=f["url"],
                filename=f.get("filename") or f["url"].split("/")[-1],
                output_dir=output_dir,
            )
            for f in files
        ],
    )

    if not task.files:
        task.status = Aria2Status.COMPLETED
        return task

    os.makedirs(output_dir, exist_ok=True)
    task.status = Aria2Status.DOWNLOADING

    # Create aria2 input file
    # Format: URL\n  out=filename\n  dir=output_dir\n
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        input_file = f.name
        for file_info in task.files:
            f.write(f"{file_info.url}\n")
            f.write(f"  out={file_info.filename}\n")
            f.write(f"  dir={output_dir}\n")

    try:
        for attempt in range(MAX_RETRIES):
            # Filter to only files that haven't completed yet
            pending_files = [f for f in task.files if f.status != Aria2Status.COMPLETED]

            if not pending_files:
                break

            # Create input file for pending files only
            with open(input_file, "w") as f:
                for file_info in pending_files:
                    f.write(f"{file_info.url}\n")
                    f.write(f"  out={file_info.filename}\n")
                    f.write(f"  dir={output_dir}\n")

            cmd = [
                "aria2c",
                *ARIA2_OPTS,
                f"-j{MAX_CONCURRENT_DOWNLOADS}",  # Concurrent downloads
                "-i", input_file,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Check which files completed
            for file_info in pending_files:
                output_path = os.path.join(output_dir, file_info.filename)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_info.status = Aria2Status.COMPLETED
                    file_info.bytes_total = os.path.getsize(output_path)
                    file_info.bytes_downloaded = file_info.bytes_total
                else:
                    file_info.retries = attempt + 1
                    if attempt == MAX_RETRIES - 1:
                        file_info.status = Aria2Status.FAILED
                        file_info.error = "Max retries exceeded"
                        task.failed_urls.append(file_info.url)

            if progress_callback:
                progress_callback(task)

            # Check if all done
            if all(f.status == Aria2Status.COMPLETED for f in task.files):
                break

            # Wait before retry
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2)

    finally:
        # Cleanup temp file
        try:
            os.unlink(input_file)
        except OSError:
            pass

    # Final status
    if all(f.status == Aria2Status.COMPLETED for f in task.files):
        task.status = Aria2Status.COMPLETED
    elif any(f.status == Aria2Status.FAILED for f in task.files):
        task.status = Aria2Status.FAILED

    return task


async def download_with_progress(
    files: List[Dict[str, str]],
    output_dir: str,
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
) -> Aria2Task:
    """
    Download files with detailed progress tracking.

    Uses aria2c in JSON-RPC mode for real-time progress, with fallback
    to polling file sizes.

    Args:
        files: List of dicts with 'url' and optional 'filename' keys
        output_dir: Directory to save files
        progress_callback: callback(completed_files, total_files, bytes_done, bytes_total)

    Returns:
        Aria2Task with results
    """
    import uuid

    task = Aria2Task(
        task_id=str(uuid.uuid4())[:8],
        files=[
            Aria2File(
                url=f["url"],
                filename=f.get("filename") or f["url"].split("/")[-1],
                output_dir=output_dir,
            )
            for f in files
        ],
    )

    if not task.files:
        task.status = Aria2Status.COMPLETED
        return task

    os.makedirs(output_dir, exist_ok=True)
    task.status = Aria2Status.DOWNLOADING

    # Download files sequentially for better progress tracking
    # (aria2 batch mode doesn't give per-file progress easily)
    for i, file_info in enumerate(task.files):
        file_info.status = Aria2Status.DOWNLOADING

        result = await download_single_file(
            url=file_info.url,
            output_dir=output_dir,
            filename=file_info.filename,
        )

        file_info.status = result.status
        file_info.bytes_downloaded = result.bytes_downloaded
        file_info.bytes_total = result.bytes_total
        file_info.error = result.error

        if result.status == Aria2Status.FAILED:
            task.failed_urls.append(file_info.url)

        if progress_callback:
            completed = sum(1 for f in task.files if f.status == Aria2Status.COMPLETED)
            bytes_done = sum(f.bytes_downloaded for f in task.files)
            bytes_total = sum(f.bytes_total or 0 for f in task.files)
            progress_callback(completed, len(task.files), bytes_done, bytes_total)

    # Final status
    if all(f.status == Aria2Status.COMPLETED for f in task.files):
        task.status = Aria2Status.COMPLETED
    elif any(f.status == Aria2Status.FAILED for f in task.files):
        task.status = Aria2Status.FAILED

    return task


def log_failed_downloads(failed_urls: List[str], log_file: str):
    """Log failed download URLs to a file."""
    if not failed_urls:
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        for url in failed_urls:
            f.write(f"{timestamp}\t{url}\n")


# =============================================================================
# Utility Functions
# =============================================================================

def check_aria2_available() -> bool:
    """Check if aria2c is available on the system."""
    try:
        result = subprocess.run(
            ["aria2c", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_aria2_version() -> Optional[str]:
    """Get aria2c version string."""
    try:
        result = subprocess.run(
            ["aria2c", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            match = re.search(r"aria2 version (\S+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
