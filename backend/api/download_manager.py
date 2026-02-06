"""
Download Manager for Mars data products.

Handles:
1. Async file downloading with progress tracking
2. CRISM bundle downloads (MTR3/TRR3, wavelength table, browse images)
3. HiRISE bundle downloads (RED JP2 + LBL)
4. GDAL conversion for HiRISE JP2 -> GeoTIFF
5. index.geojson updates
"""

import os
import asyncio
import aiohttp
import aiofiles
import json
import subprocess
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .ode_client import (
    Instrument,
    ODEFile,
    CRISMBundle,
    HiRISEBundle,
    SHARADBundle,
    SHARADHighResBundle,
    parse_crism_base_key,
    resolve_crism_bundle,
    resolve_hirise_bundle,
    resolve_sharad_bundle,
    resolve_sharad_highres_bundle,
    get_sharad_highres_footprint,
)
from .aria2_downloader import (
    download_single_file,
    download_with_progress,
    check_aria2_available,
    log_failed_downloads,
    Aria2Status,
    cancel_task as aria2_cancel_task,
    cancel_all_tasks as aria2_cancel_all,
    get_active_task_ids,
)


# =============================================================================
# Constants & Config
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
CRISM_DATA_DIR = BASE_DIR / "crism_data"
HIRISE_DATA_DIR = BASE_DIR / "hirise_data"
SHARAD_DATA_DIR = BASE_DIR / "sharad_data"
SHARAD_HIGHRES_DIR = BASE_DIR / "sharad_highres"  # Note: sharad_highres not sharad_highres_data
CRISM_QUICKVIEW_DIR = BASE_DIR / "crism_quickview"
CRISM_BROWSE_DIR = BASE_DIR / "crism_browse"
HIRISE_QUICKVIEW_DIR = BASE_DIR / "hirise_quickview"
SHARAD_QUICKVIEW_DIR = BASE_DIR / "sharad_quickview"

# Failed download log
FAILED_DOWNLOADS_LOG = BASE_DIR / "logs" / "failed_downloads.log"

# Check aria2 availability at module load
_ARIA2_AVAILABLE = check_aria2_available()
if _ARIA2_AVAILABLE:
    print("aria2 available - using accelerated downloads")
else:
    print("aria2 not found - falling back to aiohttp downloads")

# Ensure directories exist
for d in [CRISM_DATA_DIR, HIRISE_DATA_DIR, SHARAD_DATA_DIR, SHARAD_HIGHRES_DIR,
          CRISM_QUICKVIEW_DIR, CRISM_BROWSE_DIR, HIRISE_QUICKVIEW_DIR, SHARAD_QUICKVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Classes
# =============================================================================

class DownloadStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"  # GDAL conversion, etc.
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileProgress:
    """Progress for a single file download."""
    filename: str
    url: str
    status: DownloadStatus = DownloadStatus.PENDING
    bytes_downloaded: int = 0
    bytes_total: Optional[int] = None
    error: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.bytes_total and self.bytes_total > 0:
            return min(100.0, (self.bytes_downloaded / self.bytes_total) * 100)
        return 0.0


@dataclass
class DownloadTask:
    """Overall download task for a product."""
    task_id: str
    product_id: str
    base_key: str  # For CRISM this is the observation key; for HiRISE same as product_id
    instrument: Instrument
    status: DownloadStatus = DownloadStatus.PENDING
    files: List[FileProgress] = field(default_factory=list)
    target_dir: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    # Coordinates from ODE (for index.geojson)
    lat: Optional[float] = None
    lon: Optional[float] = None

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes_total or 0 for f in self.files)

    @property
    def downloaded_bytes(self) -> int:
        return sum(f.bytes_downloaded for f in self.files)

    @property
    def progress_percent(self) -> float:
        total = self.total_bytes
        if total > 0:
            return min(100.0, (self.downloaded_bytes / total) * 100)
        # If no size info, use file count
        if self.files:
            completed = sum(1 for f in self.files if f.status == DownloadStatus.COMPLETED)
            return (completed / len(self.files)) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "product_id": self.product_id,
            "base_key": self.base_key,
            "instrument": self.instrument.value,
            "status": self.status.value,
            "files": [
                {
                    "filename": f.filename,
                    "status": f.status.value,
                    "bytes_downloaded": f.bytes_downloaded,
                    "bytes_total": f.bytes_total,
                    "progress_percent": f.progress_percent,
                    "error": f.error,
                }
                for f in self.files
            ],
            "target_dir": self.target_dir,
            "progress_percent": self.progress_percent,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# =============================================================================
# Download Manager Singleton
# =============================================================================

class DownloadManager:
    """
    Manages download tasks with progress tracking.

    This is a singleton that maintains active and completed download tasks.
    Tasks are processed asynchronously and clients can poll for status updates.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.tasks: Dict[str, DownloadTask] = {}
        self._active_downloads: Dict[str, asyncio.Task] = {}
        self._progress_callbacks: Dict[str, Callable] = {}
        self._initialized = True

    def get_task(self, task_id: str) -> Optional[DownloadTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[DownloadTask]:
        """List all tasks."""
        return list(self.tasks.values())

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a specific download task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if task was cancelled, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Cancel the aria2 process if running
        cancelled = await aria2_cancel_task(task_id)

        # Cancel the asyncio task if running
        async_task = self._active_downloads.get(task_id)
        if async_task and not async_task.done():
            async_task.cancel()

        # Update task status
        task.status = DownloadStatus.FAILED
        task.error = "Cancelled by user"

        # Remove from active downloads
        self._active_downloads.pop(task_id, None)

        return True

    async def cancel_all_tasks(self) -> int:
        """
        Cancel all active download tasks.

        Returns:
            Number of tasks cancelled
        """
        cancelled_count = 0

        # Get list of active task IDs
        active_task_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.status in (DownloadStatus.PENDING, DownloadStatus.QUEUED,
                              DownloadStatus.DOWNLOADING, DownloadStatus.PROCESSING)
        ]

        for task_id in active_task_ids:
            if await self.cancel_task(task_id):
                cancelled_count += 1

        # Also cancel any aria2 processes
        await aria2_cancel_all()

        return cancelled_count

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a completed/failed task from history.

        Args:
            task_id: Task ID to delete

        Returns:
            True if task was deleted, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Only allow deleting completed or failed tasks
        if task.status not in (DownloadStatus.COMPLETED, DownloadStatus.FAILED):
            return False

        del self.tasks[task_id]
        return True

    async def start_download(
        self,
        product_id: str,
        instrument: Instrument,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        file_types: Optional[List[str]] = None,
    ) -> DownloadTask:
        """
        Start a download task for a product.

        Args:
            product_id: Product identifier
            instrument: CRISM or HiRISE
            lat: Optional latitude (for index.geojson)
            lon: Optional longitude (for index.geojson)
            file_types: Optional list of file types to download:
                        "core" (.img, .lbl), "header" (.hdr),
                        "wavelength" (.tab), "browse" (PNG files)

        Returns:
            DownloadTask with task_id for tracking
        """
        # Generate task ID
        task_id = str(uuid.uuid4())[:8]

        # Determine base_key and target directory
        if instrument == Instrument.CRISM:
            base_key = parse_crism_base_key(product_id)
            target_dir = str(CRISM_DATA_DIR / base_key)
        elif instrument == Instrument.HIRISE:
            base_key = product_id.upper()
            target_dir = str(HIRISE_DATA_DIR / base_key)
        elif instrument == Instrument.SHARAD:
            # SHARAD USRDRV2: base_key is S_XXXXXXXX
            base_key = product_id.upper().replace("_THM", "")
            target_dir = str(SHARAD_DATA_DIR)  # All SHARAD quicklooks in same dir
        elif instrument == Instrument.SHARAD_HIGHRES:
            # SHARAD RDR: base_key is full product ID
            base_key = product_id.upper()
            target_dir = str(SHARAD_HIGHRES_DIR)  # All RDR files in same dir
        else:
            base_key = product_id.upper()
            target_dir = str(BASE_DIR / "data" / base_key)

        # Create task
        task = DownloadTask(
            task_id=task_id,
            product_id=product_id,
            base_key=base_key,
            instrument=instrument,
            target_dir=target_dir,
            lat=lat,
            lon=lon,
        )
        # Store file_types filter on the task for use during download
        task._file_types = file_types

        self.tasks[task_id] = task

        # Start async download
        async_task = asyncio.create_task(self._execute_download(task))
        self._active_downloads[task_id] = async_task

        return task

    async def _execute_download(self, task: DownloadTask):
        """Execute the download task."""
        try:
            task.status = DownloadStatus.QUEUED

            # Create target directory
            os.makedirs(task.target_dir, exist_ok=True)

            # Resolve bundle files from ODE
            async with aiohttp.ClientSession() as session:
                if task.instrument == Instrument.CRISM:
                    await self._download_crism_bundle(task, session)
                elif task.instrument == Instrument.HIRISE:
                    await self._download_hirise_bundle(task, session)
                elif task.instrument == Instrument.SHARAD:
                    await self._download_sharad_bundle(task, session)
                elif task.instrument == Instrument.SHARAD_HIGHRES:
                    await self._download_sharad_highres_bundle(task, session)
                else:
                    raise ValueError(f"Unsupported instrument: {task.instrument}")

            # Check completion - core files must succeed, browse files are optional
            # Browse files have "_br" in the name (e.g., frt0000a2c2_07_brtruj_mtr3.png)
            core_files = [f for f in task.files if "_br" not in f.filename]
            browse_files = [f for f in task.files if "_br" in f.filename]

            core_completed = all(f.status == DownloadStatus.COMPLETED for f in core_files)
            core_failed = [f for f in core_files if f.status == DownloadStatus.FAILED]

            if core_completed:
                task.status = DownloadStatus.COMPLETED
                task.completed_at = datetime.utcnow().isoformat()

                # Log how many browse files succeeded
                browse_ok = sum(1 for f in browse_files if f.status == DownloadStatus.COMPLETED)
                if browse_files:
                    print(f"Downloaded {browse_ok}/{len(browse_files)} browse files")

                # Update index.geojson
                await self._update_index(task)
            elif core_failed:
                task.status = DownloadStatus.FAILED
                task.error = f"Failed to download core files: {', '.join(f.filename for f in core_failed)}"
            else:
                # Some files still pending/downloading - shouldn't happen
                task.status = DownloadStatus.FAILED
                task.error = "Download incomplete"

        except Exception as e:
            task.status = DownloadStatus.FAILED
            task.error = str(e)

        finally:
            # Remove from active downloads
            self._active_downloads.pop(task.task_id, None)

    async def _download_crism_bundle(self, task: DownloadTask, session: aiohttp.ClientSession):
        """Download CRISM bundle files (skipping existing files)."""
        bundle = await resolve_crism_bundle(task.base_key, session)

        # Get file_types filter (if any)
        file_types = getattr(task, '_file_types', None)

        # Collect all possible files
        all_files: List[ODEFile] = []

        # Core files (img, lbl)
        if bundle.img_file:
            bundle.img_file.file_type = "core"
            all_files.append(bundle.img_file)
        if bundle.lbl_file:
            bundle.lbl_file.file_type = "core"
            all_files.append(bundle.lbl_file)

        # Header file
        if bundle.hdr_file:
            bundle.hdr_file.file_type = "header"
            all_files.append(bundle.hdr_file)

        # Wavelength table
        if bundle.tab_file:
            bundle.tab_file.file_type = "wavelength"
            all_files.append(bundle.tab_file)

        # Browse files
        for bf in bundle.browse_files:
            bf.file_type = "browse"
            all_files.append(bf)

        if not all_files:
            task.status = DownloadStatus.FAILED
            task.error = f"No files found for CRISM observation {task.base_key}"
            return

        # Filter by file_types if specified
        if file_types:
            all_files = [f for f in all_files if f.file_type in file_types]

        # Skip files that already exist on disk
        target_dir = Path(task.target_dir)
        files_to_download: List[ODEFile] = []
        for f in all_files:
            file_path = target_dir / f.filename
            if file_path.exists() and file_path.stat().st_size > 0:
                # File exists and has content - mark as already completed
                task.files.append(FileProgress(
                    filename=f.filename,
                    url=f.url,
                    bytes_total=file_path.stat().st_size,
                    bytes_downloaded=file_path.stat().st_size,
                    status=DownloadStatus.COMPLETED,
                ))
                print(f"Skipping existing file: {f.filename}")
            else:
                files_to_download.append(f)
                task.files.append(FileProgress(
                    filename=f.filename,
                    url=f.url,
                    bytes_total=f.size_bytes,
                ))

        if not files_to_download:
            # All files already exist
            task.status = DownloadStatus.COMPLETED
            task.completed_at = datetime.utcnow().isoformat()
            return

        task.status = DownloadStatus.DOWNLOADING

        # Download files concurrently (with higher limit for speed)
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent downloads

        # Get indices for files that need downloading
        download_indices = []
        for i, f in enumerate(task.files):
            if f.status != DownloadStatus.COMPLETED:
                download_indices.append(i)

        async def download_with_limit(idx: int, ode_file: ODEFile):
            async with semaphore:
                await self._download_file(task, idx, ode_file, session)

        await asyncio.gather(*[
            download_with_limit(idx, files_to_download[i])
            for i, idx in enumerate(download_indices)
        ])

        # Copy browse images to quickview directory
        for f in task.files:
            if "_br" in f.filename.lower() and f.filename.lower().endswith(".png"):
                src = Path(task.target_dir) / f.filename
                # Use first browse file as quickview
                if "vna" in f.filename.lower():
                    dst = CRISM_QUICKVIEW_DIR / f"{task.base_key}.png"
                    if src.exists() and not dst.exists():
                        import shutil
                        shutil.copy(src, dst)
                # Also copy to browse directory
                browse_dst = CRISM_BROWSE_DIR / f.filename
                if src.exists() and not browse_dst.exists():
                    import shutil
                    shutil.copy(src, browse_dst)

    async def _download_hirise_bundle(self, task: DownloadTask, session: aiohttp.ClientSession):
        """Download HiRISE bundle files and convert JP2 to GeoTIFF."""
        bundle = await resolve_hirise_bundle(task.product_id, session)

        files_to_download: List[ODEFile] = []

        if bundle.jp2_file:
            files_to_download.append(bundle.jp2_file)
        if bundle.lbl_file:
            files_to_download.append(bundle.lbl_file)

        if not files_to_download:
            task.status = DownloadStatus.FAILED
            task.error = f"No files found for HiRISE product {task.product_id}"
            return

        # Create file progress entries
        for f in files_to_download:
            task.files.append(FileProgress(
                filename=f.filename,
                url=f.url,
                bytes_total=f.size_bytes,
            ))

        task.status = DownloadStatus.DOWNLOADING

        # Download files
        for i, f in enumerate(files_to_download):
            await self._download_file(task, i, f, session)

        # Convert JP2 to GeoTIFF using GDAL
        task.status = DownloadStatus.PROCESSING

        jp2_path = None
        for f in task.files:
            if f.filename.lower().endswith(".jp2"):
                jp2_path = Path(task.target_dir) / f.filename
                break

        if jp2_path and jp2_path.exists():
            tif_path = jp2_path.with_suffix(".tif")

            try:
                # Use gdal_translate to convert JP2 to GeoTIFF
                # This preserves georeferencing information
                result = subprocess.run(
                    [
                        "gdal_translate",
                        "-of", "GTiff",
                        "-co", "COMPRESS=LZW",
                        "-co", "TILED=YES",
                        str(jp2_path),
                        str(tif_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for large files
                )

                if result.returncode != 0:
                    print(f"GDAL conversion warning: {result.stderr}")
                    # Don't fail the task, JP2 can still be used

            except FileNotFoundError:
                print("GDAL not installed, skipping JP2 conversion")
            except subprocess.TimeoutExpired:
                print("GDAL conversion timed out")
            except Exception as e:
                print(f"GDAL conversion error: {e}")

    async def _download_file(
        self,
        task: DownloadTask,
        file_idx: int,
        ode_file: ODEFile,
        session: aiohttp.ClientSession,
    ):
        """Download a single file with progress tracking (uses aria2 if available)."""
        file_progress = task.files[file_idx]
        file_progress.status = DownloadStatus.DOWNLOADING

        target_path = Path(task.target_dir) / ode_file.filename

        # Determine if this is an optional file (browse images may not exist)
        is_optional = ode_file.file_type == "Browse"

        # Use aria2 for faster downloads if available
        if _ARIA2_AVAILABLE:
            try:
                result = await download_single_file(
                    url=ode_file.url,
                    output_dir=task.target_dir,
                    filename=ode_file.filename,
                    task_id=task.task_id,  # Pass task_id for cancellation tracking
                )

                if result.status == Aria2Status.COMPLETED:
                    file_progress.status = DownloadStatus.COMPLETED
                    file_progress.bytes_downloaded = result.bytes_downloaded
                    file_progress.bytes_total = result.bytes_total
                else:
                    file_progress.status = DownloadStatus.FAILED
                    file_progress.error = result.error or "aria2 download failed"
                return

            except Exception as e:
                # Fall back to aiohttp on aria2 error
                print(f"aria2 error, falling back to aiohttp: {e}")

        # Fallback: aiohttp streaming download
        try:
            # Use shorter timeout for optional files to fail fast on 404s
            timeout = aiohttp.ClientTimeout(
                total=3600,  # 1 hour total
                connect=10 if is_optional else 30,  # Quick connect timeout for browse
            )

            async with session.get(ode_file.url, timeout=timeout) as resp:
                if resp.status != 200:
                    file_progress.status = DownloadStatus.FAILED
                    file_progress.error = f"HTTP {resp.status}"
                    return

                # Update total size from Content-Length if available
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    file_progress.bytes_total = int(content_length)

                # Download with progress - use larger chunks for speed
                async with aiofiles.open(target_path, "wb") as f:
                    chunk_size = 4 * 1024 * 1024  # 4 MB chunks for faster throughput
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        file_progress.bytes_downloaded += len(chunk)

            file_progress.status = DownloadStatus.COMPLETED

        except asyncio.TimeoutError:
            file_progress.status = DownloadStatus.FAILED
            file_progress.error = "Download timeout"
        except aiohttp.ClientError as e:
            # Network errors - mark as failed but don't crash
            file_progress.status = DownloadStatus.FAILED
            file_progress.error = f"Network error: {type(e).__name__}"
        except Exception as e:
            file_progress.status = DownloadStatus.FAILED
            file_progress.error = str(e)

    async def _update_index(self, task: DownloadTask):
        """Update index.geojson after successful download."""
        if task.instrument == Instrument.CRISM:
            index_path = CRISM_DATA_DIR / "index.geojson"
            await self._update_crism_index(task, index_path)
        elif task.instrument == Instrument.HIRISE:
            index_path = HIRISE_DATA_DIR / "index.geojson"
            await self._update_hirise_index(task, index_path)
        elif task.instrument == Instrument.SHARAD:
            index_path = SHARAD_DATA_DIR / "index.geojson"
            await self._update_sharad_index(task, index_path)
        elif task.instrument == Instrument.SHARAD_HIGHRES:
            index_path = SHARAD_HIGHRES_DIR.parent / "sharad_highres_data" / "index.geojson"
            await self._update_sharad_highres_index(task, index_path)

    async def _update_crism_index(self, task: DownloadTask, index_path: Path):
        """Update CRISM index.geojson with new product."""
        # Load existing index
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                content = await f.read()
                index = json.loads(content)
        else:
            index = {"type": "FeatureCollection", "features": []}

        # Find files
        img_file = next((f.filename for f in task.files if f.filename.endswith(".img")), None)
        lbl_file = next((f.filename for f in task.files if f.filename.endswith(".lbl")), None)
        quicklook = next(
            (f"/crism/quickview/{f.filename}" for f in task.files if "_brvna" in f.filename.lower()),
            None
        )

        # Check if product already exists
        existing_idx = None
        for i, feat in enumerate(index["features"]):
            if feat["properties"].get("product_id", "").startswith(task.base_key):
                existing_idx = i
                break

        # Create new feature
        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "CRISM",
                "product_id": img_file.replace(".img", "") if img_file else task.base_key,
                "base_key": task.base_key,
                "mtr3_img": img_file,
                "mtr3_lbl": lbl_file,
                "quicklook": quicklook,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [task.lon or 0, task.lat or 0]
            }
        }

        if existing_idx is not None:
            index["features"][existing_idx] = feature
        else:
            index["features"].append(feature)

        # Write updated index
        async with aiofiles.open(index_path, "w") as f:
            await f.write(json.dumps(index, indent=2))

    async def _update_hirise_index(self, task: DownloadTask, index_path: Path):
        """Update HiRISE index.geojson with new product."""
        # Load existing index
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                content = await f.read()
                index = json.loads(content)
        else:
            index = {"type": "FeatureCollection", "features": []}

        # Find TIF file (converted from JP2) or JP2
        tif_file = None
        for f in task.files:
            if f.filename.lower().endswith(".jp2"):
                # Check if TIF was created
                tif_name = f.filename.replace(".JP2", ".tif").replace(".jp2", ".tif")
                tif_path = Path(task.target_dir) / tif_name
                if tif_path.exists():
                    tif_file = tif_name
                break

        quicklook = f"/hirise/quickview/{task.product_id}.jpg"

        # Check if product already exists
        existing_idx = None
        for i, feat in enumerate(index["features"]):
            if feat["properties"].get("product_id") == task.product_id:
                existing_idx = i
                break

        # Create new feature
        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "HIRISE",
                "product_id": task.product_id,
                "quicklook": quicklook,
                "red_tif": tif_file,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [task.lon or 0, task.lat or 0]
            }
        }

        if existing_idx is not None:
            index["features"][existing_idx] = feature
        else:
            index["features"].append(feature)

        # Write updated index
        async with aiofiles.open(index_path, "w") as f:
            await f.write(json.dumps(index, indent=2))

    async def _download_sharad_bundle(self, task: DownloadTask, session: aiohttp.ClientSession):
        """Download SHARAD USRDRV2 bundle files (THM quicklook + LBL)."""
        bundle = await resolve_sharad_bundle(task.base_key, session)

        files_to_download: List[ODEFile] = []

        if bundle.thm_file:
            files_to_download.append(bundle.thm_file)
        if bundle.lbl_file:
            files_to_download.append(bundle.lbl_file)

        if not files_to_download:
            task.status = DownloadStatus.FAILED
            task.error = f"No files found for SHARAD product {task.base_key}"
            return

        # Create file progress entries
        for f in files_to_download:
            task.files.append(FileProgress(
                filename=f.filename,
                url=f.url,
                bytes_total=f.size_bytes,
            ))

        task.status = DownloadStatus.DOWNLOADING

        # Download files
        for i, f in enumerate(files_to_download):
            await self._download_file(task, i, f, session)

        # Copy quicklook to SHARAD_QUICKVIEW_DIR
        for f in task.files:
            if f.filename.lower().endswith(".jpg") and f.status == DownloadStatus.COMPLETED:
                src = Path(task.target_dir) / f.filename
                dst = SHARAD_QUICKVIEW_DIR / f.filename
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy(src, dst)

    async def _download_sharad_highres_bundle(self, task: DownloadTask, session: aiohttp.ClientSession):
        """Download SHARAD High-Res RDR bundle files (DAT + LBL)."""
        bundle = await resolve_sharad_highres_bundle(task.base_key, session)

        files_to_download: List[ODEFile] = []

        if bundle.dat_file:
            files_to_download.append(bundle.dat_file)
        if bundle.lbl_file:
            files_to_download.append(bundle.lbl_file)

        if not files_to_download:
            task.status = DownloadStatus.FAILED
            task.error = f"No files found for SHARAD High-Res product {task.base_key}"
            return

        # Create file progress entries
        for f in files_to_download:
            task.files.append(FileProgress(
                filename=f.filename,
                url=f.url,
                bytes_total=f.size_bytes,
            ))

        task.status = DownloadStatus.DOWNLOADING

        # Download files
        for i, f in enumerate(files_to_download):
            await self._download_file(task, i, f, session)

    async def _update_sharad_index(self, task: DownloadTask, index_path: Path):
        """Update SHARAD index.geojson with new product."""
        # Load existing index
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                content = await f.read()
                index = json.loads(content)
        else:
            index = {"type": "FeatureCollection", "features": []}

        # Find quickview file
        quickview = None
        for f in task.files:
            if f.filename.lower().endswith(".jpg"):
                quickview = f"/sharad/quickview/{f.filename}"
                break

        # Check if product already exists
        existing_idx = None
        for i, feat in enumerate(index["features"]):
            if feat["properties"].get("product_id") == task.base_key:
                existing_idx = i
                break

        # Create new feature - SHARAD uses LineString geometry
        # For now, use point geometry; actual track coordinates would come from LBL
        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "SHARAD",
                "product_id": task.base_key,
                "quickview": quickview,
                "start_lat": task.lat,
                "start_lon": task.lon,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [task.lon or 0, task.lat or 0]
            }
        }

        if existing_idx is not None:
            index["features"][existing_idx] = feature
        else:
            index["features"].append(feature)

        # Write updated index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(index_path, "w") as f:
            await f.write(json.dumps(index, indent=2))

    async def _update_sharad_highres_index(self, task: DownloadTask, index_path: Path):
        """Update SHARAD High-Res index.geojson with new product."""
        # Load existing index
        if index_path.exists():
            async with aiofiles.open(index_path, "r") as f:
                content = await f.read()
                index = json.loads(content)
        else:
            index = {"type": "FeatureCollection", "features": []}

        # Find DAT file
        dat_file = None
        lbl_file = None
        for f in task.files:
            if f.filename.lower().endswith(".dat"):
                dat_file = f.filename
            elif f.filename.lower().endswith(".lbl"):
                lbl_file = f.filename

        # Check if product already exists
        existing_idx = None
        for i, feat in enumerate(index["features"]):
            if feat["properties"].get("product_id") == task.base_key:
                existing_idx = i
                break

        # Fetch footprint coordinates from ODE for LineString geometry
        footprint_coords = await get_sharad_highres_footprint(task.base_key)

        # Create geometry - use LineString if we have footprint, otherwise Point
        if footprint_coords and len(footprint_coords) >= 2:
            geometry = {
                "type": "LineString",
                "coordinates": footprint_coords
            }
        else:
            # Fallback to Point if no footprint available
            geometry = {
                "type": "Point",
                "coordinates": [task.lon or 0, task.lat or 0]
            }

        # Create new feature
        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "SHARAD_HIGHRES",
                "product_id": task.base_key,
                "dat_file": dat_file,
                "lbl_file": lbl_file,
            },
            "geometry": geometry
        }

        if existing_idx is not None:
            index["features"][existing_idx] = feature
        else:
            index["features"].append(feature)

        # Write updated index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(index_path, "w") as f:
            await f.write(json.dumps(index, indent=2))


# =============================================================================
# Existence Check
# =============================================================================

@dataclass
class ExistenceResult:
    """Detailed result of existence check."""
    exists: bool  # True only if ALL required files exist
    has_core: bool  # .img, .lbl files exist
    has_header: bool  # .hdr file exists
    has_wavelength: bool  # .tab file exists
    has_browse: bool  # At least one browse PNG exists
    missing_files: List[str]  # List of missing file types
    existing_files: List[str]  # List of existing files

    def to_dict(self) -> dict:
        return {
            "exists": self.exists,
            "has_core": self.has_core,
            "has_header": self.has_header,
            "has_wavelength": self.has_wavelength,
            "has_browse": self.has_browse,
            "missing_files": self.missing_files,
            "existing_files": self.existing_files,
        }


def check_local_existence(product_id: str, instrument: Instrument) -> bool:
    """
    Check if a product is fully downloaded (all required files exist).

    For CRISM: checks for .img, .lbl, .hdr, .tab, and at least one browse PNG
    For HiRISE: checks for .jp2/.tif and .lbl

    Args:
        product_id: Product identifier
        instrument: CRISM or HiRISE

    Returns:
        True only if ALL required files exist
    """
    result = check_local_existence_detailed(product_id, instrument)
    return result.exists


def check_local_existence_detailed(product_id: str, instrument: Instrument) -> ExistenceResult:
    """
    Detailed check of what files exist for a product.

    Args:
        product_id: Product identifier
        instrument: CRISM, HiRISE, SHARAD, or SHARAD_HIGHRES

    Returns:
        ExistenceResult with details about which files exist/are missing
    """
    if instrument == Instrument.CRISM:
        base_key = parse_crism_base_key(product_id)
        target_dir = CRISM_DATA_DIR / base_key
    elif instrument == Instrument.HIRISE:
        base_key = product_id.upper()
        target_dir = HIRISE_DATA_DIR / base_key
    elif instrument == Instrument.SHARAD:
        base_key = product_id.upper().replace("_THM", "")
        target_dir = SHARAD_DATA_DIR  # SHARAD files are not in subdirectories
    elif instrument == Instrument.SHARAD_HIGHRES:
        base_key = product_id.upper()
        target_dir = SHARAD_HIGHRES_DIR  # SHARAD High-Res files are not in subdirectories
    else:
        base_key = product_id.upper()
        target_dir = BASE_DIR / "data" / base_key

    missing = []
    existing = []

    # For SHARAD and SHARAD_HIGHRES, check for specific files in flat directory
    if instrument == Instrument.SHARAD:
        thm_file = target_dir / f"{base_key.lower()}_thm.jpg"
        lbl_file = target_dir / f"{base_key.lower()}_thm.lbl"

        has_thm = thm_file.exists() and thm_file.stat().st_size > 0
        has_lbl = lbl_file.exists() and lbl_file.stat().st_size > 0

        if has_thm:
            existing.append("thm")
        else:
            missing.append("thm")

        if has_lbl:
            existing.append("lbl")
        else:
            missing.append("lbl")

        return ExistenceResult(
            exists=has_thm,  # THM is the main required file
            has_core=has_thm,
            has_header=True,  # Not applicable
            has_wavelength=True,  # Not applicable
            has_browse=has_thm,
            missing_files=missing,
            existing_files=existing,
        )

    if instrument == Instrument.SHARAD_HIGHRES:
        dat_file = target_dir / f"{base_key.lower()}.dat"
        lbl_file = target_dir / f"{base_key.lower()}.lbl"

        has_dat = dat_file.exists() and dat_file.stat().st_size > 0
        has_lbl = lbl_file.exists() and lbl_file.stat().st_size > 0

        if has_dat:
            existing.append("dat")
        else:
            missing.append("dat")

        if has_lbl:
            existing.append("lbl")
        else:
            missing.append("lbl")

        # Both files required - partial if only one exists
        has_any = has_dat or has_lbl
        has_all = has_dat and has_lbl

        return ExistenceResult(
            exists=has_all,
            has_core=has_any,  # True if at least one file exists (for partial detection)
            has_header=True,  # Not applicable
            has_wavelength=True,  # Not applicable
            has_browse=False,  # Set to False so it doesn't trigger false "partial" status
            missing_files=missing,
            existing_files=existing,
        )

    if not target_dir.exists():
        if instrument == Instrument.CRISM:
            missing = ["img", "lbl", "hdr", "tab", "browse"]
        else:
            missing = ["jp2/tif", "lbl"]
        return ExistenceResult(
            exists=False,
            has_core=False,
            has_header=False,
            has_wavelength=False,
            has_browse=False,
            missing_files=missing,
            existing_files=[],
        )

    # Get all files in directory
    files = list(target_dir.iterdir())
    file_names = [f.name.lower() for f in files]
    file_suffixes = [f.suffix.lower() for f in files]

    # Check for aria2 partial download control files (.aria2)
    # If any .aria2 files exist, the download is incomplete
    has_aria2_partial = any(f.suffix.lower() == ".aria2" for f in files)

    # Helper to check if a file is complete (exists, non-zero size, no .aria2)
    def is_file_complete(suffix: str) -> bool:
        for f in files:
            if f.suffix.lower() == suffix and f.stat().st_size > 0:
                # Check if corresponding .aria2 file exists
                aria2_file = f.with_suffix(f.suffix + ".aria2")
                if not aria2_file.exists():
                    return True
        return False

    if instrument == Instrument.CRISM:
        # Check core files (.img, .lbl) - must be complete (no .aria2)
        has_img = is_file_complete(".img")
        has_lbl = is_file_complete(".lbl")
        has_core = has_img and has_lbl

        if has_img:
            existing.append("img")
        else:
            missing.append("img")

        if has_lbl:
            existing.append("lbl")
        else:
            missing.append("lbl")

        # Check header file (.hdr)
        has_header = is_file_complete(".hdr")
        if has_header:
            existing.append("hdr")
        else:
            missing.append("hdr")

        # Check wavelength table (.tab)
        has_wavelength = is_file_complete(".tab")
        if has_wavelength:
            existing.append("tab")
        else:
            missing.append("tab")

        # Check browse files (at least one _br*.png that is complete)
        has_browse = False
        for f in files:
            if "_br" in f.name.lower() and f.suffix.lower() == ".png" and f.stat().st_size > 0:
                aria2_file = f.with_suffix(".png.aria2")
                if not aria2_file.exists():
                    has_browse = True
                    break
        if has_browse:
            existing.append("browse")
        else:
            missing.append("browse")

        # All required files must exist and no partial downloads
        all_exist = has_core and has_header and has_wavelength and has_browse and not has_aria2_partial

        # If any aria2 partial files exist, mark as incomplete
        if has_aria2_partial:
            existing.append("partial")

        return ExistenceResult(
            exists=all_exist,
            has_core=has_core,
            has_header=has_header,
            has_wavelength=has_wavelength,
            has_browse=has_browse,
            missing_files=missing,
            existing_files=existing,
        )
    else:
        # HiRISE - check for complete files (no .aria2)
        has_jp2 = is_file_complete(".jp2")
        has_tif = is_file_complete(".tif")
        has_lbl = is_file_complete(".lbl")

        has_core = (has_jp2 or has_tif) and has_lbl

        if has_jp2 or has_tif:
            existing.append("jp2/tif")
        else:
            missing.append("jp2/tif")

        if has_lbl:
            existing.append("lbl")
        else:
            missing.append("lbl")

        # If any aria2 partial files exist, mark as incomplete
        if has_aria2_partial:
            existing.append("partial")

        return ExistenceResult(
            exists=has_core and not has_aria2_partial,
            has_core=has_core,
            has_header=True,  # HiRISE doesn't need separate header
            has_wavelength=True,  # HiRISE doesn't need wavelength table
            has_browse=True,  # Not requiring browse for HiRISE currently
            missing_files=missing,
            existing_files=existing,
        )


# Global instance
download_manager = DownloadManager()
