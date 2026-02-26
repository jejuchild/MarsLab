#!/bin/bash
# Wait for download to finish, then run the full pipeline.
# Usage: nohup bash scripts/landform_pipeline/run_after_download.sh > pipeline_full_log.txt 2>&1 &

DOWNLOAD_PID=3448520
PROJECT_ROOT="/disk1/cspark/MarsLab"
cd "$PROJECT_ROOT"

echo "$(date) — Waiting for download (PID $DOWNLOAD_PID) to finish..."

# Wait for the download process to complete
while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
    DOWNLOADED=$(ls Data/HiRISE/midlat_browse/*.jpg 2>/dev/null | wc -l)
    echo "$(date) — Download still running. $DOWNLOADED images so far."
    sleep 120
done

DOWNLOADED=$(ls Data/HiRISE/midlat_browse/*.jpg 2>/dev/null | wc -l)
echo "$(date) — Download finished! $DOWNLOADED images total."
echo ""

# Run the full pipeline
echo "$(date) — Starting full pipeline..."
python scripts/landform_pipeline/run_pipeline.py \
    --image-dirs "Data/HiRISE/midlat_browse,arcadia_hirise/jpeg" \
    --n-clusters 40 \
    --mola-weight 1.0 \
    --skip-umap \
    2>&1

echo "$(date) — Pipeline complete!"
