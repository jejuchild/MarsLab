#!/bin/bash
# Monitor background MIL training and auto-regenerate charts when done.
# Usage: nohup bash scripts/marslandform_v2/monitor_training.sh &

FROZEN_PID=$(pgrep -f "embeddings_dir Data/HiRISE/v2_output/embeddings_frozen_30" | head -1)
SSL_PID=$(pgrep -f "embeddings_dir Data/HiRISE/v2_output/embeddings_ssl" | head -1)

echo "Monitoring training..."
echo "  Frozen PID: ${FROZEN_PID:-NOT FOUND}"
echo "  SSL PID: ${SSL_PID:-NOT FOUND}"

while true; do
    FROZEN_ALIVE=0
    SSL_ALIVE=0
    
    if [ -n "$FROZEN_PID" ] && kill -0 "$FROZEN_PID" 2>/dev/null; then
        FROZEN_ALIVE=1
    fi
    if [ -n "$SSL_PID" ] && kill -0 "$SSL_PID" 2>/dev/null; then
        SSL_ALIVE=1
    fi
    
    if [ $FROZEN_ALIVE -eq 0 ] && [ $SSL_ALIVE -eq 0 ]; then
        echo "[$(date)] Both training jobs finished. Regenerating charts..."
        python3 /disk1/cspark/MarsLab/scripts/marslandform_v2/generate_ssl_comparison.py
        echo "[$(date)] Charts regenerated. Done."
        exit 0
    fi
    
    # Log progress every 10 minutes
    echo -n "[$(date)] Still running:"
    [ $FROZEN_ALIVE -eq 1 ] && echo -n " Frozen($(grep -c 'Epoch' /tmp/mil_frozen30.log 2>/dev/null)ep)"
    [ $SSL_ALIVE -eq 1 ] && echo -n " SSL($(grep -c 'Epoch' /tmp/mil_ssl30.log 2>/dev/null)ep)"
    echo ""
    
    sleep 600  # Check every 10 minutes
done
