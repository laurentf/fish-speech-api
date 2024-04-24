#!/bin/bash
set -e

CHECKPOINT_DIR="/app/checkpoints/openaudio-s1-mini"

# Auto-download model if HF_TOKEN is set and model not already present
if [ ! -f "$CHECKPOINT_DIR/codec.pth" ]; then
    if [ -n "$HF_TOKEN" ]; then
        echo "Model not found in $CHECKPOINT_DIR. Downloading openaudio-s1-mini from HuggingFace..."
        pip install -q huggingface_hub 2>/dev/null || true
        python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='fishaudio/openaudio-s1-mini',
    local_dir='$CHECKPOINT_DIR',
    token=os.environ['HF_TOKEN'],
)
print('Model downloaded successfully.')
"
    else
        echo "WARNING: No model found at $CHECKPOINT_DIR and HF_TOKEN is not set."
        echo "Either mount a volume with the model or set HF_TOKEN to auto-download."
        echo "See README.md for instructions."
        exit 1
    fi
else
    echo "Model found at $CHECKPOINT_DIR"
fi

# Run the server
exec python server.py
