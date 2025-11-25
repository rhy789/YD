#!/bin/bash
# Download Re-ID model for DeepSORT

mkdir -p models

echo "[INFO] Downloading Mars Re-ID model (mars-small128.pb)..."
echo "[INFO] Trying multiple mirror sources..."
echo ""

# Try multiple download sources
SUCCESS=0

# Method 1: Try direct link from various mirrors
URLS=(
    "https://github.com/ZQPei/deep_sort_pytorch/releases/download/v1.0/ckpt.t7"
    "https://zenodo.org/record/4322874/files/mars-small128.pb"
    "https://huggingface.co/datasets/nwojke/deep_sort_resources/resolve/main/mars-small128.pb"
)

cd models

# Try wget with different sources
for URL in "${URLS[@]}"; do
    echo "[INFO] Trying: $URL"
    if wget -q --show-progress "$URL" -O mars-small128.pb 2>/dev/null; then
        if [ -f "mars-small128.pb" ] && [ -s "mars-small128.pb" ]; then
            SUCCESS=1
            echo "[SUCCESS] Downloaded from: $URL"
            break
        else
            rm -f mars-small128.pb
        fi
    fi
done

cd ..

if [ $SUCCESS -eq 1 ]; then
    echo ""
    echo "[SUCCESS] Re-ID model downloaded successfully!"
    echo "[INFO] Model path: models/mars-small128.pb"
    echo "[INFO] Model size: $(du -h models/mars-small128.pb | cut -f1)"
else
    echo ""
    echo "[ERROR] Failed to download Re-ID model from all sources"
    echo ""
    echo "============================================"
    echo "MANUAL DOWNLOAD REQUIRED"
    echo "============================================"
    echo ""
    echo "Please download the model manually:"
    echo ""
    echo "Option 1 - Google Drive (Official):"
    echo "  1. Visit: https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6"
    echo "  2. Download 'mars-small128.pb'"
    echo "  3. Save to: $(pwd)/models/mars-small128.pb"
    echo ""
    echo "Option 2 - Alternative (from ZQPei fork):"
    echo "  wget https://github.com/ZQPei/deep_sort_pytorch/releases/download/v1.0/ckpt.t7 -O models/mars-small128.pb"
    echo ""
    echo "Option 3 - Use existing model if you have it:"
    echo "  cp /path/to/your/mars-small128.pb models/"
    echo ""
    echo "============================================"
    exit 1
fi

