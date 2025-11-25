#!/bin/bash
# Run YOLO + DeepSORT tracking on MOT17 train dataset with evaluation

# Configuration
YOLO_WEIGHTS="yolov5s.pt"
REID_MODEL="models/mars-small128.pb"
MOT_TRAIN_DIR="data/MOT17/train"
OUTPUT_DIR="results/yolo_deepsort_baseline"
DEVICE=""  # Empty for auto-detect, or "0" for GPU, "cpu" for CPU

echo "============================================"
echo "YOLO + DeepSORT Tracking Evaluation"
echo "============================================"
echo "Input: $MOT_TRAIN_DIR"
echo "Output: $OUTPUT_DIR"
echo "YOLO Weights: $YOLO_WEIGHTS"
echo "Re-ID Model: $REID_MODEL"
echo "============================================"
echo ""

# Run tracking and evaluation
python3 yolo_deepsort_integration.py \
    --input "$MOT_TRAIN_DIR" \
    --output "$OUTPUT_DIR" \
    --weights "$YOLO_WEIGHTS" \
    --reid-model "$REID_MODEL" \
    --device "$DEVICE" \
    --mode dataset \
    --save-video \
    --evaluate

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/evaluation_summary.txt"
echo "============================================"


