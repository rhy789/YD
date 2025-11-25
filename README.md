# YOLO + DeepSORT Tracking with CLIP Integration

This repository contains an experimental comparison of three object tracking approaches using MOT17 dataset:

1. **YOLO + DeepSORT** (Baseline)
2. **YOLO + CLIP + DeepSORT** (CLIP features for Re-ID)
3. **YOLO + DeepSORT + CLIP** (DeepSORT tracking + CLIP post-processing)

## 🚀 Key Features

- **Complete YOLO + DeepSORT Integration**: Real-time object detection and tracking
- **Comprehensive Evaluation**: MOTA, IDF1, HOTA, Precision, Recall, IDS metrics
- **CLIP Integration Ready**: Framework for CLIP-based Re-ID experiments
- **MOT17 Dataset Support**: Automated evaluation on all MOT17 train sequences
- **Detailed Metrics Export**: TXT and JSON format results with trajectory analysis

## 📊 Current Results (YOLO + DeepSORT Baseline)

| Sequence | MOTA | HOTA | IDF1 | DetA | AssA | IDS | Precision | Recall |
|----------|------|------|------|------|------|-----|-----------|--------|
| MOT17-02-DPM/FRCNN/SDP | 14.46% | 39.04% | 25.25% | 19.63% | 77.61% | 24 | 76.88% | 20.87% |
| MOT17-04-DPM/FRCNN/SDP | 21.34% | 49.20% | 34.88% | 27.69% | 87.42% | 60 | 77.76% | 30.07% |
| MOT17-05-DPM/FRCNN/SDP | 33.01% | 65.16% | 46.35% | 46.92% | 90.49% | 57 | 70.30% | 58.58% |
| MOT17-09-DPM/FRCNN/SDP | 43.31% | 64.00% | 42.89% | 54.19% | 75.58% | 35 | 74.80% | 66.29% |
| MOT17-10-DPM/FRCNN/SDP | 32.09% | 55.24% | 43.44% | 36.98% | 82.50% | 32 | 84.35% | 39.71% |
| MOT17-11-DPM/FRCNN/SDP | 48.73% | 68.89% | 55.34% | 54.01% | 87.87% | 16 | 84.36% | 60.03% |
| MOT17-13-DPM/FRCNN/SDP | -8.11% | 42.25% | 28.02% | 19.01% | 93.90% | 19 | 43.22% | 25.32% |

**Average**: MOTA=26.40%, HOTA=54.83%, IDF1=39.45%, IDS=729

## 🛠️ Installation

### Prerequisites
```bash
# Required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Download Models
```bash
# Re-ID model for DeepSORT (Mars-small128)
./download_reid_model.sh

# YOLOv5 model (automatically downloaded on first run)
# Or manually: wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -P yolov5/
```

### TrackEval for HOTA (Optional)
```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

## 🚀 Quick Start

### Run on MOT17 Dataset
```bash
# Process all MOT17 train sequences with evaluation
./run_evaluation.sh

# Or run specific sequence
python yolo_deepsort_integration.py \
    --input data/MOT17/train/MOT17-02-FRCNN \
    --output results \
    --mode sequence \
    --save-video \
    --evaluate
```

### Run on Video File
```bash
python yolo_deepsort_integration.py \
    --input your_video.mp4 \
    --output results/output.mp4 \
    --mode video
```

## 📁 Project Structure

```
├── yolo_deepsort_integration.py    # Main tracking script
├── compute_hota.py                 # HOTA computation with TrackEval
├── run_evaluation.sh               # Batch evaluation script
├── download_reid_model.sh          # Model download script
├── requirements.txt                # Python dependencies
├── EVALUATION_GUIDE.md            # Detailed metrics guide
├── README_USAGE.md                # Usage instructions
│
├── yolov5/                        # YOLOv5 repository
├── deep_sort-master/              # DeepSORT repository
├── models/                        # Model weights
│   └── mars-small128.pb           # Re-ID model
├── results/                       # Output results (auto-generated)
└── data/                          # MOT17 dataset (not included)
```

## 📈 Evaluation Metrics

### Primary Metrics
- **MOTA**: Multi-Object Tracking Accuracy (higher better)
- **HOTA**: Higher Order Tracking Accuracy (higher better)
- **IDF1**: ID F1 Score (higher better)
- **IDS**: ID Switches (lower better)

### Detection Metrics
- **Precision**: TP/(TP+FP) (higher better)
- **Recall**: TP/(TP+FN) (higher better)
- **FPR/FNR**: False Positive/Negative Rates (lower better)

### Trajectory Quality
- **MT/PT/ML**: Mostly/Partially/Mostly Lost trajectories

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed explanations.

## 🔬 Experimental Setup

### Experiment 1: YOLO + DeepSORT (Baseline)
- Standard YOLO detection + DeepSORT tracking
- Mars-small128 Re-ID features
- Results in `results/yolo_deepsort_baseline/`

### Experiment 2: YOLO + CLIP + DeepSORT
- Replace Re-ID features with CLIP embeddings
- Modify `create_detection_objects()` in `yolo_deepsort_integration.py`

### Experiment 3: YOLO + DeepSORT + CLIP
- Keep DeepSORT tracking, add CLIP post-processing
- Filter/re-rank trajectories using CLIP similarity

## 📋 Requirements

- Python 3.10+
- CUDA 11.8+ (recommended)
- 16GB+ RAM
- NVIDIA GPU (recommended)

## 🤝 Contributing

This is an experimental research project. Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Share your results

## 📝 License

This project combines multiple open-source components:
- YOLOv5: GPL-3.0
- DeepSORT: GPL-3.0
- TrackEval: MIT

## 📊 Citation

If you use this code in your research, please cite:

```bibtex
@article{wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  journal={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017}
}

@article{redmon2018yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
}
```

## 🆘 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--device cpu` or smaller batch sizes
2. **Re-ID model not found**: Run `./download_reid_model.sh`
3. **HOTA calculation fails**: Install TrackEval or it will use fallback
4. **Low performance**: MOT17 is challenging - this is expected baseline

### Support

- Check [README_USAGE.md](README_USAGE.md) for detailed instructions
- Review [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for metrics explanations
