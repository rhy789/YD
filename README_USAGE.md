# YOLO + DeepSORT Tracking 사용법

## 설치

### 1. Re-ID 모델 다운로드
```bash
chmod +x download_reid_model.sh
./download_reid_model.sh
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

### 옵션 1: 전체 MOT17 Train 데이터셋 처리 (권장)

```bash
# 간단한 실행
chmod +x run_evaluation.sh
./run_evaluation.sh
```

또는 직접 실행:
```bash
python yolo_deepsort_integration.py \
    --input data/MOT17/train \
    --output results/yolo_deepsort_baseline \
    --weights yolov5s.pt \
    --reid-model models/mars-small128.pb \
    --mode dataset \
    --save-video \
    --evaluate
```

### 옵션 2: 특정 시퀀스만 처리

```bash
# MOT17-02-FRCNN만 처리
python yolo_deepsort_integration.py \
    --input data/MOT17/train/MOT17-02-FRCNN \
    --output results \
    --weights yolov5s.pt \
    --reid-model models/mars-small128.pb \
    --mode sequence \
    --save-video

# MOT17-04-FRCNN만 처리
python yolo_deepsort_integration.py \
    --input data/MOT17/train/MOT17-04-FRCNN \
    --output results \
    --weights yolov5s.pt \
    --reid-model models/mars-small128.pb \
    --mode sequence \
    --save-video
```

### 옵션 3: 비디오 파일 처리

```bash
python yolo_deepsort_integration.py \
    --input video.mp4 \
    --output results/output.mp4 \
    --weights yolov5s.pt \
    --reid-model models/mars-small128.pb \
    --mode video
```

## 출력 결과

### 각 시퀀스별 결과 (예: MOT17-02-FRCNN)
```
results/
├── MOT17-02-FRCNN/
│   ├── tracking.txt      # MOT 포맷 추적 결과
│   └── tracking.mp4      # 시각화 동영상
├── MOT17-04-FRCNN/
│   ├── tracking.txt
│   └── tracking.mp4
├── ...
└── evaluation_summary.txt  # 전체 평가 결과 요약
```

### evaluation_summary.txt 예시
```
====================================================================================================
YOLO + DeepSORT Tracking Evaluation Results
====================================================================================================

Sequence                     MOTA     IDF1    IDS     Prec   Recall       FP       FN
----------------------------------------------------------------------------------------------------
MOT17-02-FRCNN              XX.XX%   XX.XX%    XXX   XX.XX%   XX.XX%     XXXX     XXXX
MOT17-04-FRCNN              XX.XX%   XX.XX%    XXX   XX.XX%   XX.XX%     XXXX     XXXX
...
----------------------------------------------------------------------------------------------------
AVERAGE                     XX.XX%   XX.XX%    XXX   XX.XX%   XX.XX%     XXXX     XXXX
====================================================================================================

Detailed Metrics:
----------------------------------------------------------------------------------------------------

MOT17-02-FRCNN:
  Frames: 600
  MOTA (Multi-Object Tracking Accuracy): XX.XX%
  MOTP (Multi-Object Tracking Precision): X.XXXX
  IDF1 (ID F1 Score): XX.XX%
  IDS (ID Switches): XXX
  Fragmentations: XXX
  False Positives: XXXX
  False Negatives (Misses): XXXX
  Precision: XX.XX%
  Recall: XX.XX%
...
```

## 평가 메트릭 설명

- **MOTA (Multi-Object Tracking Accuracy)**: 전체 추적 정확도 (높을수록 좋음)
- **IDF1 (ID F1 Score)**: ID 유지 성능 (높을수록 좋음)
- **IDS (ID Switches)**: ID 전환 횟수 (낮을수록 좋음)
- **Precision**: 검출 정밀도 (높을수록 좋음)
- **Recall**: 검출 재현율 (높을수록 좋음)
- **FP (False Positives)**: 잘못된 검출 수
- **FN (False Negatives)**: 놓친 객체 수

## 옵션 설명

```
--input: 입력 경로 (비디오 파일, 시퀀스 폴더, 또는 데이터셋 폴더)
--output: 출력 디렉토리
--weights: YOLO 가중치 파일 (기본: yolov5s.pt)
--reid-model: Re-ID 모델 파일 (기본: models/mars-small128.pb)
--device: 디바이스 ('' 자동, '0' GPU, 'cpu' CPU)
--mode: 처리 모드 (video/sequence/dataset)
--save-video: 비디오 저장 (기본: True)
--no-video: 비디오 저장 안 함
--evaluate: 평가 수행 (기본: True)
--no-evaluate: 평가 안 함
```

## GPU 사용

```bash
# GPU 0번 사용
python yolo_deepsort_integration.py \
    --input data/MOT17/train \
    --output results \
    --device 0 \
    --mode dataset

# CPU만 사용
python yolo_deepsort_integration.py \
    --input data/MOT17/train \
    --output results \
    --device cpu \
    --mode dataset
```

## 주의사항

1. **GT가 있는 데이터만 평가됨**: train 데이터셋만 자동으로 평가
2. **메모리 부족 시**: `--no-video` 옵션으로 비디오 저장 생략
3. **Re-ID 모델 필수**: `models/mars-small128.pb` 파일이 있어야 함





