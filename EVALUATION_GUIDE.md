# 평가 지표 상세 가이드

## 출력 파일 구조

```
results/yolo_deepsort_baseline/
├── MOT17-02-FRCNN/
│   ├── tracking.txt      # MOT 포맷 추적 결과
│   ├── tracking.mp4      # 시각화 비디오
│   ├── metrics.txt       # 📊 상세 평가 메트릭 (TXT)
│   └── metrics.json      # 📊 평가 메트릭 (JSON)
├── MOT17-04-FRCNN/
│   ├── tracking.txt
│   ├── tracking.mp4
│   ├── metrics.txt
│   └── metrics.json
├── ...
└── evaluation_summary.txt  # 전체 시퀀스 요약
```

## metrics.txt 내용

각 시퀀스 폴더의 `metrics.txt`에는 다음 내용이 포함됩니다:

### 1. PRIMARY TRACKING METRICS (주요 추적 메트릭)

```
MOTA (Multi-Object Tracking Accuracy):        XX.XX%
  - 전체 추적 정확도 (높을수록 좋음)
  - MOTA = 1 - (FN + FP + IDS) / GT
  
MOTP (Multi-Object Tracking Precision):       X.XXXX
  - 위치 정확도 (낮을수록 좋음, IoU 기반)
  
IDF1 (ID F1 Score):                           XX.XX%
  - ID 보존 성능 (높을수록 좋음)
  - ID를 얼마나 잘 유지하는지
  
HOTA (Higher Order Tracking Accuracy):        XX.XX%
  - 검출과 연관성의 균형 지표
  - HOTA = sqrt(DetA × AssA)
  
DetA (Detection Accuracy):                    XX.XX%
  - 검출 정확도
  
AssA (Association Accuracy):                  XX.XX%
  - 연관성 정확도 (ID 매칭)
  
Precision:                                    XX.XX%
  - 정밀도 = TP / (TP + FP)
  
Recall:                                       XX.XX%
  - 재현율 = TP / (TP + FN)
  
F1 Score:                                     XX.XX%
  - F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 2. ID SWITCHING METRICS (ID 전환 메트릭)

```
ID Switches (IDS):                            XX
  - ID가 바뀐 횟수 (낮을수록 좋음)
  
Fragmentations:                               XX
  - 트랙이 끊긴 횟수 (낮을수록 좋음)
  
ID Transfers:                                 XX
ID Ascend:                                    XX
ID Migrate:                                   XX
  - 다양한 ID 전환 유형
```

### 3. DETECTION METRICS (검출 메트릭)

```
True Positives (TP):                          XXXX
  - 올바른 검출 수
  
False Positives (FP):                         XXXX
  - 잘못된 검출 수 (낮을수록 좋음)
  
False Negatives (FN):                         XXXX
  - 놓친 객체 수 (낮을수록 좋음)
  
False Positive Rate (FPR):                    XX.XX%
  - FPR = FP / (TP + FP)
  
False Negative Rate (FNR):                    XX.XX%
  - FNR = FN / (TP + FN)
```

### 4. TRAJECTORY QUALITY (궤적 품질)

```
Mostly Tracked (MT):                          XX (XX.XX%)
  - 80% 이상 추적된 객체 수 (높을수록 좋음)
  
Partially Tracked (PT):                       XX (XX.XX%)
  - 20-80% 추적된 객체 수
  
Mostly Lost (ML):                             XX (XX.XX%)
  - 20% 미만 추적된 객체 수 (낮을수록 좋음)
```

### 5. COUNT STATISTICS (통계)

```
Total Frames:                                 XXX
Total Ground Truth Objects:                   XXXX
Total Predictions:                            XXXX
Unique Ground Truth IDs:                      XX
```

## HOTA 계산 (선택사항)

HOTA는 TrackEval 라이브러리가 필요합니다.

### 설치:

```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

### 사용:

```bash
# 방법 1: 제공된 스크립트 사용
python compute_hota.py \
    --tracker-dir results/yolo_deepsort_baseline \
    --gt-dir data/MOT17/train

# 방법 2: TrackEval 직접 사용
python -m trackeval.eval \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL yolo_deepsort_baseline \
    --GT_FOLDER data/MOT17/train \
    --TRACKERS_FOLDER results \
    --METRICS HOTA CLEAR Identity
```

## 메트릭 해석 가이드

### 좋은 성능의 기준:

| 메트릭 | 우수 | 좋음 | 보통 | 개선필요 |
|--------|------|------|------|----------|
| MOTA   | >60% | 40-60% | 20-40% | <20% |
| IDF1   | >70% | 50-70% | 30-50% | <30% |
| HOTA   | >60% | 45-60% | 30-45% | <30% |
| IDS    | <100 | 100-300 | 300-500 | >500 |
| MT     | >50% | 30-50% | 15-30% | <15% |
| ML     | <20% | 20-40% | 40-60% | >60% |

### 메트릭 간 관계:

- **MOTA ↑** = 전체 추적 성능 향상
- **IDF1 ↑** = ID 유지 성능 향상
- **IDS ↓** = ID 전환 감소
- **MT ↑ & ML ↓** = 지속적 추적 성능 향상
- **Precision ↑** = 거짓 검출 감소
- **Recall ↑** = 놓친 객체 감소

## 실험 비교 시 주의사항

세 가지 실험을 비교할 때 주목할 메트릭:

### 실험 1 vs 2 (Re-ID vs CLIP):
- **IDF1, IDS**: ID 매칭 성능 비교
- **AssA**: 연관성 정확도 비교
- **MOTA**: 전체 성능 비교

### 실험 2 vs 3 (CLIP 위치):
- **IDS, Fragmentations**: CLIP이 tracking에 미치는 영향
- **MOTA, HOTA**: 전체 성능 차이
- **MT/ML**: 지속성 차이

## JSON 포맷

각 시퀀스의 `metrics.json`은 프로그래밍 방식으로 쉽게 파싱 가능:

```python
import json

with open('results/MOT17-02-FRCNN/metrics.json') as f:
    metrics = json.load(f)
    
print(f"MOTA: {metrics['mota']:.2f}%")
print(f"IDF1: {metrics['idf1']:.2f}%")
print(f"IDS: {metrics['num_switches']}")
```

## 평가 결과 비교 스크립트

세 가지 실험 결과를 비교하려면:

```python
import json
import pandas as pd

experiments = ['yolo_deepsort', 'yolo_clip_deepsort', 'yolo_deepsort_clip']
sequences = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN']

results = []
for exp in experiments:
    for seq in sequences:
        with open(f'results/{exp}/{seq}/metrics.json') as f:
            m = json.load(f)
            m['experiment'] = exp
            results.append(m)

df = pd.DataFrame(results)
print(df[['experiment', 'sequence', 'mota', 'idf1', 'num_switches']])
```




