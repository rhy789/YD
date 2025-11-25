#!/usr/bin/env python3
"""
Simple YOLO + DeepSORT Integration
This is the base version for comparison experiments.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add yolov5 to path
sys.path.insert(0, str(Path(__file__).parent / 'yolov5'))
sys.path.insert(0, str(Path(__file__).parent / 'deep_sort-master'))
sys.path.insert(0, str(Path(__file__).parent / 'deep_sort-master' / 'tools'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

# Import Re-ID encoder from DeepSORT
from generate_detections import create_box_encoder


class YOLODeepSORT:
    """Simple YOLO + DeepSORT tracker"""
    
    def __init__(self, yolo_weights='yolov5s.pt', reid_model='models/mars-small128.pb', 
                 device='', conf_thres=0.5, iou_thres=0.45):
        """
        Initialize YOLO + DeepSORT tracker
        
        Args:
            yolo_weights: Path to YOLO weights
            reid_model: Path to Re-ID model for feature extraction
            device: Device to run on ('' for auto, '0' for GPU, 'cpu' for CPU)
            conf_thres: Confidence threshold for detection
            iou_thres: IOU threshold for NMS
        """
        # Initialize YOLO
        self.device = select_device(device)
        self.model = DetectMultiBackend(yolo_weights, device=self.device)
        self.model.warmup(imgsz=(1, 3, 640, 640))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize Re-ID encoder
        if os.path.exists(reid_model):
            self.encoder = create_box_encoder(reid_model, batch_size=32)
            print(f"[INFO] Loaded Re-ID model: {reid_model}")
        else:
            print(f"[WARNING] Re-ID model not found: {reid_model}")
            print(f"[WARNING] Download it from: https://github.com/nwojke/deep_sort/releases")
            self.encoder = None
        
        # Initialize DeepSORT
        max_cosine_distance = 0.4
        nn_budget = None
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)
        
        print(f"[INFO] Initialized YOLO model on {self.device}")
        print(f"[INFO] Initialized DeepSORT tracker")
    
    def preprocess(self, img):
        """Preprocess image for YOLO"""
        img_input = letterbox(img, 640, stride=32, auto=True)[0]
        img_input = img_input.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_input = np.ascontiguousarray(img_input)
        img_input = torch.from_numpy(img_input).to(self.device)
        img_input = img_input.float() / 255.0
        if len(img_input.shape) == 3:
            img_input = img_input[None]  # expand for batch dim
        return img_input
    
    def detect(self, img):
        """
        Run YOLO detection on image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            detections: List of [x1, y1, x2, y2, conf, cls]
        """
        img_input = self.preprocess(img)
        
        # Run inference
        pred = self.model(img_input)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        
        # Process detections
        detections = []
        if len(pred) > 0 and pred[0] is not None:
            det = pred[0]
            det[:, :4] = scale_boxes(img_input.shape[2:], det[:, :4], img.shape).round()
            detections = det.cpu().numpy()
        
        return detections
    
    def create_detection_objects(self, img, detections):
        """
        Convert YOLO detections to DeepSORT Detection objects with Re-ID features
        
        Args:
            img: Original image (BGR format)
            detections: YOLO detections [x1, y1, x2, y2, conf, cls]
            
        Returns:
            detection_list: List of DeepSORT Detection objects
        """
        detection_list = []
        
        if len(detections) == 0:
            return detection_list
        
        # Extract bounding boxes in tlwh format
        bboxes_tlwh = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bboxes_tlwh.append([x1, y1, x2 - x1, y2 - y1])
        
        # Extract Re-ID features using the encoder
        if self.encoder is not None:
            features = self.encoder(img, np.array(bboxes_tlwh))
        else:
            # Fallback to dummy features if encoder is not available
            features = np.random.randn(len(detections), 128).astype(np.float32)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Create Detection objects
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            tlwh = bboxes_tlwh[i]
            feature = features[i]
            
            detection_list.append(Detection(tlwh, conf, feature))
        
        return detection_list
    
    def track(self, img):
        """
        Run YOLO detection + DeepSORT tracking
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            tracks: List of active tracks with [x1, y1, x2, y2, track_id, conf]
        """
        # YOLO detection
        detections = self.detect(img)
        
        # Convert to DeepSORT Detection objects with Re-ID features
        detection_objects = self.create_detection_objects(img, detections)
        
        # Update tracker
        self.tracker.predict()
        self.tracker.update(detection_objects)
        
        # Get active tracks
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()  # Get bbox in [x1, y1, x2, y2] format
            track_id = track.track_id
            
            tracks.append({
                'bbox': bbox,
                'track_id': track_id,
                'confidence': 1.0  # Track confidence
            })
        
        return tracks


def process_video(video_path, output_path, yolo_weights='yolov5s.pt', 
                  reid_model='models/mars-small128.pb', device=''):
    """
    Process a video with YOLO + DeepSORT tracking
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        yolo_weights: Path to YOLO weights
        reid_model: Path to Re-ID model
        device: Device to run on
    """
    # Initialize tracker
    tracker = YOLODeepSORT(yolo_weights=yolo_weights, reid_model=reid_model, device=device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects
        tracks = tracker.track(frame)
        
        # Draw tracks
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO] Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    print(f"[INFO] Output saved to: {output_path}")


def process_mot_sequence(sequence_path, output_dir, yolo_weights='yolov5s.pt',
                        reid_model='models/mars-small128.pb', device='', save_video=True):
    """
    Process MOT17 sequence with video and txt output
    
    Args:
        sequence_path: Path to MOT sequence (e.g., data/MOT17/train/MOT17-02-FRCNN)
        output_dir: Output directory
        yolo_weights: Path to YOLO weights
        reid_model: Path to Re-ID model
        device: Device to run on
        save_video: Whether to save output video
    
    Returns:
        result_file: Path to tracking results txt file
    """
    sequence_name = Path(sequence_path).name
    img_dir = Path(sequence_path) / 'img1'
    
    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        return None
    
    # Get all images
    img_files = sorted(img_dir.glob('*.jpg'))
    
    if len(img_files) == 0:
        print(f"[ERROR] No images found in {img_dir}")
        return None
    
    print(f"\n[INFO] Processing sequence: {sequence_name}")
    print(f"[INFO] Found {len(img_files)} images")
    
    # Read seqinfo for video parameters
    seqinfo_file = Path(sequence_path) / 'seqinfo.ini'
    fps = 30  # default
    if seqinfo_file.exists():
        import configparser
        config = configparser.ConfigParser()
        config.read(seqinfo_file)
        if 'Sequence' in config and 'frameRate' in config['Sequence']:
            fps = int(config['Sequence']['frameRate'])
    
    # Initialize tracker
    tracker = YOLODeepSORT(yolo_weights=yolo_weights, reid_model=reid_model, device=device)
    
    # Create output directory
    output_path = Path(output_dir) / sequence_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize video writer if needed
    video_writer = None
    if save_video:
        first_img = cv2.imread(str(img_files[0]))
        height, width = first_img.shape[:2]
        video_file = output_path / 'tracking.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
        print(f"[INFO] Saving video to: {video_file}")
    
    # Process frames
    results = []
    
    for frame_idx, img_file in enumerate(img_files, start=1):
        img = cv2.imread(str(img_file))
        
        # Track objects
        tracks = tracker.track(img)
        
        # Save results in MOT format and draw on video
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            conf = track['confidence']
            
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            results.append(f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")
            
            # Draw on image for video
            if save_video:
                x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)
                label = f"ID: {track_id}"
                cv2.putText(img, label, (x1_int, y1_int - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write frame to video
        if save_video and video_writer:
            video_writer.write(img)
        
        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx}/{len(img_files)} frames")
    
    # Save tracking results
    result_file = output_path / 'tracking.txt'
    with open(result_file, 'w') as f:
        f.writelines(results)
    
    # Release video writer
    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved to: {video_file}")
    
    print(f"[INFO] Results saved to: {result_file}")
    return result_file


def evaluate_mot_sequence(gt_file, pred_file, sequence_name, output_dir=None):
    """
    Evaluate tracking results using motmetrics (MOTA, IDS, etc.) and HOTA
    
    Args:
        gt_file: Path to ground truth file
        pred_file: Path to prediction file
        sequence_name: Name of the sequence
        output_dir: Output directory to save metrics txt file
        
    Returns:
        metrics_dict: Dictionary of evaluation metrics
    """
    import json
    
    try:
        import motmetrics as mm
    except ImportError:
        print("[ERROR] motmetrics not installed. Install with: pip install motmetrics")
        return None
    
    # Load ground truth
    gt = mm.io.loadtxt(gt_file, fmt='mot16', min_confidence=1)
    
    # Load predictions
    pred = mm.io.loadtxt(pred_file, fmt='mot16', min_confidence=-1)
    
    # Create accumulator
    acc = mm.utils.compare_to_groundtruth(gt, pred, 'iou', distth=0.5)
    
    # Compute comprehensive metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'num_frames', 'mota', 'motp', 'idf1', 
        'num_switches', 'num_fragmentations',
        'num_false_positives', 'num_misses',
        'precision', 'recall',
        'num_detections', 'num_objects', 'num_predictions',
        'num_unique_objects', 'mostly_tracked', 'partially_tracked',
        'mostly_lost', 'num_matches', 'num_transfer', 'num_ascend', 'num_migrate'
    ], name=sequence_name)
    
    # Extract metrics
    metrics_dict = {
        'sequence': sequence_name,
        'num_frames': int(summary['num_frames'].iloc[0]),
        'mota': float(summary['mota'].iloc[0]) * 100,
        'motp': float(summary['motp'].iloc[0]),
        'idf1': float(summary['idf1'].iloc[0]) * 100,
        'num_switches': int(summary['num_switches'].iloc[0]),
        'num_fragmentations': int(summary['num_fragmentations'].iloc[0]),
        'num_false_positives': int(summary['num_false_positives'].iloc[0]),
        'num_misses': int(summary['num_misses'].iloc[0]),
        'precision': float(summary['precision'].iloc[0]) * 100,
        'recall': float(summary['recall'].iloc[0]) * 100,
        'num_detections': int(summary['num_detections'].iloc[0]),
        'num_objects': int(summary['num_objects'].iloc[0]),
        'num_predictions': int(summary['num_predictions'].iloc[0]),
        'num_unique_objects': int(summary['num_unique_objects'].iloc[0]),
        'mostly_tracked': int(summary['mostly_tracked'].iloc[0]),
        'partially_tracked': int(summary['partially_tracked'].iloc[0]),
        'mostly_lost': int(summary['mostly_lost'].iloc[0]),
        'num_matches': int(summary['num_matches'].iloc[0]),
        'num_transfer': int(summary['num_transfer'].iloc[0]),
        'num_ascend': int(summary['num_ascend'].iloc[0]),
        'num_migrate': int(summary['num_migrate'].iloc[0]),
    }
    
    # Calculate additional derived metrics
    if metrics_dict['num_unique_objects'] > 0:
        metrics_dict['mt_ratio'] = (metrics_dict['mostly_tracked'] / metrics_dict['num_unique_objects']) * 100
        metrics_dict['pt_ratio'] = (metrics_dict['partially_tracked'] / metrics_dict['num_unique_objects']) * 100
        metrics_dict['ml_ratio'] = (metrics_dict['mostly_lost'] / metrics_dict['num_unique_objects']) * 100
    else:
        metrics_dict['mt_ratio'] = 0.0
        metrics_dict['pt_ratio'] = 0.0
        metrics_dict['ml_ratio'] = 0.0
    
    if (metrics_dict['num_matches'] + metrics_dict['num_false_positives']) > 0:
        metrics_dict['fpr'] = (metrics_dict['num_false_positives'] / (metrics_dict['num_matches'] + metrics_dict['num_false_positives'])) * 100
    else:
        metrics_dict['fpr'] = 0.0
    
    if (metrics_dict['num_matches'] + metrics_dict['num_misses']) > 0:
        metrics_dict['fnr'] = (metrics_dict['num_misses'] / (metrics_dict['num_matches'] + metrics_dict['num_misses'])) * 100
    else:
        metrics_dict['fnr'] = 0.0
    
    # F1 Score
    if metrics_dict['precision'] + metrics_dict['recall'] > 0:
        metrics_dict['f1_score'] = (2 * metrics_dict['precision'] * metrics_dict['recall']) / (metrics_dict['precision'] + metrics_dict['recall'])
    else:
        metrics_dict['f1_score'] = 0.0
    
    # Compute HOTA using TrackEval if available
    hota_metrics = compute_hota_metrics(gt_file, pred_file, sequence_name)
    if hota_metrics:
        metrics_dict.update(hota_metrics)
    
    # Save metrics to txt file if output_dir provided
    if output_dir:
        # Save detailed txt
        metrics_file = Path(output_dir) / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write("="*90 + "\n")
            f.write(f"Tracking Evaluation Results: {sequence_name}\n")
            f.write("="*90 + "\n\n")
            
            f.write("PRIMARY TRACKING METRICS\n")
            f.write("="*90 + "\n")
            f.write(f"MOTA (Multi-Object Tracking Accuracy):        {metrics_dict['mota']:>8.2f}%\n")
            f.write(f"MOTP (Multi-Object Tracking Precision):       {metrics_dict['motp']:>8.4f}\n")
            f.write(f"IDF1 (ID F1 Score):                           {metrics_dict['idf1']:>8.2f}%\n")
            if 'hota' in metrics_dict:
                f.write(f"HOTA (Higher Order Tracking Accuracy):        {metrics_dict['hota']:>8.2f}%\n")
                f.write(f"DetA (Detection Accuracy):                    {metrics_dict['deta']:>8.2f}%\n")
                f.write(f"AssA (Association Accuracy):                  {metrics_dict['assa']:>8.2f}%\n")
            f.write(f"Precision:                                    {metrics_dict['precision']:>8.2f}%\n")
            f.write(f"Recall:                                       {metrics_dict['recall']:>8.2f}%\n")
            f.write(f"F1 Score:                                     {metrics_dict['f1_score']:>8.2f}%\n")
            f.write("\n")
            
            f.write("ID SWITCHING METRICS\n")
            f.write("="*90 + "\n")
            f.write(f"ID Switches (IDS):                            {metrics_dict['num_switches']:>8}\n")
            f.write(f"Fragmentations:                               {metrics_dict['num_fragmentations']:>8}\n")
            f.write(f"ID Transfers:                                 {metrics_dict['num_transfer']:>8}\n")
            f.write(f"ID Ascend:                                    {metrics_dict['num_ascend']:>8}\n")
            f.write(f"ID Migrate:                                   {metrics_dict['num_migrate']:>8}\n")
            f.write("\n")
            
            f.write("DETECTION METRICS\n")
            f.write("="*90 + "\n")
            f.write(f"True Positives (TP / Matches):                {metrics_dict['num_matches']:>8}\n")
            f.write(f"False Positives (FP):                         {metrics_dict['num_false_positives']:>8}\n")
            f.write(f"False Negatives (FN / Misses):                {metrics_dict['num_misses']:>8}\n")
            f.write(f"False Positive Rate (FPR):                    {metrics_dict['fpr']:>8.2f}%\n")
            f.write(f"False Negative Rate (FNR):                    {metrics_dict['fnr']:>8.2f}%\n")
            f.write("\n")
            
            f.write("TRAJECTORY QUALITY\n")
            f.write("="*90 + "\n")
            f.write(f"Mostly Tracked (MT):                          {metrics_dict['mostly_tracked']:>8} ({metrics_dict['mt_ratio']:>6.2f}%)\n")
            f.write(f"Partially Tracked (PT):                       {metrics_dict['partially_tracked']:>8} ({metrics_dict['pt_ratio']:>6.2f}%)\n")
            f.write(f"Mostly Lost (ML):                             {metrics_dict['mostly_lost']:>8} ({metrics_dict['ml_ratio']:>6.2f}%)\n")
            f.write("\n")
            
            f.write("COUNT STATISTICS\n")
            f.write("="*90 + "\n")
            f.write(f"Total Frames:                                 {metrics_dict['num_frames']:>8}\n")
            f.write(f"Total Ground Truth Objects:                   {metrics_dict['num_objects']:>8}\n")
            f.write(f"Total Predictions:                            {metrics_dict['num_predictions']:>8}\n")
            f.write(f"Total Detections:                             {metrics_dict['num_detections']:>8}\n")
            f.write(f"Unique Ground Truth IDs:                      {metrics_dict['num_unique_objects']:>8}\n")
            
            if 'hota' in metrics_dict and metrics_dict.get('hota_localization'):
                f.write("\n")
                f.write("HOTA COMPONENTS (if available)\n")
                f.write("="*90 + "\n")
                f.write(f"HOTA Localization:                            {metrics_dict['hota_localization']:>8.2f}%\n")
                f.write(f"HOTA Detection:                               {metrics_dict['hota_detection']:>8.2f}%\n")
                f.write(f"HOTA Association:                             {metrics_dict['hota_association']:>8.2f}%\n")
            
            f.write("\n" + "="*90 + "\n")
            
            # Add summary line at bottom
            f.write(f"\nSUMMARY: MOTA={metrics_dict['mota']:.2f}% | IDF1={metrics_dict['idf1']:.2f}% | ")
            if 'hota' in metrics_dict:
                f.write(f"HOTA={metrics_dict['hota']:.2f}% | ")
            f.write(f"IDS={metrics_dict['num_switches']} | Prec={metrics_dict['precision']:.2f}% | Rec={metrics_dict['recall']:.2f}%\n")
        
        # Also save as JSON for easy parsing
        json_file = Path(output_dir) / 'metrics.json'
        with open(json_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"[INFO] Metrics saved to: {metrics_file}")
        print(f"[INFO] JSON metrics saved to: {json_file}")
    
    return metrics_dict


def compute_hota_metrics(gt_file, pred_file, sequence_name):
    """
    Compute HOTA metrics using TrackEval library (accurate implementation)

    Args:
        gt_file: Path to ground truth file
        pred_file: Path to prediction file
        sequence_name: Name of the sequence

    Returns:
        hota_dict: Dictionary with HOTA metrics
    """
    try:
        import trackeval
        import pandas as pd
        import tempfile
        import os

        # Create temporary directory structure for TrackEval
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            gt_dir = os.path.join(temp_dir, 'gt')
            pred_dir = os.path.join(temp_dir, 'pred')

            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            # Copy GT file
            import shutil
            shutil.copy2(gt_file, os.path.join(gt_dir, f'{sequence_name}.txt'))

            # Copy prediction file
            shutil.copy2(pred_file, os.path.join(pred_dir, f'{sequence_name}.txt'))

            # Configure TrackEval for HOTA
            eval_config = {
                'USE_PARALLEL': False,
                'NUM_PARALLEL_CORES': 1,
                'BREAK_ON_ERROR': True,
                'RETURN_ON_ERROR': False,
                'LOG_ON_ERROR': '/dev/null',
                'PRINT_RESULTS': False,
                'PRINT_ONLY_COMBINED': False,
                'PRINT_CONFIG': False,
                'TIME_PROGRESS': False,
                'DISPLAY_LESS_PROGRESS': True,
                'OUTPUT_SUMMARY': False,
                'OUTPUT_EMPTY_CLASSES': True,
                'OUTPUT_DETAILED': False,
                'PLOT_CURVES': False,
            }

            dataset_config = {
                'GT_FOLDER': gt_dir,
                'TRACKERS_FOLDER': pred_dir,
                'OUTPUT_FOLDER': None,
                'TRACKERS_TO_EVAL': ['pred'],  # Use 'pred' as tracker name
                'CLASSES_TO_EVAL': ['pedestrian'],
                'BENCHMARK': 'MOT17',
                'SPLIT_TO_EVAL': 'train',
                'INPUT_AS_ZIP': False,
                'PRINT_CONFIG': False,
                'DO_PREPROC': True,
                'TRACKER_SUB_FOLDER': '',
                'OUTPUT_SUB_FOLDER': '',
                'TRACKER_DISPLAY_NAMES': None,
                'SEQMAP_FOLDER': None,
                'SEQMAP_FILE': None,
                'SEQ_INFO': None,
                'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                'SKIP_SPLIT_FOL': True,
            }

            metrics_config = {'METRICS': ['HOTA'], 'THRESHOLD': 0.5}

            # Initialize evaluator
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = [trackeval.metrics.HOTA(metrics_config)]

            # Run evaluation
            output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

            # Extract HOTA results
            if output_res and 'MotChallenge2DBox' in output_res:
                data = output_res['MotChallenge2DBox']
                if 'pred' in data and 'HOTA' in data['pred']:
                    hota_data = data['pred']['HOTA']['pedestrian']['COMBINED_SEQ']

                    return {
                        'hota': hota_data['HOTA'] * 100,
                        'deta': hota_data['DetA'] * 100,
                        'assa': hota_data['AssA'] * 100,
                        'hota_localization': hota_data['LocA'] * 100,
                        'hota_detection': hota_data['DetA'] * 100,
                        'hota_association': hota_data['AssA'] * 100,
                    }

    except Exception as e:
        print(f"[WARNING] TrackEval HOTA computation failed: {e}")
        print("[INFO] Falling back to simplified HOTA calculation")

    # Fallback to simplified implementation if TrackEval fails
    try:
        import pandas as pd
        from scipy.optimize import linear_sum_assignment
        from collections import defaultdict

        # Load data
        gt_df = pd.read_csv(gt_file, header=None,
                           names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'z'])
        pred_df = pd.read_csv(pred_file, header=None,
                             names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'z'])

        # Filter GT (only pedestrians with confidence >= 1)
        gt_df = gt_df[gt_df['conf'] >= 1].copy()

        def compute_iou(box1, box2):
            """Compute IoU between two boxes [x, y, w, h]"""
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            x1_max, y1_max = x1 + w1, y1 + h1
            x2_max, y2_max = x2 + w2, y2 + h2

            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area if union_area > 0 else 0

        # Build global GT-Pred associations across all frames
        gt_to_pred_matches = defaultdict(lambda: defaultdict(int))

        frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for frame_id in frames:
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            pred_frame = pred_df[pred_df['frame'] == frame_id]

            if len(gt_frame) == 0 or len(pred_frame) == 0:
                total_fn += len(gt_frame)
                total_fp += len(pred_frame)
                continue

            # Compute IoU matrix
            n_gt = len(gt_frame)
            n_pred = len(pred_frame)

            iou_matrix = np.zeros((n_gt, n_pred))
            gt_ids = []
            pred_ids = []

            for i, (_, gt_row) in enumerate(gt_frame.iterrows()):
                gt_ids.append(gt_row['id'])
                for j, (_, pred_row) in enumerate(pred_frame.iterrows()):
                    if i == 0:
                        pred_ids.append(pred_row['id'])
                    iou = compute_iou(
                        [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']],
                        [pred_row['x'], pred_row['y'], pred_row['w'], pred_row['h']]
                    )
                    iou_matrix[i, j] = iou

            # Match using Hungarian algorithm
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Count matches at IoU >= 0.5 threshold
            threshold = 0.5
            matched_gt = set()
            matched_pred = set()

            for gt_idx, pred_idx in zip(row_ind, col_ind):
                if iou_matrix[gt_idx, pred_idx] >= threshold:
                    total_tp += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)

                    gt_id = gt_ids[gt_idx]
                    pred_id = pred_ids[pred_idx]
                    gt_to_pred_matches[gt_id][pred_id] += 1

            total_fn += (n_gt - len(matched_gt))
            total_fp += (n_pred - len(matched_pred))

        # Calculate DetA
        deta = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

        # Calculate AssA
        total_associations = 0
        correct_associations = 0

        for gt_id, pred_matches in gt_to_pred_matches.items():
            if pred_matches:
                total_matches_for_gt = sum(pred_matches.values())
                best_pred_id = max(pred_matches, key=pred_matches.get)
                best_matches = pred_matches[best_pred_id]

                total_associations += total_matches_for_gt
                correct_associations += best_matches

        assa = correct_associations / total_associations if total_associations > 0 else 0
        hota = np.sqrt(deta * assa) if (deta > 0 and assa > 0) else 0

        return {
            'hota': hota * 100,
            'deta': deta * 100,
            'assa': assa * 100,
            'hota_localization': deta * 100,
            'hota_detection': deta * 100,
            'hota_association': assa * 100,
        }

    except Exception as e:
        print(f"[WARNING] Could not compute HOTA: {e}")
        return None


def process_mot_dataset(mot_dir, output_dir, yolo_weights='yolov5s.pt',
                       reid_model='models/mars-small128.pb', device='', 
                       save_video=True, evaluate=True):
    """
    Process entire MOT dataset (only sequences with ground truth)
    
    Args:
        mot_dir: Path to MOT dataset directory (e.g., data/MOT17/train)
        output_dir: Output directory for results
        yolo_weights: Path to YOLO weights
        reid_model: Path to Re-ID model
        device: Device to run on
        save_video: Whether to save output videos
        evaluate: Whether to evaluate against ground truth
    """
    mot_dir = Path(mot_dir)
    
    if not mot_dir.exists():
        print(f"[ERROR] MOT directory not found: {mot_dir}")
        return
    
    # Get all sequences with ground truth
    sequences = []
    for seq_dir in sorted(mot_dir.iterdir()):
        if seq_dir.is_dir():
            gt_file = seq_dir / 'gt' / 'gt.txt'
            if gt_file.exists():
                sequences.append(seq_dir)
    
    if len(sequences) == 0:
        print(f"[ERROR] No sequences with ground truth found in {mot_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing {len(sequences)} sequences from {mot_dir}")
    print(f"{'='*80}")
    
    all_metrics = []
    
    for seq_path in sequences:
        # Process sequence
        result_file = process_mot_sequence(
            str(seq_path), output_dir, yolo_weights, reid_model, device, save_video
        )
        
        # Evaluate if requested
        if evaluate and result_file:
            gt_file = seq_path / 'gt' / 'gt.txt'
            if gt_file.exists():
                print(f"[INFO] Evaluating {seq_path.name}...")
                seq_output_dir = Path(output_dir) / seq_path.name
                metrics = evaluate_mot_sequence(str(gt_file), str(result_file), seq_path.name, str(seq_output_dir))
                
                if metrics:
                    all_metrics.append(metrics)
                    print(f"\n[RESULTS] {seq_path.name}:")
                    print(f"  MOTA: {metrics['mota']:.2f}%")
                    if metrics.get('hota') is not None:
                        print(f"  HOTA: {metrics['hota']:.2f}% (DetA: {metrics['deta']:.2f}%, AssA: {metrics['assa']:.2f}%)")
                    print(f"  IDF1: {metrics['idf1']:.2f}%")
                    print(f"  IDS:  {metrics['num_switches']}")
                    print(f"  Precision: {metrics['precision']:.2f}%")
                    print(f"  Recall: {metrics['recall']:.2f}%")
                    print(f"  MT/PT/ML: {metrics['mostly_tracked']}/{metrics['partially_tracked']}/{metrics['mostly_lost']}")
    
    # Save summary
    if all_metrics:
        summary_file = Path(output_dir) / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write(f"YOLO + DeepSORT Tracking Evaluation Results\n")
            f.write("="*100 + "\n\n")
            
            # Header
            f.write(f"{'Sequence':<25} {'MOTA':>8} {'HOTA':>8} {'IDF1':>8} {'DetA':>8} {'AssA':>8} {'IDS':>6} {'Prec':>8} {'Recall':>8} {'MT':>5} {'PT':>5} {'ML':>5}\n")
            f.write("-"*130 + "\n")
            
            # Individual results
            for m in all_metrics:
                hota_str = f"{m['hota']:>7.2f}%" if m.get('hota') is not None else "    N/A"
                deta_str = f"{m['deta']:>7.2f}%" if m.get('deta') is not None else "    N/A"
                assa_str = f"{m['assa']:>7.2f}%" if m.get('assa') is not None else "    N/A"
                
                f.write(f"{m['sequence']:<25} {m['mota']:>7.2f}% {hota_str} {m['idf1']:>7.2f}% {deta_str} {assa_str} "
                       f"{m['num_switches']:>6} {m['precision']:>7.2f}% {m['recall']:>7.2f}% "
                       f"{m['mostly_tracked']:>5} {m['partially_tracked']:>5} {m['mostly_lost']:>5}\n")
            
            # Average
            f.write("-"*130 + "\n")
            avg_mota = np.mean([m['mota'] for m in all_metrics])
            avg_idf1 = np.mean([m['idf1'] for m in all_metrics])
            
            # Calculate HOTA averages (only from sequences that have HOTA)
            hota_values = [m['hota'] for m in all_metrics if m.get('hota') is not None]
            deta_values = [m['deta'] for m in all_metrics if m.get('deta') is not None]
            assa_values = [m['assa'] for m in all_metrics if m.get('assa') is not None]
            
            avg_hota = np.mean(hota_values) if hota_values else None
            avg_deta = np.mean(deta_values) if deta_values else None
            avg_assa = np.mean(assa_values) if assa_values else None
            
            total_ids = sum([m['num_switches'] for m in all_metrics])
            avg_prec = np.mean([m['precision'] for m in all_metrics])
            avg_recall = np.mean([m['recall'] for m in all_metrics])
            total_mt = sum([m['mostly_tracked'] for m in all_metrics])
            total_pt = sum([m['partially_tracked'] for m in all_metrics])
            total_ml = sum([m['mostly_lost'] for m in all_metrics])
            total_fp = sum([m['num_false_positives'] for m in all_metrics])
            total_fn = sum([m['num_misses'] for m in all_metrics])
            
            hota_str = f"{avg_hota:>7.2f}%" if avg_hota is not None else "    N/A"
            deta_str = f"{avg_deta:>7.2f}%" if avg_deta is not None else "    N/A"
            assa_str = f"{avg_assa:>7.2f}%" if avg_assa is not None else "    N/A"
            
            f.write(f"{'AVERAGE/TOTAL':<25} {avg_mota:>7.2f}% {hota_str} {avg_idf1:>7.2f}% {deta_str} {assa_str} "
                   f"{total_ids:>6} {avg_prec:>7.2f}% {avg_recall:>7.2f}% "
                   f"{total_mt:>5} {total_pt:>5} {total_ml:>5}\n")
            f.write("="*130 + "\n")
            
            # Detailed metrics
            f.write("\nDETAILED METRICS PER SEQUENCE\n")
            f.write("="*120 + "\n")
            for m in all_metrics:
                f.write(f"\n{m['sequence']}\n")
                f.write("-"*90 + "\n")
                
                f.write(f"  Primary Tracking Metrics:\n")
                f.write(f"    MOTA:                    {m['mota']:>8.2f}%\n")
                f.write(f"    MOTP:                    {m['motp']:>8.4f}\n")
                f.write(f"    IDF1:                    {m['idf1']:>8.2f}%\n")
                if m.get('hota') is not None:
                    f.write(f"    HOTA:                    {m['hota']:>8.2f}%\n")
                    f.write(f"    DetA:                    {m['deta']:>8.2f}%\n")
                    f.write(f"    AssA:                    {m['assa']:>8.2f}%\n")
                f.write(f"    Precision:               {m['precision']:>8.2f}%\n")
                f.write(f"    Recall:                  {m['recall']:>8.2f}%\n")
                f.write(f"    F1 Score:                {m['f1_score']:>8.2f}%\n\n")
                
                f.write(f"  ID Metrics:\n")
                f.write(f"    ID Switches (IDS):       {m['num_switches']:>8}\n")
                f.write(f"    Fragmentations:          {m['num_fragmentations']:>8}\n")
                f.write(f"    ID Transfers:            {m['num_transfer']:>8}\n")
                f.write(f"    ID Ascend:               {m['num_ascend']:>8}\n")
                f.write(f"    ID Migrate:              {m['num_migrate']:>8}\n\n")
                
                f.write(f"  Detection Performance:\n")
                f.write(f"    True Positives (TP):     {m['num_matches']:>8}\n")
                f.write(f"    False Positives (FP):    {m['num_false_positives']:>8}\n")
                f.write(f"    False Negatives (FN):    {m['num_misses']:>8}\n")
                f.write(f"    False Positive Rate:     {m['fpr']:>8.2f}%\n")
                f.write(f"    False Negative Rate:     {m['fnr']:>8.2f}%\n\n")
                
                f.write(f"  Trajectory Quality:\n")
                f.write(f"    Mostly Tracked (MT):     {m['mostly_tracked']:>8} ({m['mt_ratio']:>6.2f}%)\n")
                f.write(f"    Partially Tracked (PT):  {m['partially_tracked']:>8} ({m['pt_ratio']:>6.2f}%)\n")
                f.write(f"    Mostly Lost (ML):        {m['mostly_lost']:>8} ({m['ml_ratio']:>6.2f}%)\n\n")
                
                f.write(f"  Statistics:\n")
                f.write(f"    Frames:                  {m['num_frames']:>8}\n")
                f.write(f"    GT Objects:              {m['num_objects']:>8}\n")
                f.write(f"    Predictions:             {m['num_predictions']:>8}\n")
                f.write(f"    Detections:              {m['num_detections']:>8}\n")
                f.write(f"    Unique GT IDs:           {m['num_unique_objects']:>8}\n")
        
        # Calculate average F1
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        avg_fpr = np.mean([m['fpr'] for m in all_metrics])
        avg_fnr = np.mean([m['fnr'] for m in all_metrics])
        
        # Add average metrics to summary
        f.write(f"\nAVERAGE METRICS ACROSS ALL SEQUENCES\n")
        f.write("="*130 + "\n")
        f.write(f"Average MOTA:      {avg_mota:>8.2f}%\n")
        if avg_hota is not None:
            f.write(f"Average HOTA:      {avg_hota:>8.2f}%\n")
            f.write(f"Average DetA:      {avg_deta:>8.2f}%\n")
            f.write(f"Average AssA:      {avg_assa:>8.2f}%\n")
        f.write(f"Average IDF1:      {avg_idf1:>8.2f}%\n")
        f.write(f"Average Precision: {avg_prec:>8.2f}%\n")
        f.write(f"Average Recall:    {avg_recall:>8.2f}%\n")
        f.write(f"Average F1 Score:  {avg_f1:>8.2f}%\n")
        f.write(f"Average FPR:       {avg_fpr:>8.2f}%\n")
        f.write(f"Average FNR:       {avg_fnr:>8.2f}%\n")
        f.write(f"\nTotal IDS:         {total_ids:>8}\n")
        f.write(f"Total MT/PT/ML:    {total_mt:>8} / {total_pt:>8} / {total_ml:>8}\n")
        f.write(f"Total FP/FN:       {total_fp:>8} / {total_fn:>8}\n")
        f.write("="*130 + "\n")
        
        print(f"\n{'='*80}")
        print(f"[SUMMARY] Average Results Across All Sequences:")
        print(f"  MOTA: {avg_mota:.2f}%")
        if avg_hota is not None:
            print(f"  HOTA: {avg_hota:.2f}% (DetA: {avg_deta:.2f}%, AssA: {avg_assa:.2f}%)")
        print(f"  IDF1: {avg_idf1:.2f}%")
        print(f"  F1 Score: {avg_f1:.2f}%")
        print(f"  Total IDS: {total_ids}")
        print(f"  Precision: {avg_prec:.2f}% | Recall: {avg_recall:.2f}%")
        print(f"  Total MT/PT/ML: {total_mt}/{total_pt}/{total_ml}")
        print(f"\n[INFO] Summary saved to: {summary_file}")
        print(f"[INFO] Individual metrics saved to: <sequence>/metrics.txt and metrics.json")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO + DeepSORT Tracking')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input video, MOT sequence, or MOT dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='YOLO weights')
    parser.add_argument('--reid-model', type=str, default='models/mars-small128.pb', 
                       help='Re-ID model for feature extraction')
    parser.add_argument('--device', type=str, default='', help='Device (cuda:0 or cpu)')
    parser.add_argument('--mode', type=str, default='dataset', 
                       choices=['video', 'sequence', 'dataset'], 
                       help='Processing mode: video, single sequence, or entire dataset')
    parser.add_argument('--save-video', action='store_true', default=True,
                       help='Save output video')
    parser.add_argument('--no-video', dest='save_video', action='store_false',
                       help='Do not save output video')
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Evaluate against ground truth (for MOT sequences)')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false',
                       help='Do not evaluate')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        process_video(args.input, args.output, args.weights, args.reid_model, args.device)
    elif args.mode == 'sequence':
        process_mot_sequence(args.input, args.output, args.weights, args.reid_model, 
                           args.device, args.save_video)
    else:  # dataset mode
        process_mot_dataset(args.input, args.output, args.weights, args.reid_model, 
                          args.device, args.save_video, args.evaluate)

