#!/usr/bin/env python3
"""
Compute HOTA metrics using TrackEval library
Run this separately after tracking to get HOTA scores
"""

import os
import sys
import argparse
from pathlib import Path


def compute_hota_trackeval(tracker_dir, gt_dir, seqmap_file=None):
    """
    Compute HOTA using TrackEval library
    
    Args:
        tracker_dir: Directory containing tracker results
        gt_dir: Directory containing ground truth
        seqmap_file: Optional seqmap file listing sequences to evaluate
    """
    try:
        import trackeval
    except ImportError:
        print("[ERROR] TrackEval not installed!")
        print("[INFO] Install with: pip install git+https://github.com/JonathonLuiten/TrackEval.git")
        return None
    
    # Configure evaluator
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': os.devnull,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    
    # Dataset config for MOT
    dataset_config = {
        'GT_FOLDER': gt_dir,
        'TRACKERS_FOLDER': tracker_dir,
        'OUTPUT_FOLDER': None,  # Don't save TrackEval outputs
        'TRACKERS_TO_EVAL': None,  # Evaluate all
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'train',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': True,
        'DO_PREPROC': True,
        'TRACKER_SUB_FOLDER': '',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': None,
        'SEQMAP_FILE': seqmap_file,
        'SEQ_INFO': None,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': True,
    }
    
    # Metrics config
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    
    try:
        # Initialize evaluator
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        
        # Add metrics
        if 'HOTA' in metrics_config['METRICS']:
            metrics_list.append(trackeval.metrics.HOTA(metrics_config))
        if 'CLEAR' in metrics_config['METRICS']:
            metrics_list.append(trackeval.metrics.CLEAR(metrics_config))
        if 'Identity' in metrics_config['METRICS']:
            metrics_list.append(trackeval.metrics.Identity(metrics_config))
        
        # Run evaluation
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        
        return output_res
        
    except Exception as e:
        print(f"[ERROR] TrackEval error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute HOTA metrics using TrackEval')
    parser.add_argument('--tracker-dir', type=str, required=True, 
                       help='Directory containing tracker results')
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='Directory containing ground truth')
    parser.add_argument('--seqmap', type=str, default=None,
                       help='Seqmap file listing sequences')
    
    args = parser.parse_args()
    
    results = compute_hota_trackeval(args.tracker_dir, args.gt_dir, args.seqmap)
    
    if results:
        print("\n[SUCCESS] HOTA computation completed!")
        print("[INFO] Results saved by TrackEval")
    else:
        print("\n[ERROR] HOTA computation failed!")
        print("[INFO] Make sure TrackEval is installed:")
        print("  pip install git+https://github.com/JonathonLuiten/TrackEval.git")


if __name__ == "__main__":
    main()




