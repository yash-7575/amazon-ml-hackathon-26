"""LightGBM Entity Matching Pipeline for AMLC-2026.

Stage-2 matcher: candidate_pairs.tsv → P(match) → macro-F0.5 evaluation.

Usage:
    from training.lightgbm import train, evaluate, predict
"""
from .config import LightGBMConfig, get_config
from .features import compute_pair_features, compute_group_relative_features
from .dataset import prepare_dataset, load_candidates, load_ground_truth
from .train import main as train_main
from .evaluate import main as evaluate_main
from .predict import main as predict_main
from .metrics import (
    macro_f05_per_entity,
    compute_all_metrics,
    evaluate_thresholds,
    find_best_threshold,
    top_k_accuracy,
    error_analysis,
)
from .utils import set_seed, check_leakage, Timer

__all__ = [
    'LightGBMConfig',
    'get_config',
    'compute_pair_features',
    'compute_group_relative_features',
    'prepare_dataset',
    'load_candidates',
    'load_ground_truth',
    'train_main',
    'evaluate_main',
    'predict_main',
    'macro_f05_per_entity',
    'compute_all_metrics',
    'evaluate_thresholds',
    'find_best_threshold',
    'top_k_accuracy',
    'error_analysis',
    'set_seed',
    'check_leakage',
    'Timer',
]

__version__ = '0.1.0'