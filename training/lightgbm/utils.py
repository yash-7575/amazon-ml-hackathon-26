"""Utility functions for LightGBM entity matching pipeline."""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


def set_seed(seed: int = 20260926):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # LightGBM/XGBoost use their own random_state parameter
    # but we can also set environment variables
    os.environ['LIGHTGBM_SEED'] = str(seed)


def get_git_info() -> Dict[str, str]:
    """Get git commit hash and branch for reproducibility."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {'commit': commit, 'branch': branch}
    except Exception:
        return {'commit': 'unknown', 'branch': 'unknown'}


def save_experiment_config(config: Any, path: str):
    """Save experiment configuration to JSON."""
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        if hasattr(obj, '__dataclass_fields__'):  # dataclass
            return convert(obj.__dict__)
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(config), f, indent=2)


def load_experiment_config(path: str) -> Dict:
    """Load experiment configuration from JSON."""
    with open(path) as f:
        return json.load(f)


def print_dataset_summary(df: pd.DataFrame, label_col: str = 'label'):
    """Print summary statistics for a dataset."""
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    if label_col in df.columns:
        pos = (df[label_col] == 1).sum()
        neg = (df[label_col] == 0).sum()
        print(f"Positive: {pos:,} ({pos/len(df)*100:.4f}%)")
        print(f"Negative: {neg:,} ({neg/len(df)*100:.4f}%)")
        print(f"Imbalance ratio: {neg/pos:.2f}:1" if pos > 0 else "Imbalance ratio: N/A")
    
    if 's1_entity_id' in df.columns:
        print(f"Unique S1 entities: {df['s1_entity_id'].nunique():,}")
        # Candidates per S1
        cand_per_s1 = df.groupby('s1_entity_id').size()
        print(f"Candidates per S1: mean={cand_per_s1.mean():.1f}, "
              f"median={cand_per_s1.median():.1f}, max={cand_per_s1.max()}")
    
    if 'cand_entity_id' in df.columns:
        print(f"Unique candidates: {df['cand_entity_id'].nunique():,}")
    
    if 's1_country' in df.columns:
        print(f"Countries: {df['s1_country'].nunique()}")
        for c, cnt in df['s1_country'].value_counts().items():
            print(f"  {c}: {cnt:,}")
    
    print("=" * 50)


def print_feature_summary(feature_names: List[str], features: np.ndarray):
    """Print summary of feature matrix."""
    print("=" * 50)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 50)
    print(f"Shape: {features.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"NaN count: {np.isnan(features).sum():,} ({np.isnan(features).mean()*100:.2f}%)")
    print(f"Inf count: {np.isinf(features).sum():,}")
    print(f"Value range: [{np.nanmin(features):.4f}, {np.nanmax(features):.4f}]")
    print("=" * 50)


def check_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_cols: List[str] = ['s1_entity_id', 'cand_entity_id'],
) -> Dict[str, Any]:
    """Check for data leakage between train and validation sets."""
    results = {}
    
    # Check S1 entity overlap
    train_s1 = set(train_df['s1_entity_id'].unique())
    val_s1 = set(val_df['s1_entity_id'].unique())
    s1_overlap = train_s1 & val_s1
    results['s1_overlap'] = len(s1_overlap)
    results['s1_overlap_pct'] = len(s1_overlap) / len(val_s1) * 100 if val_s1 else 0
    
    # Check candidate entity overlap
    train_cand = set(train_df['cand_entity_id'].unique())
    val_cand = set(val_df['cand_entity_id'].unique())
    cand_overlap = train_cand & val_cand
    results['cand_overlap'] = len(cand_overlap)
    results['cand_overlap_pct'] = len(cand_overlap) / len(val_cand) * 100 if val_cand else 0
    
    # Check exact pair overlap
    train_pairs = set(zip(train_df['s1_entity_id'], train_df['cand_entity_id']))
    val_pairs = set(zip(val_df['s1_entity_id'], val_df['cand_entity_id']))
    pair_overlap = train_pairs & val_pairs
    results['pair_overlap'] = len(pair_overlap)
    
    print("=" * 50)
    print("LEAKAGE CHECK")
    print("=" * 50)
    print(f"S1 entity overlap: {results['s1_overlap']:,} ({results['s1_overlap_pct']:.2f}%)")
    print(f"Candidate entity overlap: {results['cand_overlap']:,} ({results['cand_overlap_pct']:.2f}%)")
    print(f"Exact pair overlap: {results['pair_overlap']:,}")
    print("=" * 50)
    
    if results['s1_overlap'] > 0:
        print("WARNING: S1 entities appear in both train and validation!")
    if results['pair_overlap'] > 0:
        print("WARNING: Exact pairs appear in both train and validation!")
    
    return results


def memory_usage_mb(obj) -> float:
    """Estimate memory usage of an object in MB."""
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024 / 1024
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / 1024 / 1024
    elif isinstance(obj, list):
        return sum(memory_usage_mb(x) for x in obj)
    else:
        import sys
        return sys.getsizeof(obj) / 1024 / 1024


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.end = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"{self.name}: {format_time(self.elapsed)}")
    
    @property
    def elapsed(self) -> float:
        if self.end is None:
            return time.time() - self.start
        return self.end - self.start


import time  # for Timer