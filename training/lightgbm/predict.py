"""Prediction script for LightGBM entity matching.

Loads trained model and generates predictions on test data.
Outputs matching_results.tsv in competition format (wide, one row per S1 entity).
"""
from __future__ import annotations
import argparse
import os
import pickle
import time
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import LightGBMConfig, get_config
from .features import compute_pair_features, compute_group_relative_features
from .dataset import load_source_data, load_candidates, join_source_data


def load_model(model_path: str):
    """Load LightGBM model from native format or pickle."""
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        feature_names = model_data.get('feature_names', [])
        print(f"Loaded model from pickle: {model_path}")
        return model, feature_names
    else:
        model = lgb.Booster(model_file=model_path)
        print(f"Loaded model from native format: {model_path}")
        return model, None


def predict_probabilities(model, features: np.ndarray, best_iteration: int = -1, batch_size: int = 100000) -> np.ndarray:
    """Predict probabilities in batches."""
    n_samples = features.shape[0]
    probs = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        probs[i:end] = model.predict(features[i:end], num_iteration=best_iteration)
    
    return probs


def apply_threshold_and_format(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    output_path: str,
) -> pd.DataFrame:
    """Apply threshold and convert to wide format (one row per S1 entity)."""
    preds = (probs >= threshold).astype(int)
    
    df = df.copy()
    df['y_prob'] = probs
    df['y_pred'] = preds
    
    # Filter to predicted matches
    matches = df[df['y_pred'] == 1].copy()
    
    # Group by S1 entity and aggregate candidate IDs
    wide = matches.groupby('s1_entity_id')['cand_entity_id'].apply(
        lambda x: ','.join(sorted(x.unique()))
    ).reset_index()
    wide.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure ALL test S1 entities have a row (even if empty)
    # We need to load test S1 to get all entity IDs
    # For now, assume df contains all S1 entities from the candidate file
    all_s1 = df['s1_entity_id'].unique()
    all_s1_df = pd.DataFrame({'source1_entity_id': all_s1})
    wide = all_s1_df.merge(wide, on='source1_entity_id', how='left')
    wide['matched_entity_ids'] = wide['matched_entity_ids'].fillna('')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wide.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    
    print(f"Saved predictions to {output_path}")
    print(f"  Total S1 entities: {len(wide):,}")
    print(f"  Entities with matches: {(wide['matched_entity_ids'] != '').sum():,}")
    print(f"  Empty predictions: {(wide['matched_entity_ids'] == '').sum():,}")
    
    return wide


def predict_on_test(
    config: LightGBMConfig,
    model_path: str,
    threshold: float,
    output_path: str,
    test_split: str = 'test',
) -> pd.DataFrame:
    """Run full prediction pipeline on test data."""
    print(f"Loading model from {model_path}...")
    model, feature_names = load_model(model_path)
    
    print("Loading test source data...")
    # Temporarily switch config to test paths
    test_s1_path = config.train_s1_path.replace('train', test_split)
    test_s2_path = config.train_s2_path.replace('train', test_split)
    test_s3_path = config.train_s3_path.replace('train', test_split)
    
    s1 = pd.read_csv(config.resolve_path(test_s1_path), sep='\t', dtype=str, keep_default_na=False)
    s2 = pd.read_csv(config.resolve_path(test_s2_path), sep='\t', dtype=str, keep_default_na=False)
    s3 = pd.read_csv(config.resolve_path(test_s3_path), sep='\t', dtype=str, keep_default_na=False)
    
    print(f"  Test S1: {len(s1):,}")
    print(f"  Test S2: {len(s2):,}")
    print(f"  Test S3: {len(s3):,}")
    
    print("Loading test candidates...")
    test_cand_path = config.candidates_path.replace('train', test_split)
    if not os.path.exists(config.resolve_path(test_cand_path)):
        # Try default location
        test_cand_path = f'blocking/candidate_pairs_{test_split}.tsv'
    candidates = load_candidates(config)
    # Update paths for test
    candidates = pd.read_csv(config.resolve_path(test_cand_path), sep='\t', dtype=str, keep_default_na=False)
    if 'candidate_entity_id' in candidates.columns:
        candidates = candidates.rename(columns={'candidate_entity_id': 'cand_entity_id'})
    if 'source' in candidates.columns:
        candidates = candidates.rename(columns={'source': 'cand_source'})
    if 'score' in candidates.columns:
        candidates['blocker_score'] = candidates['score'].astype(float)
    
    print(f"  Test candidates: {len(candidates):,}")
    
    # Join with source data
    print("Joining with source data...")
    df = join_source_data(candidates, s1, s2, s3)
    
    # Compute group-relative features
    if 'blocker_score' in df.columns:
        df = compute_group_relative_features(df, 'blocker_score', config)
    
    # Compute pairwise features
    print("Computing features...")
    feature_dicts = []
    for i, row in df.iterrows():
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i:,} pairs...")
        feats = compute_pair_features(
            row['s1_name'], row['s1_addr'], row['s1_country'],
            row['cand_name'], row['cand_addr'], row['cand_country'],
            config
        )
        feature_dicts.append(feats)
    
    feature_df = pd.DataFrame(feature_dicts)
    # Align features with training
    if feature_names is not None:
        # Add missing columns as NaN, drop extra
        for fn in feature_names:
            if fn not in feature_df.columns:
                feature_df[fn] = np.nan
        feature_df = feature_df[feature_names]
    
    features = feature_df.values.astype(np.float32)
    
    # Predict
    print("Predicting...")
    pred_start = time.time()
    probs = predict_probabilities(model, features, model.best_iteration)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f}s")
    
    # Apply threshold and format
    print(f"Applying threshold: {threshold}")
    wide = apply_threshold_and_format(df, probs, threshold, output_path)
    
    return wide


def main(config: Optional[LightGBMConfig] = None, model_path: Optional[str] = None,
         threshold: Optional[float] = None, output_path: Optional[str] = None,
         test_split: str = 'test', **overrides):
    """Main prediction pipeline."""
    if config is None:
        config = get_config(**overrides)
    
    if model_path is None:
        model_path = config.model_pkl_path if os.path.exists(config.model_pkl_path) else config.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if threshold is None:
        # Try to load best threshold from validation metrics
        metrics_path = config.metrics_path
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            threshold = metrics.get('best_threshold', 0.5)
            print(f"Using threshold from metrics: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"No metrics found, using default threshold: {threshold}")
    
    if output_path is None:
        output_path = os.path.join(config.output_dir, f'matching_results_{test_split}.tsv')
    
    import json
    predict_on_test(config, model_path, threshold, output_path, test_split)
    
    print("\nPrediction complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with LightGBM model for entity matching')
    parser.add_argument('--model', type=str, help='Path to model file (.txt or .pkl)')
    parser.add_argument('--candidates', type=str, help='Path to test candidate_pairs.tsv')
    parser.add_argument('--data-root', type=str, help='Root path for dataset')
    parser.add_argument('--output', type=str, help='Output path for matching_results.tsv')
    parser.add_argument('--threshold', type=float, help='Classification threshold')
    parser.add_argument('--test-split', type=str, default='test', choices=['train', 'test'], help='Data split')
    
    args = parser.parse_args()
    
    overrides = {}
    if args.candidates:
        overrides['candidates_path'] = args.candidates
    if args.data_root:
        overrides['data_root'] = args.data_root
    
    main(model_path=args.model, threshold=args.threshold, output_path=args.output,
         test_split=args.test_split, **overrides)