"""XGBoost prediction script for entity matching.

Loads a trained model and generates predictions on new candidate pairs.
Outputs probabilities for threshold tuning and final assignment.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import xgboost as xgb
import numpy as np
import pandas as pd

from .config import XGBoostConfig, get_config
from .features import compute_pair_features, compute_group_relative_features


def load_model(model_path: str) -> xgb.Booster:
    """Load XGBoost model from file."""
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def load_model_pkl(model_pkl_path: str) -> Dict:
    """Load model pickle with metadata."""
    import pickle
    with open(model_pkl_path, 'rb') as f:
        return pickle.load(f)


def predict_probabilities(
    model: xgb.Booster,
    features: np.ndarray,
    feature_names: list,
    batch_size: int = 100000,
) -> np.ndarray:
    """Predict probabilities in batches to manage memory."""
    n_samples = features.shape[0]
    probs = np.zeros(n_samples, dtype=np.float32)

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        dtest = xgb.DMatrix(features[i:end], feature_names=feature_names)
        probs[i:end] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    return probs


def load_test_data(config: XGBoostConfig) -> pd.DataFrame:
    """Load test source data."""
    data_root = config.resolve_path(config.data_root)
    test_dir = os.path.join(data_root, 'test')

    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    print(f"Loading test source data...")
    s1 = pd.read_csv(s1_path, sep='\t', dtype=str, keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str, keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str, keep_default_na=False)

    print(f"  S1 rows: {len(s1):,}")
    print(f"  S2 rows: {len(s2):,}")
    print(f"  S3 rows: {len(s3):,}")

    return s1, s2, s3


def load_candidates(config: XGBoostConfig) -> pd.DataFrame:
    """Load candidate pairs from blocking output."""
    cand_path = config.resolve_path(config.candidates_path)
    print(f"Loading candidates: {cand_path}")

    cand = pd.read_csv(cand_path, sep='\t', dtype=str, keep_default_na=False)

    # Rename columns for consistency
    if 'candidate_entity_id' in cand.columns:
        cand = cand.rename(columns={'candidate_entity_id': 'cand_entity_id'})
    if 'source' in cand.columns:
        cand = cand.rename(columns={'source': 'cand_source'})
    if 'score' in cand.columns:
        cand['blocker_score'] = cand['score'].astype(float)

    print(f"  Candidate pairs: {len(cand):,}")
    print(f"  Unique S1 entities: {cand['s1_entity_id'].nunique():,}")
    print(f"  Unique candidate entities: {cand['cand_entity_id'].nunique():,}")

    return cand


def join_test_data(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> pd.DataFrame:
    """Join candidate pairs with test source data."""
    print("Joining with test source data...")

    # Prepare S1 data
    s1_cols = ['entity_id', 'business_name', 'business_address', 'country']
    s1_small = s1[s1_cols].rename(columns={
        'entity_id': 's1_entity_id',
        'business_name': 's1_name',
        'business_address': 's1_addr',
        'country': 's1_country'
    })

    # Prepare S2/S3 combined
    s2_cols = ['entity_id', 'business_name', 'business_address', 'country']
    s2_small = s2[s2_cols].rename(columns={
        'entity_id': 'cand_entity_id',
        'business_name': 'cand_name',
        'business_address': 'cand_addr',
        'country': 'cand_country'
    })

    s3_small = s3[s2_cols].rename(columns={
        'entity_id': 'cand_entity_id',
        'business_name': 'cand_name',
        'business_address': 'cand_addr',
        'country': 'cand_country'
    })

    cand_combined = pd.concat([s2_small, s3_small], ignore_index=True)

    # Join
    df = candidates.merge(s1_small, on='s1_entity_id', how='left')
    df = df.merge(cand_combined, on='cand_entity_id', how='left')

    # Check for missing joins
    missing_s1 = df['s1_name'].isna().sum()
    missing_cand = df['cand_name'].isna().sum()
    if missing_s1 > 0:
        print(f"  WARNING: {missing_s1} candidates missing S1 data")
    if missing_cand > 0:
        print(f"  WARNING: {missing_cand} candidates missing candidate data")

    print(f"  Joined rows: {len(df):,}")
    return df


def compute_features_for_df(df: pd.DataFrame, config: XGBoostConfig) -> tuple:
    """Compute features for all pairs in the dataframe."""
    print("Computing pairwise features...")

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
    feature_names = list(feature_df.columns)

    # Ensure all expected features are present (fill missing with NaN)
    # This is important for test-time consistency
    feature_array = feature_df.values.astype(np.float32)

    print(f"  Feature matrix shape: {feature_array.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  NaN count: {np.isnan(feature_array).sum():,}")

    return feature_array, feature_names


def save_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    config: XGBoostConfig,
    output_path: Optional[str] = None,
):
    """Save predictions to CSV."""
    pred_df = df[['s1_entity_id', 'cand_entity_id']].copy()
    pred_df['y_prob'] = y_prob

    if output_path is None:
        output_path = os.path.join(config.output_dir, 'test_predictions.csv')

    pred_df.to_csv(output_path, index=False)
    print(f"Saved test predictions to {output_path}")


def save_wide_format(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    config: XGBoostConfig,
    output_path: Optional[str] = None,
):
    """Save predictions in wide format (one row per S1 entity)."""
    df = df.copy()
    df['y_prob'] = y_prob
    df['y_pred'] = (y_prob >= threshold).astype(int)

    # Group by S1 entity and collect predicted matches
    results = []
    for s1_id, group in df.groupby('s1_entity_id'):
        matched = group[group['y_pred'] == 1]['cand_entity_id'].tolist()
        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(matched) if matched else ''
        })

    result_df = pd.DataFrame(results)

    if output_path is None:
        output_path = os.path.join(config.output_dir, 'matching_results.tsv')

    result_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    print(f"Saved wide-format results to {output_path}")


def main(config: Optional[XGBoostConfig] = None, **overrides):
    """Main prediction pipeline."""
    if config is None:
        config = get_config(**overrides)

    print(f"Experiment: {config.experiment_name}")

    # Load model
    model_path = overrides.get('model_path', config.model_path)
    print(f"Loading model from: {model_path}")

    # Try loading pickle first to get feature names
    model_pkl_path = overrides.get('model_pkl_path', config.model_pkl_path)
    if os.path.exists(model_pkl_path):
        model_data = load_model_pkl(model_pkl_path)
        model = model_data['model']
        feature_names = model_data['feature_names']
        saved_config = model_data['config']
        print(f"Loaded model from pickle (trained at {model_data.get('timestamp', 'unknown')})")
        print(f"Model features: {len(feature_names)}")
    else:
        model = load_model(model_path)
        feature_names = model.feature_names
        print(f"Loaded model from native format")
        print(f"Model features: {len(feature_names)}")

    # Load best threshold if available
    best_threshold = 0.5
    threshold_path = overrides.get('threshold_path', config.best_threshold_path)
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        best_threshold = threshold_data['best_threshold']
        print(f"Loaded best threshold: {best_threshold:.4f}")
    else:
        print(f"No threshold file found, using default: {best_threshold}")

    # Determine if test or validation
    split = overrides.get('split', 'validation')

    if split == 'test':
        # Load test data
        s1, s2, s3 = load_test_data(config)
        candidates = load_candidates(config)
        df = join_test_data(candidates, s1, s2, s3)
    else:
        # For validation, we need to load the same split
        # Load full data and recreate the split
        from .dataset import load_source_data, load_ground_truth, load_candidates as load_cands
        from .dataset import generate_labels, join_source_data, split_by_s1_entity

        s1_train, s2_train, s3_train = load_source_data(config)
        ground_truth = load_ground_truth(config)
        candidates = load_cands(config)
        candidates = generate_labels(candidates, ground_truth)
        df = join_source_data(candidates, s1_train, s2_train, s3_train)

        # Compute group-relative features
        if 'blocker_score' in df.columns:
            df = compute_group_relative_features(df, 'blocker_score', config)

        # Split
        np.random.seed(config.val_split_seed)
        s1_entities = df['s1_entity_id'].unique()
        n_val = int(len(s1_entities) * config.val_split_ratio)
        val_entities = set(np.random.choice(s1_entities, size=n_val, replace=False))
        is_val = df['s1_entity_id'].isin(val_entities).values
        df = df[is_val]

    # Compute features
    features, computed_feature_names = compute_features_for_df(df, config)

    # Ensure feature names match (critical for XGBoost)
    if len(computed_feature_names) != len(feature_names):
        print(f"WARNING: Feature count mismatch! Model: {len(feature_names)}, Computed: {len(computed_feature_names)}")
        # Align features - add missing columns with NaN
        missing = set(feature_names) - set(computed_feature_names)
        extra = set(computed_feature_names) - set(feature_names)
        if missing:
            print(f"  Missing features: {missing}")
        if extra:
            print(f"  Extra features: {extra}")

    # Predict
    pred_start = time.time()
    probs = predict_probabilities(model, features, feature_names)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f} seconds")

    # Save predictions
    if split == 'test':
        save_predictions(df, probs, config)
        save_wide_format(df, probs, best_threshold, config)
    else:
        pred_path = os.path.join(config.output_dir, 'validation_predictions.csv')
        save_predictions(df, probs, config, pred_path)

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Samples: {len(probs):,}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Predicted positive: {(probs >= best_threshold).sum():,}")
    print(f"Prediction time: {pred_time:.2f}s")
    print("=" * 60)

    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with XGBoost for entity matching')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--model-pkl-path', type=str, help='Path to model pickle')
    parser.add_argument('--threshold-path', type=str, help='Path to best threshold JSON')
    parser.add_argument('--candidates', type=str, help='Path to candidate_pairs.tsv')
    parser.add_argument('--data-root', type=str, help='Root path for dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--split', type=str, choices=['validation', 'test'], default='validation', help='Data split to predict on')
    parser.add_argument('--seed', type=int, default=20260926, help='Random seed')

    args = parser.parse_args()

    overrides = {}
    if args.model_path:
        overrides['model_path'] = args.model_path
    if args.model_pkl_path:
        overrides['model_pkl_path'] = args.model_pkl_path
    if args.threshold_path:
        overrides['threshold_path'] = args.threshold_path
    if args.candidates:
        overrides['candidates_path'] = args.candidates
    if args.data_root:
        overrides['data_root'] = args.data_root
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.split:
        overrides['split'] = args.split
    if args.seed:
        overrides['seed'] = args.seed
        overrides['val_split_seed'] = args.seed

    main(**overrides)