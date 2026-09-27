"""Evaluation script for trained LightGBM model.

Loads a trained model and evaluates on validation or test data.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time
from datetime import datetime
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import LightGBMConfig, get_config
from .dataset import prepare_dataset, load_source_data, load_ground_truth, load_candidates, generate_labels, join_source_data
from .features import compute_pair_features, compute_group_relative_features
from .metrics import (
    compute_all_metrics, evaluate_thresholds, find_best_threshold,
    top_k_accuracy, ranking_metrics, error_analysis, compute_feature_importance
)


def load_model(model_path: str):
    """Load LightGBM model from native format or pickle."""
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        feature_names = model_data.get('feature_names', [])
        config_dict = model_data.get('config', {})
        print(f"Loaded model from pickle: {model_path}")
        print(f"  Best iteration: {model_data.get('best_iteration', 'N/A')}")
        print(f"  Timestamp: {model_data.get('timestamp', 'N/A')}")
        return model, feature_names, config_dict
    else:
        model = lgb.Booster(model_file=model_path)
        print(f"Loaded model from native format: {model_path}")
        return model, None, {}


def predict_probabilities(model, features: np.ndarray, best_iteration: int = -1, batch_size: int = 100000) -> np.ndarray:
    """Predict probabilities in batches."""
    n_samples = features.shape[0]
    probs = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        probs[i:end] = model.predict(features[i:end], num_iteration=best_iteration)
    
    return probs


def evaluate_model(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    s1_entity_ids: np.ndarray,
    cand_entity_ids: np.ndarray,
    feature_names: list,
    config: LightGBMConfig,
    thresholds: Optional[list] = None,
) -> dict:
    """Evaluate model on given data."""
    if thresholds is None:
        thresholds = config.threshold_grid
    
    # Predict
    pred_start = time.time()
    probs = predict_probabilities(model, features, model.best_iteration)
    pred_time = time.time() - pred_start
    
    # Threshold sweep
    threshold_results = evaluate_thresholds(labels, probs, s1_entity_ids, thresholds)
    best_threshold, best_metrics = find_best_threshold(threshold_results, 'macro_f05')
    
    # Final predictions
    preds = (probs >= best_threshold).astype(int)
    
    # Full metrics
    metrics, per_entity_df = compute_all_metrics(labels, preds, probs, s1_entity_ids)
    
    # Top-K and ranking
    metrics['top_k_accuracy'] = top_k_accuracy(labels, probs, s1_entity_ids, config.top_k_values)
    metrics.update(ranking_metrics(labels, probs, s1_entity_ids))
    
    # Feature importance
    feature_importance = compute_feature_importance(model, feature_names, 'gain')
    
    # Error analysis
    errors_df = error_analysis(labels, preds, probs, s1_entity_ids, cand_entity_ids)
    
    results = {
        'metrics': metrics,
        'per_entity': per_entity_df,
        'threshold_results': threshold_results,
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'feature_importance': feature_importance,
        'errors': errors_df,
        'probs': probs,
        'preds': preds,
        'pred_time': pred_time,
    }
    
    return results


def save_evaluation_results(results: dict, config: LightGBMConfig, suffix: str = ''):
    """Save evaluation results to files."""
    # Metrics
    metrics_path = config.metrics_path.replace('.json', f'{suffix}.json')
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        if isinstance(obj, pd.DataFrame): return obj.to_dict('records')
        return obj
    
    with open(metrics_path, 'w') as f:
        json.dump(convert(results['metrics']), f, indent=2)
    
    # Threshold results
    thresh_path = config.threshold_results_path.replace('.csv', f'{suffix}.csv')
    results['threshold_results'].to_csv(thresh_path, index=False)
    
    # Feature importance
    fi_path = config.feature_importance_path.replace('.csv', f'{suffix}.csv')
    results['feature_importance'].to_csv(fi_path, index=False)
    
    # Error analysis
    err_path = config.error_analysis_path.replace('.csv', f'{suffix}.csv')
    results['errors'].to_csv(err_path, index=False)
    
    # Per-entity
    pe_path = config.error_analysis_path.replace('error_analysis', 'per_entity').replace('.csv', f'{suffix}.csv')
    results['per_entity'].to_csv(pe_path, index=False)
    
    print(f"Saved evaluation results with suffix '{suffix}'")


def main(config: Optional[LightGBMConfig] = None, model_path: Optional[str] = None, **overrides):
    """Main evaluation pipeline."""
    if config is None:
        config = get_config(**overrides)
    
    # Determine model path
    if model_path is None:
        model_path = config.model_pkl_path if os.path.exists(config.model_pkl_path) else config.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model, saved_feature_names, saved_config = load_model(model_path)
    
    # Prepare dataset (or load existing split)
    print("Preparing dataset for evaluation...")
    split = prepare_dataset(config)
    
    # Use saved feature names if available
    feature_names = saved_feature_names if saved_feature_names else split.feature_names
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_model(
        model,
        split.val_features,
        split.val_labels,
        split.val_df['s1_entity_id'].values,
        split.val_df['cand_entity_id'].values,
        feature_names,
        config,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (VALIDATION)")
    print("=" * 60)
    print(f"Best Threshold: {val_results['best_threshold']:.4f}")
    print(f"Macro F0.5: {val_results['best_metrics']['macro_f05']:.6f}")
    print(f"Macro Precision: {val_results['best_metrics']['macro_precision']:.6f}")
    print(f"Macro Recall: {val_results['best_metrics']['macro_recall']:.6f}")
    print(f"Macro F1: {val_results['best_metrics']['macro_f1']:.6f}")
    print(f"PR-AUC: {val_results['best_metrics']['pr_auc']:.6f}")
    print(f"Top-1 Accuracy: {val_results['metrics']['top_k_accuracy'].get(1, 0):.6f}")
    print(f"Prediction Time: {val_results['pred_time']:.2f}s")
    print("=" * 60)
    print("Top 20 Features by Gain:")
    for _, row in val_results['feature_importance'].head(20).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:<40s} {row['importance']:.2f} ({row['importance_pct']:.2f}%)")
    
    # Save results
    save_evaluation_results(val_results, config, '_val')
    
    return val_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LightGBM model for entity matching')
    parser.add_argument('--model', type=str, help='Path to model file (.txt or .pkl)')
    parser.add_argument('--candidates', type=str, help='Path to candidate_pairs.tsv')
    parser.add_argument('--data-root', type=str, help='Root path for dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=20260926, help='Random seed')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    
    args = parser.parse_args()
    
    overrides = {}
    if args.candidates:
        overrides['candidates_path'] = args.candidates
    if args.data_root:
        overrides['data_root'] = args.data_root
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.seed:
        overrides['seed'] = args.seed
        overrides['val_split_seed'] = args.seed
    if args.val_split:
        overrides['val_split_ratio'] = args.val_split
    
    main(model_path=args.model, **overrides)