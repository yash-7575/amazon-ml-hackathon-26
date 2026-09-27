"""XGBoost evaluation script for entity matching.

Loads a trained model and evaluates on validation or test data.
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
from .dataset import prepare_dataset, save_dataset_split
from .metrics import (
    compute_all_metrics, evaluate_thresholds, find_best_threshold,
    top_k_accuracy, ranking_metrics, compute_feature_importance
)


def load_model(model_path: str) -> xgb.Booster:
    """Load XGBoost model from file."""
    model = xgb.Booster()
    model.load_model(model_path)
    return model


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


def save_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    config: XGBoostConfig,
    split_name: str = 'validation',
):
    """Save predictions to CSV."""
    pred_df = df[['s1_entity_id', 'cand_entity_id', 'label']].copy()
    pred_df['y_prob'] = y_prob
    pred_df['y_pred'] = y_pred

    path = os.path.join(config.output_dir, f'predictions_{split_name}.csv')
    pred_df.to_csv(path, index=False)
    print(f"Saved {split_name} predictions to {path}")


def save_metrics(metrics: Dict, config: XGBoostConfig, split_name: str = 'validation'):
    """Save metrics to JSON."""
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path = os.path.join(config.output_dir, f'metrics_{split_name}.json')
    with open(path, 'w') as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"Saved {split_name} metrics to {path}")


def save_threshold_results(threshold_df: pd.DataFrame, config: XGBoostConfig, split_name: str = 'validation'):
    """Save threshold evaluation results."""
    path = os.path.join(config.output_dir, f'threshold_results_{split_name}.csv')
    threshold_df.to_csv(path, index=False)
    print(f"Saved {split_name} threshold results to {path}")


def save_evaluation_report(
    metrics: Dict,
    threshold_results: pd.DataFrame,
    best_threshold: float,
    best_metrics: Dict,
    config: XGBoostConfig,
    feature_importance: pd.DataFrame,
    per_entity_df: pd.DataFrame,
    pred_time: float,
    split_name: str = 'validation',
):
    """Generate and save detailed evaluation report in Markdown."""
    path = os.path.join(config.output_dir, f'evaluation_report_{split_name}.md')
    with open(path, 'w') as f:
        f.write(f"# XGBoost Entity Matching - {split_name.capitalize()} Evaluation Report\n\n")
        f.write(f"**Experiment:** {config.experiment_name}\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")

        f.write("## Dataset Summary\n\n")
        f.write(f"- Validation pairs: {metrics.get('n_val_pairs', 'N/A'):,}\n")
        f.write(f"- Validation entities: {metrics.get('n_val_entities', 'N/A'):,}\n")
        f.write(f"- Features: {metrics.get('n_features', 'N/A')}\n")
        f.write(f"- Countries: {metrics.get('n_countries', 'N/A')}\n\n")

        f.write("## Model Performance (Best Threshold)\n\n")
        f.write(f"**Best Threshold:** {best_threshold:.4f}\n\n")
        f.write(f"- **Macro F0.5:** {best_metrics['macro_f05']:.6f}\n")
        f.write(f"- Macro Precision: {best_metrics['macro_precision']:.6f}\n")
        f.write(f"- Macro Recall: {best_metrics['macro_recall']:.6f}\n")
        f.write(f"- Macro F1: {best_metrics['macro_f1']:.6f}\n")
        f.write(f"- Global Precision: {best_metrics['global_precision']:.6f}\n")
        f.write(f"- Global Recall: {best_metrics['global_recall']:.6f}\n")
        f.write(f"- Global F1: {best_metrics['global_f1']:.6f}\n")
        f.write(f"- PR-AUC: {best_metrics['pr_auc']:.6f}\n")
        f.write(f"- ROC-AUC: {metrics.get('roc_auc', 'N/A'):.6f}\n\n")

        f.write("## Confusion Matrix\n\n")
        f.write(f"- TP: {metrics['tp']:,}\n")
        f.write(f"- FP: {metrics['fp']:,}\n")
        f.write(f"- FN: {metrics['fn']:,}\n")
        f.write(f"- TN: {metrics['tn']:,}\n\n")

        f.write("## Per-Entity Analysis\n\n")
        f.write(f"- Singletons: {metrics['n_singletons']:,} (F0.5: {metrics['singleton_f05']:.6f})\n")
        f.write(f"- Non-singletons: {metrics['n_non_singletons']:,} (F0.5: {metrics.get('non_singleton_macro_f05', 0):.6f})\n")
        f.write(f"- Empty predictions: {metrics['n_empty_predictions']:,}\n")
        f.write(f"- Multi-predictions: {metrics['n_multi_predictions']:,}\n\n")

        f.write("## Top-K Accuracy\n\n")
        for k, acc in metrics.get('top_k_accuracy', {}).items():
            f.write(f"- Top-{k}: {acc:.6f}\n")
        f.write("\n")

        f.write("## Ranking Metrics\n\n")
        f.write(f"- MRR: {metrics.get('mrr', 0):.6f}\n")
        f.write(f"- NDCG@5: {metrics.get('ndcg_at_5', 0):.6f}\n")
        f.write(f"- NDCG@10: {metrics.get('ndcg_at_10', 0):.6f}\n\n")

        f.write("## Threshold Sweep Results\n\n")
        f.write(threshold_results.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Feature Importance (Top 20 by Gain)\n\n")
        f.write(feature_importance.head(20).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Timing\n\n")
        f.write(f"- Prediction time: {pred_time:.2f} seconds\n\n")

    print(f"Saved evaluation report to {path}")


def main(config: Optional[XGBoostConfig] = None, **overrides):
    """Main evaluation pipeline."""
    if config is None:
        config = get_config(**overrides)

    print(f"Experiment: {config.experiment_name}")

    # Load model
    model_path = overrides.get('model_path', config.model_path)
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Prepare dataset (only need validation)
    split = prepare_dataset(config)

    # Predict on validation
    pred_start = time.time()
    val_probs = predict_probabilities(model, split.val_features, split.feature_names)
    pred_time = time.time() - pred_start
    print(f"Validation prediction time: {pred_time:.2f} seconds")

    # Evaluate thresholds
    threshold_results = evaluate_thresholds(
        split.val_labels,
        val_probs,
        split.val_df['s1_entity_id'].values,
        config.threshold_grid,
    )

    # Find best threshold
    best_threshold, best_metrics = find_best_threshold(threshold_results, 'macro_f05')
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Best macro F0.5: {best_metrics['macro_f05']:.6f}")
    print(f"Best macro Precision: {best_metrics['macro_precision']:.6f}")
    print(f"Best macro Recall: {best_metrics['macro_recall']:.6f}")

    # Final predictions at best threshold
    val_preds = (val_probs >= best_threshold).astype(int)

    # Compute full metrics
    metrics, per_entity_df = compute_all_metrics(
        split.val_labels,
        val_preds,
        val_probs,
        split.val_df['s1_entity_id'].values,
    )

    # Add dataset info
    metrics.update(split.split_info)
    metrics['best_iteration'] = model.best_iteration
    metrics['best_threshold'] = best_threshold

    # Top-K accuracy
    metrics['top_k_accuracy'] = top_k_accuracy(
        split.val_labels, val_probs, split.val_df['s1_entity_id'].values, config.top_k_values
    )

    # Ranking metrics
    metrics.update(ranking_metrics(
        split.val_labels, val_probs, split.val_df['s1_entity_id'].values
    ))

    # Feature importance
    feature_importance = compute_feature_importance(model, split.feature_names, 'gain')

    # Save outputs
    save_predictions(split.val_df, val_probs, val_preds, config, 'validation')
    save_metrics(metrics, config, 'validation')
    save_threshold_results(threshold_results, config, 'validation')
    save_evaluation_report(
        metrics, threshold_results, best_threshold, best_metrics, config,
        feature_importance, per_entity_df, pred_time, 'validation'
    )

    # Save feature importance
    fi_path = os.path.join(config.output_dir, 'feature_importance_validation.csv')
    feature_importance.to_csv(fi_path, index=False)

    # Save per-entity analysis
    per_entity_path = os.path.join(config.output_dir, 'per_entity_validation.csv')
    per_entity_df.to_csv(per_entity_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Macro F0.5: {best_metrics['macro_f05']:.6f}")
    print(f"Macro Precision: {best_metrics['macro_precision']:.6f}")
    print(f"Macro Recall: {best_metrics['macro_recall']:.6f}")
    print(f"Macro F1: {best_metrics['macro_f1']:.6f}")
    print(f"PR-AUC: {best_metrics['pr_auc']:.6f}")
    print(f"Top-1 Accuracy: {metrics['top_k_accuracy'].get(1, 0):.6f}")
    print(f"Prediction Time: {pred_time:.2f}s")
    print("=" * 60)
    print("Top 20 Features by Gain:")
    for _, row in feature_importance.head(20).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:<40s} {row['importance']:.2f} ({row['importance_pct']:.2f}%)")

    return metrics, best_threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate XGBoost for entity matching')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--candidates', type=str, help='Path to candidate_pairs.tsv')
    parser.add_argument('--data-root', type=str, help='Root path for dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=20260926, help='Random seed')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')

    args = parser.parse_args()

    overrides = {}
    if args.model_path:
        overrides['model_path'] = args.model_path
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

    main(**overrides)