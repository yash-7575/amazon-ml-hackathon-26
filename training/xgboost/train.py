"""XGBoost training script for entity matching.

Trains a binary classifier on candidate pairs to predict P(match).
Uses early stopping on validation set. Handles class imbalance via scale_pos_weight.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import warnings
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
from .utils import save_experiment_config


def train_xgboost(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    feature_names: list,
    config: XGBoostConfig,
    scale_pos_weight: Optional[float] = None,
) -> tuple:
    """Train XGBoost model with early stopping.

    Returns:
        model: Trained XGBoost Booster
        eval_results: Dict with training history
    """
    print("=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    # Prepare XGBoost DMatrix
    dtrain = xgb.DMatrix(train_features, label=train_labels, feature_names=feature_names)
    dval = xgb.DMatrix(val_features, label=val_labels, feature_names=feature_names)

    # Set scale_pos_weight for class imbalance
    params = config.xgb_params.copy()
    if config.use_scale_pos_weight and scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
        print(f"Using scale_pos_weight = {scale_pos_weight:.4f}")

    print(f"Parameters: {json.dumps(params, indent=2)}")
    print(f"Train samples: {len(train_labels):,}, pos rate: {train_labels.mean():.6f}")
    print(f"Val samples: {len(val_labels):,}, pos rate: {val_labels.mean():.6f}")
    print(f"Num features: {len(feature_names)}")
    print(f"Num boost rounds: {config.num_boost_round}")
    print(f"Early stopping rounds: {config.early_stopping_rounds}")

    # Train
    start_time = time.time()
    eval_results = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        evals_result=eval_results,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=50,
    )

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best validation logloss: {eval_results['valid']['logloss'][model.best_iteration - 1]:.6f}")

    return model, eval_results, train_time


def predict_probabilities(
    model,
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


def save_model(model, config: XGBoostConfig, feature_names: list, split_info: Dict, train_time: float):
    """Save model in both XGBoost native format and pickle."""
    import pickle

    # Native format
    model.save_model(config.model_path)
    print(f"Saved model to {config.model_path}")

    # Pickle with metadata
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'config': config.to_dict(),
        'split_info': split_info,
        'train_time': train_time,
        'best_iteration': model.best_iteration,
        'timestamp': datetime.now().isoformat(),
    }
    with open(config.model_pkl_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved model pickle to {config.model_pkl_path}")


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

    if split_name == 'validation':
        path = config.val_predictions_path
    else:
        path = os.path.join(config.output_dir, f'predictions_{split_name}.csv')

    pred_df.to_csv(path, index=False)
    print(f"Saved {split_name} predictions to {path}")


def save_metrics(metrics: Dict, config: XGBoostConfig):
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

    with open(config.metrics_path, 'w') as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"Saved metrics to {config.metrics_path}")


def save_threshold_results(threshold_df: pd.DataFrame, config: XGBoostConfig):
    """Save threshold evaluation results."""
    threshold_df.to_csv(config.threshold_results_path, index=False)
    print(f"Saved threshold results to {config.threshold_results_path}")


def save_best_threshold(best_threshold: float, best_metrics: Dict, config: XGBoostConfig):
    """Save best threshold to JSON."""
    data = {
        'best_threshold': float(best_threshold),
        'best_metrics': best_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    with open(config.best_threshold_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved best threshold to {config.best_threshold_path}")


def save_evaluation_report(
    metrics: Dict,
    threshold_results: pd.DataFrame,
    best_threshold: float,
    best_metrics: Dict,
    config: XGBoostConfig,
    feature_importance: pd.DataFrame,
    per_entity_df: pd.DataFrame,
    train_time: float,
    pred_time: float,
):
    """Generate and save detailed evaluation report in Markdown."""
    with open(config.eval_report_path, 'w') as f:
        f.write("# XGBoost Entity Matching - Evaluation Report\n\n")
        f.write(f"**Experiment:** {config.experiment_name}\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n")
        f.write(f"**Config:** {json.dumps(config.to_dict(), indent=2)}\n\n")

        f.write("## Dataset Summary\n\n")
        f.write(f"- Training pairs: {metrics.get('n_train_pairs', 'N/A'):,}\n")
        f.write(f"- Validation pairs: {metrics.get('n_val_pairs', 'N/A'):,}\n")
        f.write(f"- Training entities: {metrics.get('n_train_entities', 'N/A'):,}\n")
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
        f.write(f"- Training time: {train_time:.2f} seconds\n")
        f.write(f"- Prediction time: {pred_time:.2f} seconds\n\n")

        f.write("## XGBoost Parameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(config.xgb_params, indent=2))
        f.write("\n```\n")

    print(f"Saved evaluation report to {config.eval_report_path}")


def main(config: Optional[XGBoostConfig] = None, **overrides):
    """Main training pipeline."""
    if config is None:
        config = get_config(**overrides)

    # Set experiment timestamp
    config = XGBoostConfig(**{**config.to_dict(), 'experiment_timestamp': datetime.now().isoformat()})

    print(f"Experiment: {config.experiment_name}")
    print(f"Timestamp: {config.experiment_timestamp}")

    # Prepare dataset
    split = prepare_dataset(config)
    save_dataset_split(split, config)

    # Compute scale_pos_weight
    n_neg = (split.train_labels == 0).sum()
    n_pos = (split.train_labels == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class balance: pos={n_pos:,}, neg={n_neg:,}, ratio={scale_pos_weight:.2f}")

    # Train
    model, eval_results, train_time = train_xgboost(
        split.train_features,
        split.train_labels,
        split.val_features,
        split.val_labels,
        split.feature_names,
        config,
        scale_pos_weight=scale_pos_weight,
    )

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
    metrics['scale_pos_weight'] = scale_pos_weight
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
    save_model(model, config, split.feature_names, split.split_info, train_time)
    save_predictions(split.val_df, val_probs, val_preds, config, 'validation')
    save_metrics(metrics, config)
    save_threshold_results(threshold_results, config)
    save_best_threshold(best_threshold, best_metrics, config)
    save_evaluation_report(
        metrics, threshold_results, best_threshold, best_metrics, config,
        feature_importance, per_entity_df, train_time, pred_time
    )

    # Save feature importance
    feature_importance.to_csv(config.feature_importance_path, index=False)

    # Save per-entity analysis
    per_entity_path = config.error_analysis_path.replace('error_analysis', 'per_entity')
    per_entity_df.to_csv(per_entity_path, index=False)

    # Save experiment config for reproducibility
    save_experiment_config(config, metrics, train_time, pred_time)

    # Generate plots
    try:
        from .utils import create_plots
        create_plots(
            threshold_results,
            feature_importance,
            split.val_labels,
            val_probs,
            split.val_df['s1_entity_id'].values,
            config.output_dir,
        )
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Macro F0.5: {best_metrics['macro_f05']:.6f}")
    print(f"Macro Precision: {best_metrics['macro_precision']:.6f}")
    print(f"Macro Recall: {best_metrics['macro_recall']:.6f}")
    print(f"Macro F1: {best_metrics['macro_f1']:.6f}")
    print(f"PR-AUC: {best_metrics['pr_auc']:.6f}")
    print(f"Top-1 Accuracy: {metrics['top_k_accuracy'].get(1, 0):.6f}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Prediction Time: {pred_time:.2f}s")
    print("=" * 60)
    print("Top 20 Features by Gain:")
    for _, row in feature_importance.head(20).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:<40s} {row['importance']:.2f} ({row['importance_pct']:.2f}%)")

    return model, metrics, best_threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost for entity matching')
    parser.add_argument('--candidates', type=str, help='Path to candidate_pairs.tsv')
    parser.add_argument('--data-root', type=str, help='Root path for dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=20260926, help='Random seed')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--num-boost-round', type=int, default=2000, help='Number of boosting rounds')
    parser.add_argument('--early-stopping', type=int, default=100, help='Early stopping rounds')
    parser.add_argument('--learning-rate', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--max-depth', type=int, default=6, help='Max depth')
    parser.add_argument('--min-child-weight', type=int, default=1, help='Min child weight')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio')
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='Colsample bytree')
    parser.add_argument('--reg-alpha', type=float, default=0.0, help='L1 regularization')
    parser.add_argument('--reg-lambda', type=float, default=1.0, help='L2 regularization')

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
    if args.num_boost_round:
        overrides['num_boost_round'] = args.num_boost_round
    if args.early_stopping:
        overrides['early_stopping_rounds'] = args.early_stopping

    # Update xgb_params
    xgb_overrides = {}
    if args.learning_rate:
        xgb_overrides['learning_rate'] = args.learning_rate
    if args.max_depth:
        xgb_overrides['max_depth'] = args.max_depth
    if args.min_child_weight:
        xgb_overrides['min_child_weight'] = args.min_child_weight
    if args.subsample:
        xgb_overrides['subsample'] = args.subsample
    if args.colsample_bytree:
        xgb_overrides['colsample_bytree'] = args.colsample_bytree
    if args.reg_alpha:
        xgb_overrides['reg_alpha'] = args.reg_alpha
    if args.reg_lambda:
        xgb_overrides['reg_lambda'] = args.reg_lambda

    if xgb_overrides:
        overrides['xgb_params'] = {**XGBoostConfig().xgb_params, **xgb_overrides}

    main(**overrides)