"""Utility functions for XGBoost entity matching pipeline."""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime


def setup_output_dirs(config) -> None:
    """Create all output directories."""
    dirs = [
        config.output_dir,
        os.path.dirname(config.model_path),
        os.path.dirname(config.model_pkl_path),
        os.path.dirname(config.val_predictions_path),
        os.path.dirname(config.metrics_path),
        os.path.dirname(config.threshold_results_path),
        os.path.dirname(config.best_threshold_path),
        os.path.dirname(config.eval_report_path),
        os.path.dirname(config.error_analysis_path),
        os.path.dirname(config.feature_importance_path),
        os.path.dirname(config.experiment_config_path),
    ]
    for d in dirs:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def save_experiment_config(config, metrics: Dict, train_time: float, pred_time: float) -> None:
    """Save complete experiment configuration for reproducibility."""
    exp_config = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'data_root': config.data_root,
        'candidates_path': config.candidates_path,
        'ground_truth_path': config.ground_truth_path,
        'train_s1_path': config.train_s1_path,
        'train_s2_path': config.train_s2_path,
        'train_s3_path': config.train_s3_path,
        'val_split_ratio': config.val_split_ratio,
        'val_split_seed': config.val_split_seed,
        'stratified_by_country': config.stratified_by_country,
        'xgb_params': config.xgb_params,
        'num_boost_round': config.num_boost_round,
        'early_stopping_rounds': config.early_stopping_rounds,
        'use_scale_pos_weight': config.use_scale_pos_weight,
        'threshold_grid': config.threshold_grid,
        'top_k_values': config.top_k_values,
        'feature_flags': {k: v for k, v in config.to_dict().items()
                          if k.startswith('use_')},
        'metrics': metrics,
        'train_time_seconds': train_time,
        'pred_time_seconds': pred_time,
    }

    # Try to get git commit
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            exp_config['git_commit'] = result.stdout.strip()
    except Exception:
        pass

    # Save
    with open(config.experiment_config_path, 'w') as f:
        json.dump(exp_config, f, indent=2, default=str)

    print(f"Saved experiment config to {config.experiment_config_path}")


def load_experiment_config(path: str) -> Dict:
    """Load experiment configuration."""
    with open(path, 'r') as f:
        return json.load(f)


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20) -> None:
    """Print feature importance table."""
    print(f"\nTop {top_n} Features by Gain:")
    print("-" * 70)
    print(f"{'Rank':>4}  {'Feature':<40} {'Importance':>12} {'Pct':>8}")
    print("-" * 70)
    for _, row in feature_importance.head(top_n).iterrows():
        print(f"{row['rank']:>4}  {row['feature']:<40} {row['importance']:>12.2f} {row['importance_pct']:>7.2f}%")


def print_metrics_summary(metrics: Dict, best_threshold: float, best_metrics: Dict) -> None:
    """Print metrics summary."""
    print("\n" + "=" * 60)
    print("XGBOOST ENTITY RESOLUTION RESULTS")
    print("=" * 60)
    print(f"Dataset: {metrics.get('data_root', 'N/A')}")
    print(f"Candidate pairs: {metrics.get('n_train_pairs', 0) + metrics.get('n_val_pairs', 0):,}")
    print(f"Positive pairs: {int(metrics.get('n_train_pairs', 0) * metrics.get('train_pos_rate', 0) + metrics.get('n_val_pairs', 0) * metrics.get('val_pos_rate', 0)):,}")
    print(f"Negative pairs: {int(metrics.get('n_train_pairs', 0) * (1 - metrics.get('train_pos_rate', 0)) + metrics.get('n_val_pairs', 0) * (1 - metrics.get('val_pos_rate', 0))):,}")
    print(f"Positive ratio: {metrics.get('train_pos_rate', 0):.6f}")
    print(f"Features: {metrics.get('n_features', 'N/A')}")
    print(f"Training samples: {metrics.get('n_train_pairs', 'N/A'):,}")
    print(f"Validation samples: {metrics.get('n_val_pairs', 'N/A'):,}")
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"\nMacro F0.5: {best_metrics['macro_f05']:.6f}")
    print(f"Macro Precision: {best_metrics['macro_precision']:.6f}")
    print(f"Macro Recall: {best_metrics['macro_recall']:.6f}")
    print(f"Macro F1: {best_metrics['macro_f1']:.6f}")
    print(f"PR-AUC: {best_metrics['pr_auc']:.6f}")
    print(f"Top-1 Accuracy: {metrics.get('top_k_accuracy', {}).get(1, 0):.6f}")
    print(f"Top-3 Accuracy: {metrics.get('top_k_accuracy', {}).get(3, 0):.6f}")
    print(f"Top-5 Accuracy: {metrics.get('top_k_accuracy', {}).get(5, 0):.6f}")
    print(f"\nTP: {metrics.get('tp', 'N/A'):,}")
    print(f"FP: {metrics.get('fp', 'N/A'):,}")
    print(f"TN: {metrics.get('tn', 'N/A'):,}")
    print(f"FN: {metrics.get('fn', 'N/A'):,}")
    print(f"\nTraining time: {metrics.get('train_time', 'N/A'):.2f}s")
    print(f"Prediction time: {metrics.get('pred_time', 'N/A'):.2f}s")
    print("=" * 60)


def compute_leakage_warnings(feature_names: List[str]) -> List[str]:
    """Check for potential data leakage in features."""
    warnings = []

    for feat in feature_names:
        # Check for suspicious patterns
        lower = feat.lower()
        if any(kw in lower for kw in ['label', 'target', 'ground', 'truth', 'matched', 'is_match']):
            warnings.append(f"POTENTIAL LEAKAGE: Feature '{feat}' contains target-related keyword")
        if 'validation' in lower or 'test' in lower:
            warnings.append(f"POTENTIAL LEAKAGE: Feature '{feat}' references validation/test split")

    return warnings


def create_plots(
    threshold_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
    output_dir: str,
) -> None:
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available, skipping plots")
        return

    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # 1. Threshold vs Macro F0.5
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_results['threshold'], threshold_results['macro_f05'], 'b-o', label='Macro F0.5')
    plt.plot(threshold_results['threshold'], threshold_results['macro_precision'], 'g--', label='Macro Precision')
    plt.plot(threshold_results['threshold'], threshold_results['macro_recall'], 'r--', label='Macro Recall')
    plt.plot(threshold_results['threshold'], threshold_results['macro_f1'], 'm--', label='Macro F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Macro Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plots', 'threshold_vs_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plots', 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Feature importance (top 20)
    plt.figure(figsize=(10, 8))
    top20 = feature_importance.head(20).iloc[::-1]  # Reverse for horizontal bar
    plt.barh(range(len(top20)), top20['importance'])
    plt.yticks(range(len(top20)), top20['feature'])
    plt.xlabel('Gain Importance')
    plt.title('Top 20 Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'feature_importance_top20.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Negative', density=True)
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Positive', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plots', 'probability_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Positive vs Negative probability distribution (boxplot)
    plt.figure(figsize=(8, 6))
    data = [y_prob[y_true == 0], y_prob[y_true == 1]]
    plt.boxplot(data, labels=['Negative', 'Positive'])
    plt.ylabel('Predicted Probability')
    plt.title('Probability Distribution by True Class')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plots', 'prob_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Confusion matrix heatmap
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix (threshold=0.5)')
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {os.path.join(output_dir, 'plots')}")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def estimate_memory_usage(n_samples: int, n_features: int, dtype=np.float32) -> float:
    """Estimate memory usage for feature matrix in MB."""
    bytes_per_sample = n_features * np.dtype(dtype).itemsize
    total_bytes = n_samples * bytes_per_sample
    return total_bytes / (1024 * 1024)  # MB


if __name__ == '__main__':
    # Test utilities
    print("Utils module loaded successfully")