"""Evaluation metrics for entity matching.

Implements competition metric: macro-averaged F0.5 per S1 entity.
Also computes Precision, Recall, F1, PR-AUC, Top-K accuracy, confusion matrix.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix, roc_auc_score
)


def macro_f05_per_entity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s1_entity_ids: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    """Compute macro-averaged F0.5 per S1 entity.
    
    F0.5 = (1.25 * P * R) / (0.25 * P + R)
    
    For each S1 entity:
    - TP = predicted matches that are correct
    - FP = predicted matches that are wrong
    - FN = true matches that were missed
    - P = TP / (TP + FP) if TP+FP > 0 else 1.0 (for singleton entities)
    - R = TP / (TP + FN) if TP+FN > 0 else 1.0 (for singleton entities)
    - If both true and predicted are empty: F0.5 = 1.0 (free points)
    
    Then average F0.5 across all S1 entities.
    
    Args:
        y_true: Binary ground truth labels (0/1)
        y_pred: Binary predictions (0/1)
        s1_entity_ids: S1 entity ID for each pair
    
    Returns:
        macro_f05: Float macro-averaged F0.5
        per_entity_df: DataFrame with per-entity metrics
    """
    df = pd.DataFrame({
        's1_entity_id': s1_entity_ids,
        'y_true': y_true,
        'y_pred': y_pred,
    })
    
    # Group by S1 entity
    grouped = df.groupby('s1_entity_id')
    
    results = []
    for s1_id, group in grouped:
        tp = ((group['y_true'] == 1) & (group['y_pred'] == 1)).sum()
        fp = ((group['y_true'] == 0) & (group['y_pred'] == 1)).sum()
        fn = ((group['y_true'] == 1) & (group['y_pred'] == 0)).sum()
        tn = ((group['y_true'] == 0) & (group['y_pred'] == 0)).sum()
        
        n_true = tp + fn
        n_pred = tp + fp
        
        if n_pred == 0 and n_true == 0:
            # Singleton entity - correct empty prediction
            precision = 1.0
            recall = 1.0
            f05 = 1.0
        elif n_pred == 0:
            # Predicted nothing but there are true matches
            precision = 1.0  # No false positives
            recall = 0.0
            f05 = 0.0
        elif n_true == 0:
            # Predicted something but no true matches (shouldn't happen if blocking is perfect)
            precision = 0.0
            recall = 1.0  # No false negatives
            f05 = 0.0
        else:
            precision = tp / n_pred
            recall = tp / n_true
            if precision == 0 and recall == 0:
                f05 = 0.0
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        
        results.append({
            's1_entity_id': s1_id,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'n_true': int(n_true),
            'n_pred': int(n_pred),
            'precision': precision,
            'recall': recall,
            'f05': f05,
        })
    
    per_entity_df = pd.DataFrame(results)
    macro_f05 = per_entity_df['f05'].mean()
    
    return macro_f05, per_entity_df


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
) -> Dict:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Binary ground truth (0/1)
        y_pred: Binary predictions (0/1)
        y_prob: Predicted probabilities (0-1)
        s1_entity_ids: S1 entity ID for each pair
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Global (micro) metrics
    metrics['global_precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['global_recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['global_f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # PR-AUC (pair-level)
    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    # Competition metric: macro F0.5
    macro_f05, per_entity_df = macro_f05_per_entity(y_true, y_pred, s1_entity_ids)
    metrics['macro_f05'] = macro_f05
    
    # Per-entity precision/recall/F1 averages
    metrics['macro_precision'] = per_entity_df['precision'].mean()
    metrics['macro_recall'] = per_entity_df['recall'].mean()
    metrics['macro_f1'] = (2 * metrics['macro_precision'] * metrics['macro_recall'] / 
                           (metrics['macro_precision'] + metrics['macro_recall']) 
                           if (metrics['macro_precision'] + metrics['macro_recall']) > 0 else 0)
    
    # Singleton analysis
    singleton_mask = per_entity_df['n_true'] == 0
    n_singletons = singleton_mask.sum()
    singleton_f05 = per_entity_df.loc[singleton_mask, 'f05'].mean() if n_singletons > 0 else 0
    metrics['n_singletons'] = int(n_singletons)
    metrics['singleton_f05'] = float(singleton_f05)
    
    # Non-singleton analysis
    non_singleton_mask = ~singleton_mask
    n_non_singletons = non_singleton_mask.sum()
    metrics['n_non_singletons'] = int(n_non_singletons)
    if n_non_singletons > 0:
        metrics['non_singleton_macro_f05'] = float(
            per_entity_df.loc[non_singleton_mask, 'f05'].mean()
        )
        metrics['non_singleton_macro_precision'] = float(
            per_entity_df.loc[non_singleton_mask, 'precision'].mean()
        )
        metrics['non_singleton_macro_recall'] = float(
            per_entity_df.loc[non_singleton_mask, 'recall'].mean()
        )
    
    # Entities with no predicted match (empty predictions)
    empty_pred_mask = per_entity_df['n_pred'] == 0
    metrics['n_empty_predictions'] = int(empty_pred_mask.sum())
    metrics['empty_pred_f05'] = float(per_entity_df.loc[empty_pred_mask, 'f05'].mean()) if empty_pred_mask.any() else 0
    
    # Entities with multiple predictions
    multi_pred_mask = per_entity_df['n_pred'] > 1
    metrics['n_multi_predictions'] = int(multi_pred_mask.sum())
    
    return metrics, per_entity_df


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    """Evaluate multiple thresholds and return results DataFrame.
    
    Args:
        y_true: Binary ground truth
        y_prob: Predicted probabilities
        s1_entity_ids: S1 entity IDs
        thresholds: List of thresholds to evaluate
    
    Returns:
        DataFrame with columns: threshold, macro_f05, precision, recall, f1, n_predicted, pr_auc
    """
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics, _ = compute_all_metrics(y_true, y_pred, y_prob, s1_entity_ids)
        
        # Also compute number of predicted matches
        n_predicted = y_pred.sum()
        
        results.append({
            'threshold': thresh,
            'macro_f05': metrics['macro_f05'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'macro_f1': metrics['macro_f1'],
            'global_precision': metrics['global_precision'],
            'global_recall': metrics['global_recall'],
            'global_f1': metrics['global_f1'],
            'pr_auc': metrics['pr_auc'],
            'n_predicted': int(n_predicted),
            'n_singletons': metrics['n_singletons'],
        })
    
    return pd.DataFrame(results)


def find_best_threshold(
    threshold_results: pd.DataFrame,
    metric: str = 'macro_f05',
) -> Tuple[float, Dict]:
    """Find the best threshold based on validation metric.
    
    Args:
        threshold_results: DataFrame from evaluate_thresholds
        metric: Column name to optimize (default: macro_f05)
    
    Returns:
        best_threshold, best_metrics_dict
    """
    best_idx = threshold_results[metric].idxmax()
    best_row = threshold_results.loc[best_idx]
    best_threshold = best_row['threshold']
    best_metrics = best_row.to_dict()
    
    return best_threshold, best_metrics


def top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[int, float]:
    """Compute Top-K accuracy: for each S1 entity, check if any true match is in top-K predictions.
    
    Args:
        y_true: Binary ground truth
        y_prob: Predicted probabilities
        s1_entity_ids: S1 entity IDs
        k_values: K values to evaluate
    
    Returns:
        Dict mapping K -> accuracy
    """
    df = pd.DataFrame({
        's1_entity_id': s1_entity_ids,
        'y_true': y_true,
        'y_prob': y_prob,
    })
    
    results = {}
    for k in k_values:
        correct = 0
        total = 0
        for s1_id, group in df.groupby('s1_entity_id'):
            total += 1
            # Get top-K by probability
            top_k = group.nlargest(k, 'y_prob')
            if (top_k['y_true'] == 1).any():
                correct += 1
        results[k] = correct / total if total > 0 else 0.0
    
    return results


def ranking_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
) -> Dict:
    """Compute ranking metrics: MRR, NDCG@K."""
    df = pd.DataFrame({
        's1_entity_id': s1_entity_ids,
        'y_true': y_true,
        'y_prob': y_prob,
    })
    
    mrr_sum = 0.0
    ndcg_at_5_sum = 0.0
    ndcg_at_10_sum = 0.0
    n_entities = 0
    
    for s1_id, group in df.groupby('s1_entity_id'):
        n_entities += 1
        # Sort by probability descending
        group = group.sort_values('y_prob', ascending=False).reset_index(drop=True)
        
        # MRR: reciprocal rank of first true match
        true_positions = group.index[group['y_true'] == 1].tolist()
        if true_positions:
            first_true = true_positions[0]
            mrr_sum += 1.0 / (first_true + 1)
            
            # NDCG@5 and @10
            for k in [5, 10]:
                k = min(k, len(group))
                dcg = sum(1.0 / np.log2(i + 2) for i in range(k) if group.loc[i, 'y_true'] == 1)
                # Ideal DCG: all true matches at top
                n_true = len(true_positions)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_true)))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                if k == 5:
                    ndcg_at_5_sum += ndcg
                else:
                    ndcg_at_10_sum += ndcg
    
    return {
        'mrr': mrr_sum / n_entities if n_entities > 0 else 0.0,
        'ndcg_at_5': ndcg_at_5_sum / n_entities if n_entities > 0 else 0.0,
        'ndcg_at_10': ndcg_at_10_sum / n_entities if n_entities > 0 else 0.0,
    }


def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
    cand_entity_ids: np.ndarray,
    df_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate error analysis DataFrame with false positives and false negatives."""
    df = pd.DataFrame({
        's1_entity_id': s1_entity_ids,
        'cand_entity_id': cand_entity_ids,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    })
    
    # False positives: predicted 1, true 0
    fp = df[(df['y_true'] == 0) & (df['y_pred'] == 1)].copy()
    fp['error_type'] = 'false_positive'
    
    # False negatives: predicted 0, true 1
    fn = df[(df['y_true'] == 1) & (df['y_pred'] == 0)].copy()
    fn['error_type'] = 'false_negative'
    
    # True positives
    tp = df[(df['y_true'] == 1) & (df['y_pred'] == 1)].copy()
    tp['error_type'] = 'true_positive'
    
    # True negatives (sample for analysis)
    tn = df[(df['y_true'] == 0) & (df['y_pred'] == 0)].copy()
    tn['error_type'] = 'true_negative'
    
    # Combine
    errors = pd.concat([fp, fn, tp], ignore_index=True)
    errors = errors.sort_values(['s1_entity_id', 'error_type', 'y_prob'], ascending=[True, True, False])
    
    return errors


def compute_feature_importance(
    model,
    feature_names: List[str],
    importance_type: str = 'gain',
) -> pd.DataFrame:
    """Extract feature importance from trained LightGBM model."""
    if importance_type == 'gain':
        importance = model.feature_importance(importance_type='gain')
    elif importance_type == 'split':
        importance = model.feature_importance(importance_type='split')
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")
    
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    fi_df['rank'] = range(1, len(fi_df) + 1)
    fi_df['importance_pct'] = fi_df['importance'] / fi_df['importance'].sum() * 100
    
    return fi_df