"""Error analysis for XGBoost entity matching.

Generates detailed error analysis reports including false positives,
false negatives, and difficult case categorization.
"""
from __future__ import annotations
import os
import json
from typing import Optional, Dict, List
import numpy as np
import pandas as pd


def generate_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    s1_entity_ids: np.ndarray,
    cand_entity_ids: np.ndarray,
    df_features: Optional[pd.DataFrame] = None,
    df_source: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate detailed error analysis DataFrame.

    Args:
        y_true: Binary ground truth (0/1)
        y_pred: Binary predictions (0/1)
        y_prob: Predicted probabilities (0-1)
        s1_entity_ids: S1 entity IDs
        cand_entity_ids: Candidate entity IDs
        df_features: Optional feature DataFrame for additional context
        df_source: Optional source data DataFrame with business_name, business_address

    Returns:
        DataFrame with error analysis
    """
    df = pd.DataFrame({
        's1_entity_id': s1_entity_ids,
        'cand_entity_id': cand_entity_ids,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    })

    # Add source data if available
    if df_source is not None:
        # Merge with source data for business names/addresses
        source_cols = ['s1_entity_id', 'cand_entity_id', 's1_name', 'cand_name', 's1_addr', 'cand_addr', 's1_country', 'cand_country']
        available_cols = [c for c in source_cols if c in df_source.columns]
        if len(available_cols) > 2:  # At least s1_entity_id and cand_entity_id
            df = df.merge(df_source[available_cols], on=['s1_entity_id', 'cand_entity_id'], how='left')

    # Add features if available
    if df_features is not None:
        # Only add key features for readability
        key_features = [c for c in df_features.columns if any(k in c for k in
            ['exact', 'jaro', 'jarowinkler', 'levenshtein', 'token_jaccard', 'token_dice',
             'token_set', 'token_sort', 'char3gram', 'char4gram', 'postal_match',
             'city_match', 'housenumber', 'phone', 'cross_script', 'transliteration',
             'group_rank', 'score_margin', 'score_ratio', 'group_size', 'name_high_addr_low',
             'name_low_addr_high', 'sim_disagreement', 'missing'])]
        if key_features:
            feat_df = df_features[key_features].copy()
            feat_df.index = df.index
            df = pd.concat([df, feat_df], axis=1)

    # Categorize errors
    df['error_type'] = 'true_negative'
    df.loc[(df['y_true'] == 1) & (df['y_pred'] == 1), 'error_type'] = 'true_positive'
    df.loc[(df['y_true'] == 0) & (df['y_pred'] == 1), 'error_type'] = 'false_positive'
    df.loc[(df['y_true'] == 1) & (df['y_pred'] == 0), 'error_type'] = 'false_negative'

    # Add confidence level
    df['confidence'] = np.abs(y_prob - 0.5) * 2  # 0 to 1

    # Categorize false positives by type
    fp_mask = df['error_type'] == 'false_positive'
    fn_mask = df['error_type'] == 'false_negative'

    if 'name_token_jaccard' in df.columns and 'addr_token_jaccard' in df.columns:
        # High name similarity, low address = generic name collision
        df.loc[fp_mask & (df['name_token_jaccard'] > 0.7) & (df['addr_token_jaccard'] < 0.3), 'fp_category'] = 'generic_name_collision'
        # High address similarity, low name = address-only match missed
        df.loc[fn_mask & (df['name_token_jaccard'] < 0.3) & (df['addr_token_jaccard'] > 0.5), 'fn_category'] = 'address_only_match'
        # Both high = ambiguous
        df.loc[fp_mask & (df['name_token_jaccard'] > 0.7) & (df['addr_token_jaccard'] > 0.7), 'fp_category'] = 'ambiguous_both_high'
        df.loc[fn_mask & (df['name_token_jaccard'] > 0.7) & (df['addr_token_jaccard'] > 0.7), 'fn_category'] = 'ambiguous_both_high'
        # Cross-script
        if 'is_cross_script' in df.columns:
            df.loc[fp_mask & (df['is_cross_script'] == 1), 'fp_category'] = 'cross_script'
            df.loc[fn_mask & (df['is_cross_script'] == 1), 'fn_category'] = 'cross_script'
        # Phone conflict
        if 'phone_exact' in df.columns:
            df.loc[fp_mask & (df['phone_exact'] == 0), 'fp_category'] = 'phone_mismatch'
            df.loc[fn_mask & (df['phone_exact'] == 0), 'fn_category'] = 'phone_mismatch'
        # Postal code mismatch
        if 'addr_postal_match' in df.columns:
            df.loc[fp_mask & (df['addr_postal_match'] == 0), 'fp_category'] = 'postal_mismatch'
            df.loc[fn_mask & (df['addr_postal_match'] == 0), 'fn_category'] = 'postal_mismatch'
        # Name digit conflict
        if 'name_digits_conflict' in df.columns:
            df.loc[fp_mask & (df['name_digits_conflict'] == 1), 'fp_category'] = 'digit_conflict'
            df.loc[fn_mask & (df['name_digits_conflict'] == 1), 'fn_category'] = 'digit_conflict'

    # Sort by entity, then error type, then probability
    df = df.sort_values(['s1_entity_id', 'error_type', 'y_prob'], ascending=[True, True, False])

    return df


def save_error_analysis(
    error_df: pd.DataFrame,
    config,
    split_name: str = 'validation',
):
    """Save error analysis to CSV and Markdown report."""
    # Save CSV
    csv_path = os.path.join(config.output_dir, f'error_analysis_{split_name}.csv')
    error_df.to_csv(csv_path, index=False)
    print(f"Saved error analysis CSV to {csv_path}")

    # Generate Markdown report
    md_path = os.path.join(config.output_dir, f'error_analysis_{split_name}.md')
    with open(md_path, 'w') as f:
        f.write(f"# Error Analysis - {split_name.capitalize()}\n\n")

        # Summary counts
        f.write("## Summary\n\n")
        counts = error_df['error_type'].value_counts()
        for etype, count in counts.items():
            f.write(f"- {etype}: {count:,}\n")
        f.write("\n")

        # False positive categories
        if 'fp_category' in error_df.columns:
            f.write("## False Positive Categories\n\n")
            fp_cats = error_df[error_df['error_type'] == 'false_positive']['fp_category'].value_counts()
            for cat, count in fp_cats.items():
                f.write(f"- {cat}: {count:,}\n")
            f.write("\n")

        # False negative categories
        if 'fn_category' in error_df.columns:
            f.write("## False Negative Categories\n\n")
            fn_cats = error_df[error_df['error_type'] == 'false_negative']['fn_category'].value_counts()
            for cat, count in fn_cats.items():
                f.write(f"- {cat}: {count:,}\n")
            f.write("\n")

        # Top false positives by probability
        f.write("## Top False Positives (Highest Probability)\n\n")
        fps = error_df[error_df['error_type'] == 'false_positive'].head(20)
        if len(fps) > 0:
            display_cols = ['s1_entity_id', 'cand_entity_id', 'y_prob', 'error_type']
            if 'fp_category' in fps.columns:
                display_cols.insert(-1, 'fp_category')
            if 's1_name' in fps.columns and 'cand_name' in fps.columns:
                display_cols.extend(['s1_name', 'cand_name'])
            if 's1_addr' in fps.columns and 'cand_addr' in fps.columns:
                display_cols.extend(['s1_addr', 'cand_addr'])
            f.write(fps[display_cols].to_markdown(index=False))
            f.write("\n\n")

        # Top false negatives by probability (missed true matches with high probability)
        f.write("## Top False Negatives (Missed True Matches)\n\n")
        fns = error_df[error_df['error_type'] == 'false_negative'].head(20)
        if len(fns) > 0:
            display_cols = ['s1_entity_id', 'cand_entity_id', 'y_prob', 'error_type']
            if 'fn_category' in fns.columns:
                display_cols.insert(-1, 'fn_category')
            if 's1_name' in fns.columns and 'cand_name' in fns.columns:
                display_cols.extend(['s1_name', 'cand_name'])
            if 's1_addr' in fns.columns and 'cand_addr' in fns.columns:
                display_cols.extend(['s1_addr', 'cand_addr'])
            f.write(fns[display_cols].to_markdown(index=False))
            f.write("\n\n")

        # Probability distributions
        f.write("## Prediction Probability Distribution by Error Type\n\n")
        for etype in ['true_positive', 'true_negative', 'false_positive', 'false_negative']:
            subset = error_df[error_df['error_type'] == etype]['y_prob']
            if len(subset) > 0:
                f.write(f"### {etype} (n={len(subset):,})\n")
                f.write(f"- Mean: {subset.mean():.4f}\n")
                f.write(f"- Std: {subset.std():.4f}\n")
                f.write(f"- Min: {subset.min():.4f}\n")
                f.write(f"- Max: {subset.max():.4f}\n")
                f.write(f"- Median: {subset.median():.4f}\n\n")

    print(f"Saved error analysis report to {md_path}")


def generate_leakage_audit(
    feature_names: List[str],
    config,
) -> pd.DataFrame:
    """Generate a leakage audit report for all features.

    This documents each feature and whether it has potential leakage risk.
    """
    audit_rows = []

    for feat in feature_names:
        risk = "LOW"
        decision = "KEEP"
        reason = "Standard pairwise similarity feature"

        # Check for potential leakage indicators
        leak_keywords = [
            'label', 'target', 'ground_truth', 'matched', 'true_pair',
            'is_match', 'ground', 'truth', 'validation', 'test'
        ]

        for kw in leak_keywords:
            if kw in feat.lower():
                risk = "HIGH"
                decision = "REVIEW"
                reason = f"Contains keyword '{kw}' suggesting potential target leakage"
                break

        # Check for features that might encode group-level target info
        if any(k in feat for k in ['group_', 'entity_', 'rank_in_group', 'mutual_best']):
            if 'label' not in feat.lower() and 'target' not in feat.lower():
                risk = "MEDIUM"
                reason = "Group-relative feature - ensure computed without target labels"

        # Check for features derived from global statistics
        if any(k in feat for k in ['global_', 'dataset_', 'overall_', 'population_']):
            risk = "MEDIUM"
            reason = "May encode dataset-wide statistics"

        # Country match is constant within partition (not leakage, but not useful)
        if 'country_match' in feat:
            risk = "NONE"
            decision = "DROP"
            reason = "Constant within country partition (Finding 10)"

        audit_rows.append({
            'feature': feat,
            'leakage_risk': risk,
            'decision': decision,
            'reason': reason,
        })

    audit_df = pd.DataFrame(audit_rows)

    # Save
    audit_path = os.path.join(config.output_dir, 'leakage_audit.csv')
    audit_df.to_csv(audit_path, index=False)
    print(f"Saved leakage audit to {audit_path}")

    # Save markdown
    md_path = os.path.join(config.output_dir, 'leakage_audit.md')
    with open(md_path, 'w') as f:
        f.write("# Feature Leakage Audit\n\n")
        f.write("| Feature | Leakage Risk | Decision | Reason |\n")
        f.write("|---------|--------------|----------|--------|\n")
        for _, row in audit_df.iterrows():
            f.write(f"| {row['feature']} | {row['leakage_risk']} | {row['decision']} | {row['reason']} |\n")

    print(f"Saved leakage audit report to {md_path}")

    return audit_df


def analyze_entity_level_errors(
    error_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """Analyze errors at the entity level."""
    # Group by S1 entity
    entity_analysis = []

    for s1_id, group in error_df.groupby('s1_entity_id'):
        tp = (group['error_type'] == 'true_positive').sum()
        fp = (group['error_type'] == 'false_positive').sum()
        fn = (group['error_type'] == 'false_negative').sum()
        tn = (group['error_type'] == 'true_negative').sum()

        n_true = tp + fn
        n_pred = tp + fp

        # Entity-level metrics
        if n_pred > 0:
            precision = tp / n_pred
        elif n_true == 0:
            precision = 1.0
        else:
            precision = 0.0

        if n_true > 0:
            recall = tp / n_true
        else:
            recall = 1.0

        if precision > 0 and recall > 0:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 0.0

        # Categorize entity difficulty
        if n_true == 0:
            difficulty = 'singleton'
        elif fp > 0 and fn > 0:
            difficulty = 'mixed_errors'
        elif fp > 0:
            difficulty = 'over_predicting'
        elif fn > 0:
            difficulty = 'under_predicting'
        else:
            difficulty = 'correct'

        entity_analysis.append({
            's1_entity_id': s1_id,
            'n_candidates': len(group),
            'n_true': n_true,
            'n_pred': n_pred,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f05': f05,
            'difficulty': difficulty,
            'max_prob': group['y_prob'].max(),
            'top1_correct': (group.nlargest(1, 'y_prob')['y_true'].values[0] == 1) if len(group) > 0 else False,
        })

    entity_df = pd.DataFrame(entity_analysis)

    # Save
    entity_path = os.path.join(config.output_dir, f'entity_level_analysis_{split_name}.csv')
    entity_df.to_csv(entity_path, index=False)
    print(f"Saved entity-level analysis to {entity_path}")

    return entity_df


if __name__ == '__main__':
    # This module is meant to be imported
    pass