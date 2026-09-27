"""Dataset loading and label generation for XGBoost training.

Loads candidate pairs, joins with source data, generates labels from ground truth.
Handles train/validation split by S1 entity to avoid leakage.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from .config import XGBoostConfig, get_config
from .features import compute_pair_features, compute_group_relative_features


@dataclass
class DatasetSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    feature_names: List[str]
    split_info: Dict


def load_source_data(config: XGBoostConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load S1, S2, S3 source data for training."""
    s1_path = config.resolve_path(config.train_s1_path)
    s2_path = config.resolve_path(config.train_s2_path)
    s3_path = config.resolve_path(config.train_s3_path)

    print(f"Loading source data...")
    print(f"  S1: {s1_path}")
    print(f"  S2: {s2_path}")
    print(f"  S3: {s3_path}")

    s1 = pd.read_csv(s1_path, sep='\t', dtype=str, keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str, keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str, keep_default_na=False)

    print(f"  S1 rows: {len(s1):,}")
    print(f"  S2 rows: {len(s2):,}")
    print(f"  S3 rows: {len(s3):,}")

    return s1, s2, s3


def load_ground_truth(config: XGBoostConfig) -> pd.DataFrame:
    """Load ground truth and expand to long format (one row per match)."""
    gt_path = config.resolve_path(config.ground_truth_path)
    print(f"Loading ground truth: {gt_path}")

    gt = pd.read_csv(gt_path, sep='\t', dtype=str, keep_default_na=False)
    print(f"  Ground truth rows (S1 entities): {len(gt):,}")

    # Expand comma-separated matched_entity_ids to long format
    rows = []
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if matched and matched.strip():
            for cand_id in matched.split(','):
                rows.append({'s1_entity_id': s1_id, 'matched_entity_id': cand_id.strip()})

    gt_long = pd.DataFrame(rows)
    print(f"  Ground truth pairs: {len(gt_long):,}")
    print(f"  Unique S1 entities with matches: {gt_long['s1_entity_id'].nunique():,}")
    print(f"  Unique S2/S3 entities matched: {gt_long['matched_entity_id'].nunique():,}")

    return gt_long


def load_candidates(config: XGBoostConfig) -> pd.DataFrame:
    """Load candidate pairs from blocking output."""
    cand_path = config.resolve_path(config.candidates_path)
    print(f"Loading candidates: {cand_path}")

    # Candidate pairs format: s1_entity_id, candidate_entity_id, source, score
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


def generate_labels(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Generate binary labels by joining candidates with ground truth.

    Label = 1 if (s1_entity_id, cand_entity_id) is in ground truth, else 0.
    """
    print("Generating labels...")

    # Create a set of true pairs for fast lookup
    true_pairs = set(zip(ground_truth['s1_entity_id'], ground_truth['matched_entity_id']))
    print(f"  True pair set size: {len(true_pairs):,}")

    # Vectorized label generation
    candidate_pairs = list(zip(candidates['s1_entity_id'], candidates['cand_entity_id']))
    labels = np.array([1.0 if pair in true_pairs else 0.0 for pair in candidate_pairs])

    candidates = candidates.copy()
    candidates['label'] = labels

    pos = labels.sum()
    neg = len(labels) - pos
    print(f"  Positive pairs: {int(pos):,}")
    print(f"  Negative pairs: {int(neg):,}")
    print(f"  Positive ratio: {pos/len(labels):.6f}")

    return candidates


def join_source_data(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> pd.DataFrame:
    """Join candidate pairs with source data to get name/address/country for both sides."""
    print("Joining with source data...")

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
    print(f"  Combined S2+S3 rows: {len(cand_combined):,}")

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


def compute_features(df: pd.DataFrame, config: XGBoostConfig) -> Tuple[np.ndarray, List[str]]:
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

    # Handle NaN - XGBoost handles NaN natively, but we need to ensure consistency
    feature_array = feature_df.values.astype(np.float32)

    print(f"  Feature matrix shape: {feature_array.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  NaN count: {np.isnan(feature_array).sum():,}")

    return feature_array, feature_names


def split_by_s1_entity(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: np.ndarray,
    config: XGBoostConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Split data by S1 entity to avoid leakage.

    All candidate pairs for a given S1 entity go to either train or validation,
    never both. This prevents the model from memorizing entity-specific patterns.
    """
    print("Splitting by S1 entity...")

    s1_entities = df['s1_entity_id'].unique()
    n_entities = len(s1_entities)
    n_val = int(n_entities * config.val_split_ratio)

    # Deterministic split using hash of entity_id
    np.random.seed(config.val_split_seed)
    val_entities = set(np.random.choice(s1_entities, size=n_val, replace=False))

    is_val = df['s1_entity_id'].isin(val_entities).values

    train_features = features[~is_val]
    train_labels = labels[~is_val]
    val_features = features[is_val]
    val_labels = labels[is_val]

    train_entities = (~is_val).sum()
    val_entities_count = is_val.sum()

    print(f"  Train pairs: {len(train_labels):,} ({train_labels.mean():.6f} pos rate)")
    print(f"  Val pairs: {len(val_labels):,} ({val_labels.mean():.6f} pos rate)")
    print(f"  Train S1 entities: {n_entities - n_val:,}")
    print(f"  Val S1 entities: {n_val:,}")

    split_info = {
        'n_train_pairs': int(len(train_labels)),
        'n_val_pairs': int(len(val_labels)),
        'n_train_entities': n_entities - n_val,
        'n_val_entities': n_val,
        'train_pos_rate': float(train_labels.mean()),
        'val_pos_rate': float(val_labels.mean()),
        'val_split_ratio': config.val_split_ratio,
        'val_split_seed': config.val_split_seed,
    }

    return train_features, train_labels, val_features, val_labels, split_info


def prepare_dataset(config: Optional[XGBoostConfig] = None) -> DatasetSplit:
    """Full pipeline: load data, generate labels, compute features, split."""
    if config is None:
        config = get_config()

    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    # Load data
    s1, s2, s3 = load_source_data(config)
    ground_truth = load_ground_truth(config)
    candidates = load_candidates(config)

    # Generate labels
    candidates = generate_labels(candidates, ground_truth)

    # Join with source data
    df = join_source_data(candidates, s1, s2, s3)

    # Compute group-relative features (need blocker_score)
    if 'blocker_score' in df.columns:
        df = compute_group_relative_features(df, 'blocker_score', config)

    # Compute pairwise features
    features, feature_names = compute_features(df, config)
    labels = df['label'].values.astype(np.float32)

    # Split
    train_features, train_labels, val_features, val_labels, split_info = split_by_s1_entity(
        df, labels, features, config
    )

    # Add split info
    split_info.update({
        'n_features': len(feature_names),
        'n_countries': df['s1_country'].nunique() if 's1_country' in df.columns else 0,
        'countries': sorted(df['s1_country'].unique().tolist()) if 's1_country' in df.columns else [],
    })

    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    # Recreate val_entities for DatasetSplit
    np.random.seed(config.val_split_seed)
    s1_entities_all = df['s1_entity_id'].unique()
    n_val = int(len(s1_entities_all) * config.val_split_ratio)
    val_entities = set(np.random.choice(s1_entities_all, size=n_val, replace=False))

    return DatasetSplit(
        train_df=df[~df['s1_entity_id'].isin(val_entities)],
        val_df=df[df['s1_entity_id'].isin(val_entities)],
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        feature_names=feature_names,
        split_info=split_info,
    )


def save_dataset_split(split: DatasetSplit, config: XGBoostConfig):
    """Save dataset split info for reproducibility."""
    info_path = os.path.join(config.output_dir, "dataset_split_info.json")
    with open(info_path, 'w') as f:
        # Convert numpy types to Python types
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
        json.dump(convert(split.split_info), f, indent=2)
    print(f"Saved dataset split info to {info_path}")