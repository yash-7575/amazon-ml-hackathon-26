"""Feature engineering for XGBoost entity matching.

Computes pairwise features for (S1, candidate) pairs.
All features derived ONLY from the four input columns: entity_id, business_name, business_address, country.
No ground truth leakage.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein, Jaro, JaroWinkler

# Reuse normalization from blocking to ensure train/test consistency
import sys
import os
blocking_path = os.path.join(os.path.dirname(__file__), '..', '..', 'blocking')
sys.path.insert(0, os.path.abspath(blocking_path))
from normalize import (
    base_normalize, name_tokens, name_core, addr_tokens,
    has_devanagari, transliterate_deva, country_key, SUFFIX, ADDR_STOPWORDS
)


# -----------------------------------------------------------------------------
# Low-level similarity functions
# -----------------------------------------------------------------------------

def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity (0-1)."""
    if not s1 or not s2:
        return 0.0
    return Jaro.normalized_similarity(s1, s2)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (0-1)."""
    if not s1 or not s2:
        return 0.0
    return JaroWinkler.normalized_similarity(s1, s2)


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (0-1). 1 - distance/max_len."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = Levenshtein.distance(s1, s2)
    return 1.0 - dist / max_len


def token_jaccard(s1: str, s2: str, tokenizer=None) -> float:
    """Token Jaccard similarity."""
    if tokenizer is None:
        tokenizer = lambda x: set(x.split())
    t1 = tokenizer(s1)
    t2 = tokenizer(s2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


def token_dice(s1: str, s2: str, tokenizer=None) -> float:
    """Token Dice coefficient (2*|A∩B| / (|A|+|B|))."""
    if tokenizer is None:
        tokenizer = lambda x: set(x.split())
    t1 = tokenizer(s1)
    t2 = tokenizer(s2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    return 2.0 * inter / (len(t1) + len(t2))


def token_set_ratio(s1: str, s2: str) -> float:
    """RapidFuzz token_set_ratio (0-100) normalized to 0-1."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """RapidFuzz token_sort_ratio (0-100) normalized to 0-1."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    if len(s1) < n or len(s2) < n:
        return 0.0
    ng1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    ng2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    if not ng1 and not ng2:
        return 1.0
    inter = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return inter / union if union > 0 else 0.0


def exact_match(s1: str, s2: str) -> float:
    """Exact string match (1.0 or 0.0)."""
    return 1.0 if s1 == s2 and s1 != "" else 0.0


def containment_ratio(s1: str, s2: str, tokenizer=None) -> float:
    """Asymmetric containment: |A∩B| / |A|. Returns both directions."""
    if tokenizer is None:
        tokenizer = lambda x: set(x.split())
    t1 = tokenizer(s1)
    t2 = tokenizer(s2)
    if not t1:
        return 0.0
    inter = len(t1 & t2)
    return inter / len(t1)


def extract_postal_code(addr: str) -> Optional[str]:
    """Extract postal code from address (simple regex for US/India/UK patterns)."""
    if not addr:
        return None
    # US ZIP (5 or 9 digit), India PIN (6 digit), UK postcode patterns
    patterns = [
        r'\b\d{5}(?:-\d{4})?\b',  # US ZIP
        r'\b\d{6}\b',              # India PIN
        r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b',  # UK postcode
    ]
    for pat in patterns:
        m = re.search(pat, addr, re.IGNORECASE)
        if m:
            return m.group(0).upper()
    return None


def extract_house_number(addr: str) -> Optional[str]:
    """Extract leading house/building number from address."""
    if not addr:
        return None
    # Match leading digits possibly with suffix (e.g., "123", "123A", "123-45")
    m = re.match(r'^\s*(\d+[A-Za-z]?(?:[-\s]\d+)?)', addr)
    if m:
        return m.group(1)
    return None


def extract_city(addr: str) -> Optional[str]:
    """Extract city from address (heuristic: token before state/zip)."""
    if not addr:
        return None
    tokens = addr_tokens(addr, drop_stopwords=False)
    if not tokens:
        return None
    # Heuristic: city is often the token before state abbreviation or postal code
    # For simplicity, return the last non-stopword token before a known pattern
    # This is approximate - in practice, address parsing is hard without external libs
    return tokens[-1] if tokens else None


def normalize_phone(phone: str) -> str:
    """Normalize phone number to digits only."""
    if not phone:
        return ""
    return re.sub(r'\D', '', phone)


def phone_suffix_match(phone1: str, phone2: str, suffix_len: int = 7) -> float:
    """Match on last N digits of phone numbers."""
    p1 = normalize_phone(phone1)
    p2 = normalize_phone(phone2)
    if not p1 or not p2:
        return 0.0
    if len(p1) < suffix_len or len(p2) < suffix_len:
        return 0.0
    return 1.0 if p1[-suffix_len:] == p2[-suffix_len:] else 0.0


def is_cross_script(name1: str, name2: str) -> float:
    """Check if names are in different scripts (ASCII vs Devanagari)."""
    deva1 = has_devanagari(name1)
    deva2 = has_devanagari(name2)
    return 1.0 if deva1 != deva2 else 0.0


def transliteration_similarity(name1: str, name2: str) -> float:
    """Similarity after transliterating Devanagari to Latin."""
    if has_devanagari(name1):
        n1 = transliterate_deva(name1)
    else:
        n1 = base_normalize(name1)
    if has_devanagari(name2):
        n2 = transliterate_deva(name2)
    else:
        n2 = base_normalize(name2)
    if not n1 or not n2:
        return 0.0
    return token_set_ratio(n1, n2)


def name_digits_conflict(name1: str, name2: str) -> float:
    """Detect conflicting digits in names (e.g., Store 14 vs Store 15)."""
    d1 = re.findall(r'\d+', name1)
    d2 = re.findall(r'\d+', name2)
    if not d1 or not d2:
        return 0.0
    # If any digit sequence differs, it's a conflict
    return 1.0 if set(d1) != set(d2) else 0.0


# -----------------------------------------------------------------------------
# Feature extraction for a single pair
# -----------------------------------------------------------------------------

def compute_pair_features(
    s1_name: str, s1_addr: str, s1_country: str,
    cand_name: str, cand_addr: str, cand_country: str,
    config,
) -> dict:
    """Compute all features for one (S1, candidate) pair.

    Returns dict of feature_name -> value (float or np.nan).
    """
    feats = {}

    # Normalized forms
    s1_name_n = base_normalize(s1_name)
    s1_addr_n = base_normalize(s1_addr)
    cand_name_n = base_normalize(cand_name)
    cand_addr_n = base_normalize(cand_addr)

    s1_name_core = name_core(s1_name)
    cand_name_core = name_core(cand_name)

    s1_name_toks = name_tokens(s1_name, drop_suffix=True)
    cand_name_toks = name_tokens(cand_name, drop_suffix=True)

    s1_addr_toks = addr_tokens(s1_addr, drop_stopwords=True)
    cand_addr_toks = addr_tokens(cand_addr, drop_stopwords=True)

    # Missing indicators
    s1_addr_missing = 1.0 if not s1_addr_n else 0.0
    cand_addr_missing = 1.0 if not cand_addr_n else 0.0

    # --- NAME FEATURES ---
    if config.use_name_exact:
        feats['name_exact'] = exact_match(s1_name_n, cand_name_n)
        feats['name_core_exact'] = exact_match(s1_name_core, cand_name_core)

    if config.use_name_jaro:
        feats['name_jaro'] = jaro_similarity(s1_name_n, cand_name_n)

    if config.use_name_jarowinkler:
        feats['name_jarowinkler'] = jaro_winkler_similarity(s1_name_n, cand_name_n)

    if config.use_name_levenshtein:
        feats['name_levenshtein'] = levenshtein_similarity(s1_name_n, cand_name_n)

    if config.use_name_token_jaccard:
        feats['name_token_jaccard'] = token_jaccard(s1_name_n, cand_name_n)
        feats['name_token_jaccard_nosuffix'] = token_jaccard(
            ' '.join(s1_name_toks), ' '.join(cand_name_toks)
        )

    if config.use_name_token_dice:
        feats['name_token_dice'] = token_dice(s1_name_n, cand_name_n)
        feats['name_token_dice_nosuffix'] = token_dice(
            ' '.join(s1_name_toks), ' '.join(cand_name_toks)
        )

    if config.use_name_token_set_ratio:
        feats['name_token_set_ratio'] = token_set_ratio(s1_name_n, cand_name_n)

    if config.use_name_token_sort_ratio:
        feats['name_token_sort_ratio'] = token_sort_ratio(s1_name_n, cand_name_n)

    if config.use_name_char_ngram_jaccard:
        feats['name_char3gram_jaccard'] = char_ngram_jaccard(s1_name_n, cand_name_n, n=3)
        feats['name_char4gram_jaccard'] = char_ngram_jaccard(s1_name_n, cand_name_n, n=4)

    # Name containment (asymmetric)
    feats['name_containment_s1_in_cand'] = containment_ratio(s1_name_n, cand_name_n)
    feats['name_containment_cand_in_s1'] = containment_ratio(cand_name_n, s1_name_n)

    # Name digit conflict
    feats['name_digits_conflict'] = name_digits_conflict(s1_name_n, cand_name_n)

    # --- ADDRESS FEATURES ---
    if config.use_addr_exact:
        feats['addr_exact'] = exact_match(s1_addr_n, cand_addr_n)

    if config.use_addr_token_jaccard:
        feats['addr_token_jaccard'] = token_jaccard(s1_addr_n, cand_addr_n)
        feats['addr_token_jaccard_nostop'] = token_jaccard(
            ' '.join(s1_addr_toks), ' '.join(cand_addr_toks)
        )

    if config.use_addr_token_dice:
        feats['addr_token_dice'] = token_dice(s1_addr_n, cand_addr_n)
        feats['addr_token_dice_nostop'] = token_dice(
            ' '.join(s1_addr_toks), ' '.join(cand_addr_toks)
        )

    if config.use_addr_char_jaccard:
        feats['addr_char3gram_jaccard'] = char_ngram_jaccard(s1_addr_n, cand_addr_n, n=3)

    if config.use_addr_levenshtein:
        feats['addr_levenshtein'] = levenshtein_similarity(s1_addr_n, cand_addr_n)

    # Address sub-components
    if config.use_addr_city_match:
        s1_city = extract_city(s1_addr)
        cand_city = extract_city(cand_addr)
        feats['addr_city_match'] = 1.0 if (s1_city and cand_city and s1_city == cand_city) else 0.0

    if config.use_addr_postal_match:
        s1_postal = extract_postal_code(s1_addr)
        cand_postal = extract_postal_code(cand_addr)
        feats['addr_postal_match'] = 1.0 if (s1_postal and cand_postal and s1_postal == cand_postal) else 0.0

    if config.use_addr_housenumber_match:
        s1_hnum = extract_house_number(s1_addr)
        cand_hnum = extract_house_number(cand_addr)
        feats['addr_housenumber_match'] = 1.0 if (s1_hnum and cand_hnum and s1_hnum == cand_hnum) else 0.0

    # Address containment
    feats['addr_containment_s1_in_cand'] = containment_ratio(s1_addr_n, cand_addr_n)
    feats['addr_containment_cand_in_s1'] = containment_ratio(cand_addr_n, s1_addr_n)

    # --- PHONE FEATURES (extracted from address field) ---
    if config.use_phone_exact or config.use_phone_normalized or config.use_phone_suffix:
        # Try to extract phone from address (common patterns)
        s1_phone = _extract_phone_from_address(s1_addr)
        cand_phone = _extract_phone_from_address(cand_addr)

        if config.use_phone_exact:
            feats['phone_exact'] = 1.0 if (s1_phone and cand_phone and s1_phone == cand_phone) else 0.0

        if config.use_phone_normalized:
            s1_phone_n = normalize_phone(s1_phone) if s1_phone else ""
            cand_phone_n = normalize_phone(cand_phone) if cand_phone else ""
            feats['phone_normalized_match'] = 1.0 if (s1_phone_n and cand_phone_n and s1_phone_n == cand_phone_n) else 0.0

        if config.use_phone_suffix:
            feats['phone_suffix7_match'] = phone_suffix_match(s1_phone, cand_phone, 7)
            feats['phone_suffix10_match'] = phone_suffix_match(s1_phone, cand_phone, 10)

        if config.use_phone_missing:
            feats['phone_missing_s1'] = 1.0 if not s1_phone else 0.0
            feats['phone_missing_cand'] = 1.0 if not cand_phone else 0.0
            feats['phone_missing_either'] = 1.0 if (not s1_phone or not cand_phone) else 0.0

    # --- COUNTRY ---
    if config.use_country_match:
        s1_ck = country_key(s1_country)
        cand_ck = country_key(cand_country)
        feats['country_match'] = 1.0 if s1_ck == cand_ck else 0.0

    # --- CROSS-SCRIPT ---
    if config.use_cross_script:
        feats['is_cross_script'] = is_cross_script(s1_name, cand_name)

    if config.use_transliteration_sim:
        feats['transliteration_sim'] = transliteration_similarity(s1_name, cand_name)

    # --- MISSINGNESS INDICATORS ---
    if config.use_missing_indicators:
        feats['addr_missing_s1'] = s1_addr_missing
        feats['addr_missing_cand'] = cand_addr_missing
        feats['addr_missing_either'] = 1.0 if (s1_addr_missing or cand_addr_missing) else 0.0
        feats['addr_missing_both'] = 1.0 if (s1_addr_missing and cand_addr_missing) else 0.0
        feats['name_len_s1'] = float(len(s1_name_n))
        feats['name_len_cand'] = float(len(cand_name_n))
        feats['addr_len_s1'] = float(len(s1_addr_n))
        feats['addr_len_cand'] = float(len(cand_addr_n))
        feats['name_len_diff'] = abs(len(s1_name_n) - len(cand_name_n))
        feats['addr_len_diff'] = abs(len(s1_addr_n) - len(cand_addr_n))
        feats['name_token_count_s1'] = float(len(s1_name_toks))
        feats['name_token_count_cand'] = float(len(cand_name_toks))
        feats['addr_token_count_s1'] = float(len(s1_addr_toks))
        feats['addr_token_count_cand'] = float(len(cand_addr_toks))

    # --- CROSS-FIELD FEATURES ---
    if config.use_name_high_addr_low or config.use_name_low_addr_high or config.use_sim_disagreement:
        name_sim = feats.get('name_token_jaccard', 0.0)
        addr_sim = feats.get('addr_token_jaccard', 0.0)

        if config.use_name_high_addr_low:
            # High name sim, low addr sim = potential false positive (generic name)
            feats['name_high_addr_low'] = 1.0 if (name_sim > 0.7 and addr_sim < 0.3) else 0.0

        if config.use_name_low_addr_high:
            # Low name sim, high addr sim = address-only match (FINDING 3)
            feats['name_low_addr_high'] = 1.0 if (name_sim < 0.3 and addr_sim > 0.5) else 0.0

        if config.use_sim_disagreement:
            feats['sim_disagreement'] = abs(name_sim - addr_sim)

    # Encode missing as NaN for tree-based models (XGBoost/LightGBM handle NaN natively)
    if config.encode_missing_as_nan:
        for k, v in feats.items():
            if isinstance(v, float) and (v != v):  # NaN check
                feats[k] = np.nan

    return feats


def _extract_phone_from_address(addr: str) -> Optional[str]:
    """Extract phone number from address string (heuristic)."""
    if not addr:
        return None
    # Look for phone patterns: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, etc.
    patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # International
        r'\b\d{10}\b',  # 10 digits
    ]
    for pat in patterns:
        m = re.search(pat, addr)
        if m:
            return m.group(0)
    return None


# -----------------------------------------------------------------------------
# Group-relative features (computed per S1 entity's candidate group)
# -----------------------------------------------------------------------------

def compute_group_relative_features(
    df: pd.DataFrame,
    score_col: str = 'blocker_score',
    config=None,
) -> pd.DataFrame:
    """Add group-relative features per S1 entity.

    Args:
        df: DataFrame with columns including 's1_entity_id' and score_col
        score_col: Column with blocker/retrieval score for ranking
        config: XGBoostConfig with feature flags

    Returns:
        DataFrame with added group-relative feature columns.
    """
    if config is None:
        config = XGBoostConfig()

    df = df.copy()

    # Rank within each S1 group (1 = best score)
    if config.use_group_rank:
        df['group_rank'] = df.groupby('s1_entity_id')[score_col].rank(
            ascending=False, method='min'
        ).astype(int)
        df['is_top1'] = (df['group_rank'] == 1).astype(float)

    if config.use_group_margin_to_2nd:
        # Margin to 2nd best score in group
        def margin_to_2nd(group):
            scores = group[score_col].values
            if len(scores) < 2:
                return np.nan
            sorted_scores = np.sort(scores)[::-1]  # descending
            return sorted_scores[0] - sorted_scores[1]

        margins = df.groupby('s1_entity_id').apply(margin_to_2nd)
        df['score_margin_to_2nd'] = df['s1_entity_id'].map(margins)

    if config.use_group_score_ratio_to_max:
        # Ratio to max score in group
        max_scores = df.groupby('s1_entity_id')[score_col].transform('max')
        df['score_ratio_to_max'] = df[score_col] / max_scores.replace(0, np.nan)

    if config.use_group_size:
        df['group_size'] = df.groupby('s1_entity_id')['s1_entity_id'].transform('size').astype(float)

    if config.use_n_competitors:
        # Number of candidates with score within 10% of top score
        max_scores = df.groupby('s1_entity_id')[score_col].transform('max')
        df['n_competitors'] = ((df[score_col] >= 0.9 * max_scores) & (df[score_col] > 0)).groupby(
            df['s1_entity_id']
        ).transform('sum').astype(float)

    # is_mutual_best requires S2/S3 perspective - would need reverse index
    # Placeholder - can be computed if we have full bipartite data
    if config.use_is_mutual_best:
        df['is_mutual_best'] = 0.0  # Requires cross-source computation

    return df


# Import config for type hints
from .config import XGBoostConfig