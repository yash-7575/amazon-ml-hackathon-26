"""THE single frozen featurization entry point.

Both the training path and the inference path call `featurize()` and nothing else.
That is the whole point: two code paths that each build "the same" feature matrix
drift, and the drift is silent -- the model simply gets worse and no test fails.

Order of columns is fixed by construction (base -> extra -> group -> reciprocal) and
asserted against the saved column list at inference time.

No labels enter here. `leakage_check` enforces the import boundary on `features.py`;
this module is held to the same rule by review and by `_forbidden_imports` below.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np

from .featconfig import FeatureConfig
from . import features as F
from . import features_extra as FX
from .group_features import group_features, reciprocal_features


def prepare(names: Sequence[str], addrs: Sequence[str],
            cfg: FeatureConfig) -> Dict[str, list]:
    """Per-record normalization for both feature modules, done once."""
    base = F.prepare_records(names, addrs, cfg)
    base.update(FX.prepare_extra(names, addrs, cfg))
    return base


def featurize(li: np.ndarray, ri: np.ndarray,
              L: Dict[str, list], R: Dict[str, list],
              cfg: FeatureConfig,
              retrieval_score: np.ndarray,
              source_tag: Sequence[str],
              s1_group: np.ndarray,
              cand_group: np.ndarray,
              name_idf: Dict[str, float] | None = None,
              a_idf: Dict[str, float] | None = None) -> Tuple[np.ndarray, List[str]]:
    """Build the full pair-feature matrix.

    s1_group / cand_group are INTEGER group keys (not id strings) used by the
    group-relative family. They must be formed identically at train and inference --
    changing K or the blocker's rescore weights shifts every rank and margin here.
    """
    M1, c1 = F.extract(li, ri, L, R, cfg, retrieval_score=retrieval_score, idf=name_idf)
    M2, c2 = FX.extract_extra(li, ri, L, R, cfg, source_tag=source_tag, a_idf=a_idf)

    # Group-relative features rank by the RETRIEVAL score, which is available at
    # inference before the model has scored anything -- so they are legitimately
    # computable on test data.
    M3, c3 = group_features(s1_group, retrieval_score)
    M4, c4 = reciprocal_features(s1_group, cand_group, retrieval_score)

    M = np.hstack([M1, M2, M3, M4]).astype(cfg.dtype)
    cols = list(c1) + list(c2) + list(c3) + list(c4)
    assert M.shape[1] == len(cols), f'{M.shape[1]} columns vs {len(cols)} names'
    return M, cols
