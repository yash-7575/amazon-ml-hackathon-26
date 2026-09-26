"""The ONLY module in `matcher/` permitted to read ground truth.

`leakage_check.py` enforces that `features.py` cannot import this. That is a
structural guarantee rather than a convention: the extractor literally cannot encode
a label because it cannot see one.

Note on singletons: `blocking.measure.load_gt` DISCARDS ground-truth rows with an
empty match list, which makes "singleton" and "absent from the file" indistinguishable.
Every S1 row in train does have a ground-truth row (verified: 2,206,821 of 2,206,821),
and 123,247 of them are singletons -- 5.58% of the evaluation set, each worth a full
1.0 of macro-F0.5. So this module keeps the empties.
"""
from __future__ import annotations
import os
from typing import Dict, Iterable, List, Set
import numpy as np

GT_PATH = ('/home/yash/Downloads/Amazon-ML-dataset/student_resource/'
           'dataset/train/train_ground_truth.tsv')


def load_truth(s1_ids: Iterable[str] | None = None,
               path: str = GT_PATH) -> Dict[str, Set[str]]:
    """{s1_id: set(matched_ids)}. KEEPS empty sets -- they are the singletons.

    Pass `s1_ids` to restrict the read to the entities you care about; the file is
    127 MB and holding all 2.2M sets is unnecessary for a sample.
    """
    want = set(s1_ids) if s1_ids is not None else None
    truth: Dict[str, Set[str]] = {}
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            sid, _, rest = line.rstrip('\n').partition('\t')
            if want is not None and sid not in want:
                continue
            truth[sid] = {x for x in rest.split(',') if x}
    return truth


def labels_for_pairs(s1_ids: List[str], cand_ids: List[str],
                     truth: Dict[str, Set[str]]) -> np.ndarray:
    """y aligned to the pair arrays. 1 iff the candidate is a true match for that S1."""
    y = np.zeros(len(s1_ids), dtype=np.int8)
    for k in range(len(s1_ids)):
        t = truth.get(s1_ids[k])
        if t and cand_ids[k] in t:
            y[k] = 1
    return y


def group_stats(s1_ids_unique: List[str], truth: Dict[str, Set[str]]) -> dict:
    """Distribution facts used to sanity-check a sample against the evaluation set."""
    sizes = np.asarray([len(truth.get(s, set())) for s in s1_ids_unique])
    return {
        'n_s1': len(sizes),
        'n_singleton': int((sizes == 0).sum()),
        'pct_singleton': float(100.0 * (sizes == 0).mean()) if len(sizes) else 0.0,
        'n_exactly_one': int((sizes == 1).sum()),
        'mean_true_size': float(sizes.mean()) if len(sizes) else 0.0,
        'max_true_size': int(sizes.max()) if len(sizes) else 0,
        'total_true_pairs': int(sizes.sum()),
    }
