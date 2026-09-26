"""Load ONLY the record text the candidate pairs actually reference.

The full sources are 5.03M + 5.29M rows. A candidate file for a few thousand S1
entities references a small fraction of that, so we collect the distinct ids first
and then stream each source file exactly once, keeping only the rows we need.
Never load a whole source into memory.

Deliberately no pandas: `read_tsv` streaming keeps this bounded and matches the
blocker's own I/O path, so normalization sees identical raw strings.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from blocking.io_utils import read_tsv

DATA_ROOT = '/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/'


@dataclass
class Side:
    """Parallel arrays for one side of the pairing, plus an id -> row-index map."""
    ids: List[str]
    names: List[str]
    addrs: List[str]
    countries: List[str]
    index: Dict[str, int]

    def __len__(self) -> int:
        return len(self.ids)


def read_long_pairs(path: str) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """Read the long candidate file.

    Returns (s1_id_per_pair, cand_id_per_pair, retrieval_score, source_tag_per_pair).
    Row order is preserved exactly -- every downstream array is aligned to it.
    """
    s1: List[str] = []
    cand: List[str] = []
    score: List[float] = []
    src: List[str] = []
    with open(path, encoding='utf-8') as f:
        header = next(f).rstrip('\n').split('\t')
        want = ['s1_entity_id', 'candidate_entity_id', 'source', 'score']
        if header != want:
            raise ValueError(f'{path}: expected header {want}, got {header}')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 4:
                continue
            s1.append(parts[0]); cand.append(parts[1])
            src.append(parts[2]); score.append(float(parts[3]))
    return s1, cand, np.asarray(score, dtype=np.float32), src


def load_sides(s1_ids: List[str], cand_ids: List[str], split: str = 'train',
               data_root: str = DATA_ROOT) -> Tuple[Side, Side]:
    """Stream the three source files once each, keeping only referenced rows.

    Returns (left, right) where left holds the S1 records and right holds the S2/S3
    records in one combined table -- the entity_id prefix already says which source a
    row came from, so a single table keeps the pair indexing simple.
    """
    need_s1 = set(s1_ids)
    need_cand = set(cand_ids)
    d = os.path.join(data_root, split)

    def _collect(fname: str, wanted: set) -> Tuple[list, list, list, list]:
        ids, nm, ad, co = [], [], [], []
        for row in read_tsv(os.path.join(d, fname)):
            eid = row['entity_id']
            if eid in wanted:
                ids.append(eid)
                nm.append(row.get('business_name', '') or '')
                ad.append(row.get('business_address', '') or '')
                co.append(row.get('country', '') or '')
        return ids, nm, ad, co

    prefix = f'{split}_source'
    li, ln, la, lc = _collect(f'{prefix}1.tsv', need_s1)
    left = Side(li, ln, la, lc, {e: k for k, e in enumerate(li)})

    ri, rn, ra, rc = [], [], [], []
    for n in (2, 3):
        a, b, c, e = _collect(f'{prefix}{n}.tsv', need_cand)
        ri += a; rn += b; ra += c; rc += e
    right = Side(ri, rn, ra, rc, {e: k for k, e in enumerate(ri)})

    missing_l = len(need_s1) - len(left)
    missing_r = len(need_cand) - len(right)
    if missing_l or missing_r:
        # Not fatal, but it means the candidate file and the split disagree -- the
        # usual cause is pointing at train candidates with split='test'.
        print(f'[records] WARNING: {missing_l:,} S1 and {missing_r:,} candidate ids '
              f'in the pair file were not found in split={split!r}')
    return left, right


def pair_indices(s1_ids: List[str], cand_ids: List[str],
                 left: Side, right: Side) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the per-pair id strings onto row indices into `left` / `right`.

    Returns (li, ri, keep) -- keep is the boolean mask of pairs whose BOTH sides were
    resolved. Pairs referencing a missing record are dropped rather than silently
    featurized against a wrong row.
    """
    n = len(s1_ids)
    li = np.full(n, -1, dtype=np.int64)
    ri = np.full(n, -1, dtype=np.int64)
    lm, rm = left.index, right.index
    for k in range(n):
        li[k] = lm.get(s1_ids[k], -1)
        ri[k] = rm.get(cand_ids[k], -1)
    keep = (li >= 0) & (ri >= 0)
    return li, ri, keep
