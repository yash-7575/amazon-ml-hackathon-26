"""Predictions -> the two official submission files.

Memory: India test alone is ~121M scored pairs. Holding them as Python tuples would not
fit, so ids are interned to int32 and everything below is array work. Countries never mix
(ground truth never crosses a country), so each is processed independently and appended.

Produces WIDE format, per the official spec:
    source1_entity_id \\t matched_entity_ids
one row per S1 entity in test_source1.tsv, comma-joined, EMPTY allowed and required for
singletons. Rows are emitted by iterating the required id list read from test_source1.tsv
-- never by iterating predictions, which would silently drop every entity we found nothing
for and forfeit the singleton credit that is worth the most.

Usage:
  PYTHONPATH=. python -m matcher.submit --preds p1.tsv p2.tsv --cands c1.tsv c2.tsv \
      --out-dir output --max-k 8
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from collections import defaultdict
import numpy as np

from .evaluate import f_beta_05

DATA_ROOT = '/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/'
csv.field_size_limit(10**9)


def required_s1(path: str) -> list[str]:
    ids = []
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            i = line.find('\t')
            ids.append(line[:i] if i > 0 else line.rstrip('\n'))
    return ids


def best_n(probs: np.ndarray, max_k: int) -> int:
    """Expected-F0.5 choice of how many to keep. n=0 (empty) is a real option -- that is
    how a singleton is chosen, and it is worth a full 1.0 when correct."""
    tot = float(probs.sum())
    if tot <= 0:
        return 0
    best_v = float(np.prod(1.0 - probs))       # expected score of predicting nothing
    best_i, run = 0, 0.0
    for n in range(1, min(max_k, len(probs)) + 1):
        run += float(probs[n - 1])
        v = f_beta_05(run / n, run / tot)
        if v > best_v:
            best_i, best_v = n, v
    return best_i


def load_preds(paths: list[str]):
    s1s, cds, ps = [], [], []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            next(f)
            for line in f:
                a, b, c = line.rstrip('\n').split('\t')
                s1s.append(a); cds.append(b); ps.append(c)
    return s1s, cds, np.asarray(ps, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', nargs='+', required=True)
    ap.add_argument('--cands', nargs='+', required=True,
                    help='long candidate files -> candidate_pairs.tsv')
    ap.add_argument('--out-dir', default='output')
    ap.add_argument('--max-k', type=int, default=8)
    ap.add_argument('--split', default='test')
    ap.add_argument('--no-conflict', action='store_true')
    ap.add_argument('--wide-cands', action='store_true',
                    help='--cands are already wide (s1 <TAB> comma-joined ids). The long '
                         'files are deleted per country to reclaim disk, so this is the '
                         'normal path.')
    a = ap.parse_args()
    t0 = time.time()
    def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
    os.makedirs(a.out_dir, exist_ok=True)

    req = required_s1(f'{DATA_ROOT}{a.split}/{a.split}_source1.tsv')
    log(f'required S1 = {len(req):,}')

    log('loading predictions ...')
    s1s, cds, prob = load_preds(a.preds)
    log(f'  {len(prob):,} scored pairs')

    # intern to int32 -- string tuples at this scale do not fit
    su, sidx = np.unique(np.asarray(s1s), return_inverse=True)
    cu, cidx = np.unique(np.asarray(cds), return_inverse=True)
    del s1s, cds
    log(f'  {len(su):,} S1 / {len(cu):,} candidates interned')

    keep = np.ones(len(prob), dtype=bool)
    if not a.no_conflict:
        # Ground truth is a PARTITION of the candidate side: no S2/S3 record belongs to
        # two S1 entities. Keep each candidate only for its most confident claimant.
        order = np.argsort(-prob, kind='stable')
        seen = np.full(len(cu), -1, dtype=np.int64)
        keep[:] = False
        for k in order:
            c = cidx[k]
            if seen[c] == -1:
                seen[c] = sidx[k]; keep[k] = True
            elif seen[c] == sidx[k]:
                keep[k] = True
        log(f'  conflict resolution kept {keep.sum():,} of {len(keep):,}')

    si, ci, pp = sidx[keep], cidx[keep], prob[keep]
    order = np.lexsort((-pp, si))
    si, ci, pp = si[order], ci[order], pp[order]
    bounds = np.flatnonzero(np.diff(si)) + 1
    starts = np.concatenate([[0], bounds])
    sizes = np.diff(np.concatenate([starts, [len(si)]]))

    preds: dict[str, list[str]] = {}
    n_empty = 0
    for st, sz in zip(starts, sizes):
        n = best_n(pp[st:st + sz].astype(np.float64), a.max_k)
        if n == 0:
            n_empty += 1
            continue
        preds[su[si[st]]] = [cu[c] for c in ci[st:st + n]]
    log(f'  {len(preds):,} S1 with matches; {n_empty:,} predicted empty')

    log('loading candidate lists ...')
    cand: dict[str, list[str]] = defaultdict(list)
    if a.wide_cands:
        for p in a.cands:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    i = line.find('\t')
                    if i < 0:
                        continue
                    sid, rest = line[:i], line[i + 1:].rstrip('\n')
                    if rest:
                        cand[sid].extend(rest.split(','))
    else:
        seen_pair: set = set()
        for p in a.cands:
            with open(p, encoding='utf-8') as f:
                next(f)
                for line in f:
                    i = line.find('\t'); j = line.find('\t', i + 1)
                    s_, c_ = line[:i], line[i + 1:j]
                    if (s_, c_) in seen_pair:
                        continue
                    seen_pair.add((s_, c_)); cand[s_].append(c_)
        del seen_pair

    def write(path, col, mapping, extra=None):
        n_rows = n_blank = 0
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE)
            w.writerow(['source1_entity_id', col])
            for sid in req:                       # iterate REQUIRED ids, not predictions
                ids = mapping.get(sid, [])
                if extra is not None:
                    # every matched id must also appear as a candidate
                    ids = list(dict.fromkeys(list(extra.get(sid, [])) + list(ids)))
                else:
                    ids = list(dict.fromkeys(ids))
                w.writerow([sid, ','.join(ids)])
                n_rows += 1
                n_blank += (not ids)
        log(f'  {path}: {n_rows:,} rows, {n_blank:,} empty')

    write(os.path.join(a.out_dir, 'matching_results.tsv'), 'matched_entity_ids', preds)
    # candidate_pairs must be a SUPERSET of matching_results -- union them so a matched id
    # can never be missing from the candidate file (the validator rejects that).
    write(os.path.join(a.out_dir, 'candidate_pairs.tsv'), 'candidate_entity_ids',
          cand, extra=preds)
    log('done')


if __name__ == '__main__':
    main()
