"""Long candidate file -> (X, y, groups) on disk.

Usage:
    PYTHONPATH=. python -m matcher.build_dataset \
        --pairs blocking/results/smoke_test_india_2000_candidate_pairs.long.tsv \
        --tag india_capped

Writes matcher/data/<tag>_{X.npy,y.npy,meta.npz,cols.json}.
`meta.npz` carries the id strings so assign/evaluate can map rows back to entities
without re-reading the pair file.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .featconfig import FeatureConfig
from .featurize import prepare, featurize
from . import features as F
from . import features_extra as FX
from . import records as RC
from . import label as LB

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')


def build(pairs_path: str, tag: str, split: str = 'train',
          with_labels: bool = True, cfg: FeatureConfig | None = None) -> dict:
    cfg = cfg or FeatureConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

    log(f'reading pairs {pairs_path}')
    s1_ids, cand_ids, rscore, src = RC.read_long_pairs(pairs_path)
    log(f'  {len(s1_ids):,} pairs')

    log(f'loading referenced records from split={split!r}')
    left, right = RC.load_sides(s1_ids, cand_ids, split=split)
    log(f'  left(S1)={len(left):,}  right(S2+S3)={len(right):,}')

    li, ri, keep = RC.pair_indices(s1_ids, cand_ids, left, right)
    if not keep.all():
        n_drop = int((~keep).sum())
        log(f'  WARNING dropping {n_drop:,} pairs with an unresolved side')
        li, ri, rscore = li[keep], ri[keep], rscore[keep]
        s1_ids = [s for s, k in zip(s1_ids, keep) if k]
        cand_ids = [c for c, k in zip(cand_ids, keep) if k]
        src = [s for s, k in zip(src, keep) if k]

    log('preparing per-record normalization')
    L = prepare(left.names, left.addrs, cfg)
    R = prepare(right.names, right.addrs, cfg)

    # Corpus IDF over the records actually in play. An unsupervised statistic --
    # computing it per split is correct, not leakage.
    name_idf = F.token_idf(L['name_tokens'] + R['name_tokens'])
    a_idf = FX.addr_idf(L['addr_tokens'] + R['addr_tokens'])

    # Integer group keys. li already IS the S1 row index, so it is the group key.
    s1_group = li.astype(np.int64)
    cand_group = ri.astype(np.int64)

    log('extracting features')
    X, cols = featurize(li, ri, L, R, cfg, retrieval_score=rscore, source_tag=src,
                        s1_group=s1_group, cand_group=cand_group,
                        name_idf=name_idf, a_idf=a_idf)
    log(f'  X={X.shape} ({X.nbytes/2**20:.0f} MiB)  {len(cols)} features')

    y = None
    stats = {}
    if with_labels:
        log('labelling')
        uniq_s1 = left.ids
        truth = LB.load_truth(uniq_s1)
        y = LB.labels_for_pairs(s1_ids, cand_ids, truth)
        stats = LB.group_stats(uniq_s1, truth)
        n_pos = int(y.sum())
        log(f'  positives={n_pos:,} / {len(y):,} ({100*n_pos/max(len(y),1):.3f}%)  '
            f'neg:pos={(len(y)-n_pos)/max(n_pos,1):.0f}:1')
        log(f'  candidate recall = {n_pos:,}/{stats["total_true_pairs"]:,} = '
            f'{100*n_pos/max(stats["total_true_pairs"],1):.2f}%')
        log(f'  S1 singletons in sample: {stats["n_singleton"]:,} '
            f'({stats["pct_singleton"]:.2f}%)   mean true size={stats["mean_true_size"]:.2f}')

    np.save(os.path.join(OUT_DIR, f'{tag}_X.npy'), X)
    if y is not None:
        np.save(os.path.join(OUT_DIR, f'{tag}_y.npy'), y)
    np.savez_compressed(
        os.path.join(OUT_DIR, f'{tag}_meta.npz'),
        s1_ids=np.array(s1_ids), cand_ids=np.array(cand_ids),
        src=np.array(src), rscore=rscore,
        s1_group=s1_group, cand_group=cand_group,
        left_ids=np.array(left.ids), left_countries=np.array(left.countries),
    )
    with open(os.path.join(OUT_DIR, f'{tag}_cols.json'), 'w') as f:
        json.dump({'cols': cols, 'cfg': json.loads(cfg.to_json()),
                   'fingerprint': cfg.fingerprint(), 'pairs_path': pairs_path,
                   'split': split, 'stats': stats}, f, indent=2)
    log(f'wrote {OUT_DIR}/{tag}_*  (fingerprint {cfg.fingerprint()})')
    return {'X': X, 'y': y, 'cols': cols, 'stats': stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--no-labels', action='store_true')
    a = ap.parse_args()
    build(a.pairs, a.tag, split=a.split, with_labels=not a.no_labels)


if __name__ == '__main__':
    main()
