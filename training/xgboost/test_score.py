"""TEST scoring: stream LONG candidates -> 54 exact training features -> XGBoost P(match).

Same code path as training: training.xgboost.features.compute_pair_features
with default XGBoostConfig flags. Feature-name ORDER is forced to the saved
model's list; any mismatch aborts (train/test drift guard).

Multiprocessing: main process holds id->fields dicts and streams batches;
Pool workers featurize string tuples only (no big shared state). Predict in
main process. Output per-country scored TSV: s1_entity_id, cand_entity_id, prob.

Usage (from repo root):
    python -m training.xgboost.test_score --country france \
        --long output/candidates_france.long.tsv --out output/scored_france.tsv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'blocking'))

from training.xgboost.config import XGBoostConfig
from training.xgboost.features import compute_pair_features

CFG = None
ORDER = None


def _init(cfg_dict: dict, order: list):
    global CFG, ORDER
    from training.xgboost.config import XGBoostConfig
    CFG = XGBoostConfig(**cfg_dict)
    ORDER = order


def _feat(tup) -> list:
    s1n, s1a, s1c, cn, ca, cc = tup
    f = compute_pair_features(s1n, s1a, s1c, cn, ca, cc, CFG)
    return [f[k] for k in ORDER]


def load_fields(path: str, country: str, tag: str):
    """Load {entity_id: (name, addr, country)} for one country (key matched)."""
    from blocking.normalize import country_key
    d = {}
    with open(path, encoding='utf-8', newline='') as fh:
        r = csv.DictReader(fh, delimiter='\t')
        for row in r:
            if country_key(row.get('country', '')) != country:
                continue
            d[row['entity_id']] = (row.get('business_name', '') or '',
                                   row.get('business_address', '') or '',
                                   row.get('country', '') or '')
    print(f'[score] {tag}: {len(d):,} rows for country={country!r}', flush=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--country', required=True)
    ap.add_argument('--long', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=r'D:\Amazon_challenge_ML\Amazon-ML-dataset\student_resource\dataset\test')
    ap.add_argument('--model-pkl', default=r'D:\Amazon_challenge_ML\amazon-ml-hackathon-26\training\xgboost\outputs\models\xgboost_model.pkl')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--batch-pairs', type=int, default=20000)
    args = ap.parse_args()

    import pickle
    import xgboost as xgb
    with open(args.model_pkl, 'rb') as fh:
        m = pickle.load(fh)
    booster = m['model']
    order = list(m['feature_names'])
    print(f'[score] model: {len(order)} features, best_iter={m.get("best_iteration")}', flush=True)
    try:
        it_range = (0, booster.best_iteration + 1)
    except Exception:
        it_range = None

    cfg = XGBoostConfig()
    s1 = load_fields(os.path.join(args.data_root, 'test_source1.tsv'), args.country, 'S1')
    s2 = load_fields(os.path.join(args.data_root, 'test_source2.tsv'), args.country, 'S2')
    s3 = load_fields(os.path.join(args.data_root, 'test_source3.tsv'), args.country, 'S3')
    cand = dict(s2)
    cand.update(s3)
    del s2, s3
    print(f'[score] cand dict: {len(cand):,}', flush=True)

    pool = Pool(args.workers, initializer=_init,
                initargs=(cfg.to_dict(), order))
    t0 = time.time()
    n_pairs = n_batches = 0
    checked = False
    batch_tups, batch_keys = [], []
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fout = open(args.out, 'w', encoding='utf-8', newline='')
    w = csv.writer(fout, delimiter='\t', lineterminator='\n')

    def flush():
        global checked, n_pairs, n_batches
        if not batch_tups:
            return
        rows = pool.map(_feat, batch_tups, chunksize=500)
        if not checked:
            probe = compute_pair_features(*batch_tups[0], cfg)
            assert set(probe.keys()) == set(order), \
                f'FEATURE DRIFT: {set(probe.keys()) ^ set(order)}'
            print('[score] feature parity OK (54/54)', flush=True)
            checked = True
        X = np.asarray(rows, dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=order)
        probs = booster.predict(dm, iteration_range=it_range) if it_range \
            else booster.predict(dm)
        for (s, c), p in zip(batch_keys, probs):
            w.writerow([s, c, f'{float(p):.6f}'])
        n_pairs += len(batch_keys)
        n_batches += 1
        if n_batches % 20 == 0:
            el = time.time() - t0
            print(f'[score] {n_pairs:,} pairs, {n_batches} batches, '
                  f'{n_pairs/max(el,1):,.0f} pairs/s, el {el:.0f}s', flush=True)

    with open(args.long, encoding='utf-8', newline='') as fh:
        r = csv.DictReader(fh, delimiter='\t')
        for row in r:
            s = row['s1_entity_id']
            c = row['candidate_entity_id']
            try:
                s1t = s1[s]
                ct = cand[c]
            except KeyError:
                continue
            batch_tups.append((s1t[0], s1t[1], s1t[2], ct[0], ct[1], ct[2]))
            batch_keys.append((s, c))
            if len(batch_tups) >= args.batch_pairs:
                flush()
                batch_tups, batch_keys = [], []
    flush()
    pool.close()
    pool.join()
    fout.close()
    el = time.time() - t0
    print(f'[score] DONE {n_pairs:,} pairs in {el:.0f}s '
          f'({n_pairs/max(el,1):,.0f} pairs/s)', flush=True)


if __name__ == '__main__':
    main()
