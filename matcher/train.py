"""LightGBM pairwise matcher.

The split is BY S1 GROUP, never within one. Group-relative features (rank_in_group,
score_margin_to_2nd, is_mutual_best, n_competitors_for_cand) are computed across a whole
candidate group, so putting some of a group's rows in train and the rest in validation
leaks: the validation row's rank was partly determined by rows the model trained on.
That inflates validation scores and the inflation is invisible.

LightGBM handles NaN natively, which is why `features.py` emits NaN for undefined
similarity instead of 0.0 -- 0.0 would assert "these disagree", a false statement about
a pair where one side simply has no address.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import lightgbm as lgb

DATA = os.path.join(os.path.dirname(__file__), 'data')
MODELS = os.path.join(os.path.dirname(__file__), 'models')

PARAMS = dict(
    objective='binary',
    metric=['binary_logloss', 'auc'],
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    num_threads=6,          # leave cores for the concurrent candidate-generation job
    verbosity=-1,
    seed=20260926,
)


def load(tag: str):
    X = np.load(os.path.join(DATA, f'{tag}_X.npy'))
    y = np.load(os.path.join(DATA, f'{tag}_y.npy'))
    meta = np.load(os.path.join(DATA, f'{tag}_meta.npz'), allow_pickle=False)
    cols = json.load(open(os.path.join(DATA, f'{tag}_cols.json')))['cols']
    return X, y, meta, cols


def group_split(s1_ids: np.ndarray, frac_val: float = 0.25, seed: int = 20260926):
    """Split on distinct S1 ids, then map back to rows. Returns (train_mask, val_mask)."""
    uniq = np.unique(s1_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * frac_val)))
    val_ids = set(uniq[:n_val].tolist())
    val = np.array([s in val_ids for s in s1_ids])
    return ~val, val


def train(tag: str, frac_val: float = 0.25, rounds: int = 2000,
          early: int = 100, out_tag: str | None = None) -> dict:
    out_tag = out_tag or tag
    os.makedirs(MODELS, exist_ok=True)
    X, y, meta, cols = load(tag)
    s1 = meta['s1_ids']

    tr, va = group_split(s1, frac_val=frac_val)
    # the leak this guards against is invisible in any metric -- assert it
    assert not (set(s1[tr].tolist()) & set(s1[va].tolist())), \
        'an S1 group appears in both train and validation'
    print(f'[train] rows train={tr.sum():,} val={va.sum():,}  '
          f'S1 train={len(set(s1[tr].tolist())):,} val={len(set(s1[va].tolist())):,}')
    print(f'[train] positives train={int(y[tr].sum()):,} val={int(y[va].sum()):,}')

    dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=cols, free_raw_data=False)
    dva = lgb.Dataset(X[va], label=y[va], feature_name=cols, reference=dtr,
                      free_raw_data=False)
    evals: dict = {}
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=rounds, valid_sets=[dtr, dva],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(early, verbose=True),
                   lgb.log_evaluation(100),
                   lgb.record_evaluation(evals)],
    )
    booster.save_model(os.path.join(MODELS, f'{out_tag}.lgb'),
                       num_iteration=booster.best_iteration)

    imp = sorted(zip(cols, booster.feature_importance('gain')),
                 key=lambda t: -t[1])
    with open(os.path.join(MODELS, f'{out_tag}_importance.json'), 'w') as f:
        json.dump([{'feature': c, 'gain': float(g)} for c, g in imp], f, indent=2)
    print('\n[train] top 20 features by gain:')
    for c, g in imp[:20]:
        print(f'   {g:14,.0f}  {c}')
    zero = [c for c, g in imp if g == 0]
    if zero:
        print(f'[train] {len(zero)} features contributed ZERO gain: {zero}')

    np.save(os.path.join(MODELS, f'{out_tag}_val_mask.npy'), va)
    return {'booster': booster, 'val': va, 'cols': cols,
            'best_iter': booster.best_iteration,
            'best_auc': booster.best_score['val']['auc']}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--frac-val', type=float, default=0.25)
    ap.add_argument('--out-tag', default=None)
    a = ap.parse_args()
    r = train(a.tag, frac_val=a.frac_val, out_tag=a.out_tag)
    print(f"\n[train] best_iteration={r['best_iter']}  val AUC={r['best_auc']:.5f}")


if __name__ == '__main__':
    main()
