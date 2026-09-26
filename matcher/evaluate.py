"""Macro-averaged F-beta (beta=0.5), exactly as the official README specifies.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Computed PER Source-1 entity, then averaged over ALL S1 entities in the evaluation
set -- singletons included. A singleton scores 1.0 when you correctly predict an empty
list and 0.0 when you predict anything. So correct singleton identification earns real
credit and false merges on singletons are penalised.

Why this file has unit tests at the bottom: every threshold decision downstream is read
off this number. If the metric is wrong the whole pipeline optimises the wrong thing and
nothing else would reveal it.

The asymmetry this metric creates is NOT uniform, and it is the single most important
fact for assign.py. Loss from one error, by true group size T:

    T | one false positive | one miss
    1 |       0.444        |  1.000   <- a MISS is 2.3x worse
    2 |       0.286        |  0.167
    3 |       0.211        |  0.091
    4 |       0.167        |  0.063
    6 |       0.118        |  0.039

For T >= 2 precision dominates, as the metric's design intends. At T = 1 it REVERSES,
because an empty prediction scores a flat 0.0. Hence: never emit an empty list unless
you positively believe the entity is a singleton.
"""
from __future__ import annotations
from typing import Dict, Iterable, Set, Tuple
import numpy as np

BETA2 = 0.25          # beta^2 for beta = 0.5
ONE_PLUS_BETA2 = 1.25


def f_beta_05(precision: float, recall: float) -> float:
    denom = BETA2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (ONE_PLUS_BETA2 * precision * recall) / denom


def score_one(pred: Set[str], truth: Set[str]) -> float:
    """F0.5 for a single S1 entity.

    Both empty -> 1.0: a correctly identified singleton. This is the branch that is
    easy to get wrong (0/0 in P and R) and it is worth 5.58% of the entities.
    """
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    hits = len(pred & truth)
    if hits == 0:
        return 0.0
    return f_beta_05(hits / len(pred), hits / len(truth))


def macro_f05(preds: Dict[str, Set[str]], truth: Dict[str, Set[str]],
              s1_ids: Iterable[str]) -> Tuple[float, dict]:
    """Average score_one over EVERY id in s1_ids.

    s1_ids is passed explicitly rather than inferred from `preds`, because an S1 entity
    the model produced nothing for still counts in the denominator -- silently dropping
    it would inflate the score.
    """
    s1_ids = list(s1_ids)
    per = np.empty(len(s1_ids), dtype=np.float64)
    tp = fp = fn = 0
    n_pred_empty = n_true_empty = 0
    for k, sid in enumerate(s1_ids):
        p = preds.get(sid, set())
        t = truth.get(sid, set())
        per[k] = score_one(p, t)
        tp += len(p & t); fp += len(p - t); fn += len(t - p)
        n_pred_empty += (not p); n_true_empty += (not t)

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    stats = {
        'macro_f05': float(per.mean()) if len(per) else 0.0,
        'n_s1': len(s1_ids),
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f05': f_beta_05(micro_p, micro_r),
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_pred_empty': n_pred_empty,
        'n_true_empty': n_true_empty,
        'mean_pred_size': float(np.mean([len(preds.get(s, set())) for s in s1_ids])) if s1_ids else 0.0,
    }
    return stats['macro_f05'], stats


def breakdown_by_true_size(preds: Dict[str, Set[str]], truth: Dict[str, Set[str]],
                           s1_ids: Iterable[str]) -> list[dict]:
    """Per-true-group-size scores. This is where over/under-prediction shows up:
    a good global number can hide a collapse on singletons or on T=1."""
    buckets: Dict[int, list] = {}
    for sid in s1_ids:
        t = truth.get(sid, set())
        buckets.setdefault(len(t), []).append(score_one(preds.get(sid, set()), t))
    out = []
    for size in sorted(buckets):
        v = np.asarray(buckets[size], dtype=np.float64)
        out.append({'true_size': size, 'n_s1': len(v), 'mean_f05': float(v.mean()),
                    'share_of_eval': len(v) / max(sum(len(x) for x in buckets.values()), 1)})
    return out


def _test() -> None:
    S = set
    # --- the four hand-checked cases from the plan ---
    assert abs(score_one(S({'a'}), S({'a', 'b'})) - 0.8333333) < 1e-6, \
        'T=2, found 1 of 2 -> 0.8333'
    assert abs(score_one(S({'a', 'b', 'x'}), S({'a', 'b'})) - 0.7142857) < 1e-6, \
        'T=2, found both plus one error -> 0.7143'
    assert score_one(S(), S()) == 1.0, 'correct empty on a singleton -> 1.0'
    assert score_one(S({'a'}), S()) == 0.0, 'any prediction on a singleton -> 0.0'

    # --- the T=1 reversal that drives assign.py ---
    miss_T1 = 1.0 - score_one(S(), S({'a'}))
    fp_T1 = 1.0 - score_one(S({'a', 'x'}), S({'a'}))
    assert abs(miss_T1 - 1.0) < 1e-9 and abs(fp_T1 - 0.4444444) < 1e-6
    assert miss_T1 > fp_T1, 'at T=1 a miss must cost more than a false positive'

    # --- and that it flips the other way by T=3 ---
    miss_T3 = 1.0 - score_one(S({'a', 'b'}), S({'a', 'b', 'c'}))
    fp_T3 = 1.0 - score_one(S({'a', 'b', 'c', 'x'}), S({'a', 'b', 'c'}))
    assert abs(miss_T3 - 0.0909091) < 1e-6 and abs(fp_T3 - 0.2105263) < 1e-6
    assert fp_T3 > miss_T3, 'at T=3 a false positive must cost more than a miss'

    # --- zero overlap is 0.0, not a partial credit ---
    assert score_one(S({'z'}), S({'a', 'b'})) == 0.0

    # --- macro counts every s1_id, including ones with no prediction ---
    truth = {'s1': {'a'}, 's2': {'b'}, 's3': set()}
    preds = {'s1': {'a'}}                      # s2 missed entirely, s3 correctly empty
    m, st = macro_f05(preds, truth, ['s1', 's2', 's3'])
    assert abs(m - (1.0 + 0.0 + 1.0) / 3) < 1e-9, f'macro over all ids, got {m}'
    assert st['n_s1'] == 3 and st['n_true_empty'] == 1

    # dropping an id from s1_ids would inflate the score -- prove the guard works
    m2, _ = macro_f05(preds, truth, ['s1', 's3'])
    assert m2 > m, 'omitting a failed entity inflates the macro average'

    print('evaluate self-test: PASS')


if __name__ == '__main__':
    _test()
