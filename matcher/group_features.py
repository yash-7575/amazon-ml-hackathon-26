"""Group-relative and reciprocal features.

THE highest-value family on one-to-one matching problems, and the one most pipelines omit.

Absolute similarity cannot express "this is the best of its group." When 50 candidates all
score 0.97 on name similarity, a sixth string metric adds nothing -- the problem is not that
similarity is LOW, it is that similarity is AMBIGUOUS. Position within the candidate group
resolves it.

Cost: one lexsort + vectorized groupby arithmetic = Theta(P log P). Seconds at 1e8 pairs,
which makes this the best value-to-cost ratio in the entire feature set.

No labels are used. These features are NOT leakage -- with one condition: the group must be
formed IDENTICALLY at train and test. Change K and rank/group_size/ratio all shift.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def _group_bounds(sorted_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start index and size of each run of equal keys."""
    if sorted_keys.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    change = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate([[0], change])
    sizes = np.diff(np.concatenate([starts, [sorted_keys.size]]))
    return starts, sizes


def group_features(s1_idx: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Features describing each pair's position within its s1 group.

    Returns a matrix aligned to the ORIGINAL pair order.
    """
    P = s1_idx.size
    score = np.asarray(score, dtype=np.float64)
    # sort by (group asc, score desc) -- one pass, then all stats are run-wise
    order = np.lexsort((-score, s1_idx))
    gk, gs = s1_idx[order], score[order]
    starts, sizes = _group_bounds(gk)

    rank = np.empty(P, np.float32)
    gmax = np.empty(P, np.float32)
    g2nd = np.empty(P, np.float32)
    gmean = np.empty(P, np.float32)
    gstd = np.empty(P, np.float32)
    gsize = np.empty(P, np.float32)
    gap_below = np.empty(P, np.float32)

    for st, sz in zip(starts, sizes):
        sl = slice(st, st + sz)
        v = gs[sl]
        rank[sl] = np.arange(1, sz + 1, dtype=np.float32)
        gmax[sl] = v[0]
        # runner-up undefined for a singleton -> NaN, never 0
        g2nd[sl] = v[1] if sz > 1 else np.nan
        gmean[sl] = v.mean()
        gstd[sl] = v.std() if sz > 1 else np.nan
        gsize[sl] = sz
        gb = np.empty(sz, np.float32)
        gb[:-1] = v[:-1] - v[1:]
        gb[-1] = np.nan                       # nothing below the last
        gap_below[sl] = gb

    # NOTE on score_margin_to_2nd: defined as own_score - group_runner_up_score.
    # For rank 1 it is the decisive margin (the useful case). For rank 2 it is exactly 0
    # by construction, and negative below. That is consistent and the model uses it as
    # "how far above the runner-up am I"; it is not a bug.
    cols = ["rank_in_group", "is_top1", "score_margin_to_2nd", "score_ratio_to_max",
            "score_minus_group_mean", "z_score_in_group", "group_size",
            "score_gap_below", "frac_above"]

    own = gs.astype(np.float32)
    feats = np.vstack([
        rank,
        (rank == 1).astype(np.float32),
        # the margin to the runner-up: "clear winner" vs "coin flip". Absolute
        # similarity is IDENTICAL in both cases; this feature separates them.
        np.where(np.isnan(g2nd), np.nan, own - g2nd),
        np.where(gmax > 0, own / np.maximum(gmax, 1e-9), np.nan),
        own - gmean,
        np.where((gstd > 0) & np.isfinite(gstd), (own - gmean) / np.maximum(gstd, 1e-9), np.nan),
        gsize,
        gap_below,
        (rank - 1) / np.maximum(gsize, 1),
    ]).T.astype(np.float32)

    out = np.empty_like(feats)
    out[order] = feats                        # restore original pair order
    return out, cols


def reciprocal_features(s1_idx: np.ndarray, cand_idx: np.ndarray,
                        score: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """The reverse direction: rank each pair from the CANDIDATE's perspective.

    Under a one-to-one constraint (ground truth is a partition), two source records cannot
    both claim the same candidate. `is_mutual_best` encodes "neither of us has a better
    option" -- the structure of a stable matching. Cheap, test-time computable, and strong.
    `n_competitors_for_cand > 1` is the contested-claim flag: exactly where precision is lost.
    """
    rev, _ = group_features(cand_idx, score)
    rank_rev = rev[:, 0]
    n_comp = rev[:, 6]
    cand_best = np.where(rev[:, 3] > 0, np.asarray(score, np.float32) / rev[:, 3], np.nan)

    fwd, _ = group_features(s1_idx, score)
    is_mutual = ((fwd[:, 1] == 1) & (rank_rev == 1)).astype(np.float32)

    cols = ["rank_reverse", "is_mutual_best", "n_competitors_for_cand",
            "cand_best_score", "cand_score_margin"]
    return np.vstack([rank_rev, is_mutual, n_comp, cand_best,
                      np.asarray(score, np.float32) - cand_best]).T.astype(np.float32), cols


if __name__ == "__main__":
    # group 0: a clear winner (margin 0.35). group 1: a coin flip (margin 0.01).
    # Absolute score is ~identical; the margin feature is what separates them.
    s1 = np.array([0, 0, 0, 1, 1, 2])
    cd = np.array([10, 11, 12, 13, 14, 10])
    sc = np.array([0.97, 0.62, 0.55, 0.97, 0.96, 0.91])
    G, gc = group_features(s1, sc)
    R, rc = reciprocal_features(s1, cd, sc)
    import numpy as _np
    _np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print("group cols:", gc)
    for k in range(len(s1)):
        print(f"  s1={s1[k]} cand={cd[k]} score={sc[k]:.2f}  "
              f"rank={G[k,0]:.0f} top1={G[k,1]:.0f} margin2nd={G[k,2]:.3f} "
              f"ratio={G[k,3]:.3f} gsize={G[k,6]:.0f}")
    print("\nreciprocal cols:", rc)
    for k in range(len(s1)):
        print(f"  s1={s1[k]} cand={cd[k]}  rank_rev={R[k,0]:.0f} "
              f"mutual_best={R[k,1]:.0f} competitors={R[k,2]:.0f}")
    assert G[0, 1] == 1 and G[3, 1] == 1, "top1 flags"
    assert abs(G[0, 2] - 0.35) < 1e-6, "clear winner margin"
    assert abs(G[3, 2] - 0.01) < 1e-6, "coin-flip margin"
    assert np.isnan(G[5, 2]), "singleton runner-up must be NaN, not 0"
    # cand 10 is claimed by s1=0 (0.97) and s1=2 (0.91): contested
    assert R[0, 2] == 2 and R[5, 2] == 2, "contested claim detected"
    assert R[0, 1] == 1 and R[5, 1] == 0, "mutual best resolves the contest"
    print("\ngroup_features self-test: PASS")
