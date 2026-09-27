"""Per-pair probabilities -> per-S1 predicted match sets.

This file, not the model, is where macro-F0.5 is won or lost. Two measured facts drive
every choice here:

FACT A -- the ground truth is a PARTITION of the candidate side. 7,638,365 matched slots
hold 7,638,365 distinct ids: no S2/S3 record is ever claimed by two different S1
entities. So when several S1 entities claim one candidate, at most one is right.
Dropping all but the most confident claim is a pure precision gain at zero recall cost,
and precision is weighted 2x.

FACT B -- the error asymmetry REVERSES at true-group-size 1. Cost of one mistake:

    T | one false positive | one miss
    1 |       0.444        |  1.000   <- a miss is 2.3x worse
    3 |       0.211        |  0.091   <- a false positive is 2.3x worse

An empty prediction on a non-singleton scores a flat 0.0, however good the rest of the
list would have been. So a single global threshold is the wrong instrument: it silently
empties the groups where emptiness is most expensive. Instead the argmax candidate is
always kept unless a separate, deliberately conservative gate says "this entity is a
singleton" -- 5.58% of entities are, and each correct empty is worth a full 1.0.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set
import numpy as np

from .evaluate import f_beta_05


@dataclass(frozen=True)
class AssignConfig:
    tau: float = 0.50            # keep a candidate when p >= tau
    tau_singleton: float = 0.30  # if the group's BEST p is below this, predict empty
    max_k: int = 6               # cap per S1; true sizes run 0..11, mean 3.46
    resolve_conflicts: bool = True   # FACT A
    always_keep_argmax: bool = True  # FACT B


def assign(s1_ids: Sequence[str], cand_ids: Sequence[str], prob: np.ndarray,
           cfg: AssignConfig) -> Dict[str, Set[str]]:
    """Returns {s1_id: set(cand_id)}. S1 entities with an empty set are singletons
    by prediction; callers must still emit a row for them."""
    prob = np.asarray(prob, dtype=np.float64)
    order = np.argsort(-prob, kind='stable')     # confident claims first

    # --- FACT A: one candidate, one claimant ---
    if cfg.resolve_conflicts:
        claimed: Dict[str, str] = {}
        allowed = np.zeros(len(prob), dtype=bool)
        for k in order:
            c = cand_ids[k]
            owner = claimed.get(c)
            if owner is None:
                claimed[c] = s1_ids[k]
                allowed[k] = True
            elif owner == s1_ids[k]:
                allowed[k] = True                # duplicate row for the same pair
    else:
        allowed = np.ones(len(prob), dtype=bool)

    # Every S1 in the input must appear in the output, even if conflict resolution
    # stripped its entire candidate list -- the submission requires one row per S1,
    # and a caller iterating `out` must not silently skip an entity.
    out: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}

    groups: Dict[str, List[tuple]] = defaultdict(list)
    for k in range(len(prob)):
        if allowed[k]:
            groups[s1_ids[k]].append((prob[k], cand_ids[k]))

    for sid, lst in groups.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        best = lst[0][0]
        # --- FACT B: the singleton gate is the ONLY way to produce an empty list ---
        if best < cfg.tau_singleton:
            out[sid] = set()
            continue
        picked = [c for p, c in lst[: cfg.max_k] if p >= cfg.tau]
        if not picked and cfg.always_keep_argmax:
            # The gate did not fire, so we believe this entity has a match. Emitting
            # nothing here would score 0.0; the argmax costs at most 0.444.
            picked = [lst[0][1]]
        out[sid] = set(picked)
    return out


# ---------------------------------------------------------------- expected-F0.5 rule
def _expected_f05_best_n(probs: np.ndarray, max_k: int) -> int:
    """Choose how many of the top-ranked candidates to keep by maximising EXPECTED F0.5.

    A single global threshold answers "is this pair probably a match?" But the metric does
    not score pairs, it scores ENTITIES -- and the value of adding the n-th candidate
    depends on how many you have already taken and on how confident the others are.
    Taking a 0.6-probability candidate is right when it is your only one and wrong when
    you already hold three at 0.99.

    Treat each candidate as an independent Bernoulli with parameter p (the model's
    probability). For a prediction of size n over the top-n sorted probabilities:
        E[TP] = sum(p_1..p_n)          E[T] = sum(all p in the group)
        P = E[TP]/n                    R = E[TP]/E[T]
    and F0.5 is evaluated on those expectations. Independence is an approximation -- the
    candidates compete for the same entity -- but it captures the effect that actually
    matters: the marginal candidate must clear a bar that RISES as n grows, because
    precision is weighted 2x.

    n = 0 (predict empty) is included, which is how a singleton gets chosen: when every
    probability is low, the best expected score really is the empty list.
    """
    tot = float(probs.sum())
    if tot <= 0:
        return 0
    best_n, best_v = 0, f_beta_05(1.0, 0.0) if tot == 0 else 0.0
    # expected score of predicting nothing: right only if the entity truly has no match
    best_v = float(np.prod(1.0 - probs))          # P(no true match exists) -> scores 1.0
    run = 0.0
    for n in range(1, min(max_k, len(probs)) + 1):
        run += float(probs[n - 1])
        P = run / n
        R = run / tot
        v = f_beta_05(P, R)
        if v > best_v:
            best_n, best_v = n, v
    return best_n


def assign_expected_f05(s1_ids: Sequence[str], cand_ids: Sequence[str],
                        prob: np.ndarray, cfg: AssignConfig) -> Dict[str, Set[str]]:
    """Per-entity adaptive match count. Same conflict resolution as `assign`."""
    prob = np.asarray(prob, dtype=np.float64)
    order = np.argsort(-prob, kind='stable')
    if cfg.resolve_conflicts:
        claimed: Dict[str, str] = {}
        allowed = np.zeros(len(prob), dtype=bool)
        for k in order:
            c = cand_ids[k]
            owner = claimed.get(c)
            if owner is None:
                claimed[c] = s1_ids[k]; allowed[k] = True
            elif owner == s1_ids[k]:
                allowed[k] = True
    else:
        allowed = np.ones(len(prob), dtype=bool)

    out: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
    groups: Dict[str, List[tuple]] = defaultdict(list)
    for k in range(len(prob)):
        if allowed[k]:
            groups[s1_ids[k]].append((prob[k], cand_ids[k]))
    for sid, lst in groups.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        p = np.array([x[0] for x in lst], dtype=np.float64)
        n = _expected_f05_best_n(p, cfg.max_k)
        out[sid] = {c for _, c in lst[:n]}
    return out


def _test_expected() -> None:
    # a clear single match: keep exactly one
    assert _expected_f05_best_n(np.array([0.98, 0.02, 0.01]), 8) == 1
    # three confident matches: keep all three
    assert _expected_f05_best_n(np.array([0.97, 0.95, 0.93, 0.02]), 8) == 3
    # everything weak -> predict EMPTY (the singleton case), no threshold needed
    assert _expected_f05_best_n(np.array([0.05, 0.03, 0.01]), 8) == 0
    # the bar rises with n: 0.60 is worth taking alone...
    assert _expected_f05_best_n(np.array([0.60]), 8) == 1
    # ...but not as a 4th behind three near-certain ones
    n = _expected_f05_best_n(np.array([0.99, 0.99, 0.99, 0.60]), 8)
    assert n == 3, f'marginal candidate must clear a rising bar, got n={n}'
    print('expected-F0.5 self-test: PASS')


def _test() -> None:
    # --- FACT A: cand X is claimed by s1=A (0.9) and s1=B (0.6). Only A keeps it. ---
    s1 = ['A', 'A', 'B', 'B']
    cd = ['X', 'Y', 'X', 'Z']
    p = np.array([0.9, 0.7, 0.6, 0.55])
    r = assign(s1, cd, p, AssignConfig(tau=0.5, tau_singleton=0.3, max_k=6))
    assert r['A'] == {'X', 'Y'}, r['A']
    assert r['B'] == {'Z'}, f"B must lose the contested X, got {r['B']}"

    # with conflict resolution off, both keep X -- one of them is necessarily wrong
    r2 = assign(s1, cd, p, AssignConfig(tau=0.5, resolve_conflicts=False))
    assert 'X' in r2['A'] and 'X' in r2['B']

    # --- FACT B: every p below tau but above the singleton gate -> keep the argmax ---
    r3 = assign(['C', 'C'], ['M', 'N'], np.array([0.45, 0.10]),
                AssignConfig(tau=0.5, tau_singleton=0.3))
    assert r3['C'] == {'M'}, f'must not empty a group the gate did not reject: {r3["C"]}'

    # --- the gate IS what produces an empty list ---
    r4 = assign(['D', 'D'], ['M', 'N'], np.array([0.20, 0.05]),
                AssignConfig(tau=0.5, tau_singleton=0.3))
    assert r4['D'] == set(), f'gate must fire below tau_singleton: {r4["D"]}'

    # --- max_k caps the list even when many clear tau ---
    r5 = assign(['E'] * 5, list('VWXYZ'), np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
                AssignConfig(tau=0.5, max_k=2))
    assert len(r5['E']) == 2

    # --- argmax rescue must respect the conflict resolution, not resurrect a loser ---
    r6 = assign(['F', 'G'], ['Q', 'Q'], np.array([0.80, 0.40]),
                AssignConfig(tau=0.9, tau_singleton=0.3))
    assert r6['F'] == {'Q'}, r6['F']
    assert r6['G'] == set(), f'G had only the lost candidate, so it must be empty: {r6["G"]}'

    _test_expected()
    print('assign self-test: PASS')


if __name__ == '__main__':
    _test()
