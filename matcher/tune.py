"""Sweep the assignment rule and report the whole surface, not just the argmax.

The argmax alone is misleading: if macro-F0.5 is nearly flat across a wide band of tau
then the exact value does not matter and we should not pretend it does, while a sharp
peak means the number is fragile and will not survive the shift from a capped-negative
sample to the real pool. Seeing the surface is how we tell those apart.

Tuning happens on the VALIDATION groups only. The reported best is therefore optimistic
by construction -- it is chosen on the same data it is measured on. `--holdout` carves a
third split off validation so the final number is measured on groups the sweep never saw.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
from typing import Dict, List, Sequence, Set
import numpy as np

from .assign import AssignConfig, assign
from .evaluate import macro_f05, breakdown_by_true_size

TAUS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
TAU_SINGLE = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
MAX_KS = (2, 3, 4, 5, 6, 8)


def sweep(s1_ids: Sequence[str], cand_ids: Sequence[str], prob: np.ndarray,
          truth: Dict[str, Set[str]], eval_s1: Sequence[str],
          taus=TAUS, tau_singles=TAU_SINGLE, max_ks=MAX_KS,
          resolve=(True, False)) -> List[dict]:
    rows: List[dict] = []
    for tau, ts, k, rc in itertools.product(taus, tau_singles, max_ks, resolve):
        if ts > tau:
            continue          # a gate above tau would empty groups tau would have filled
        cfg = AssignConfig(tau=tau, tau_singleton=ts, max_k=k, resolve_conflicts=rc)
        preds = assign(s1_ids, cand_ids, prob, cfg)
        m, st = macro_f05(preds, truth, eval_s1)
        rows.append({'tau': tau, 'tau_singleton': ts, 'max_k': k, 'resolve': rc,
                     'macro_f05': m, 'micro_p': st['micro_precision'],
                     'micro_r': st['micro_recall'],
                     'mean_pred': st['mean_pred_size'],
                     'n_pred_empty': st['n_pred_empty'],
                     'n_true_empty': st['n_true_empty']})
    rows.sort(key=lambda r: -r['macro_f05'])
    return rows


def report(rows: List[dict], top: int = 15) -> None:
    print(f'\n{"macroF0.5":>10} {"tau":>5} {"tauS":>5} {"K":>3} {"rslv":>5} '
          f'{"microP":>7} {"microR":>7} {"pred/S1":>8} {"predEmpty":>10} {"trueEmpty":>10}')
    for r in rows[:top]:
        print(f'{r["macro_f05"]:>10.5f} {r["tau"]:>5.2f} {r["tau_singleton"]:>5.2f} '
              f'{r["max_k"]:>3} {str(r["resolve"]):>5} {r["micro_p"]:>7.4f} '
              f'{r["micro_r"]:>7.4f} {r["mean_pred"]:>8.2f} {r["n_pred_empty"]:>10,} '
              f'{r["n_true_empty"]:>10,}')

    best = rows[0]
    # How flat is the surface? If many settings land within 0.005 of the best, the exact
    # threshold is not the thing that matters and we should not over-fit it.
    near = [r for r in rows if best['macro_f05'] - r['macro_f05'] <= 0.005]
    print(f'\n  {len(near)} of {len(rows)} settings are within 0.005 of the best '
          f'({best["macro_f05"]:.5f}).')
    if len(near) > 1:
        print(f'    tau in [{min(r["tau"] for r in near):.2f}, '
              f'{max(r["tau"] for r in near):.2f}]  '
              f'tau_singleton in [{min(r["tau_singleton"] for r in near):.2f}, '
              f'{max(r["tau_singleton"] for r in near):.2f}]  '
              f'max_k in [{min(r["max_k"] for r in near)}, '
              f'{max(r["max_k"] for r in near)}]')

    on = [r for r in rows if r['resolve']]
    off = [r for r in rows if not r['resolve']]
    if on and off:
        print(f'  conflict resolution: best ON  = {on[0]["macro_f05"]:.5f}   '
              f'best OFF = {off[0]["macro_f05"]:.5f}   '
              f'delta = {on[0]["macro_f05"] - off[0]["macro_f05"]:+.5f}')


def best_config(rows: List[dict]) -> AssignConfig:
    b = rows[0]
    return AssignConfig(tau=b['tau'], tau_singleton=b['tau_singleton'],
                        max_k=b['max_k'], resolve_conflicts=b['resolve'])
