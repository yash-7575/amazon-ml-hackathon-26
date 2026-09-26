"""End-to-end: build -> train -> predict -> tune the assignment rule -> report.

One command so the whole chain is reproducible and the reported number always comes
from the same path.

Usage:
    PYTHONPATH=. python -m matcher.run_experiment --tag india_capped
    PYTHONPATH=. python -m matcher.run_experiment --tag india_capped --eval-tag us_capped
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import lightgbm as lgb

from . import train as TR
from . import label as LB
from .assign import AssignConfig, assign
from .evaluate import macro_f05, breakdown_by_true_size
from .tune import sweep, report, best_config
from .features import variance_report

DATA = TR.DATA
MODELS = TR.MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='dataset tag to train on')
    ap.add_argument('--eval-tag', default=None,
                    help='score on a DIFFERENT dataset (cross-country transfer test)')
    ap.add_argument('--frac-val', type=float, default=0.25)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--all-s1-file', default=None,
                    help='Newline-separated list of EVERY sampled S1 id. An S1 that '
                         'blocking gave zero candidates never appears in the pair file, '
                         'so without this it silently leaves the metric denominator -- '
                         'which is wrong in both directions: it hides a real miss, and '
                         'it also discards a correct empty prediction on a true '
                         'singleton, each of which is worth a full 1.0.')
    a = ap.parse_args()

    # ---------------- train ----------------
    if a.skip_train:
        booster = lgb.Booster(model_file=os.path.join(MODELS, f'{a.tag}.lgb'))
        va = np.load(os.path.join(MODELS, f'{a.tag}_val_mask.npy'))
        cols = json.load(open(os.path.join(DATA, f'{a.tag}_cols.json')))['cols']
    else:
        r = TR.train(a.tag, frac_val=a.frac_val)
        booster, va, cols = r['booster'], r['val'], r['cols']
        print(f"\n[run] best_iteration={r['best_iter']}  val AUC={r['best_auc']:.5f}")

    # ---------------- pick the evaluation rows ----------------
    if a.eval_tag:
        # transfer test: the whole other dataset is unseen, so use all of it
        Xe = np.load(os.path.join(DATA, f'{a.eval_tag}_X.npy'))
        ye = np.load(os.path.join(DATA, f'{a.eval_tag}_y.npy'))
        me = np.load(os.path.join(DATA, f'{a.eval_tag}_meta.npz'))
        ecols = json.load(open(os.path.join(DATA, f'{a.eval_tag}_cols.json')))['cols']
        assert ecols == cols, 'feature columns differ between train and eval datasets'
        mask = np.ones(len(ye), dtype=bool)
        label = f'{a.eval_tag} (ALL rows -- transfer from {a.tag})'
    else:
        Xe, ye, me, _ = TR.load(a.tag)
        mask = va
        label = f'{a.tag} validation groups'

    X, y = Xe[mask], ye[mask]
    s1 = me['s1_ids'][mask]
    cand = me['cand_ids'][mask]
    eval_s1 = sorted(set(s1.tolist()))
    if a.all_s1_file and not a.eval_tag:
        raise SystemExit(
            '--all-s1-file is only valid with --eval-tag (whole-dataset scoring).\n'
            'In validation-subset mode the pair file already restricts scoring to the\n'
            'val groups, so folding in every sampled S1 adds TRAINING groups to the\n'
            'denominator and scores them as total misses -- inflation in reverse.')
    if a.all_s1_file:
        every = [ln.strip() for ln in open(a.all_s1_file) if ln.strip()]
        extra = sorted(set(every) - set(eval_s1))
        eval_s1 = sorted(set(eval_s1) | set(every))
        print(f'[run] +{len(extra):,} S1 with ZERO candidates folded into the '
              f'denominator (total eval S1 = {len(eval_s1):,})')

    print(f'\n[run] scoring {label}: {len(y):,} pairs over {len(eval_s1):,} S1')
    prob = booster.predict(X, num_iteration=booster.best_iteration)

    truth = LB.load_truth(eval_s1)
    st = LB.group_stats(eval_s1, truth)
    print(f'[run] eval set: {st["n_singleton"]:,} singletons ({st["pct_singleton"]:.2f}%), '
          f'mean true size {st["mean_true_size"]:.2f}, '
          f'{st["total_true_pairs"]:,} true pairs')

    # ---------------- the reachable ceiling ----------------
    # Recall of the CANDIDATE set bounds anything the matcher can do. Reporting the
    # model's number without it hides whether a loss came from blocking or matching.
    reachable = int(y.sum())
    print(f'[run] candidate recall on this eval set = {reachable:,}/'
          f'{st["total_true_pairs"]:,} = '
          f'{100*reachable/max(st["total_true_pairs"],1):.2f}%  <- matcher ceiling')

    # ---------------- sweep the assignment rule ----------------
    rows = sweep(s1.tolist(), cand.tolist(), prob, truth, eval_s1)
    report(rows)

    cfg = best_config(rows)
    preds = assign(s1.tolist(), cand.tolist(), prob, cfg)
    m, stats = macro_f05(preds, truth, eval_s1)
    print(f'\n[run] BEST macro-F0.5 = {m:.5f}   ({cfg})')
    print(f'[run] micro P={stats["micro_precision"]:.4f} R={stats["micro_recall"]:.4f} '
          f'tp={stats["tp"]:,} fp={stats["fp"]:,} fn={stats["fn"]:,}')
    print(f'[run] predicted empty for {stats["n_pred_empty"]:,} S1; '
          f'{stats["n_true_empty"]:,} are truly singletons')

    print('\n[run] breakdown by TRUE group size:')
    print(f'   {"T":>3} {"n_S1":>8} {"share":>7} {"mean F0.5":>10}')
    for b in breakdown_by_true_size(preds, truth, eval_s1):
        print(f'   {b["true_size"]:>3} {b["n_s1"]:>8,} {100*b["share_of_eval"]:>6.2f}% '
              f'{b["mean_f05"]:>10.5f}')

    # ---------------- feature health ----------------
    vr = variance_report(X, cols)
    const = [v['feature'] for v in vr if v['constant']]
    if const:
        print(f'\n[run] CONSTANT features on this eval set ({len(const)}): {const}')
    high_nan = [(v['feature'], v['pct_nan']) for v in vr if v['pct_nan'] > 50]
    if high_nan:
        print(f'[run] features >50% NaN: {high_nan}')

    with open(os.path.join(MODELS, f'{a.tag}_sweep.json'), 'w') as f:
        json.dump({'eval': label, 'best_macro_f05': m, 'best_cfg': cfg.__dict__,
                   'ceiling_recall': 100*reachable/max(st['total_true_pairs'],1),
                   'rows': rows[:60]}, f, indent=2)


if __name__ == '__main__':
    main()
