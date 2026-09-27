"""measure_recall_at_k.py — Recall@K sweep for the union blocker.

Answers the four "First things to measure" from PROJECT_CONTEXT in one run:
  1. Recall@K curve for P2 alone, K ∈ {10,20,50,100,200} — find the knee.
  2. Marginal recall of P3 over P2 (per-K).
  3. Stratified PC by country and by script (in-script vs cross-script).
  4. Peak RSS on the largest tested partition.

Design:
- Blocking runs ONCE per (source, partition) pair at K_max = max(k_sweep).
- Recall@K is evaluated by truncating the per-S1 sorted candidate list.
- Ground truth is loaded here (not in the generator), consulted ONLY for the
  recall calculation. It does NOT filter or augment the candidate set.
- Script stratification: a "cross-script" true pair is one where the S2/S3 name
  contains Devanagari (S1 is 100% ASCII).

Usage:
    PYTHONPATH=. python blocking/measure_recall_at_k.py --only india --sample 5000
    PYTHONPATH=. python blocking/measure_recall_at_k.py --sample 20000     # both partitions
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from blocking.config import BlockerConfig
from blocking.io_utils import read_tsv, peak_rss_mb
from blocking.normalize import country_key, has_devanagari
from blocking.blocker_v1 import (
    PartitionData, load_partition,
    block_p1_exact,
    _build_name_index, _build_addr_index,
)
from blocking.index import transform_queries, topk_search
from blocking.measure import load_gt


# ---------------------------------------------------------------------------
# Per-source ranked candidates at K_max
# ---------------------------------------------------------------------------
def _blocker_ranked_lists(
    s1: PartitionData, target: PartitionData, cfg: BlockerConfig,
    field: str, k_max: int, prebuilt=None,
) -> dict[int, list[tuple[str, float]]]:
    """Return {s1_idx: [(cand_id, score), ...]} sorted by score desc, len <= k_max.

    field ∈ {'name','addr'}. Devanagari transliteration aliases are folded into
    the name index (P4).

    `prebuilt`: optional (index, ids) tuple to skip index building — lets the
    caller time build vs query separately.
    """
    if prebuilt is not None:
        # unpack the pre-built index bundle (name: (index, ids, _mask); addr: (index, ids))
        index = prebuilt[0]
        idx_ids = prebuilt[1]
        docs = s1.name_doc if field == 'name' else s1.addr_doc
    elif field == 'name':
        # _build_name_index returns (index, ids, translit_mask). The mask is used
        # by the main pipeline to tag hits as P2 vs P4; the K-sweep does not
        # differentiate them (P4 aliases share the same target id), so we drop it.
        index, idx_ids, _ = _build_name_index(target, cfg)
        docs = s1.name_doc
    else:
        idx_bundle = _build_addr_index(target, cfg)
        index, idx_ids = idx_bundle
        docs = s1.addr_doc
    if index is None:
        return {}

    q_docs, q_pos_to_s1 = [], []
    for i, d in enumerate(docs):
        if d:
            q_docs.append(d)
            q_pos_to_s1.append(i)
    if not q_docs:
        return {}
    Q = transform_queries(index, q_docs)
    top_idx, top_scr = topk_search(
        Q, index.matrix, k_max,
        chunk_rows=(cfg.p2_chunk_rows if field == 'name' else cfg.p3_chunk_rows),
        dtype=(cfg.p2_score_dtype if field == 'name' else cfg.p3_score_dtype),
    )
    out: dict[int, list[tuple[str, float]]] = {}
    for qi, s1_idx in enumerate(q_pos_to_s1):
        lst = []
        for j in range(top_idx.shape[1]):
            ii = int(top_idx[qi, j])
            if ii < 0:
                break
            sc = float(top_scr[qi, j])
            if not np.isfinite(sc) or sc <= 0.0:
                continue
            lst.append((idx_ids[ii], sc))
        if lst:
            out[s1_idx] = lst
    return out


def _p1_hits(s1: PartitionData, targets: list[PartitionData]) -> dict[int, set[str]]:
    hits = block_p1_exact(s1, targets)
    return {i: {t[1] for t in lst} for i, lst in hits.items()}


# ---------------------------------------------------------------------------
# Per-partition sweep
# ---------------------------------------------------------------------------
def _load_partition_capped(
    path: str, target_ck: str, source_tag: str, label: str,
    need_ids: set[str], cap: int | None, seed: int,
) -> PartitionData:
    """Like load_partition, but if cap is set, keep ALL rows whose id is in
    need_ids (ground-truth partners of sampled S1) plus a reservoir-sampled
    distractor set to reach `cap`. This is faithful to blocking (the model
    never sees the label; it still has to find the needle among distractors)
    but bounds memory to O(cap) instead of O(country_partition_size).

    Reservoir sampling: uniform without replacement over the distractor stream,
    seeded for determinism. Peak in-memory rows is mandatory + cap.
    """
    if cap is None:
        return load_partition(path, target_ck, source_tag, label)
    from blocking.io_utils import stream_country_rows
    rng = random.Random(seed)
    slots = max(0, cap)                     # number of distractor slots
    mandatory: list[dict] = []
    reservoir: list[dict] = []
    seen_optional = 0
    for row in stream_country_rows(path, target_ck):
        if row['entity_id'] in need_ids:
            mandatory.append(row)
            continue
        # Reservoir sampling among distractors.
        seen_optional += 1
        if len(reservoir) < slots:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen_optional - 1)
            if j < slots:
                reservoir[j] = row
    keep = mandatory + reservoir
    rng.shuffle(keep)
    p = PartitionData(label=label, source_tag=source_tag)
    for r in keep:
        p.add(r)
    return p


def _load_s1_sampled(path: str, ck: str, sample_ids: set[str] | None) -> PartitionData:
    """Load only S1 rows in `ck` whose entity_id is in sample_ids (if given).
    This avoids ballooning memory with 800k-1.3M S1 rows we never score."""
    from blocking.io_utils import stream_country_rows
    p = PartitionData(label=f'S1[{ck}]', source_tag='S1')
    for row in stream_country_rows(path, ck):
        if sample_ids is not None and row['entity_id'] not in sample_ids:
            continue
        p.add(row)
    return p


def sweep_partition(
    ck: str, cfg: BlockerConfig, gt: dict[str, set[str]],
    k_sweep: tuple[int, ...],
    sample_s1_ids: set[str] | None,
    s2s3_cap: int | None = None,
) -> dict:
    t0 = time.time()
    d = cfg.data_root + ('train/' if cfg.split == 'train' else 'test/')
    # Load only the sampled S1 rows in this partition (memory-tight).
    s1 = _load_s1_sampled(d + f'{cfg.split}_source1.tsv', ck, sample_s1_ids)
    # Pre-compute need_ids so capped S2/S3 always contain the true partners.
    need_ids: set[str] = set()
    if sample_s1_ids is not None:
        for sid in sample_s1_ids:
            if sid in gt:
                need_ids.update(gt[sid])
    else:
        for sid, ids in gt.items():
            need_ids.update(ids)
    s2 = _load_partition_capped(
        d + f'{cfg.split}_source2.tsv', ck, 'S2', f'S2[{ck}]',
        need_ids, s2s3_cap, cfg.seed,
    )
    s3 = _load_partition_capped(
        d + f'{cfg.split}_source3.tsv', ck, 'S3', f'S3[{ck}]',
        need_ids, s2s3_cap, cfg.seed,
    )
    t_load = time.time() - t0
    print(f'  loaded s1={s1.n:,}  s2={s2.n:,}  s3={s3.n:,}   ({t_load:.1f}s)', flush=True)

    # Restrict scope to sampled S1 rows with ground truth.
    def _in_scope(sid: str) -> bool:
        if sid not in gt:
            return False
        if sample_s1_ids is not None and sid not in sample_s1_ids:
            return False
        return True
    scope_idx = [i for i, sid in enumerate(s1.ids) if _in_scope(sid)]
    print(f'  scope s1 = {len(scope_idx):,}', flush=True)

    ids_here = set(s2.ids) | set(s3.ids)
    # Cross-script (Devanagari) target ids for stratification.
    deva_ids = {s2.ids[i] for i in range(s2.n) if has_devanagari(s2.name_raw[i])}
    deva_ids |= {s3.ids[i] for i in range(s3.n) if has_devanagari(s3.name_raw[i])}

    truth: dict[int, set[str]] = {}
    truth_cs: dict[int, set[str]] = {}
    truth_is: dict[int, set[str]] = {}
    for i in scope_idx:
        sid = s1.ids[i]
        want = gt[sid] & ids_here
        if not want:
            continue
        truth[i] = want
        cs = want & deva_ids
        if cs:
            truth_cs[i] = cs
        rest = want - deva_ids
        if rest:
            truth_is[i] = rest

    total_truth = sum(len(v) for v in truth.values())
    total_cs = sum(len(v) for v in truth_cs.values())
    total_is = sum(len(v) for v in truth_is.values())
    print(f'  truth pairs (in-country): {total_truth:,}  '
          f'(in-script {total_is:,} + cross-script {total_cs:,})', flush=True)
    if total_truth == 0:
        return {'country': ck, 'skipped': True}

    k_max = max(k_sweep)

    # ---- INDEX BUILD (fixed cost per partition; must be timed separately for
    #      the runtime extrapolation in RUNTIME_ESTIMATE.md) ----
    # Parallel over the independent builds (name/addr x S2/S3). Bit-identical
    # indexes to the sequential version; only wall-clock changes. TF-IDF
    # fitting is CPU-bound and the GIL is released in the heavy C loops.
    t = time.time()
    _build_jobs: list[tuple[str, str, PartitionData]] = []
    if s2.n:
        _build_jobs.append(('name_s2', 'name', s2))
        _build_jobs.append(('addr_s2', 'addr', s2))
    if s3.n:
        _build_jobs.append(('name_s3', 'name', s3))
        _build_jobs.append(('addr_s3', 'addr', s3))

    def _build_one(job: tuple[str, str, PartitionData]):
        key, field, target = job
        if field == 'name':
            return key, _build_name_index(target, cfg)
        return key, _build_addr_index(target, cfg)

    _built: dict = {}
    if _build_jobs:
        with ThreadPoolExecutor(max_workers=min(4, len(_build_jobs))) as _ex:
            for _key, _bundle in _ex.map(_build_one, _build_jobs):
                _built[_key] = _bundle
    name_s2 = _built.get('name_s2')
    name_s3 = _built.get('name_s3')
    addr_s2 = _built.get('addr_s2')
    addr_s3 = _built.get('addr_s3')
    index_build_sec = time.time() - t
    print(f'  indexes built ({index_build_sec:.2f}s)', flush=True)

    # ---- QUERIES (P1 + P2 + P3) — this is the portion that scales with n_s1 ----
    # The 4 ranked-list searches are independent (read-only indexes) so they run
    # in parallel threads. scipy/numpy release the GIL in matmul/argpartition,
    # so this uses ~4 cores. Outputs are identical to sequential execution.
    t_q = time.time()

    t = time.time()
    p1 = _p1_hits(s1, [s2, s3])
    print(f'  P1 hits computed ({time.time()-t:.2f}s)', flush=True)

    _query_jobs: list[tuple] = []
    if name_s2:
        _query_jobs.append(('p2_s2', s2, 'name', k_max, name_s2))
    if name_s3:
        _query_jobs.append(('p2_s3', s3, 'name', k_max, name_s3))
    if addr_s2:
        _query_jobs.append(('p3_s2', s2, 'addr', k_max, addr_s2))
    if addr_s3:
        _query_jobs.append(('p3_s3', s3, 'addr', k_max, addr_s3))

    def _query_one(job: tuple):
        key, target, field, kk, pre = job
        tq = time.time()
        res = _blocker_ranked_lists(s1, target, cfg, field, kk, prebuilt=pre)
        return key, res, time.time() - tq

    _ranked: dict = {}
    if _query_jobs:
        with ThreadPoolExecutor(max_workers=min(4, len(_query_jobs))) as _ex:
            for _key, _res, _dt in _ex.map(_query_one, _query_jobs):
                _ranked[_key] = _res
                print(f'  {_key} ranked lists ({_dt:.2f}s)', flush=True)
    p2_s2 = _ranked.get('p2_s2', {})
    p2_s3 = _ranked.get('p2_s3', {})
    p3_s2 = _ranked.get('p3_s2', {})
    p3_s3 = _ranked.get('p3_s3', {})

    query_sec = time.time() - t_q

    # ---- Per-K union: for each S1, take top-K from EACH ranked list, union with P1 ----
    def _topk_ids(ranked: dict[int, list[tuple[str, float]]], i: int, K: int) -> set[str]:
        return {cid for cid, _ in ranked.get(i, ())[:K]}

    # Combined-score ranked candidate: per S1, union all candidates from
    # (P1 + top-K_max name + top-K_max addr), rescore by w_name*cos_name + w_addr*cos_addr,
    # sort, then evaluate recall at each K by truncating the sorted list.
    # This mirrors the shipped rescore, but exposes the full K sweep.
    #
    # For plain per-blocker recall we truncate each ranked list to K and union.
    #
    # Deliverables asked for:
    #   - P2 only (name blocker) at K
    #   - P3 marginal (name+addr union) at K
    #   - P1+P2+P3+P4 union at K (P4 folded via name index)
    #   - final rescored top-K
    stats: dict = {
        'country': ck,
        'n_s1_scope': len(scope_idx),
        'n_s2': s2.n, 'n_s3': s3.n,
        'truth_pairs': total_truth,
        'truth_in_script': total_is,
        'truth_cross_script': total_cs,
        'load_sec': round(t_load, 2),
        'index_build_sec': round(index_build_sec, 2),
        'query_sec': round(query_sec, 2),
        'k_sweep': list(k_sweep),
    }

    # ----- Straight blocker recalls at each K -----
    per_k: dict[int, dict[str, dict[str, float]]] = {}
    for K in k_sweep:
        cand_p1 = p1
        cand_p2 = defaultdict(set)
        cand_p12 = defaultdict(set)
        cand_p123 = defaultdict(set)
        for i in truth:
            # P2 union across sources (top-K per source)
            n2 = _topk_ids(p2_s2, i, K) | _topk_ids(p2_s3, i, K)
            n3 = _topk_ids(p3_s2, i, K) | _topk_ids(p3_s3, i, K)
            cand_p2[i] = n2
            cand_p12[i] = cand_p1.get(i, set()) | n2
            cand_p123[i] = cand_p12[i] | n3

        def _recall(cands: dict, truth_d: dict) -> tuple[int, int]:
            hit = tot = 0
            for i, want in truth_d.items():
                hit += len(want & cands.get(i, set()))
                tot += len(want)
            return hit, tot

        row = {}
        for label, cands in [('P2', cand_p2), ('P1+P2', cand_p12),
                             ('P1+P2+P3', cand_p123)]:
            h_all, t_all = _recall(cands, truth)
            h_is, t_is = _recall(cands, truth_is) if truth_is else (0, 0)
            h_cs, t_cs = _recall(cands, truth_cs) if truth_cs else (0, 0)
            n_cand = sum(len(v) for v in cands.values())
            row[label] = {
                'recall': h_all / t_all if t_all else 0.0,
                'recall_in_script': (h_is / t_is) if t_is else None,
                'recall_cross_script': (h_cs / t_cs) if t_cs else None,
                'candidates': n_cand,
                'per_s1': n_cand / max(len(scope_idx), 1),
            }
        per_k[K] = row

    stats['per_k'] = per_k

    # ----- Peak RSS -----
    stats['peak_rss_mib'] = round(peak_rss_mb(), 1)
    stats['elapsed_sec'] = round(time.time() - t0, 2)
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _stratified_sample(
    s1_path: str, gt: dict[str, set[str]], n: int, seed: int,
    restrict_countries: set[str] | None,
) -> tuple[set[str], dict[str, str]]:
    rng = random.Random(seed)
    by_country: dict[str, list[str]] = defaultdict(list)
    sid_to_ck: dict[str, str] = {}
    for row in read_tsv(s1_path):
        sid = row['entity_id']
        if sid not in gt:
            continue
        ck = country_key(row['country'])
        if restrict_countries and ck not in restrict_countries:
            continue
        by_country[ck].append(sid)
        sid_to_ck[sid] = ck
    total = sum(len(v) for v in by_country.values())
    n = min(n, total)
    picked: set[str] = set()
    for ck, ids in by_country.items():
        alloc = max(1, int(round(n * len(ids) / total)))
        alloc = min(alloc, len(ids))
        picked.update(rng.sample(ids, alloc))
    return picked, sid_to_ck


def _print_report(all_stats: list[dict], k_sweep: tuple[int, ...]):
    print('\n' + '=' * 78)
    print('RECALL@K REPORT'.center(78))
    print('=' * 78)

    # Per-partition, per-K table.
    for s in all_stats:
        if s.get('skipped'):
            continue
        print(f'\nPartition: {s["country"]}  '
              f'(scope S1={s["n_s1_scope"]:,}, '
              f'truth={s["truth_pairs"]:,}, '
              f'in-script={s["truth_in_script"]:,}, '
              f'cross-script={s["truth_cross_script"]:,})')
        print(f'  peak RSS = {s["peak_rss_mib"]} MiB, elapsed = {s["elapsed_sec"]}s')
        header = f'  {"K":>5}  {"strategy":10}  {"recall":>8}  {"in_scr":>7}  {"x_scr":>7}  {"cands":>10}  {"per_S1":>8}'
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for K in k_sweep:
            for strat in ('P2', 'P1+P2', 'P1+P2+P3'):
                r = s['per_k'][K][strat]
                iscr = f'{100*r["recall_in_script"]:.2f}' if r['recall_in_script'] is not None else '   -  '
                xscr = f'{100*r["recall_cross_script"]:.2f}' if r['recall_cross_script'] is not None else '   -  '
                print(f'  {K:>5}  {strat:10}  {100*r["recall"]:>7.2f}%  {iscr:>7}  {xscr:>7}  '
                      f'{r["candidates"]:>10,}  {r["per_s1"]:>8.2f}')

        # Marginal P3 over P2 per K.
        print('\n  Marginal of +P3 over P1+P2 (recall points added):')
        for K in k_sweep:
            r2 = s['per_k'][K]['P1+P2']['recall']
            r3 = s['per_k'][K]['P1+P2+P3']['recall']
            print(f'    K={K:>3}:  {100*r2:>6.2f}%  ->  {100*r3:>6.2f}%   (+{100*(r3-r2):>5.2f} pp)')

    # Global aggregate across all measured partitions.
    print('\n' + '=' * 78)
    print('AGGREGATE (sum across partitions)'.center(78))
    print('=' * 78)
    header = f'  {"K":>5}  {"strategy":10}  {"recall":>8}  {"in_scr":>7}  {"x_scr":>7}  {"cands":>12}'
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for K in k_sweep:
        for strat in ('P2', 'P1+P2', 'P1+P2+P3'):
            hit_all = tot_all = 0
            hit_is = tot_is = 0
            hit_cs = tot_cs = 0
            n_cand = 0
            for s in all_stats:
                if s.get('skipped'):
                    continue
                r = s['per_k'][K][strat]
                # reconstruct absolute counts using totals
                hit_all += r['recall'] * s['truth_pairs']
                tot_all += s['truth_pairs']
                if r['recall_in_script'] is not None:
                    hit_is += r['recall_in_script'] * s['truth_in_script']
                    tot_is += s['truth_in_script']
                if r['recall_cross_script'] is not None:
                    hit_cs += r['recall_cross_script'] * s['truth_cross_script']
                    tot_cs += s['truth_cross_script']
                n_cand += r['candidates']
            iscr = f'{100*hit_is/tot_is:.2f}' if tot_is else '   -  '
            xscr = f'{100*hit_cs/tot_cs:.2f}' if tot_cs else '   -  '
            print(f'  {K:>5}  {strat:10}  {100*hit_all/tot_all:>7.2f}%  {iscr:>7}  {xscr:>7}  {n_cand:>12,}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', dest='only', default=None,
                    help='Restrict to one country_key (e.g. india, us)')
    ap.add_argument('--sample', type=int, default=5000,
                    help='Stratified S1 sample size (total across partitions)')
    ap.add_argument('--k', dest='k_sweep', default='10,20,50,100,200',
                    help='Comma-separated K values')
    ap.add_argument('--s2s3-cap', type=int, default=None,
                    help='Cap S2/S3 rows per partition (keeps all true partners '
                         'plus random distractors). Use to bound RAM for the sweep.')
    ap.add_argument('--out', dest='out_json', default=None,
                    help='Write per-partition results as JSON to this path.')
    args = ap.parse_args()

    k_sweep = tuple(int(x) for x in args.k_sweep.split(','))
    cfg = BlockerConfig(split='train')
    # Bump target-side K to the max we sweep — the topk_search must return enough.
    # We reuse p2_top_k / p3_top_k as the ceiling in _blocker_ranked_lists via k_max.

    d = cfg.data_root + 'train/'
    print(f'[measure] loading ground truth ...', flush=True)
    gt = load_gt(d + 'train_ground_truth.tsv')
    print(f'[measure] {len(gt):,} S1 entities have ground truth', flush=True)

    countries = None
    if args.only:
        countries = {args.only}

    print(f'[measure] stratified sample = {args.sample:,} S1 rows'
          f'{" (restricted to " + args.only + ")" if args.only else ""}', flush=True)
    sample_ids, _ = _stratified_sample(
        d + 'train_source1.tsv', gt, args.sample, cfg.seed, countries,
    )
    print(f'[measure] picked {len(sample_ids):,} S1 rows', flush=True)

    # Iterate partitions in ascending size (small first, safer).
    from blocking.io_utils import country_row_counts
    counts = country_row_counts(d + 'train_source1.tsv')
    order = sorted([k for k in counts if counts[k] > 0], key=lambda k: counts[k])
    if args.only:
        order = [args.only] if args.only in counts else []

    all_stats = []
    for ck in order:
        print(f'\n[measure] === partition {ck!r} (S1 rows = {counts[ck]:,}) ===',
              flush=True)
        try:
            s = sweep_partition(ck, cfg, gt, k_sweep, sample_ids,
                                s2s3_cap=args.s2s3_cap)
        except MemoryError as e:
            print(f'  MemoryError: {e}. Aborting this partition.')
            continue
        all_stats.append(s)

    _print_report(all_stats, k_sweep)
    print(f'\n[measure] overall peak RSS = {peak_rss_mb():.1f} MiB')

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        payload = {
            'k_sweep': list(k_sweep),
            'sample_size_requested': args.sample,
            'sample_size_actual': len(sample_ids),
            's2s3_cap': args.s2s3_cap,
            'only_country': args.only,
            'peak_rss_mib_overall': round(peak_rss_mb(), 1),
            'partitions': all_stats,
        }
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f'[measure] wrote {args.out_json}')


if __name__ == '__main__':
    main()
