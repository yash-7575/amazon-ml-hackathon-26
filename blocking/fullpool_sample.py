"""Full-pool candidate generation for a UNIFORM S1 sample — the honest training set.

Why this exists (see plan Phase 1):
  smoke_test.py caps S2/S3 at 60k rows each. That cap is harmless for RECALL --
  `_load_capped` always admits every ground-truth partner -- but it makes the
  NEGATIVES 34x thinner than production (120k pool vs 4.13M). A threshold tuned on
  that file is badly over-permissive in production, and precision is weighted 2x.

  smoke_test.py also samples `if row['entity_id'] in gt`, and measure.load_gt drops
  empty rows -- so it contains ZERO singletons. Singletons are 5.58% of entities and
  each is worth a full 1.0 of macro-F0.5. They must be in the sample.

This script fixes both: no cap, and a uniform sample over ALL S1 rows for the country.

Memory: S2 and S3 are processed SEQUENTIALLY and each partial is written to disk,
because holding both partitions plus all four indexes peaks near the ~6.6 GiB
available. Per-source peak is ~4 GiB (measured: 2.0M-row India S2 = 0.62 GiB raw
load, 0.54 GiB name index, 0.39 GiB addr index, ~4.9 GiB with query temps).

Taking top-K per source and then top-K of the merge is EXACTLY the global top-K over
the union: the merge keeps the K best of a superset of the global K best. Scores are
already compared across sources in rescore_and_union, so nothing changes semantically.

Usage:
    PYTHONPATH=. python blocking/fullpool_sample.py india 6000
"""
from __future__ import annotations
import os
import random
import sys
import time
from collections import defaultdict

from blocking.config import BlockerConfig
from blocking.io_utils import read_tsv, peak_rss_mb, CandidateWriter
from blocking.normalize import country_key
from blocking.blocker_v1 import (
    PartitionData, block_p1_exact, block_p2_name_tfidf, block_p3_addr_tfidf,
    rescore_and_union, _build_name_index, _build_addr_index,
)

S1_BATCH = 1500  # rescore holds every pair for the batch in Python dicts; cap it.


def slice_s1(s1: PartitionData, lo: int, hi: int) -> PartitionData:
    """A view-like copy of rows [lo:hi). The indexes are already built, so batching
    S1 costs nothing and keeps rescore_and_union's per-pair Python dicts bounded."""
    sub = PartitionData(s1.label, s1.source_tag)
    for attr in ('ids', 'name_raw', 'addr_raw', 'name_doc', 'addr_doc',
                 'name_core', 'name_translit', 'addr_nonempty_mask'):
        getattr(sub, attr).extend(getattr(s1, attr)[lo:hi])
    return sub


CK = sys.argv[1] if len(sys.argv) > 1 else 'india'
N_S1 = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

# config.py's data_root points at the SageMaker path; override locally.
LOCAL_ROOT = '/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/'
cfg = BlockerConfig(data_root=LOCAL_ROOT)
D = cfg.data_root + 'train/'
OUT = f'blocking/results/fullpool_{CK}_{N_S1}_candidate_pairs.long.tsv'
os.makedirs('blocking/results', exist_ok=True)

T0 = time.time()
def log(msg: str) -> None:
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)

log(f'== FULL-POOL SAMPLE — country={CK!r}  n_s1={N_S1:,}  K={cfg.final_top_k} ==')
random.seed(cfg.seed)

# ---------------------------------------------------------------- 1) uniform S1 sample
# NOT filtered on ground truth. That filter is what removed every singleton from the
# smoke sample, and it also skews the group-size distribution away from evaluation.
log('sampling S1 uniformly (no ground-truth filter) ...')
all_s1: list[str] = []
for row in read_tsv(D + 'train_source1.tsv'):
    if country_key(row['country']) == CK:
        all_s1.append(row['entity_id'])
random.shuffle(all_s1)
pick = set(all_s1[:N_S1])
log(f'  {len(all_s1):,} S1 rows in {CK}; sampled {len(pick):,}')

ids_path = f'blocking/results/fullpool_{CK}_{N_S1}_sampled_ids.txt'
with open(ids_path, 'w', encoding='utf-8') as f:
    for sid in sorted(pick):
        f.write(sid + '\n')
log(f'  wrote {ids_path}')

s1 = PartitionData(f'{CK}_fullpool', 'S1')
for row in read_tsv(D + 'train_source1.tsv'):
    if row['entity_id'] in pick:
        s1.add(row)
log(f'  s1 partition built: {s1.n:,} rows   RSS={peak_rss_mb()/1024:.2f} GiB')

# ---------------------------------------------------------------- 2) per-source pass
partials: list[str] = []
for tag, fname in (('S2', 'train_source2.tsv'), ('S3', 'train_source3.tsv')):
    part_path = f'blocking/results/.fullpool_{CK}_{N_S1}_{tag}.partial.tsv'
    if os.path.exists(part_path) and os.path.getsize(part_path) > 0:
        # Each source pass is ~15 min of retrieval and its partial is self-contained,
        # so never redo one that already completed.
        with open(part_path) as _f:
            _n = sum(1 for _ in _f) - 1
        log(f'--- {tag}: reusing existing partial ({_n:,} pairs) -> skipping')
        partials.append(part_path)
        continue
    log(f'--- {tag}: loading FULL {CK} partition (no cap) ...')
    t = time.time()
    tgt = PartitionData(f'{CK}_fullpool', tag)
    for row in read_tsv(D + fname):
        if country_key(row['country']) == CK:
            tgt.add(row)
    log(f'  {tag} loaded {tgt.n:,} rows in {time.time()-t:.1f}s   RSS={peak_rss_mb()/1024:.2f} GiB')

    t = time.time()
    name_idx = _build_name_index(tgt, cfg)
    addr_idx = _build_addr_index(tgt, cfg)
    log(f'  {tag} indexes built in {time.time()-t:.1f}s   RSS={peak_rss_mb()/1024:.2f} GiB')

    part = part_path
    n_part = 0
    t_stage = time.time()
    with CandidateWriter(part, write_header=True) as w:
        for lo in range(0, s1.n, S1_BATCH):
            hi = min(lo + S1_BATCH, s1.n)
            sb = slice_s1(s1, lo, hi)
            h1 = block_p1_exact(sb, [tgt])
            h2 = block_p2_name_tfidf(sb, tgt, cfg, prebuilt=name_idx)
            h3 = block_p3_addr_tfidf(sb, tgt, cfg, prebuilt=addr_idx)
            # Rescore against THIS source's indexes only. The absent source's pair
            # mask is empty, so passing None for it never reaches _cos_for.
            final = rescore_and_union(
                sb, [h1, h2, h3], s2_by_id={}, s3_by_id={}, cfg=cfg,
                name_index_s2=name_idx[0] if tag == 'S2' else None,
                name_index_s3=name_idx[0] if tag == 'S3' else None,
                addr_index_s2=addr_idx[0] if tag == 'S2' else None,
                addr_index_s3=addr_idx[0] if tag == 'S3' else None,
            )
            for k in range(sb.n):
                hits = final.get(k, [])
                if hits:
                    w.write_s1(sb.ids[k], [(cid, src_, sc) for (src_, cid, sc, _t) in hits])
                    n_part += len(hits)
            del sb, h1, h2, h3, final
            log(f'  {tag} batch {lo:,}-{hi:,}  pairs={n_part:,}  '
                f'{time.time()-t_stage:.0f}s  RSS={peak_rss_mb()/1024:.2f} GiB')
    partials.append(part)

    del tgt, name_idx, addr_idx
    import gc; gc.collect()

# ---------------------------------------------------------------- 3) merge -> top-K
log('merging partials and trimming to top-K ...')
merged: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
for part in partials:
    with open(part, encoding='utf-8') as f:
        next(f)
        for line in f:
            a, b, src, sc = line.rstrip('\n').split('\t')
            merged[a].append((b, src, float(sc)))

n_pairs = 0
with CandidateWriter(OUT, write_header=cfg.write_header) as w:
    for sid in s1.ids:
        lst = merged.get(sid, [])
        if not lst:
            continue
        lst.sort(key=lambda t: (-t[2], t[0]))
        keep = lst[: cfg.final_top_k]
        w.write_s1(sid, keep)
        n_pairs += len(keep)

for part in partials:
    os.remove(part)

log(f'DONE  wrote {n_pairs:,} pairs for {len(merged):,} S1 -> {OUT}')
log(f'  S1 with zero candidates: {s1.n - len(merged):,}')
log(f'  peak RSS: {peak_rss_mb()/1024:.2f} GiB')
