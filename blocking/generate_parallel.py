"""Parallel candidate generation for a WHOLE split+country. Built for the test run.

Why threads and not processes: the dominant cost is the sparse matmul in
`index.topk_search`, and scipy releases the GIL for it. Threads therefore give real
parallelism while SHARING one copy of the index. Processes would need one copy of a
multi-GB index per worker and would exhaust RAM long before they finished.

Two structural costs removed versus the sequential path:
  - the index transpose is built ONCE and pre-warmed (see index.transposed_index);
    previously every batch rebuilt it.
  - S2 and S3 are processed one at a time, each writing a self-contained partial, so a
    crash never costs more than one source and a rerun resumes.

Usage:
    PYTHONPATH=. python blocking/generate_parallel.py --split test --country us --threads 8
"""
from __future__ import annotations
import argparse, gc, os, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from blocking.config import BlockerConfig
from blocking.io_utils import read_tsv, peak_rss_mb, CandidateWriter
from blocking.normalize import country_key
from blocking.index import transposed_index
from blocking.blocker_v1 import (
    PartitionData, block_p1_exact, block_p2_name_tfidf, block_p3_addr_tfidf,
    rescore_and_union, _build_name_index, _build_addr_index,
)

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='test', choices=['train', 'test'])
ap.add_argument('--country', required=True)
ap.add_argument('--threads', type=int, default=os.cpu_count())
ap.add_argument('--chunk', type=int, default=400, help='S1 rows per work unit')
ap.add_argument('--data-root', default=None)
ap.add_argument('--out', default=None)
A = ap.parse_args()

cfg = BlockerConfig(data_root=A.data_root) if A.data_root else BlockerConfig()
D = cfg.data_root + A.split + '/'
CK = country_key(A.country)
OUT = A.out or f'blocking/results/{A.split}_{CK}_candidate_pairs.long.tsv'
os.makedirs('blocking/results', exist_ok=True)

T0 = time.time()
_pl = threading.Lock()
def log(m):
    with _pl:
        print(f'[{time.time()-T0:8.1f}s] {m}', flush=True)

log(f'== PARALLEL GENERATE  split={A.split} country={CK} threads={A.threads} K={cfg.final_top_k} ==')

log('loading S1 ...')
s1 = PartitionData(f'{CK}_{A.split}', 'S1')
for row in read_tsv(D + f'{A.split}_source1.tsv'):
    if country_key(row['country']) == CK:
        s1.add(row)
log(f'  S1 rows = {s1.n:,}   RSS={peak_rss_mb()/1024:.2f} GiB')
if s1.n == 0:
    sys.exit(f'no S1 rows for country {CK!r} in split {A.split!r}')


def slice_s1(lo, hi):
    sub = PartitionData(s1.label, s1.source_tag)
    for attr in ('ids', 'name_raw', 'addr_raw', 'name_doc', 'addr_doc',
                 'name_core', 'name_translit', 'addr_nonempty_mask'):
        getattr(sub, attr).extend(getattr(s1, attr)[lo:hi])
    return sub


partials = []
for tag, fname in (('S2', f'{A.split}_source2.tsv'), ('S3', f'{A.split}_source3.tsv')):
    part = f'blocking/results/.{A.split}_{CK}_{tag}.partial.tsv'
    if os.path.exists(part) and os.path.getsize(part) > 0:
        log(f'--- {tag}: reusing existing partial -> skip'); partials.append(part); continue

    log(f'--- {tag}: loading full {CK} partition ...')
    t = time.time()
    tgt = PartitionData(f'{CK}_{A.split}', tag)
    for row in read_tsv(D + fname):
        if country_key(row['country']) == CK:
            tgt.add(row)
    log(f'  {tag} rows = {tgt.n:,} in {time.time()-t:.1f}s   RSS={peak_rss_mb()/1024:.2f} GiB')

    t = time.time()
    name_idx = _build_name_index(tgt, cfg)
    addr_idx = _build_addr_index(tgt, cfg)
    log(f'  {tag} indexes built in {time.time()-t:.1f}s')

    # Pre-warm both transposes before any thread runs, so they are built once rather
    # than raced for by every worker.
    t = time.time()
    for ix in (name_idx[0], addr_idx[0]):
        if ix is not None:
            transposed_index(ix.matrix)
    log(f'  {tag} transposes pre-warmed in {time.time()-t:.1f}s   RSS={peak_rss_mb()/1024:.2f} GiB')

    bounds = [(lo, min(lo + A.chunk, s1.n)) for lo in range(0, s1.n, A.chunk)]
    done = {'n': 0, 'pairs': 0}
    wlock = threading.Lock()
    t_stage = time.time()

    with CandidateWriter(part, write_header=True) as w:
        def work(b):
            lo, hi = b
            sb = slice_s1(lo, hi)
            h1 = block_p1_exact(sb, [tgt])
            h2 = block_p2_name_tfidf(sb, tgt, cfg, prebuilt=name_idx)
            h3 = block_p3_addr_tfidf(sb, tgt, cfg, prebuilt=addr_idx)
            final = rescore_and_union(
                sb, [h1, h2, h3], s2_by_id={}, s3_by_id={}, cfg=cfg,
                name_index_s2=name_idx[0] if tag == 'S2' else None,
                name_index_s3=name_idx[0] if tag == 'S3' else None,
                addr_index_s2=addr_idx[0] if tag == 'S2' else None,
                addr_index_s3=addr_idx[0] if tag == 'S3' else None,
            )
            rows = [(sb.ids[k], [(cid, src, sc) for (src, cid, sc, _t) in final[k]])
                    for k in range(sb.n) if final.get(k)]
            with wlock:
                for sid, r in rows:
                    w.write_s1(sid, r); done['pairs'] += len(r)
                done['n'] += 1
                if done['n'] % 25 == 0 or done['n'] == len(bounds):
                    el = time.time() - t_stage
                    frac = done['n'] / len(bounds)
                    log(f'  {tag} {done["n"]}/{len(bounds)} chunks  '
                        f'pairs={done["pairs"]:,}  {el:.0f}s  '
                        f'eta={el/max(frac,1e-9)-el:.0f}s  RSS={peak_rss_mb()/1024:.2f} GiB')

        with ThreadPoolExecutor(max_workers=A.threads) as ex:
            list(ex.map(work, bounds))

    partials.append(part)
    log(f'  {tag} DONE {done["pairs"]:,} pairs in {time.time()-t_stage:.0f}s')
    del tgt, name_idx, addr_idx
    gc.collect()

log('merging partials -> global top-K ...')
merged = defaultdict(list)
for p in partials:
    with open(p, encoding='utf-8') as f:
        next(f)
        for line in f:
            a, b, src, sc = line.rstrip('\n').split('\t')
            merged[a].append((b, src, float(sc)))

n = 0
with CandidateWriter(OUT, write_header=cfg.write_header) as w:
    for sid in s1.ids:
        lst = merged.get(sid)
        if not lst:
            continue
        lst.sort(key=lambda t: (-t[2], t[0]))
        keep = lst[: cfg.final_top_k]
        w.write_s1(sid, keep); n += len(keep)

for p in partials:
    os.remove(p)
log(f'DONE  {n:,} pairs for {len(merged):,} S1 -> {OUT}')
log(f'  S1 with zero candidates: {s1.n - len(merged):,}   peak RSS {peak_rss_mb()/1024:.2f} GiB')
