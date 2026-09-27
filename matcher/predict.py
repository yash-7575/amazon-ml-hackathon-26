"""Streaming inference over a test candidate file.

Never materializes the full feature matrix. At K=150 the test set is ~260M pairs; at 77
float32 features that is 80 GB, against 12 GiB of free disk. So each chunk is featurized,
scored, and thrown away, keeping only (s1_id, cand_id, probability).

Chunks are cut on S1-GROUP BOUNDARIES because the group-relative features (rank_in_group,
score_margin_to_2nd, is_mutual_best) are computed across a whole candidate group. Splitting
a group across chunks would change those values versus training.

Parallelism: measured 47 us/pair for feature extraction, so 260M pairs is ~3.4 h on one
core. Workers are forked so they share the parent's record text via copy-on-write instead
of each loading its own copy.

Usage:
  PYTHONPATH=. python -m matcher.predict --pairs <long.tsv> --split test \
      --model matcher/models/india_kb150.lgb --idf matcher/data/india_kb150_idf.pkl \
      --out preds.tsv --workers 14
"""
from __future__ import annotations
import argparse, json, os, pickle, resource, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import lightgbm as lgb

from .featconfig import FeatureConfig
from .featurize import prepare, featurize
from blocking.io_utils import read_tsv

DATA_ROOT = '/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/'
_G: dict = {}


def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _load_text(split: str, ids_needed: set, srcs=(1, 2, 3)) -> dict:
    """id -> (name, addr). Strings only; the expensive prepared structures are built
    per chunk, because holding them for millions of records does not fit."""
    out = {}
    for n in srcs:
        for row in read_tsv(f'{DATA_ROOT}{split}/{split}_source{n}.tsv'):
            e = row['entity_id']
            if e in ids_needed:
                out[e] = (row.get('business_name', '') or '',
                          row.get('business_address', '') or '')
    return out


def _work(chunk):
    """Featurize + score one S1-group-aligned chunk. Runs in a forked worker."""
    s1_ids, cand_ids, rscore, src, txt = chunk
    cfg, cols, booster, idf = _G['cfg'], _G['cols'], _G['booster'], _G['idf']
    lu = sorted({s for s in s1_ids}); ru = sorted({c for c in cand_ids})
    li_map = {e: i for i, e in enumerate(lu)}
    ri_map = {e: i for i, e in enumerate(ru)}
    L = prepare([txt[e][0] for e in lu], [txt[e][1] for e in lu], cfg)
    R = prepare([txt[e][0] for e in ru], [txt[e][1] for e in ru], cfg)
    li = np.fromiter((li_map[e] for e in s1_ids), dtype=np.int64, count=len(s1_ids))
    ri = np.fromiter((ri_map[e] for e in cand_ids), dtype=np.int64, count=len(cand_ids))
    X, c = featurize(li, ri, L, R, cfg, retrieval_score=rscore, source_tag=src,
                     s1_group=li, cand_group=ri,
                     name_idf=idf['name'], a_idf=idf['addr'])
    assert c == cols, 'feature columns diverged from training'
    p = booster.predict(X, num_iteration=booster.best_iteration)
    return s1_ids, cand_ids, p.astype(np.float32)


def _init(model_path):
    # txt/cfg/cols/idf arrive via fork inheritance; only the booster is built per worker
    # (LightGBM Boosters do not survive pickling cleanly).
    _G['booster'] = lgb.Booster(model_file=model_path)


def iter_chunks(path, max_pairs, max_per_s1=0):
    """Yield chunks that never split an S1 group."""
    s1, cand, sc, src = [], [], [], []
    cur = None
    seen = 0
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            a, b, s, v = line.rstrip('\n').split('\t')
            if a != cur:
                if len(s1) >= max_pairs:
                    yield s1, cand, np.asarray(sc, np.float32), src
                    s1, cand, sc, src = [], [], [], []
                seen = 0
            cur = a
            seen += 1
            if max_per_s1 and seen > max_per_s1:
                continue          # rows are already in descending score order
            s1.append(a); cand.append(b); src.append(s); sc.append(float(v))
    if s1:
        yield s1, cand, np.asarray(sc, np.float32), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--model', required=True)
    ap.add_argument('--idf', required=True)
    ap.add_argument('--cols', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--chunk-pairs', type=int, default=25_000,
                    help='Pairs per work unit. The binding cost is prepare() over a\n'
                         'chunk DISTINCT candidate records: ~18 structures each,\n'
                         'including two n-gram sets. At 150k pairs that was ~2 GB per\n'
                         'worker and pinned the machine at 14/14 GB. 25k keeps a\n'
                         'worker near 400 MB.')
    ap.add_argument('--max-per-s1', type=int, default=0,
                    help='Emergency time lever. Score only the top-N candidates per S1. '
                         'The long file is already written in descending score order, so '
                         'this truncates the tail rather than sampling. N=60 cuts work ~60%% '
                         'at a measured recall cost (86.05%% at K=150 vs 78.41%% at K=50). '
                         '0 = keep all.')
    ap.add_argument('--min-prob', type=float, default=0.02,
                    help='Drop pairs scoring below this. At K=150 most candidates score '
                         'near zero; writing all 121M for India would cost ~4 GB of the '
                         '9.5 GB free. A pair this weak can never be chosen -- it loses '
                         'every conflict and expected-F0.5 would never include it.')
    a = ap.parse_args()
    t0 = time.time()
    def log(m): print(f'[{time.time()-t0:7.1f}s] {m}  RSS={rss():.2f}G', flush=True)

    cfg = FeatureConfig()
    cols = json.load(open(a.cols))['cols']
    idf = pickle.load(open(a.idf, 'rb'))

    log('scanning pair file for referenced ids ...')
    need = set()
    with open(a.pairs, encoding='utf-8') as f:
        next(f)
        for line in f:
            i = line.find('\t'); j = line.find('\t', i + 1)
            need.add(line[:i]); need.add(line[i + 1:j])
    log(f'  {len(need):,} distinct ids')
    txt = _load_text(a.split, need)
    log(f'  loaded text for {len(txt):,} records')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    # Populate globals BEFORE the pool forks so children inherit them copy-on-write.
    # Passing them via initargs PICKLES a full ~1 GB copy into each worker, which pinned
    # the machine at 14/14 GB and froze it.
    # Workers must NOT each hold the full record dict. fork+CoW does not save us --
    # CPython refcounting writes to every object header and copies the page, so 6 workers
    # each materialized ~1.9 GB and pinned the machine at 14/14 GB. Instead the parent
    # slices out only the records a chunk references and ships them with the task.
    _G.update(cfg=cfg, cols=cols, idf=idf)

    def with_text(it):
        for _s1, _cd, _sc, _sr in it:
            sub = {}
            for e in _s1:
                if e not in sub:
                    sub[e] = txt[e]
            for e in _cd:
                if e not in sub:
                    sub[e] = txt[e]
            yield _s1, _cd, _sc, _sr, sub
    n = 0; n_kept = 0
    with open(a.out, 'w', encoding='utf-8') as fo, \
         ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.model,)) as ex:
        fo.write('s1_entity_id\tcandidate_entity_id\tprob\n')
        # Executor.map consumes its entire input iterable up front, so every chunk's
        # text slice was built and queued immediately and the parent climbed past 7.5 GB.
        # Keep a bounded number of tasks in flight instead.
        from concurrent.futures import FIRST_COMPLETED, wait
        src_iter = with_text(iter_chunks(a.pairs, a.chunk_pairs, a.max_per_s1))
        inflight = set()

        def drain(done_set):
            nonlocal n, n_kept
            for fut in done_set:
                _s1, _cd, _pr = fut.result()
                for x, y, z in zip(_s1, _cd, _pr):
                    if z >= a.min_prob:
                        fo.write(f'{x}\t{y}\t{z:.6f}\n')
                        n_kept += 1
                n += len(_s1)
                if n % 2_000_000 < len(_s1):
                    log(f'  scored {n:,} pairs, kept {n_kept:,}')

        for chunk in src_iter:
            inflight.add(ex.submit(_work, chunk))
            if len(inflight) >= a.workers * 2:
                done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                drain(done_set)
        while inflight:
            done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            drain(done_set)
    log(f'DONE scored {n:,} pairs, wrote {n_kept:,} above min_prob={a.min_prob} -> {a.out}')


if __name__ == '__main__':
    main()
