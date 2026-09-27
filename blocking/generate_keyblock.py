"""Candidate generation with inverted-index key blocking + TF-IDF rescore.

Two stages, because each fixes the other's weakness:
  1. The key index PROPOSES ~100-175 candidates per S1 in ~0.1 ms. Cheap, but unranked
     and recall-bounded.
  2. TF-IDF DISPOSES: vectors are built only for that S1's own candidates and scored with
     the same 0.6*name + 0.4*addr weights the old blocker used, then trimmed to top-K.
     Same ranking quality, but the matmul is over ~150 documents instead of 4.1M.

The rescore is what makes the group-relative features meaningful -- rank_in_group and
score_margin_to_2nd are only informative if the ordering means something.

Usage:
  PYTHONPATH=. python blocking/generate_keyblock.py --split test --country India --out X.tsv
  PYTHONPATH=. python blocking/generate_keyblock.py --split train --country India --sample 20000
"""
from __future__ import annotations
import argparse, os, random, resource, sys, time
from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, '.')
from blocking import keyblock as KB
from blocking.io_utils import CandidateWriter

DEFAULT_ROOT = '/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/'

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='test', choices=['train', 'test'])
ap.add_argument('--country', required=True)
ap.add_argument('--out', default=None)
ap.add_argument('--data-root', default=DEFAULT_ROOT)
ap.add_argument('--cap', type=int, default=400)
ap.add_argument('--n-rt', type=int, default=10)
ap.add_argument('--n-ra', type=int, default=6)
ap.add_argument('--fams', default='RT,RA,G4,X')
ap.add_argument('--top-k', type=int, default=40, help='candidates kept per S1 after rescore')
ap.add_argument('--sample', type=int, default=0, help='train only: uniform S1 sample size')
ap.add_argument('--skip-first', type=int, default=0,
                help='Resume: skip the first N S1 rows in FILE ORDER. A partial run covers '
                     'a prefix of the file, so this generates exactly the remainder instead '
                     'of redoing work. Only valid without --sample (which shuffles).')
ap.add_argument('--w-name', type=float, default=0.6)
ap.add_argument('--w-addr', type=float, default=0.4)
A = ap.parse_args()
KB.FAMS, KB.N_RT, KB.N_RA = set(A.fams.split(',')), A.n_rt, A.n_ra

D = os.path.join(A.data_root, A.split) + '/'
OUT = A.out or f'blocking/results/{A.split}_{A.country.lower()}_keyblock.long.tsv'
os.makedirs('blocking/results', exist_ok=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
T0 = time.time()
def log(m): print(f'[{time.time()-T0:7.1f}s] {m}  RSS={rss():.2f}G', flush=True)


def stream(src: int):
    with open(f'{D}{A.split}_source{src}.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4 and p[3] == A.country:
                yield p[0], p[1], p[2]


log(f'== KEYBLOCK  split={A.split} country={A.country} K={A.top_k} cap={A.cap} ==')

# ---- pass 1: document frequencies over the target corpus ----
nd, ad, gd = defaultdict(int), defaultdict(int), defaultdict(int)
n_tgt = 0
for src in (2, 3):
    for eid, nm, adr in stream(src):
        R = KB.record_terms(nm, adr)
        for t in set(R['nt']): nd[t] += 1
        for t in set(R['at']): ad[t] += 1
        if 'G4' in KB.FAMS:
            for g in set(R['grams']): gd[g] += 1
        n_tgt += 1
nd, ad, gd = dict(nd), dict(ad), dict(gd)
log(f'DF done. targets={n_tgt:,}')

# ---- pass 2: inverted index + keep target text for the rescore ----
inv = defaultdict(list)
tgt_ids: list[str] = []
tgt_name: list[str] = []
tgt_addr: list[str] = []
for src in (2, 3):
    for eid, nm, adr in stream(src):
        i = len(tgt_ids)
        tgt_ids.append(eid)
        R = KB.record_terms(nm, adr)
        tgt_name.append(R['n'] if not R['tr'] else R['n'] + ' ' + R['tr'])
        tgt_addr.append(R['a'])
        for k in KB.keys_for(R, nd, ad, gd, domain_stem=KB.domain_stem_of(nm)):
            inv[k].append(i)
nb = len(inv)
inv = {k: v for k, v in inv.items() if len(v) <= A.cap}
log(f'index built. keys={len(inv):,} of {nb:,}')

# ---- TF-IDF vocabularies, fit once over the whole target corpus ----
vn = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2,
                     sublinear_tf=True, lowercase=False, norm='l2', dtype=np.float32)
Xn = vn.fit_transform(tgt_name)
va = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2,
                     sublinear_tf=True, lowercase=False, norm='l2', dtype=np.float32)
Xa = va.fit_transform(tgt_addr)
del tgt_name, tgt_addr
log(f'tfidf fitted. name nnz={Xn.nnz:,} addr nnz={Xa.nnz:,}')

# ---- S1 ----
s1 = list(stream(1))
if A.skip_first:
    assert not A.sample, '--skip-first assumes file order; --sample shuffles'
    log(f'resuming: skipping first {A.skip_first:,} of {len(s1):,} S1 rows')
    s1 = s1[A.skip_first:]
if A.sample and A.sample < len(s1):
    random.seed(20260926); random.shuffle(s1); s1 = s1[:A.sample]
    ids_path = OUT.replace('.long.tsv', '_sampled_ids.txt')
    with open(ids_path, 'w') as f:
        for e, _, _ in s1: f.write(e + '\n')
    log(f'sampled {len(s1):,} S1 -> {ids_path}')
log(f'S1 rows = {len(s1):,}')

t = time.time(); n_pairs = 0; n_empty = 0; done = 0
with CandidateWriter(OUT, write_header=True) as w:
    for eid, nm, adr in s1:
        R = KB.record_terms(nm, adr)
        cand = KB.query(R, inv, nd, ad, gd, domain_stem=KB.domain_stem_of(nm))
        if not cand:
            n_empty += 1; done += 1; continue
        idx = np.fromiter(cand, dtype=np.int64, count=len(cand))
        qn = vn.transform([R['n']])
        qa = va.transform([R['a']])
        # cosine against ONLY this entity's candidates: ~150 rows, not 4.1M
        sn = np.asarray((Xn[idx] @ qn.T).todense()).ravel()
        sa = np.asarray((Xa[idx] @ qa.T).todense()).ravel()
        sc = A.w_name * sn + A.w_addr * sa
        if len(idx) > A.top_k:
            keep = np.argpartition(-sc, A.top_k - 1)[:A.top_k]
        else:
            keep = np.arange(len(idx))
        order = keep[np.argsort(-sc[keep], kind='stable')]
        rows = [(tgt_ids[int(idx[j])],
                 'S2' if tgt_ids[int(idx[j])].startswith('S2') else 'S3',
                 float(sc[j])) for j in order]
        w.write_s1(eid, rows)
        n_pairs += len(rows)
        done += 1
        if done % 50000 == 0:
            el = time.time() - t
            log(f'  {done:,}/{len(s1):,}  pairs={n_pairs:,}  {el:.0f}s  '
                f'eta={el/done*(len(s1)-done):.0f}s')

el = time.time() - t
log(f'DONE {n_pairs:,} pairs for {len(s1)-n_empty:,} S1 in {el:.0f}s -> {OUT}')
log(f'  zero-candidate S1: {n_empty:,} ({100*n_empty/len(s1):.2f}%)   {1000*el/len(s1):.3f} ms/S1')
