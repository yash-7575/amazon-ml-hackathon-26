"""Recall harness for keyblock.

STREAMS the target corpus twice (once for document frequencies, once to build the index)
and never holds per-record term structures. Caching them OOM'd at 4.1M records: the
char-4gram lists alone are ~82M strings. Recomputing normalization is far cheaper than
storing it.

Usage: PYTHONPATH=. python blocking/measure_keyblock.py --cap 100 --n-s1 8000 --fams RT,RA,G4,X
"""
from __future__ import annotations
import argparse, random, resource, sys, time
from collections import defaultdict
sys.path.insert(0, '.')
from blocking import keyblock as KB

D = '/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/train/'
ap = argparse.ArgumentParser()
ap.add_argument('--cap', type=int, default=100)
ap.add_argument('--n-s1', type=int, default=8000)
ap.add_argument('--country', default='India')
ap.add_argument('--fams', default='RT,RA,G4,X',
                help='comma list: RT rare-name, RA rare-addr, G4 char4gram, X cross-field')
ap.add_argument('--n-rt', type=int, default=3)
ap.add_argument('--n-ra', type=int, default=2)
A = ap.parse_args()
FAMS = set(A.fams.split(',')) if A.fams else set()
KB.FAMS, KB.N_RT, KB.N_RA = FAMS, A.n_rt, A.n_ra

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
T0 = time.time()
def log(m): print(f'[{time.time()-T0:6.1f}s] {m}  RSS={rss():.2f}G', flush=True)


def stream_targets():
    for fn in ('train_source2.tsv', 'train_source3.tsv'):
        with open(D + fn, encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4 and p[3] == A.country:
                    yield p[0], p[1], p[2]


log(f'pass 1/2: document frequencies (fams={sorted(FAMS)}) ...')
nd, ad, gd = defaultdict(int), defaultdict(int), defaultdict(int)
n_tgt = 0
for eid, nm, adr in stream_targets():
    R = KB.record_terms(nm, adr)
    for t in set(R['nt']): nd[t] += 1
    for t in set(R['at']): ad[t] += 1
    if 'G4' in FAMS:
        for g in set(R['grams']): gd[g] += 1
    n_tgt += 1
nd, ad, gd = dict(nd), dict(ad), dict(gd)
log(f'  targets={n_tgt:,}  |name|={len(nd):,} |addr|={len(ad):,} |gram|={len(gd):,}')

log('pass 2/2: building inverted index ...')
inv = defaultdict(list)
tgt_ids = []
for i, (eid, nm, adr) in enumerate(stream_targets()):
    tgt_ids.append(eid)
    R = KB.record_terms(nm, adr)
    for k in KB.keys_for(R, nd, ad, gd, domain_stem=KB.domain_stem_of(nm)):
        inv[k].append(i)
n_before = len(inv)
inv = {k: v for k, v in inv.items() if len(v) <= A.cap}
log(f'  keys {len(inv):,} kept of {n_before:,} (cap={A.cap}, dropped {n_before-len(inv):,})')

# S1 sample + ground truth
random.seed(7)
s1_all = []
with open(D + 'train_source1.tsv', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4 and p[3] == A.country:
            s1_all.append((p[0], p[1], p[2]))
random.shuffle(s1_all)
s1 = s1_all[:A.n_s1]
del s1_all
want = {e for e, _, _ in s1}
gt = {}
with open(D + 'train_ground_truth.tsv', encoding='utf-8') as f:
    next(f)
    for line in f:
        a, _, b = line.rstrip('\n').partition('\t')
        if a in want:
            gt[a] = {x for x in b.split(',') if x}
tot = sum(len(v) for v in gt.values())
log(f'S1 sample={len(s1):,}  singletons={sum(1 for v in gt.values() if not v):,}  true pairs={tot:,}')

t = time.time(); hit = nc = empty = 0
for eid, nm, adr in s1:
    R = KB.record_terms(nm, adr)
    c = KB.query(R, inv, nd, ad, gd, domain_stem=KB.domain_stem_of(nm))
    nc += len(c)
    if not c: empty += 1
    truth = gt.get(eid, set())
    if truth:
        hit += len(truth & {tgt_ids[j] for j in c})
el = time.time() - t
print()
print(f'  FAMS={sorted(FAMS)} n_rt={A.n_rt} n_ra={A.n_ra} cap={A.cap}')
print(f'  RECALL      = {hit:,}/{tot:,} = {100*hit/max(tot,1):.2f}%')
print(f'  cands/S1    = {nc/len(s1):.1f}')
print(f'  zero-cand   = {empty:,} ({100*empty/len(s1):.2f}%)')
print(f'  query       = {1000*el/len(s1):.3f} ms/S1  -> 1.73M S1 = {1.73e6*el/len(s1)/60:.1f} min')
print(f'  peak RSS    = {rss():.2f} GiB')
