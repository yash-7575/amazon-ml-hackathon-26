"""Smoke test — run the full pipeline on a small SUBSAMPLE of India, with ground
truth, to verify every stage before committing to a big run.

This is NOT a leak of ground truth into generation: we filter the loaded rows
to a subsample and pass that same subsample to the evaluator. The blocker itself
doesn't know about the labels — it only knows about a smaller partition."""
import sys, random, time
from dataclasses import replace
from collections import defaultdict

from blocking.config import BlockerConfig
from blocking.io_utils import read_tsv, peak_rss_mb
from blocking.normalize import country_key, has_devanagari
from blocking.blocker_v1 import (
    PartitionData, block_p1_exact, block_p2_name_tfidf, block_p3_addr_tfidf,
    rescore_and_union, _build_name_index, _build_addr_index,
)
from blocking.measure import load_gt


CK = sys.argv[1] if len(sys.argv) > 1 else 'india'
SAMPLE_S1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

cfg = BlockerConfig()
D = cfg.data_root + 'train/'

print(f'== SMOKE TEST — country={CK!r}, sample_s1={SAMPLE_S1:,} ==')
random.seed(cfg.seed)

# 1) Load ground truth and sample S1 ids for country CK.
print('[1/5] loading ground truth ...', flush=True)
gt = load_gt(D + 'train_ground_truth.tsv')

# 2) First pass over S1: pick SAMPLE_S1 rows in this country that have ground truth.
print('[2/5] sampling S1 ...', flush=True)
s1_country_gt_ids = []
for row in read_tsv(D + 'train_source1.tsv'):
    if country_key(row['country']) != CK: continue
    if row['entity_id'] in gt: s1_country_gt_ids.append(row['entity_id'])
random.shuffle(s1_country_gt_ids)
s1_pick = set(s1_country_gt_ids[:SAMPLE_S1])
print(f'   sampled {len(s1_pick):,} S1 rows (of {len(s1_country_gt_ids):,} in gt for {CK})')

# 3) Load PartitionData for those S1 rows, and for S2/S3 rows that are ground-truth
#    matches of those S1 rows PLUS some distractors. To keep it a real blocking
#    test (not a leak), we pull ALL S2/S3 for the country. If the country is too
#    big, we cap S2/S3 at N random rows plus all the ground-truth partners.
S2_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
S3_CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
print(f'[3/5] loading S1/S2/S3 (S2/S3 capped at {S2_CAP:,}/{S3_CAP:,}) ...', flush=True)

need_ids = set()
for sid in s1_pick:
    need_ids.update(gt[sid])

s1 = PartitionData('india_smoke', 'S1')
for row in read_tsv(D + 'train_source1.tsv'):
    if country_key(row['country']) != CK: continue
    if row['entity_id'] in s1_pick:
        s1.add(row)

def _load_capped(path, tag, cap):
    p = PartitionData('smoke', tag)
    mandatory = []
    optional = []
    for row in read_tsv(path):
        if country_key(row['country']) != CK: continue
        if row['entity_id'] in need_ids: mandatory.append(row)
        else: optional.append(row)
    random.shuffle(optional)
    rows = mandatory + optional[:max(0, cap - len(mandatory))]
    random.shuffle(rows)
    for r in rows: p.add(r)
    return p

t = time.time()
s2 = _load_capped(D + 'train_source2.tsv', 'S2', S2_CAP)
s3 = _load_capped(D + 'train_source3.tsv', 'S3', S3_CAP)
print(f'   loaded s1={s1.n:,}  s2={s2.n:,}  s3={s3.n:,}   in {time.time()-t:.1f}s')

# 4) Run stages.
print('[4/5] running stages ...', flush=True)
t = time.time()
hits_p1 = block_p1_exact(s1, [s2, s3])
print(f'   P1 done  ({time.time()-t:.2f}s)  s1 with any P1 hit = {len(hits_p1):,}')

t = time.time()
name_s2 = _build_name_index(s2, cfg); name_s3 = _build_name_index(s3, cfg)
addr_s2 = _build_addr_index(s2, cfg); addr_s3 = _build_addr_index(s3, cfg)
print(f'   indexes built ({time.time()-t:.2f}s)')

t = time.time()
hits_p2_s2 = block_p2_name_tfidf(s1, s2, cfg, prebuilt=name_s2)
hits_p2_s3 = block_p2_name_tfidf(s1, s3, cfg, prebuilt=name_s3)
print(f'   P2 done  ({time.time()-t:.2f}s)  s1 with any P2 hit = {len(set(hits_p2_s2)|set(hits_p2_s3)):,}')

t = time.time()
hits_p3_s2 = block_p3_addr_tfidf(s1, s2, cfg, prebuilt=addr_s2)
hits_p3_s3 = block_p3_addr_tfidf(s1, s3, cfg, prebuilt=addr_s3)
print(f'   P3 done  ({time.time()-t:.2f}s)  s1 with any P3 hit = {len(set(hits_p3_s2)|set(hits_p3_s3)):,}')

t = time.time()
final = rescore_and_union(
    s1, [hits_p1, hits_p2_s2, hits_p2_s3, hits_p3_s2, hits_p3_s3],
    s2_by_id={}, s3_by_id={}, cfg=cfg,
    name_index_s2=name_s2[0], name_index_s3=name_s3[0],
    addr_index_s2=addr_s2[0], addr_index_s3=addr_s3[0],
)
print(f'   rescore  ({time.time()-t:.2f}s)  s1 with any final hit = {len(final):,}')

# 5) Recall.
print('[5/5] recall ...', flush=True)
ids_here = set(s2.ids) | set(s3.ids)
truth = {}
for i, sid in enumerate(s1.ids):
    want = gt.get(sid, set()) & ids_here
    if want: truth[i] = want

def hits_to_sets(*hits_dicts):
    out = defaultdict(set)
    for hd in hits_dicts:
        for i, lst in hd.items():
            for (_src, cid, _sc, _tag) in lst:
                out[i].add(cid)
    return out

def recall(cand):
    hit = tot = 0
    for i, w in truth.items():
        hit += len(w & cand.get(i, set()))
        tot += len(w)
    return hit, tot

cand_p1     = hits_to_sets(hits_p1)
cand_p12    = hits_to_sets(hits_p1, hits_p2_s2, hits_p2_s3)
cand_p123   = hits_to_sets(hits_p1, hits_p2_s2, hits_p2_s3, hits_p3_s2, hits_p3_s3)
cand_final  = {i: {t[1] for t in lst} for i, lst in final.items()}

for name, c in [('P1', cand_p1), ('P1+P2', cand_p12), ('P1+P2+P3', cand_p123),
                ('final top-K', cand_final)]:
    h, t = recall(c)
    print(f'   recall {name:14}: {h:,}/{t:,} = {100*h/max(t,1):.2f}%')

# per-source total truth in scope
truth_total = sum(len(v) for v in truth.values())
print(f'\n   truth pairs in scope   : {truth_total:,}')
print(f'   |cand_final|           : {sum(len(v) for v in cand_final.values()):,}')
print(f'   |cand_final| per S1    : {sum(len(v) for v in cand_final.values())/max(len(s1.ids),1):.2f}')
print(f'   peak RSS               : {peak_rss_mb():.1f} MiB')

# cross-script sanity
deva_targets = {s2.ids[i] for i in range(s2.n) if has_devanagari(s2.name_raw[i])}
deva_targets |= {s3.ids[i] for i in range(s3.n) if has_devanagari(s3.name_raw[i])}
truth_cs = {i: w & deva_targets for i, w in truth.items()}
truth_cs = {i: w for i, w in truth_cs.items() if w}
if truth_cs:
    h = sum(len(w & cand_final.get(i,set())) for i,w in truth_cs.items())
    t = sum(len(w) for w in truth_cs.values())
    print(f'   cross-script recall    : {h:,}/{t:,} = {100*h/max(t,1):.2f}%')
