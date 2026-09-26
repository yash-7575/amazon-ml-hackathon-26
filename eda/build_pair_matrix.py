"""Build the numeric pair-feature matrix that the EDA checklist operates on.
Positives = ground-truth matches. Negatives = hard candidates from a real blocking pass.
Stdlib only (runs before deps are needed). Output: pairs.csv"""
import csv, random, re, sys, unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

csv.field_size_limit(10**9)
D = "/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/"
OUT = "/home/yash/dev/clg/amlc-2026/eda/pairs.csv"
N_S1        = 20000    # S1 entities sampled
POOL_FRAC   = 0.25     # fraction of S2/S3 indexed as the negative pool
MAX_DF      = 120      # tokens rarer than this are usable blocking keys
MAX_CAND    = 40       # candidates retrieved per S1 entity
random.seed(11)

DEVA = re.compile(r'[ऀ-ॿ]')
SUFFIX = {'inc','llc','ltd','limited','corp','corporation','co','company','pvt','private',
          'llp','plc','sarl','sas','sa','eurl','gmbh','and','the','of'}
NUMTOK = re.compile(r'\d+')

def norm(s):
    """Unicode-category strip: keeps Devanagari matras/virama (cat Mn/Mc) that \\w drops."""
    s = unicodedata.normalize('NFKC', s).lower().replace('&', ' and ')
    out = ''.join(' ' if (unicodedata.category(c)[0] in 'PSZ' and not c.isspace()) else c
                  for c in s)
    return ' '.join(out.split())
def ntoks(s):
    t = [x for x in norm(s).split() if x not in SUFFIX]
    return t or norm(s).split()
def grams(s, n=3):
    s = norm(s).replace(' ', '')
    return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))
def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0
def cont(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0

# ---------- 1. sample S1 entities + ground truth ----------
with open(D+"train/train_ground_truth.tsv", encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t'); next(r)
    gt_rows = [row for row in r]
sample = random.sample(gt_rows, N_S1)
truth = {sid: set(x for x in ids.split(',') if x) for sid, ids in sample}
need  = set().union(*truth.values()) if truth else set()
del gt_rows
print(f"[1] sampled {len(truth):,} S1 entities, {len(need):,} true partners", flush=True)

s1 = {}
with open(D+"train/train_source1.tsv", encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t'); next(r)
    for row in r:
        if row[0] in truth: s1[row[0]] = row
print(f"[2] resolved {len(s1):,} S1 records", flush=True)

# ---------- 2. build negative pool: true partners + random slice of S2/S3 ----------
pool = []
for src in ('2', '3'):
    with open(D+f"train/train_source{src}.tsv", encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t'); next(r)
        for row in r:
            if row[0] in need or random.random() < POOL_FRAC:
                pool.append(row)
print(f"[3] pool size {len(pool):,} records", flush=True)

# ---------- 3. inverted index on rare name tokens + address numeric tokens ----------
df_ct = Counter()
pool_ntoks = []
for row in pool:
    t = set(ntoks(row[1])); pool_ntoks.append(t); df_ct.update(t)
name_ix = defaultdict(list)
for i, t in enumerate(pool_ntoks):
    for tok in t:
        if df_ct[tok] <= MAX_DF: name_ix[tok].append(i)
addr_ix = defaultdict(list)
for i, row in enumerate(pool):
    for tok in set(NUMTOK.findall(row[2]))    :
        if len(tok) >= 3: addr_ix[tok].append(i)
pool_ix = {row[0]: i for i, row in enumerate(pool)}
print(f"[4] index: {len(name_ix):,} name keys, {len(addr_ix):,} address keys", flush=True)

# ---------- 4. retrieve candidates, emit features ----------
COLS = ["s1_id","cand_id","source","country","is_match",
        "name_tok_jaccard","name_tok_containment","name_3gram_jaccard","name_seqratio",
        "name_len_diff","name_tok_count_s1","name_tok_count_c","name_prefix4_match",
        "addr_tok_jaccard","addr_3gram_jaccard","addr_num_overlap","addr_empty",
        "rare_tokens_shared","same_script","country_match"]
n_pos = n_neg = 0; retrieved_true = 0; total_true = 0
with open(OUT, "w", newline='', encoding='utf-8') as fo:
    w = csv.writer(fo); w.writerow(COLS)
    for k, (sid, tru) in enumerate(truth.items()):
        if sid not in s1: continue
        r1 = s1[sid]
        t1, g1 = set(ntoks(r1[1])), grams(r1[1])
        a1t, a1g = set(norm(r1[2]).split()), grams(r1[2])
        a1n = set(NUMTOK.findall(r1[2])); n1 = norm(r1[1])
        deva1 = bool(DEVA.search(r1[1]))
        total_true += len(tru)
        cand = Counter()
        for tok in t1:
            if df_ct.get(tok, 0) <= MAX_DF:
                for i in name_ix.get(tok, ()): cand[i] += 2
        for tok in a1n:
            if len(tok) >= 3:
                for i in addr_ix.get(tok, ()): cand[i] += 1
        top = [i for i, _ in cand.most_common(MAX_CAND)]
        got = set(i for i in top if pool[i][0] in tru)
        # unbiased positives: every true partner, retrieved or not
        forced = [pool_ix[x] for x in tru if x in pool_ix]
        seen = set()
        for i in forced + [i for i in top if pool[i][0] not in tru]:
            if i in seen: continue
            seen.add(i)
            r2 = pool[i]
            if r2[3] != r1[3]: continue          # country partition
            lbl = 1 if r2[0] in tru else 0
            t2, g2 = pool_ntoks[i], grams(r2[1])
            a2t, a2g = set(norm(r2[2]).split()), grams(r2[2])
            a2n = set(NUMTOK.findall(r2[2])); n2 = norm(r2[1])
            w.writerow([sid, r2[0], r2[0][:2], r1[3], lbl,
                round(jac(t1,t2),5), round(cont(t1,t2),5), round(jac(g1,g2),5),
                round(SequenceMatcher(None, n1, n2).ratio(),5),
                abs(len(n1)-len(n2)), len(t1), len(t2),
                int(n1[:4]==n2[:4] and len(n1)>=4),
                round(jac(a1t,a2t),5), round(jac(a1g,a2g),5),
                round(jac(a1n,a2n),5), int(not r2[2].strip()),
                sum(1 for x in (t1&t2) if df_ct.get(x,0)<=MAX_DF),
                int(deva1 == bool(DEVA.search(r2[1]))), int(r1[3]==r2[3])])
            n_pos += lbl; n_neg += (1-lbl)
        retrieved_true += len(set(pool[i][0] for i in got))
        if (k+1) % 5000 == 0: print(f"    ...{k+1:,} entities", flush=True)

print(f"\n[5] pairs written: {n_pos+n_neg:,}  (pos={n_pos:,}  neg={n_neg:,})")
print(f"[6] BLOCKING RECALL @K={MAX_CAND}: {retrieved_true:,}/{total_true:,} "
      f"= {100*retrieved_true/max(total_true,1):.2f}%")
print(f"    (pool is {POOL_FRAC:.0%} of S2/S3 + all true partners)")
print(f"[7] -> {OUT}")
