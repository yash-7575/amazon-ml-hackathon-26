"""Phase B: sampled analysis of TRUE matching pairs. Answers: how much signal is in
name vs address, does country ever cross, and what blocking recall is achievable."""
import csv, random, re, unicodedata, sys
from collections import Counter, defaultdict

csv.field_size_limit(10**9)
D = "/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/"
SAMPLE = 40000
random.seed(7)
DEVA = re.compile(r'[ऀ-ॿ]')
SUFFIX = {'inc','llc','ltd','limited','corp','corporation','co','company','pvt','private',
          'llp','plc','sarl','sas','sa','eurl','sarlu','gmbh','and','the','of'}

def norm(s):
    s = unicodedata.normalize('NFKC', s).lower().replace('&',' and ')
    return ' '.join(re.sub(r'[^\w\s]', ' ', s).split())
def toks(s, drop=True):
    t = norm(s).split()
    return set(x for x in t if not drop or x not in SUFFIX) or set(t)
def grams(s, n=3):
    s = norm(s).replace(' ','')
    return set(s[i:i+n] for i in range(max(0,len(s)-n+1)))
def jac(a,b):
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

# --- sample S1 entities + their truth ---
gt = {}
with open(D+"train/train_ground_truth.tsv", encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t'); next(r)
    rows = [row for row in r]
sample = random.sample(rows, SAMPLE)
need = set()
for sid, ids in sample:
    lst = [x for x in ids.split(',') if x]
    gt[sid] = lst
    need.update(lst)
print(f"sampled S1 entities: {len(gt):,}   matched ids needed: {len(need):,}", flush=True)

# --- pull the S1 records and the needed S2/S3 records ---
s1 = {}
with open(D+"train/train_source1.tsv", encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t'); next(r)
    for row in r:
        if row[0] in gt: s1[row[0]] = row
other = {}
for src in ('2','3'):
    with open(D+f"train/train_source{src}.tsv", encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t'); next(r)
        for row in r:
            if row[0] in need: other[row[0]] = row
print(f"resolved S1: {len(s1):,}   resolved S2/S3: {len(other):,}", flush=True)

# --- analyse true pairs ---
nj, gj, aj = [], [], []
cross_country = 0; cross_script = 0; pairs = 0
exact_name = 0; share_tok = 0; share_gram = 0; no_name_signal = 0
addr_saves = 0; empty_addr = 0
src_mix = Counter(); per_entity_src = Counter()
for sid, lst in gt.items():
    if sid not in s1: continue
    r1 = s1[sid]; n1t, n1g = toks(r1[1]), grams(r1[1]); a1 = toks(r1[2], drop=False)
    kinds = set()
    for oid in lst:
        r2 = other.get(oid)
        if not r2: continue
        pairs += 1; kinds.add(oid[:2]); src_mix[oid[:2]] += 1
        if r1[3] != r2[3]: cross_country += 1
        if DEVA.search(r2[1]) and not DEVA.search(r1[1]): cross_script += 1
        n2t, n2g = toks(r2[1]), grams(r2[1]); a2 = toks(r2[2], drop=False)
        j = jac(n1t, n2t); nj.append(j); gj.append(jac(n1g, n2g))
        if norm(r1[1]) == norm(r2[1]): exact_name += 1
        if n1t & n2t: share_tok += 1
        if len(n1g & n2g) >= 2: share_gram += 1
        if not r2[2].strip(): empty_addr += 1
        else:
            av = jac(a1, a2); aj.append(av)
            if j == 0.0:
                no_name_signal += 1
                if av > 0.2: addr_saves += 1
    per_entity_src[''.join(sorted(kinds)) or 'none'] += 1

def dist(v, label):
    if not v: print(f"  {label}: n/a"); return
    v = sorted(v); m = len(v)
    q = lambda p: v[min(m-1, int(m*p))]
    print(f"  {label:28} p10={q(.10):.3f} p25={q(.25):.3f} p50={q(.50):.3f} "
          f"p75={q(.75):.3f} p90={q(.90):.3f}  mean={sum(v)/m:.3f}")

P = pairs or 1
print(f"\n=== TRUE PAIRS ANALYSED: {pairs:,} ===")
print("\n-- similarity distributions (true pairs) --")
dist(nj, "name token Jaccard"); dist(gj, "name char-3gram Jaccard"); dist(aj, "address token Jaccard")
print("\n-- blocking signal --")
print(f"  exact normalized name match : {exact_name:>9,}  {100*exact_name/P:.2f}%")
print(f"  share >=1 name token        : {share_tok:>9,}  {100*share_tok/P:.2f}%")
print(f"  share >=2 name char-3grams  : {share_gram:>9,}  {100*share_gram/P:.2f}%")
print(f"  ZERO name-token overlap     : {no_name_signal:>9,}  {100*no_name_signal/P:.2f}%")
print(f"     ...of those, address Jaccard>0.2 : {addr_saves:,}")
print("\n-- structural --")
print(f"  cross-COUNTRY true pairs    : {cross_country:>9,}  {100*cross_country/P:.2f}%")
print(f"  cross-SCRIPT (S1 latin vs Deva): {cross_script:>9,}  {100*cross_script/P:.2f}%")
print(f"  matched partner has EMPTY address: {empty_addr:,}  {100*empty_addr/P:.2f}%")
print(f"  match source mix: {dict(src_mix)}")
print(f"\n-- per-entity source pattern --")
for k,v in per_entity_src.most_common():
    print(f"  {k or 'none':8} {v:>8,}  {100*v/len(gt):.2f}%")
