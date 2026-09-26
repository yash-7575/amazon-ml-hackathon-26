"""Phase A: streaming profile of each source file. Stdlib only, O(1) memory per file
except for the entity_id / name-collision counters."""
import sys, csv, unicodedata, re
from collections import Counter

csv.field_size_limit(10**9)
DEVA = re.compile(r'[ऀ-ॿ]')
ARAB = re.compile(r'[؀-ۿ]')
CJK  = re.compile(r'[一-鿿]')
NONASCII = re.compile(r'[^\x00-\x7F]')

def script_of(s):
    if DEVA.search(s): return 'devanagari'
    if ARAB.search(s): return 'arabic'
    if CJK.search(s):  return 'cjk'
    if NONASCII.search(s): return 'latin-extended'
    return 'ascii'

def norm_name(s):
    s = unicodedata.normalize('NFKC', s).lower()
    s = s.replace('&', ' and ')
    s = re.sub(r'[^\w\s]', ' ', s)
    return ' '.join(s.split())

def profile(path):
    n=0; empty=Counter(); country=Counter(); script=Counter()
    namelen=[]; addrlen=[]; ntok=[]
    ids=set(); dup_ids=0
    normname=Counter()
    with open(path, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        hdr = next(r)
        for row in r:
            if len(row) != len(hdr):
                empty['MALFORMED_ROW'] += 1
                continue
            n += 1
            eid, name, addr, cty = row[0], row[1], row[2], row[3]
            if not eid.strip():   empty['entity_id'] += 1
            if not name.strip():  empty['business_name'] += 1
            if not addr.strip():  empty['business_address'] += 1
            if not cty.strip():   empty['country'] += 1
            country[cty] += 1
            if eid in ids: dup_ids += 1
            else: ids.add(eid)
            script[script_of(name)] += 1
            nn = norm_name(name)
            normname[nn] += 1
            if n <= 400000:                      # sample for length stats
                namelen.append(len(name)); addrlen.append(len(addr))
                ntok.append(len(nn.split()))
    return dict(path=path, rows=n, empty=empty, country=country, script=script,
                dup_ids=dup_ids, namelen=namelen, addrlen=addrlen, ntok=ntok,
                normname=normname, n_unique_ids=len(ids))

def pct(a,b): return f"{100*a/b:.2f}%" if b else "n/a"
def stats(v):
    if not v: return "n/a"
    v=sorted(v); m=len(v)
    return f"min={v[0]} p50={v[m//2]} p90={v[int(m*.9)]} p99={v[int(m*.99)]} max={v[-1]}"

for path in sys.argv[1:]:
    p = profile(path)
    n = p['rows']
    print(f"\n{'='*72}\n{path}\n{'='*72}")
    print(f"rows: {n:,}   unique entity_id: {p['n_unique_ids']:,}   duplicate ids: {p['dup_ids']:,}")
    print("\n-- empty/missing --")
    for k in ['entity_id','business_name','business_address','country','MALFORMED_ROW']:
        print(f"   {k:20} {p['empty'][k]:>10,}  {pct(p['empty'][k], n)}")
    print("\n-- country --")
    for k,v in p['country'].most_common(): print(f"   {k:20} {v:>10,}  {pct(v,n)}")
    print("\n-- name script --")
    for k,v in p['script'].most_common(): print(f"   {k:20} {v:>10,}  {pct(v,n)}")
    print("\n-- lengths (first 400k rows) --")
    print(f"   name chars  : {stats(p['namelen'])}")
    print(f"   addr chars  : {stats(p['addrlen'])}")
    print(f"   name tokens : {stats(p['ntok'])}")
    nm = p['normname']
    collide = sum(c for c in nm.values() if c > 1)
    print("\n-- normalized-name ambiguity --")
    print(f"   distinct normalized names : {len(nm):,}")
    print(f"   rows sharing a name with >=1 other row : {collide:,} ({pct(collide,n)})")
    print(f"   top collisions: {[f'{k!r}x{v}' for k,v in nm.most_common(5)]}")
