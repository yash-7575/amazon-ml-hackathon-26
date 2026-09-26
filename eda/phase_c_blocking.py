"""Phase C: how big do blocks get? Token document-frequency in S2, per country."""
import csv, re, unicodedata
from collections import Counter
csv.field_size_limit(10**9)
D="/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/"
SUFFIX={'inc','llc','ltd','limited','corp','corporation','co','company','pvt','private',
        'llp','plc','sarl','sas','sa','eurl','gmbh','and','the','of'}
def toks(s):
    s=unicodedata.normalize('NFKC',s).lower().replace('&',' and ')
    return [t for t in re.sub(r'[^\w\s]',' ',s).split() if t not in SUFFIX]
df=Counter(); n=0
with open(D+"train/train_source2.tsv",encoding='utf-8') as f:
    r=csv.reader(f,delimiter='\t'); next(r)
    for row in r:
        n+=1
        for t in set(toks(row[1])): df[t]+=1
print(f"S2 rows: {n:,}   distinct name tokens: {len(df):,}")
print("\n-- most common tokens (these create giant blocks) --")
for t,c in df.most_common(15): print(f"   {t:18} {c:>9,}  {100*c/n:5.2f}% of all S2 rows")
buckets=[(1,1),(2,10),(11,100),(101,1000),(1001,10000),(10001,100000),(100001,10**9)]
print("\n-- token rarity distribution --")
for lo,hi in buckets:
    k=sum(1 for c in df.values() if lo<=c<=hi)
    print(f"   df {lo:>7,}-{hi if hi<10**8 else '∞':>9}: {k:>9,} tokens")
import statistics
print("\n-- implication: block size if you key on a token --")
for pct,lbl in [(0.50,'median token'),(0.90,'p90 token'),(0.99,'p99 token')]:
    vals=sorted(df.values()); v=vals[int(len(vals)*pct)]
    print(f"   {lbl:14}: df={v:,} -> a block of ~{v:,} candidates")
