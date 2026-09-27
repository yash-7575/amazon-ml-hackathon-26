#!/usr/bin/env bash
set -u
cd /home/yash/dev/clg/amlc-2026
source .venv/bin/activate
export PYTHONPATH=.
say(){ echo "[$(date +%H:%M:%S)] $*"; }

until grep -qE "DONE|Traceback" matcher/results/gen_india_rest.log 2>/dev/null; do
  pgrep -f "country India" >/dev/null || { say "GEN DIED"; tail -5 matcher/results/gen_india_rest.log; exit 1; }
  sleep 30
done
grep -q DONE matcher/results/gen_india_rest.log || { say "gen failed"; exit 1; }
say "$(tail -2 matcher/results/gen_india_rest.log | head -1)"

say "predicting India remainder ..."
python -u -m matcher.predict --pairs blocking/results/test_india_rest.long.tsv --split test \
    --model matcher/models/india_kb150.lgb --idf matcher/data/india_kb150_idf.pkl \
    --cols matcher/data/india_kb150_cols.json \
    --out blocking/results/preds_india_rest.tsv --workers 8 --min-prob 0.02 \
    --chunk-pairs 25000 > matcher/results/predict_india_rest.log 2>&1 \
  || { say "PREDICT FAILED"; tail -5 matcher/results/predict_india_rest.log; exit 1; }
say "$(tail -1 matcher/results/predict_india_rest.log)"

say "wide candidates for the remainder ..."
python - blocking/results/test_india_rest.long.tsv blocking/results/wide_cand_india_rest.tsv <<'PY'
import sys, csv
src, out = sys.argv[1], sys.argv[2]
g = {}
with open(src, encoding='utf-8') as f:
    next(f)
    for line in f:
        i = line.find('\t'); j = line.find('\t', i+1)
        g.setdefault(line[:i], []).append(line[i+1:j])
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE)
    for s, cs in g.items():
        w.writerow([s, ','.join(dict.fromkeys(cs))])
print(f'{len(g):,} S1 -> {out}')
PY
rm -f blocking/results/test_india_rest.long.tsv

say "rebuilding submission with COMPLETE India ..."
cp output/matching_results.tsv output/matching_results_65pct.tsv.bak
python -u -m matcher.submit \
    --preds blocking/results/preds_france.tsv blocking/results/preds_us.tsv \
            blocking/results/preds_india.tsv blocking/results/preds_india_rest.tsv \
    --cands blocking/results/wide_cand_france.tsv blocking/results/wide_cand_us.tsv \
            blocking/results/wide_cand_india.tsv blocking/results/wide_cand_india_rest.tsv \
    --wide-cands --out-dir output --max-k 8 2>&1 | tail -6

say "integrity check ..."
awk -F'\t' 'NR==1{if($1=="source1_entity_id"&&$2=="matched_entity_ids")h=1}
  NR>1{n++; if($2!=""){m++; k=split($2,a,","); for(i=1;i<=k;i++) if(a[i]!~/^S[23]-/) bad++}}
  END{printf "header_ok=%d rows=%d (need 1732544) with_matches=%d bad_ids=%d\n",h,n,m,bad+0}' \
  output/matching_results.tsv
say "=== COMPLETE SUBMISSION READY ==="
