#!/usr/bin/env bash
set -u
cd /home/yash/dev/clg/amlc-2026
source .venv/bin/activate
export PYTHONPATH=.
say(){ echo "[$(date +%H:%M:%S)] $*"; }

until grep -qE "DONE scored" matcher/results/predict_india.log 2>/dev/null; do
  pgrep -f "matcher.predict" >/dev/null || { say "predict died"; exit 1; }
  sleep 15
done
say "india predicted: $(grep 'DONE scored' matcher/results/predict_india.log)"

say "building wide candidates for india ..."
python - blocking/results/test_india_kb.long.tsv blocking/results/wide_cand_india.tsv <<'PY'
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
rm -f blocking/results/test_india_kb.long.tsv
say "freed india long file; disk $(df -h / | tail -1 | awk '{print $4}')"

say "building submission from all three countries ..."
python -u -m matcher.submit --preds blocking/results/preds_france.tsv \
    blocking/results/preds_us.tsv blocking/results/preds_india.tsv \
    --cands blocking/results/wide_cand_france.tsv blocking/results/wide_cand_us.tsv \
    blocking/results/wide_cand_india.tsv --wide-cands \
    --out-dir output --max-k 8 2>&1 | tail -8

say "validating ..."
python3 /home/yash/Downloads/Amazon-ML-dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir /home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/test 2>&1 | tail -6
say "entities with matches: $(awk -F'\t' 'NR>1 && $2!=""{n++} END{print n}' output/matching_results.tsv)"
say "=== FINAL SUBMISSION READY ==="
