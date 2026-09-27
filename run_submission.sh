#!/usr/bin/env bash
# STRICTLY SEQUENTIAL. The previous run froze the machine by holding India's 4.7M-record
# index (~7 GB) while France prediction forked 12 workers. Nothing here overlaps: one
# process at a time, and each country's long candidate file is deleted the moment its
# predictions are written, because 9 GB of disk cannot hold all three at K=150.
set -u
cd /home/yash/dev/clg/amlc-2026
source .venv/bin/activate
export PYTHONPATH=.
R=matcher/results
M="--model matcher/models/india_kb150.lgb --idf matcher/data/india_kb150_idf.pkl --cols matcher/data/india_kb150_cols.json"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
free_gb(){ df -h / | tail -1 | awk '{print $4}'; }

predict_and_free(){  # $1=country_lower  $2=long file
  local c=$1 long=$2
  [ -s "$long" ] || { say "SKIP $c: $long missing"; return 1; }
  say "predicting $c (8 workers) ..."
  python -u -m matcher.predict --pairs "$long" --split test $M \
      --out blocking/results/preds_${c}.tsv --workers 6 --min-prob 0.02 --chunk-pairs 25000 \
      > $R/predict_${c}.log 2>&1 || { say "PREDICT $c FAILED"; tail -6 $R/predict_${c}.log; return 1; }
  say "  $(tail -1 $R/predict_${c}.log)"
  python - "$long" "blocking/results/wide_cand_${c}.tsv" <<'PY'
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
print(f'wide candidates: {len(g):,} S1 -> {out}')
PY
  rm -f "$long"
  say "  freed $long ; disk now $(free_gb)"
}

generate(){  # $1=Country  $2=country_lower
  say "generating $1 (disk $(free_gb)) ..."
  python -u blocking/generate_keyblock.py --split test --country "$1" --top-k 150 \
      --cap 400 --n-rt 10 --n-ra 6 --out blocking/results/test_${2}_kb.long.tsv \
      > $R/gen_test_${2}.log 2>&1
  grep -q DONE $R/gen_test_${2}.log || { say "GEN $1 FAILED"; tail -6 $R/gen_test_${2}.log; return 1; }
  say "  $(tail -2 $R/gen_test_${2}.log | head -1)"
}

# France is already generated -- predict it first to reclaim 1.4 GB before India needs it.
predict_and_free france blocking/results/test_france_kb.long.tsv
generate India india   && predict_and_free india blocking/results/test_india_kb.long.tsv
# US may already have been generated in parallel by run_us_gen.sh; wait for it rather
# than regenerating. Two processes writing one file is what corrupted an earlier run.
if [ -s blocking/results/test_us_kb.long.tsv ] || pgrep -f "country US" >/dev/null; then
  say "waiting for parallel US generation ..."
  while pgrep -f "country US" >/dev/null; do sleep 20; done
  grep -q DONE $R/gen_test_us.log && say "  US gen complete (parallel)"
else
  generate US us
fi
predict_and_free us blocking/results/test_us_kb.long.tsv

say "=== building submission ==="
python -u -m matcher.submit --preds blocking/results/preds_*.tsv \
    --cands blocking/results/wide_cand_*.tsv --wide-cands \
    --out-dir output --max-k 8 > $R/submit.log 2>&1
tail -10 $R/submit.log

say "=== validating ==="
python3 /home/yash/Downloads/Amazon-ML-dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir /home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/test 2>&1 | tail -20
say "=== PIPELINE COMPLETE ==="
