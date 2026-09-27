#!/usr/bin/env bash
set -u
cd /home/yash/dev/clg/amlc-2026
source .venv/bin/activate
export PYTHONPATH=.
# Wait until India GENERATION is done (not its prediction) -- then generation capacity is
# free while prediction runs on the other cores.
until grep -q "DONE" matcher/results/gen_test_india.log 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] India gen done -> starting US generation in parallel with India predict"
python -u blocking/generate_keyblock.py --split test --country US --top-k 150 \
    --cap 400 --n-rt 10 --n-ra 6 --out blocking/results/test_us_kb.long.tsv \
    > matcher/results/gen_test_us.log 2>&1
echo "[$(date +%H:%M:%S)] US generation finished"
tail -2 matcher/results/gen_test_us.log
