# Business Entity Resolution — Amazon ML Challenge 2026

Matches each Source-1 business record to all Source-2/3 records describing the same real
business. CPU only; no external data, APIs, or pretrained models.

## Pipeline

```
sources ──▶ 1. key blocking ──▶ 2. TF-IDF rescore ──▶ 3. LightGBM ──▶ 4. expected-F0.5 ──▶ output/
            inverted index       top-K = 150           77 features     + conflict
            (9 key families)     per entity                            resolution
```

Stage 1 proposes ~560 candidates per entity in **0.047 ms** (vs 0.28 s for TF-IDF cosine over
the full index). Stage 2 ranks them by building vectors for *only those candidates*. Stage 3
scores each pair. Stage 4 chooses how many to keep per entity.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set `DATA_ROOT` in `matcher/records.py`, `matcher/label.py` and `blocking/generate_keyblock.py`
to the directory containing `train/` and `test/`, or pass `--data-root`.

## Reproduce end-to-end

`./run_submission.sh` runs everything. It processes countries **strictly sequentially** and
deletes each country's candidate file after prediction — the full test set at K=150 exceeds
available disk, and running stages concurrently exhausts RAM.

Or step by step:

```bash
export PYTHONPATH=.

# 1. Candidates for one split + country (repeat for India / US / France)
python blocking/generate_keyblock.py --split test --country India \
    --top-k 150 --cap 400 --n-rt 10 --n-ra 6 \
    --out blocking/results/test_india_kb.long.tsv

# 2. Score them (chunked and parallel; nothing is materialised in full)
python -m matcher.predict --pairs blocking/results/test_india_kb.long.tsv --split test \
    --model matcher/models/india_kb150.lgb \
    --idf matcher/data/india_kb150_idf.pkl \
    --cols matcher/data/india_kb150_cols.json \
    --out blocking/results/preds_india.tsv --workers 8 --chunk-pairs 25000

# 3. Build the two submission files
python -m matcher.submit --preds blocking/results/preds_*.tsv \
    --cands blocking/results/wide_cand_*.tsv --wide-cands --out-dir output --max-k 8

# 4. Validate before uploading
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Retraining

```bash
python blocking/generate_keyblock.py --split train --country India --sample 20000 --top-k 150 \
    --out blocking/results/train_india_kb150.long.tsv
python -m matcher.build_dataset --pairs blocking/results/train_india_kb150.long.tsv --tag india_kb150
python -m matcher.run_experiment --tag india_kb150     # train + tune + report
```

The S1 sample is drawn **uniformly**, not filtered to entities that have matches — filtering
removes every singleton, and singletons are 5.58% of the evaluation set with each correct empty
prediction worth a full 1.0.

## Self-tests

```bash
python -m matcher.evaluate          # macro-F0.5, incl. 4 hand-checked cases
python -m matcher.assign            # conflict resolution + expected-F0.5
python -m matcher.group_features    # group-relative features
python -m matcher.features_extra    # domain stems, transliteration
python matcher/leakage_check.py     # the extractor cannot import labels
python blocking/measure_keyblock.py --cap 400 --n-rt 10 --n-ra 6   # pair completeness
```

## Operational notes

- **Memory.** `--chunk-pairs` is the lever: the binding cost is `prepare()` over a chunk's
  distinct candidate records (~18 structures each, two of them n-gram sets). 25,000 keeps a
  worker near 400 MB; 150,000 costs ~2 GB per worker.
- **Determinism.** Seed 20260926 throughout; sorts are stable and ties break on entity id.
- **Reported score.** Macro-F₀.₅ **0.89696** on held-out India entities against the full
  4.13M-record pool. Blocking pair completeness is **86.05%**, which caps it.
