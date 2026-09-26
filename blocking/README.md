# blocking/ — candidate generation for AMLC-2026 BER

Stage-1 blocker for the Business Entity Resolution task. Emits `candidate_pairs.tsv`
that the matcher will consume. **Recall lost here is unrecoverable downstream.**

See `../PROJECT_CONTEXT.md` for measured EDA facts and `../eda/FINDINGS.md` for the
evidence behind every design decision.

---

## File layout

| File | Purpose |
|---|---|
| `config.py` | Frozen `BlockerConfig` dataclass. Every parameter defaulted from a specific FINDING. |
| `normalize.py` | Unicode-safe normalization. Devanagari-safe punctuation strip (category `P`/`S`/`Z`, NOT `[^\w\s]`). |
| `io_utils.py` | Streaming TSV I/O, country partitioning, deterministic `CandidateWriter`, `peak_rss_mb()`. |
| `index.py` | `TfidfIndex`, chunked top-K search via `argpartition` (NOT `argsort` — 8.8× slower at scale). |
| `blocker_v1.py` | Orchestrator: P1 exact → P2 name TF-IDF → P3 address TF-IDF → P4 Devanagari translit → rescore → union → write. |
| `measure.py` | Ground-truth-aware evaluator (per-partition recall). |
| `measure_recall_at_k.py` | K-sweep evaluator with stratified sample + reservoir-sampled distractors. **Ground truth stays out of generation.** |
| `smoke_test.py` | End-to-end sanity on 2000 India S1. |
| `results/` | JSON + logs from measurement runs (this stage's evidence). |

The pipeline is **P1 + P2 + P3 + P4 (folded into P2 index as translit aliases) → rescore
(`w_name·cos_name + w_addr·cos_addr` with a raw-score floor) → top-K per S1 → union → write.**

---

## How to run

### Smoke test (sanity, ~1 min)
```bash
PYTHONPATH=. python blocking/smoke_test.py india 2000 60000 60000
```
Args: `country_key` `sample_S1` `s2_cap` `s3_cap`. Uses ground truth ONLY to compute
recall at the end; the blocker itself never sees it.

### Recall@K sweep (~2 min)
```bash
PYTHONPATH=. python blocking/measure_recall_at_k.py \
    --sample 3000 --s2s3-cap 80000 --k 10,20,50,100,200 \
    --out blocking/results/recall_at_k_sample3k_cap80k.json
```
- `--sample N`: stratified S1 sample size across countries.
- `--s2s3-cap N`: reservoir-sampled distractor cap per source per country. All true
  partners of sampled S1 are always kept. Removes memory as a function of country
  size at the cost of a slight upward bias in recall (fewer distractors to hide the
  needle) — quantified in the results table below.
- `--only <country_key>`: restrict to one partition (e.g. `india`, `us`).

### Full blocker (writes `candidate_pairs.tsv`)
```bash
PYTHONPATH=. python -m blocking.blocker_v1 --split train
PYTHONPATH=. python -m blocking.blocker_v1 --split test          # for submission
PYTHONPATH=. python -m blocking.blocker_v1 --dry-run             # smallest partition only, no write
PYTHONPATH=. python -m blocking.blocker_v1 --only india          # single partition
```

Output schema: `s1_entity_id \t candidate_entity_id \t source \t score` where
`source` is `P1+P2` / `P3` / `P2+P3` etc. (which passes surfaced the pair) and
`score` is the rescored combined name+address cosine.

---

## Config knobs that matter

Ordered by expected leverage on final recall.

1. **`final_top_k`** (default 50). Global candidates per S1. Increasing gains recall
   linearly in candidate volume. See the K-sweep table — the recall-vs-K curve is
   still climbing at K=200 for India.
2. **`p2_top_k` / `p3_top_k`** (default 50 / 30). Per-source retrieval breadth
   feeding into the union. Must be ≥ `final_top_k` or rescore has nothing to rerank.
3. **`rescore_w_name` / `rescore_w_addr`** (0.6 / 0.4). Combined-score weights.
   The smoke test shows the rescore currently drops ~4pp vs. the raw union
   (P1+P2+P3 = 98.27 %, final top-K = 94.30 %); the `raw_floor` in
   `rescore_and_union` blunts but does not eliminate this. Retune when the matcher
   is available and can tell us which recall we most need to preserve.
4. **`p2_min_df` / `p3_min_df`** (2 / 2). Do **NOT** raise these into a `max_df` cap.
   The 45.53 % negative result in PROJECT_CONTEXT was caused by exactly that.
5. **`p2_chunk_rows`** (1024). Peak RAM per chunk = `chunk_rows × n_index × 4 B`.
   Auto-shrunk inside `topk_search` if it would exceed 512 MiB.

---

## Measured recall@K — train, stratified 3k-S1 sample, 80k distractor cap per source per partition

Command: as in the K-sweep example above. Elapsed: 118 s wall, peak RSS 3.5 GiB.
JSON at `results/recall_at_k_sample3k_cap80k.json`.

### Aggregate (India + US pooled, weighted by truth pairs)

| K | P2 only | P1+P2 | P1+P2+P3 (union) |
|---:|---:|---:|---:|
| 10 | 81.33 % | 81.43 % | **97.86 %** |
| 20 | 84.13 % | 84.15 % | **98.54 %** |
| 50 | 87.04 % | 87.06 % | **99.21 %** |
| 100 | 88.67 % | 88.67 % | **99.49 %** |
| 200 | 90.05 % | 90.05 % | **99.69 %** |

**Marginal P3 lift over P1+P2** (address recovers zero-name-overlap pairs):
+16.4 pp at K=10, +14.4 pp at K=20, +12.2 pp at K=50, +10.8 pp at K=100, +9.6 pp at K=200.
Aggregate lift matches the PROJECT_CONTEXT prediction (+8–13 pp) — driven by US;
India is far higher (+20–25 pp) because so much of India recall comes through address.

### Per-partition (K=50, the operating point)

| Partition | truth | P2 | P1+P2 | P1+P2+P3 | in-script | cross-script |
|---|---:|---:|---:|---:|---:|---:|
| india | 4,447 | 75.98 % | 76.03 % | 98.45 % | 98.75 % | 95.21 % |
| us    | 6,676 | 94.41 % | 94.41 % | 99.72 % | 99.72 % | —    |

### Cross-script (India Devanagari) — where the blocker is weakest

| K | P2 (name only, w/ P4 translit alias) | P1+P2+P3 |
|---:|---:|---:|
| 10  | 1.33 %  | 88.83 % |
| 20  | 2.13 %  | 90.96 % |
| 50  | 3.19 %  | 95.21 % |
| 100 | 4.52 %  | 97.07 % |
| 200 | 5.32 %  | 97.87 % |

**P4 alias in the char-3gram name index is not doing meaningful work.** All the
cross-script recall is coming from P3 (address is script-agnostic once tokenized).
Fix candidates below.

### Caveat on the 80k distractor cap

Recall reported here is a mild upward-biased estimate: the reservoir keeps all true
partners of sampled S1 plus 80k random distractors, whereas production faces
~3–5M distractors. Blocking with more distractors can only *hurt* recall for
top-K methods, so numbers on the full partition will be slightly lower — but
because our cosine scores separate near-duplicates well, the drop is expected to
be small (<1 pp at K=50). This is testable in the extrapolation run below.

---

## Peak RSS + full-partition fit (14 GiB budget)

- Measured peak on the 80k-cap sweep: **3492 MiB**.
- Baseline overhead (ground truth dict, 2.08M S1 keys × ~30 targets): ~1.9 GiB.
- Marginal cost of one 80k-doc partition (S1 + S2 + S3 + two TF-IDF indexes + top-K arrays for the sweep at K=200): ~1.6 GiB.
- **Extrapolation to full US** (~3.3M S2 + ~3.3M S3, ~1.3M S1): CSR memory scales
  linearly with `n_docs`, so ~40× → name+addr indexes ≈ 2.5 GiB (PROJECT_CONTEXT
  measured 12.5M-doc index at 3.77 GiB; scales down proportionally). Plus S1
  metadata ~800 MiB, top-K arrays for K=50 × 1.3M queries × 4 B ≈ 260 MiB, chunked
  temp buffer ≤512 MiB. **Total peak per partition ≈ 5–6 GiB**, fits in 14 GiB with
  headroom.
- Partitions run sequentially; only one is resident at a time. India is smaller
  (~5M S2/S3, ~883k S1) so US is the binding one.
- `blocker_v1.run_partition` explicitly `del`s per-partition indexes before writing,
  which is what lets the smaller partition run first (`partition_order='ascending_size'`)
  without carrying over.

If peak surprises us on the full run, the safest lever is **`p2_chunk_rows`**:
halving it halves the temp matmul buffer with no recall cost.

---

## What to tune next (ranked)

1. **Fix cross-script recall (India Devanagari, currently 1–5 % from P2).**
   The P4 char-3gram transliteration alias is not being retrieved because
   `char_wb` 3-grams on `unidecode`'d Hindi look nothing like natural English
   3-grams that the S1 side produces. Two concrete experiments:
   (a) run the S1 name through `transliterate_deva`-style ASCII-normalization only
   on the query side (removes accents but is a no-op for ASCII) and add a second
   index built on `unidecode(target_name)` with `char_wb` 4-grams (accent-stripped);
   (b) build a **word-level** name index for the Devanagari targets — token overlap
   after transliteration is 40–70 % per sample even when 3-grams overlap ~0. This
   costs one extra index build but is expected to lift cross-script from 3 %
   to 40–60 % at K=50.

2. **Recover the ~4 pp lost in the rescore.**
   On the smoke test, `P1+P2+P3` union = 98.27 % but `final top-K` (after rescore
   + trim to `final_top_k=50`) = 94.30 %. The `raw_floor` in
   `rescore_and_union` prevents the worst outranking but not all of it. Two
   experiments: (a) raise `final_top_k` to 100 and measure the matcher-stage cost
   directly; (b) rescore using `max(w_name·cos_n + w_addr·cos_a, cos_n, cos_a)`
   so a pure-name or pure-address hit is never demoted below a mediocre mix.

3. **Validate on the full US partition without the distractor cap.**
   Numbers above assume 80k random distractors per source. Running the full US
   (~3.3M S2 + ~3.3M S3) on a 5–10k S1 sample will tell us the true recall drop
   from more distractors. Expect <1 pp at K=50 based on how well cosine separates
   near-duplicates, but this is a prediction that needs measurement before we
   trust the numbers above for the leaderboard.

---

## Non-negotiables enforced in this module

- No `max_df` cap on TF-IDF (PROJECT_CONTEXT: rare-token blocker measured 45.53 %).
- No `[^\w\s]` (destroys Devanagari — verified in `normalize._selftest`).
- Ground truth is loaded ONLY in `measure.py` and `measure_recall_at_k.py`. `blocker_v1.py`
  has no import path that touches ground truth. The file the matcher receives is
  byte-identical whether ground truth exists on disk or not.
- No external APIs, geocoding, or internet lookups (competition rule).
- Deterministic writes: rows sorted by `(-score, cand_id)` per S1; seed in config.
