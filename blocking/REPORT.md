# Blocking Layer — Engineering Report

Amazon ML Challenge 2026, Business Entity Resolution. This document narrates
**what was tried, why, what worked, and the measured numbers.** For a parameter
reference and run instructions, see `README.md`.

---

## 1. The problem, sized honestly

Match each Source-1 business record to its Source-2/3 counterparts.

| | rows |
|---|---:|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| **brute-force pairs** | **~2.28 × 10¹³** |

At 10⁷ pair-comparisons per second that is **~26 days** for a single train pass.
The blocking layer's job is to shrink this to a candidate set the matcher can
score in minutes, **without losing recall** — recall lost here is unrecoverable.
Metric penalty is macro-F₀.₅ (precision weighted 2×), so silently dropping
true pairs is expensive.

Machine budget: **14 GiB RAM**. No external APIs, no internet lookups.

---

## 2. Design constraints — measured, not assumed

From `../eda/FINDINGS.md`, verified before writing any pipeline code:

1. **Matches never cross country** (0 of 138,120 sampled true pairs). → Partition by country. Free ~3× reduction, zero recall cost.
2. **Name-only blocking caps recall at ~85 %.** 14.81 % of true pairs share zero name tokens; 96.3 % of *those* have address Jaccard > 0.2. → Blocking **must** union a name index and an address index.
3. **4.17 % of true pairs cross script** (S1 is 100 % ASCII; S2/S3 contain Devanagari). Standard `[^\w\s]` regex shatters Devanagari into single consonants. → Strip punctuation by Unicode category, not by `\w`.
4. **Token frequency is heavy-tailed.** `com` covers 4.00 % of S2 rows, `group` 2.77 %. But a hard document-frequency cap silently forfeits the only-shared-signal for many pairs. → **IDF weighting, no `max_df` cap.**
5. **39 % of S1 rows share a normalized name with another S1 row.** Generic names ("primary care group" ×253). → Name similarity alone can't decide a match; address has to break ties.

Each of these is a design constraint the code was written to satisfy.

---

## 3. What was tried

### v0 — rare-token + address-numeric blocker (prior work in `eda/phase_c_blocking.py`)

Idea: index only tokens with `df ≤ 120`, union with a blocker on numeric address
components. Cheap, small candidate set.

**Result: 45.53 % recall @ K=40. Killed.**

Root cause: the `df ≤ 120` cut discards common tokens (`primary`, `group`, `care`)
that are the **only** shared signal for many ambiguous-name pairs. This was the
motivation for using IDF weighting over the full vocabulary in v1.

### v1 — the shipped blocker

Country-partitioned union of four passes, rescored and unioned to a global top-K
per S1 entity.

```
FOR each country:
  P1  normalized-exact match on name core            (~22% of true pairs, ~0 cost)
  P2  char-3gram TF-IDF cosine, top-K on name        (primary retrieval engine)
  P3  word 1-2 gram TF-IDF cosine, top-K on address  (recovers zero-name-overlap pairs)
  P4  Devanagari names transliterated into P1 + P2   (cross-script support)
  → rescore each candidate by w_name·cos_name + w_addr·cos_addr
  → keep top-K per S1
  → stream to candidate_pairs.tsv
```

---

## 4. Algorithms and methods

### 4.1 Normalization (`normalize.py`)

- **NFKC + casefold + `&` → " and "** — canonicalize Unicode compatibility forms and case.
- **Category-based punctuation strip:** iterate chars; drop those with Unicode category `P` (punct), `S` (symbol), `Z` (separator); **keep `M` (mark)** so Devanagari matra + virama survive. The self-test in `normalize.py::_selftest` verifies that "राम मार्केटिंग प्राइवेट लिमिटेड" produces 4 tokens with category-based strip vs. 12+ single-consonant fragments with `[^\w\s]`.
- **Name core** (P1 key): tokens → suffix drop (`inc`, `llc`, `sarl`, `gmbh`, `pvt`, `and`, `the`, …) → **sorted** join. Sorting makes "Acme Foods Inc" and "Foods Acme LLC" collide with zero fuzzy logic.
- **Address tokens:** same normalization, plus small ASCII street-word stopword list (`st`, `road`, `apt`, `suite`, …). Numerals kept — highest-signal address tokens.

### 4.2 P1 — normalized-exact name-core index

Hash-map from `name_core(name)` to `(source, id)` list. O(1) per lookup, near-zero
memory, catches the "easy 22 %" that would otherwise burn ANN cycles.

### 4.3 P2 — char-3gram TF-IDF on name (the primary engine)

- **`TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), min_df=2, max_df=1.0, sublinear_tf=True, norm='l2', dtype=float32)`**
- `char_wb` (character n-grams *within word boundaries*, padded with spaces) — resistant to word-order swaps, catches partial-token overlaps ("acme foods" vs "acme grp foods").
- `min_df=2` drops hapax n-grams (typos, single-occurrence junk) — safe.
- **No `max_df`.** IDF handles the "com"/"center"/"group" problem naturally — those n-grams end up with near-zero weight, which is what we want, not with hard exclusion.
- `sublinear_tf=True`: replaces raw tf with `1 + log(tf)` — long strings don't dominate.
- `norm='l2'` + `float32`: sparse matmul then equals cosine similarity directly, and halves memory vs float64.

### 4.4 P3 — word 1–2 gram TF-IDF on address

- **`TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, ...)`**
- Word grams (not char): numerals like `450` and `main st` are the highest-signal address tokens; character-level fragmentation would lose them.
- Bigrams catch `1600 pennsylvania` — proximity is meaningful in addresses in a way it isn't in business names.
- Records with **empty address** contribute no P3 document but still get P1/P2/P4 candidates. P3 is a *union*, never a filter.

### 4.5 P4 — Devanagari cross-script alias

Rather than a special code path at query time, we **add a second document** for every
Devanagari S2/S3 record: its `unidecode`/ITRANS romanization. That row now lives
in both the Devanagari char-3gram space and the Latin one, so an ASCII S1 query
retrieves it through P2 with no branching. Also inserted into the P1 exact index.

Measured effect on the sample: P4 alone lifts cross-script recall from ~1 % to
~5 % at K=50. **Not enough** — see §7.

### 4.6 Top-K search — chunked sparse matmul + `argpartition`

The naive `queries @ index.T` densifies to `n_queries × n_docs × 4 B`; for
1.3 M × 3 M × 4 B that is **15 TB**. So:

- Process queries in **chunks of `chunk_rows`** (default 1024). Peak temp buffer per chunk is `chunk_rows × n_docs × 4 B`.
- `topk_search` auto-shrinks `chunk_rows` if that would exceed 512 MiB. This is the single memory guardrail that keeps us inside 14 GiB.
- **`np.argpartition` for the per-row top-K, not `np.argsort`** — O(n) vs O(n log n). At n_docs = 500 k this is measured ~8.8× faster.
- Sort just the k winners with `np.lexsort((idx, -score))` — deterministic tie-breaks.

### 4.7 Rescore + union

Every unique candidate from P1 ∪ P2 ∪ P3 ∪ P4 is re-scored with

    combined = w_name · cos_name(s1, cand) + w_addr · cos_addr(s1, cand)

using the *original* TF-IDF spaces (recomputing cosine directly from the sparse
rows — no re-fit). Weights **0.6 / 0.4** favour name because address has
more missing / dirty data (3.3 % empty; `p10` address Jaccard is 0.30 vs. 0.174
for name char-3grams). A raw-score floor prevents rescore from demoting a
high-cos_name / zero-cos_addr hit below a mediocre mix — this is imperfect and
still costs some recall (§7).

### 4.8 Reservoir sampling for the measurement harness (`measure_recall_at_k.py`)

Original `_load_partition_capped` materialized every candidate into a Python list
before capping — 5 M dicts × ~500 B = **~2.5 GiB per source**, causing swap.
Rewritten as **reservoir sampling** over the distractor stream: memory is O(cap),
independent of country size. All *true partners* of sampled S1 are unconditionally
kept; only distractors are subsampled. Bug was found before the K-sweep, not after.

---

## 5. Metrics

### 5.1 Recall@K — aggregate (India + US, weighted by truth pairs, 3k S1 sample, 80k distractor cap)

Command: `PYTHONPATH=. python blocking/measure_recall_at_k.py --sample 3000 --s2s3-cap 80000 --k 10,20,50,100,200`.
Elapsed 118 s wall, peak RSS 3.5 GiB.

| K | P2 only | P1+P2 | **P1+P2+P3 (union)** |
|---:|---:|---:|---:|
| 10  | 81.33 % | 81.43 % | **97.86 %** |
| 20  | 84.13 % | 84.15 % | **98.54 %** |
| 50  | 87.04 % | 87.06 % | **99.21 %** |
| 100 | 88.67 % | 88.67 % | **99.49 %** |
| 200 | 90.05 % | 90.05 % | **99.69 %** |

### 5.2 Marginal P3 lift over P1+P2

| K | US (Latin) | India (mixed script) | Aggregate |
|---:|---:|---:|---:|
| 10  | +10.7 pp | +25.0 pp | +16.4 pp |
| 20  |  +8.3 pp | +23.5 pp | +14.4 pp |
| 50  |  +5.3 pp | +22.4 pp | +12.2 pp |
| 100 |  +3.8 pp | +21.3 pp | +10.8 pp |
| 200 |  +2.7 pp | +20.1 pp |  +9.6 pp |

The PROJECT_CONTEXT prediction from FINDING 3 was +8–13 pp; confirmed on US. India is far
higher because most India recall comes through address.

### 5.3 Per-partition at K=50 (the operating point)

| Partition | truth pairs | P2 | P1+P2 | P1+P2+P3 | in-script | cross-script |
|---|---:|---:|---:|---:|---:|---:|
| india | 4,447 | 75.98 % | 76.03 % | 98.45 % | 98.75 % | 95.21 % |
| us    | 6,676 | 94.41 % | 94.41 % | 99.72 % | 99.72 % | — |

### 5.4 Cross-script (India Devanagari, 376 truth pairs)

| K | P2 (name only, w/ P4 alias) | Full union |
|---:|---:|---:|
| 10  | 1.33 % | 88.83 % |
| 50  | 3.19 % | 95.21 % |
| 200 | 5.32 % | 97.87 % |

**P4 in its shipped form is not doing meaningful work.** All observed
cross-script recall comes from P3 (address is script-agnostic once tokenized).

### 5.5 Smoke test — end-to-end sanity (India, 2000 S1, 60k caps)

| stage | recall | note |
|---|---:|---|
| P1 alone | 40.63 % | exact-name-core matches |
| P1 + P2 | 75.04 % | name blocking |
| **P1 + P2 + P3 union** | **98.27 %** | before rescore |
| **Final top-K (rescored, K=50)** | **94.30 %** | ~4 pp lost in rescore |

Runtime 65 s, peak RSS 4.3 GiB. See §7 for the rescore leak.

### 5.6 Memory

Measured peak: **3.5 GiB** on the K=200 sweep with 80k caps. Extrapolation to the
full US partition (~3.3 M S2 + ~3.3 M S3, ~1.3 M S1):

- CSR matrices scale linearly in `n_docs`. PROJECT_CONTEXT measured a 12.5 M-doc index at 3.77 GiB → full US name+addr indexes ≈ 2.5 GiB.
- S1 metadata ~800 MiB, top-K int64 arrays for K=50 × 1.3 M queries × 4 B ≈ 260 MiB, chunk buffer ≤ 512 MiB.
- **Total peak per partition ≈ 5–6 GiB. Fits 14 GiB with headroom.**

Partitions run sequentially (`partition_order='ascending_size'`); only one resident at a time.

---

## 6. What worked

- **Country partitioning first, before any expensive work.** ~3× reduction at zero recall cost — cheapest win in the pipeline.
- **IDF weighting, no `max_df` cap.** Directly addresses the v0 failure mode.
- **Char-3gram TF-IDF as the primary retrieval.** Robust to word-order, partial-word, and small typos without needing a fuzzy metric.
- **Address as a peer of name, not an afterthought.** +12 pp at K=50 aggregate, +25 pp on India.
- **Union of independent retrievals + rescore.** Simple, debuggable, and each pass is separately measurable — if recall drops in prod, we know which pass to blame.
- **`argpartition` + chunked matmul + auto-shrink to 512 MiB budget.** The three lines that make 14 GiB actually work.
- **Reservoir-sampled distractors in the harness.** Kept the measurement runnable on 14 GiB instead of thrashing swap.

## 7. What did not work (yet)

- **P4 Devanagari alias via char-3gram.** Retrieval on `unidecode`'d Hindi in a Latin char-3gram index barely fires — the 3-gram profile of romanized Devanagari doesn't look like the 3-gram profile of natural English names. Currently pulling ~5 % of cross-script pairs; the rescue is entirely from P3. Fix in §8.
- **Rescore recall leak.** Union hits 98.27 % on the smoke test; the rescored final top-K hits 94.30 %. The raw-score floor blunts but does not eliminate the drop. Weight retuning without matcher signal is guesswork — deferred until the matcher can tell us which recall we most need to preserve.
- **The 80k distractor cap** introduces mild upward bias in reported recall (fewer distractors to hide the needle). Expected drop at K=50 on the full partition is < 1 pp, but that is a *prediction* — needs a validating run on the uncapped US partition.

## 8. What to tune next (ranked)

1. **Word-level Devanagari alias in P4.** After transliteration, token overlap is 40–70 % per sample even where 3-gram overlap is near zero. Add a second name index over `unidecode(target)` with `char_wb` 4-grams *and* a word-token index. Expected cross-script lift 3 % → 40–60 % at K=50, worth ~1.5 pp aggregate.
2. **Recover the ~4 pp lost in rescore.** Two experiments: (a) `combined = max(w_n·cos_n + w_a·cos_a, cos_n, cos_a)` so pure-name and pure-address hits are never demoted; (b) raise `final_top_k` to 100 and let the matcher pay the extra 2× per-pair cost.
3. **Full-partition validation without the distractor cap.** 5–10k S1 sample on the full US partition. Quantifies the true recall drop from more distractors before we trust these numbers on the leaderboard.

---

## 9. Non-negotiables enforced

- No `max_df` cap on TF-IDF. (v0 hit exactly this wall at 45.53 %.)
- No `[^\w\s]` punctuation regex. (Destroys Devanagari; `normalize._selftest` guards this.)
- Ground truth is imported **only** in `measure.py` and `measure_recall_at_k.py`. `blocker_v1.py` has no import path that touches it; the candidate file is byte-identical whether ground truth exists on disk or not.
- Deterministic writes: rows sorted by `(-score, cand_id)` per S1; seed in config.
- No external APIs, no geocoding, no internet lookups (competition DQ rule).
