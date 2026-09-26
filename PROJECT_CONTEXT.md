# PROJECT_CONTEXT — Amazon ML Challenge 2026, Business Entity Resolution

**Read by `candidate-generation-research-architect` before designing anything.**
These are MEASURED facts about this dataset. They override every general heuristic.
Source: `eda/FINDINGS.md`, `eda/EDA_entity_resolution.ipynb`.

## Task
For each Source-1 record, find all Source-2/3 records describing the same real business.
Fields: `entity_id`, `business_name`, `business_address`, `country`. Tab-separated (`sep="\t"`).
Metric: **macro-averaged F₀.₅** — precision weighted 2×, computed per S1 entity then averaged,
singletons included (a correct empty prediction scores 1.0).

`F0.5 = (1.25 · P · R) / (0.25 · P + R)`

## Scale
| File | Rows |
|---|---|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

Brute force = 2.21e6 × 1.03e7 = **2.28e13 pairs ≈ 26 days** at 1e7 pairs/sec. Impossible.
Hardware: 14GB RAM. Ground truth: 7,638,365 matched slots over 2,206,821 S1 entities.

## Measured findings — these ARE the design constraints

**1. Matching is ONE-TO-ONE.** 7,638,365 matched slots, 7,638,365 distinct ids, **zero reuse**.
Ground truth is a partition, not an arbitrary bipartite graph. → Resolve to a global assignment;
when two S1 entities claim one record, at most one is right. Pure precision gain, which is what
F₀.₅ rewards double. (Matcher-stage exploit, but generation must preserve both candidates for it.)

**2. Matches NEVER cross country.** 0 of 138,120 sampled true pairs. → **Partition by country.**
Free ~3× reduction at zero recall cost, bounds peak memory to the largest country, and makes
France (15% of test S1, absent from train) its own problem instead of distribution shift.
Decision-tree Gate 1 is already answered: partition.

**3. Name-only blocking caps recall at ~85%.**
- exact normalized name: **21.90%**
- share ≥1 name token: **85.19%**  ← ceiling of any token-keyed blocker
- share ≥2 name char-3grams: **90.98%**  ← ceiling of any n-gram blocker
- **zero name-token overlap: 14.81%**

Of those 20,449 zero-name-signal pairs, **19,684 (96.3%) have address Jaccard > 0.2**.
→ Blocking MUST union a name blocker AND an address blocker. Name-only silently forfeits ~15%.
Gate 2's conditional measurement is already done and the answer is unambiguous.

**4. `[^\w\s]` destroys Devanagari.** S1 is 100% ASCII; Devanagari appears in S2 (5.35%) and
S3 (2.99%), so **4.17% of true pairs are genuinely cross-script**. The standard punctuation regex
shatters Hindi names into single consonants with no error — vowel signs and virama are Unicode
`Mn`/`Mc`, which `\w` does not match. Strip by Unicode category instead (implemented in
`scripts/normalize.py:strip_punct_by_category`, self-tested).

**5. Ambiguity is the hard half.** 39.2% of S1 rows share a normalized name with ≥1 other S1 row
(`primary care group` ×253, `ear nose and throat group` ×251). S2: 27.7%.
→ Name similarity alone cannot decide a match. **Use ARCH-5: combine name + address at ranking
time**, not only as post-hoc features. This is the highest-leverage fix here.

**6. Token frequency.** 810,495 distinct S2 tokens; `com` 4.00% of rows, `center` 3.85%,
`partners` 3.03%, `group` 2.77%. 640,329 tokens appear once; p90 df=6, p99 df=113.
Blocking on `group` = a 139,390-record block. → IDF weighting, not a hard df cap.

**7. Data is clean.** No nulls, no dup ids, no malformed rows. Only gap: ~3.3% of S2/S3 have an
empty address → those need name-only matching and must still produce candidates (taxonomy #6).

**8. True-pair similarity percentiles** (138,120 sampled) — use for the similarity floor:
| | p10 | p50 | p90 | mean |
|---|---|---|---|---|
| name token Jaccard | 0.000 | 0.667 | 1.000 | 0.656 |
| name char-3gram Jaccard | 0.174 | 0.700 | 1.000 | 0.654 |
| address token Jaccard | 0.300 | 0.636 | 1.000 | 0.623 |

Set `sim_floor` from the **1st** percentile of true-pair similarity, not from p10 and not from a
round number.

## Known negative result — do not repeat it
A rare-token (df ≤ 120) + address-numeric blocker measured **45.53% recall at K=40** — far below
the 85% token ceiling. Cause: the hard df cut discards common tokens that were the only shared
signal for many pairs. **Fix is IDF-weighted TF-IDF cosine over ALL tokens/n-grams plus ANN or
chunked exact matmul — not a rarity threshold.** This is failure class #11 + a df-cap mistake.

## Starting architecture (justify any deviation)
**ARCH-3 partitioned union + ARCH-5 ambiguity-hardened ranking + ARCH-4 for cross-script.**
```
FOR each country:
  P1 normalized-exact on name core        (~22% at near-zero cost)
  P2 char-3gram TF-IDF on name, top-K     (primary engine; IDF-weighted, min_df=2)
  P3 address-keyed TF-IDF, top-K          (the ~15% with no name overlap — do not omit)
  P4 transliterated-name key for Devanagari records, unioned  (the 4.17%)
  → rescore by w_name·cos_name + w_addr·cos_addr, keep top-K
  → union, dedup, stream to candidate_pairs.tsv
```
Expected PC 93–98%. Measure it; do not assume it.

## Competition constraints — hard
- **No external databases, APIs, or services to look up business identities or resolve entities.**
  Includes geocoding APIs and any internet data augmentation. Violation = disqualification.
- Final model must be MIT/Apache-2.0 licensed, ≤8B parameters.
- Private leaderboard decides final ranking — do not overfit the public one.
- Deadline **27 Sep 2026 23:59 IST**.

## First things to measure
1. Recall@K curve for P2 alone, K ∈ {10,20,50,100,200} — find the knee.
2. Marginal recall of P3 over P2. Expect large (~+8–13%) given finding 3.
3. Stratified PC by country and by script — expect cross-script to lag badly.
4. Peak RSS on the largest country partition; confirm it fits 14GB.

---

# Algorithmic hot spots (for `dsa-algorithm-engineering-architect`)

Measured with `~/.claude/skills/algorithm-engineering/scripts/`. These are decisions, not guesses.

| Concern | Number | Consequence |
|---|---|---|
| Brute force | 2.2e6 × 1.03e7 = **2.28e13 pairs ≈ 26 days** | must index; no loop survives |
| Candidate budget | 2.2e6 × K=50 = **1.1e8 pairs** | ~1.5 h feature extraction single-threaded at 50 µs/pair |
| Inverted index, flat int32 postings | **1.96 GB** | fits 14 GB comfortably |
| Same index as dict-of-lists | **16.8 GB** | **does not fit** — layout alone decides feasibility |
| TF-IDF CSR, 12.5M × 40 nnz | **3.77 GB** | fits, but partition by country to be safe |
| 12.5M strings as a Python list | 1.01 GB | 0.44 GB as one buffer + offsets (2.3×) |
| Top-50 of 2e6 scores | argsort 74 ms vs **argpartition 8.4 ms** | 8.8× per query; × 2.2M queries = hours |

## Decisions that follow
1. **Never `np.argsort` a full score row to take top-K.** Use `np.argpartition` — measured 8.8× at
   2e6, and it runs once per source record.
2. **Inverted index as flat int32 postings + int64 offsets**, never dict-of-lists. 8.6× less memory,
   and the difference between fitting in 14 GB and not.
3. **Partition by country** (proven lossless) → peak memory is the largest country, and partitions
   are embarrassingly parallel across processes. Set `OMP_NUM_THREADS=1` in workers or 3 processes ×
   BLAS threads will thrash.
4. **Chunk the query side** of the sparse matmul (1–10K rows). Bounds peak RSS; BLAS keeps it fast.
5. **One-to-one resolution:** Hungarian is Θ(n³) — 1e19 at n=2.2e6, impossible. Build the candidate
   graph, run **DSU** (≈Θ(1) amortized per edge) to find connected components, then **plot the
   component-size distribution**. Small components → exact assignment per component, which is
   optimal and cheap. One giant component → greedy by descending score, Θ(E log E), near-optimal
   when scores separate. Measure `max(component_size)` before choosing.
6. **Dedup/sort the union** with int32 pairs + `np.unique`, not a Python set of tuples. For
   out-of-core, `external_sort.py` or `sort -S 2G -u`.
7. **Validate the complexity of the retrieval loop** with `complexity_validator.py` on a size sweep
   before the full run — a measured exponent above ~1.2 means something is scanning that should be
   indexed.

---

# Feature engineering notes (for `feature-engineering-matching-architect`)

## Start from SET-3 (ambiguity-hardened), not SET-2
The measured findings above make this a decision, not a preference:

| Measured finding | Feature consequence |
|---|---|
| **39.2% of S1 rows share a normalized name** (`primary care group` ×253) | Absolute name similarity cannot discriminate. **Group-relative features are mandatory**, not optional |
| **One-to-one: 7,638,365 slots, 7,638,365 distinct ids, zero reuse** | `is_mutual_best` and `n_competitors_for_cand` directly encode the partition structure. Strong, cheap, test-time computable |
| **Zero cross-country matches** (blocking partitions on country) | `country_match` has **variance 0** → drop as a feature. It is a blocking rule, not evidence |
| **14.81% of true pairs share no name token**; 96.3% of those have address Jaccard > 0.2 | Address features are mandatory. Also add cross-field `name_low_addr_high` |
| **~3.3% of S2/S3 have empty address** | `addr_sim` must be **NaN**, never 0. Plus `addr_missing_either`. 0 would assert disagreement |
| **4.17% of true pairs are cross-script** (S1 100% ASCII, Devanagari in S2/S3 only) | `is_cross_script` + transliterated similarity. **But 4.17% is small — measure the subgroup before investing days** |
| **Metric is macro-F₀.₅ (precision ×2)** | Features that **reject** are worth more than features that accept. Ask of each: what does it look like on a near-miss? |
| **1.1e8 candidate pairs at K=50** | SET-3 ≈ 17 µs/pair ≈ 28 min single-threaded, ~6 min with `workers=-1`. SET-5 ≈ 1.7 h. Budget accordingly |

## The near-miss classes this dataset will produce
1. **Generic-name collision** — `primary care group` in two cities. Fixed by `score_margin_to_2nd`,
   `is_mutual_best`, `name_high_addr_low`. **Not** by more string metrics.
2. **Sequential numbering** — `Store 14` vs `Store 15` score 0.875 on `fuzz.ratio` and are different
   businesses. Fixed only by `name_digits_conflict`. The fixture in
   `~/.claude/skills/pairwise-features/scripts/test_fixture.py` demonstrates both.
3. **Subset names** — `Ram Trading` vs `Ram Trading Textiles Pvt Ltd`. Needs asymmetric containment
   in **both** directions. Note that legal-suffix stripping alone resolves `X` vs `X Pvt Ltd` to
   exact equality, which is why `name_core_exact` is a separate cheap feature.

## Hard rules for this project
- **Freeze K** across train and test. Every group-relative feature shifts if K changes, and no
  leakage check will flag it.
- **Rank by a combined name+address score**, not name alone — ranking by name reproduces the
  ambiguity you are trying to escape.
- Pass the **retrieval score and pass-provenance through** from blocking. Free, and `n_passes_found_by`
  encodes agreement between independent blockers.
- **No external data of any kind** — competition rules forbid APIs, geocoding, and internet lookup.
  Every feature must derive from the four provided columns.
- Evaluate with **PR-AUC**, not ROC-AUC (imbalance ~1:6 at pair level), and stratify by country,
  script, address-completeness, and name ambiguity.
