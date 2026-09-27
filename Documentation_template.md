# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** NavaDrishti
**Team Members:** Kalyani Bhintade, Pranish Belsare, Yash Bhagyawant
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We use a three-stage pipeline: **inverted-index composite-key blocking** proposes candidates,
a **TF-IDF rescore over each entity's own candidates** ranks them, and a **LightGBM pairwise
classifier over 77 engineered features** scores each pair. The core contributions are (a) a
blocking stage that is ~6,000× faster per query than TF-IDF cosine over the full index while
retaining 86% pair completeness, and (b) a decision layer that picks the **number** of matches
per entity by maximising *expected* F₀.₅ rather than applying a global threshold — which is what
the macro-averaged, precision-weighted metric actually rewards.

---

## 2. Methodology

### 2.1 Problem Analysis

Every design decision below traces to a measurement on the training data, not a heuristic.

| Finding | Measured | Consequence |
|---|---|---|
| Matches never cross country | 0 of 138,120 sampled true pairs | Partition by country. Free ~3× reduction at zero recall cost; makes France its own problem rather than distribution shift. |
| Name-only blocking is capped | share ≥1 name token **85.19%**; zero name-token overlap **14.81%** | A name-only blocker silently forfeits ~15% of matches. |
| Address rescues the rest | of the zero-name-overlap pairs, **96.3%** have address Jaccard > 0.2 | Blocking must **union** a name and an address channel. |
| Ground truth is a partition | 7,638,365 matched slots, 7,638,365 **distinct** ids | No S2/S3 record belongs to two S1 entities → candidate-side conflict resolution is a free precision gain. |
| One-to-many from S1 | mean **3.46** matches/entity (mode 3, range 0–11) | Top-1 assignment is wrong; we must predict *sets*. |
| Singletons are large and valuable | **123,247 / 2,206,821 = 5.58%** have zero matches | Each correct empty prediction scores a full 1.0 — up to 5.58 points of macro-F₀.₅ in one decision. |
| Script asymmetry | Devanagari: S2 **5.32%**, S3 **2.99%**, S1 **0%** | Cross-script matching is one-sided; romanisation must be folded into both index and features. |
| Hidden field type | bare web domains as names: S2 **3.38%**, S3 **3.36%**, S1 **0%** | For these, no string metric on the raw name works — the domain stem must be extracted. |
| Missing addresses | S2/S3 **3.32%** empty, S1 **0%** | Undefined similarity must be NaN, never 0.0. |

**A negative result worth recording.** An early blocker applied a hard `max_df` cut to the
TF-IDF vocabulary and reached only **45.53%** recall. Discarding common n-grams discards the
evidence that links noisy names; IDF weighting already down-weights them correctly. We removed
the cap.

**A second negative result.** In the rescore we tested replacing the weighted score floor with
an unweighted `max(raw, cos_addr)`. Recall *fell* from 94.30% to 93.30%, and cross-script from
76.39% to 73.68%. Cause: real `cos_name` runs 0.5–0.95 while `cos_addr` runs 0.10–0.35, so an
unweighted max systematically promotes the higher-dynamic-range field and lets name-strong
distractors displace address-only true pairs. Reverted.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — inverted-index blocking → TF-IDF rescore → GBDT pairwise classifier
→ expected-F₀.₅ set selection.

**Core Innovation:** Two parts.

1. **Blocking that is fast *and* multi-channel.** TF-IDF cosine against the full index costs
   **0.28 s per S1 entity** (measured on the full 6.19M-record US pool) — ~85 h for the test set,
   which is not runnable. Replacing it with composite-key lookup over an inverted index costs
   **0.047 ms per entity**, and the two-stage version with rescore costs **4.76 ms**. The index
   proposes; TF-IDF disposes, over ~560 candidates instead of 4.1M documents.

2. **A decision layer matched to the metric.** The metric is macro-averaged per entity and
   weights precision 2×, so the value of the *n*-th predicted match depends on how many you
   already hold. We choose *n* per entity by maximising expected F₀.₅ under the model's
   probabilities. Crucially the cost asymmetry **reverses at T = 1**:

   | true matches T | cost of one false positive | cost of one miss |
   |---|---|---|
   | 1 | 0.444 | **1.000** ← a miss is 2.3× worse |
   | 2 | 0.286 | 0.167 |
   | 3 | 0.211 | 0.091 |
   | 4 | 0.167 | 0.063 |
   | 6 | 0.118 | 0.039 |

   For T ≥ 2 precision dominates as intended; at T = 1 an empty prediction scores a flat 0.0.
   A single global threshold cannot express this, and silently empties exactly the entities where
   emptiness is most expensive.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** — nine families over an inverted index, all built from the *same* function
for index and query so the two can never drift:

| Key | Purpose |
|---|---|
| `C` normalised name core (legal suffixes stripped) | exact anchor |
| `S` sorted name tokens | word-order invariance |
| `RT` 10 rarest name tokens by document frequency | **main recall driver** — survives reordering and partial names |
| `RA` 6 rarest address tokens (stop-words removed) | reaches the 14.81% zero-name-overlap pairs |
| `G4` rarest character 4-grams | typo tolerance, which exact keys structurally lack |
| `NA` / `NN` / `AN` / `A2` cross-field anchors | name+region, name+house-number, house-number+region, address head |
| transliterated name core | cross-script (Devanagari → Latin) |
| domain stem | the 3.4% of names that are bare web domains |

Rarity is by document frequency in the target corpus — a common token produces a bucket nobody
benefits from. Buckets larger than **400 postings are dropped entirely**: a key matching thousands
of records is not evidence, it is a scan.

**Two-stage ranking.** The index proposes ~560 candidates per entity; TF-IDF vectors are then built
for *only those candidates* and scored as `0.6·cos_name + 0.4·cos_addr`, trimmed to **K = 150**.

**Candidate pairs generated (shipped run):**

| Partition | S1 entities | K | Candidate pairs |
|---|---|---|---|
| France | 259,449 | 150 | 37,180,196 |
| US | 663,106 | 150 | ~99,000,000 |
| India (first pass) | 527,359 | **80** | ~42,000,000 |
| India (second pass) | 282,622 | 150 | 41,383,765 |

India was generated in two passes because the first was lost to an out-of-memory kill and the
restart ran under a disk constraint that forced K=80; the remainder was completed at K=150 once
space was reclaimed. The two passes are disjoint by construction — the second resumed at the
exact S1 file offset where the first stopped — and 5 entities across the whole test set received
no candidates at all.

**How we ensured true matches were not lost.** Pair completeness was measured against ground truth
at every iteration on a fixed 8,000-entity uniform sample — never assumed:

| Key set | Recall |
|---|---|
| 6 keys, initial | 67.62% |
| + rare name/address tokens | 82.78% |
| + more tokens, larger bucket cap | 86.16% |
| + char-4grams, cap 400 (**shipped**, pre-trim) | **90.51%** |
| after the K=150 rescore trim | **86.05%** |

**Stated honestly:** the TF-IDF blocker we replaced reaches **94.41%**. We traded **8.4 points of
pair completeness for a stage that finishes in minutes instead of ~85 hours**. That gap — not the
classifier — is the dominant remaining error source, and closing it is the first thing we would do
with more time.

The sample used for measurement is drawn **uniformly over all S1 entities**, not filtered to those
with matches, so singletons appear at their true rate (6.1% observed vs 5.58% population) and the
group-size distribution matches evaluation (mean 3.45 vs 3.46).

---

## 4. Matching Model

**Features used — 77 total, all label-free.** The extractor is structurally barred from importing
ground truth, and an automated check enforces it; a rule in a comment gets forgotten, an import
boundary does not.

- **Name:** exact / core-exact / first- and last-token exact, token Jaccard, Dice, containment in
  **both directions** (asymmetric by design, never averaged away), character 3-gram Jaccard and
  containment, IDF-weighted token overlap, max shared IDF, `token_sort_ratio`, `token_set_ratio`,
  `ratio` (RapidFuzz `cpdist`, C++, multithreaded), length and token-count ratios, and an explicit
  **digit-conflict guard** — "Store 14" vs "Store 15" scores ~0.97 on every string metric but is a
  different business, and under a precision-weighted metric that false positive costs double.
- **Address:** token/n-gram/numeric Jaccard, Dice, containment both directions, exact match,
  numeric-set exact, IDF-weighted overlap, max shared IDF, last-token (region) exact, house-number
  equality and conflict, token-count ratio.
- **Cross-script:** dominant-script match, cross-script flag, and — the part flag-only approaches
  miss — an actual **romanised comparison** (`translit_ratio`, `translit_token_set_ratio`) using the
  same transliteration as the blocking index.
- **Domain-aware:** domain-stem detection plus name similarities recomputed on the stem, for the
  3.4% of records whose name is a bare URL.
- **Cross-field:** name-high/address-low state flags (the "same chain, different branch" signature
  that no single similarity value can express), disagreement, min/mean field similarity.
- **Missingness:** explicit flags; undefined similarity is **NaN, never 0.0** — a zero asserts
  "these disagree", a false statement about a pair where one side simply has no address. LightGBM
  handles NaN natively.
- **Group-relative and reciprocal (14):** rank in group, margin to runner-up, score ratio to max,
  z-score within group, group size, gap below, reverse rank, **`is_mutual_best`**, and
  **`n_competitors_for_cand`** — the last two encode the measured partition property directly.

**Model type:** LightGBM binary classifier (`binary`, lr 0.05, 63 leaves, `min_data_in_leaf` 50,
feature_fraction 0.9, bagging 0.8, L2 1.0, seed 20260926), early stopping on a held-out split,
**best iteration 364**.

**Validation design.** The train/validation split is **by S1 entity, never by pair**. Group-relative
features are computed across a whole candidate group, so splitting inside a group leaks rank
information from train into validation — and the inflation is invisible to every metric. The split
is asserted disjoint at training time.

**Threshold selection method.** Per-entity **expected-F₀.₅** maximisation over the sorted
probabilities, including *n* = 0 (predict empty), preceded by candidate-side conflict resolution
where each candidate is kept only for its highest-probability claimant.

---

## 5. Results & Error Analysis

**Validation (held-out India entities, full 4.13M-record candidate pool):**

| Metric | Value |
|---|---|
| **Macro F₀.₅** | **0.89696** |
| Precision (micro) | 0.9772 |
| Recall (micro) | 0.8050 |
| ROC-AUC | 0.99979 |
| Accuracy | 0.9950 |
| Blocking recall ceiling | 86.05% |

Training set: 8,000 S1 entities, 1,174,554 pairs, 23,762 positives, 48:1 class ratio.
Confusion at the operating point: **TP 5,574 · FP 130 · FN 1,350 · TN 286,799**.

**We do not report accuracy as a headline.** At 48:1, predicting "no match" everywhere already
scores ~97.9%; the 0.9950 above is worth 1.9 points over that baseline and tells you nothing. The
same applies to ROC-AUC, which is diluted by the enormous negative pool. Macro-F₀.₅ is the number.

**Top features by gain:** `addr_contain_rev` (606,881) and `retrieval_score` (420,094) dominate,
followed by `addr_numeric_jaccard`, `addr_token_jaccard`, `z_score_in_group`, `translit_ratio`,
`eff_name_ratio`. Two observations: address features outrank name features, consistent with the EDA
finding that address carries the pairs names cannot; and both purpose-built families —
transliteration and domain stems — earn top-10 places, confirming they were worth adding.

**Eight features contributed zero gain**, and these are findings rather than defects:
`is_domain_l` is constant because S1 never contains a domain; `addr_missing_*` and
`name_missing_either` are near-constant because S1 has no empty addresses; `name_exact` is subsumed
by stronger continuous features.

**Common false positives (wrong merges).** Same-chain-different-branch pairs: names identical,
addresses in the same city. The cross-field `name_high_addr_low` flag and the numeric-conflict
features exist specifically for these. Residual errors concentrate where a business has several
genuinely similar sibling records and conflict resolution must pick one.

**Common false negatives (missed matches).** Overwhelmingly **blocking failures, not classifier
failures** — 13.95% of true pairs never reach the model. These are pairs sharing no rare token and
no address anchor: heavy typos across every token, or aggressive abbreviation. Within the
candidates that *do* reach the model, recall is high; the weakest score bucket is **T = 1**
(0.797), exactly as the cost-asymmetry analysis predicts, followed by singletons (0.886).

---

## 6. Conclusion

We built a country-partitioned pipeline whose blocking stage is roughly 6,000× faster per query
than full TF-IDF retrieval, and paired it with a 77-feature LightGBM matcher and a decision layer
that optimises the competition metric directly rather than thresholding probabilities. The key
lesson was that **the metric, not the model, dictates the design**: macro-averaging per entity,
2× precision weighting, and the reversal of error cost at a single true match together mean that
choosing *how many* matches to emit matters more than ranking them better. The honest limiting
factor is blocking recall (86.05%), not classifier quality — we knowingly traded pair completeness
for a stage that could finish within the competition window, and we measured that trade rather than
hiding it.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` contains two packages:

```
blocking/
  keyblock.py            composite-key families + inverted index (shared by index and query)
  generate_keyblock.py   ENTRY POINT — candidates for one split+country, two-stage
  measure_keyblock.py    pair-completeness harness against ground truth
  normalize.py           Unicode-category punctuation stripping, transliteration, country keys
  index.py, io_utils.py  TF-IDF index, streaming TSV I/O, deterministic writer
matcher/
  featconfig.py          single source of truth for feature parameters
  features.py            41-feature core extractor (label-free; enforced)
  features_extra.py      22 features: domain stems, transliteration, address depth
  group_features.py      14 group-relative and reciprocal features
  featurize.py           THE frozen entry point — train and inference both call only this
  label.py               the ONLY module permitted to read ground truth
  build_dataset.py       ENTRY POINT — pairs → (X, y, groups)
  train.py               ENTRY POINT — LightGBM, split by S1 group
  predict.py             ENTRY POINT — streaming chunked inference
  assign.py              conflict resolution + expected-F₀.₅ set selection
  evaluate.py            macro-F₀.₅ exactly per spec, with unit tests
  submit.py              ENTRY POINT — predictions → the two output TSVs
```

**Reproduce end-to-end** (see `README.md` for full commands):

```bash
python blocking/generate_keyblock.py --split test --country India --top-k 150
python -m matcher.predict --pairs <long.tsv> --model <model.lgb> --out preds.tsv
python -m matcher.submit --preds preds_*.tsv --cands wide_cand_*.tsv --out-dir output
```

`run_submission.sh` chains all of it. Countries are processed strictly sequentially and each
country's intermediate candidate file is deleted after prediction, because the full test set at
K=150 exceeds available disk.

**Self-tests.** `matcher/evaluate.py`, `matcher/assign.py`, `matcher/group_features.py` and
`matcher/features_extra.py` each run assertions via `python -m`. `matcher/leakage_check.py` verifies
the extractor cannot import labels.

### B. Additional Results

- Blocking recall by key set: 67.62% → 82.78% → 86.16% → **90.51%** (pre-trim) → **86.05%** (K=150).
- Blocking throughput: **0.047 ms/entity** key lookup; **4.76 ms/entity** with TF-IDF rescore;
  peak RSS 3.3 GB on a 4.1M-record pool.
- Threshold sensitivity: only **46 of 600** swept assignment configurations lie within 0.005 of the
  best. An earlier experiment on a 60k-capped candidate pool had **188 of 600** within 0.005 — a
  flat surface that made threshold choice look free. Harder negatives make the surface **4× sharper**,
  which is why all reported figures use the full uncapped pool.
- Cross-country transfer: a model trained on India scored **0.97534** on 2,000 unseen US entities
  versus 0.97257 on held-out India under matched conditions — no degradation. This is our only
  evidence that France (17.6% of the test set, absent from training) transfers, and it is a proxy,
  not proof.

**Leaderboard evidence.** Partial submissions let us isolate per-country performance. An
all-empty submission scored **0.056**, matching our own scorer's prediction exactly and confirming
the metric implementation. A France-only submission scored **0.177**; backing out the empty
baseline gives France **~0.865 on its own** — achieved with **zero French training data**, which is
the strongest available evidence that the feature set is language-agnostic rather than fitted to
India.

**Limitations.** Blocking recall caps the achievable score (86.05% at K=150, lower for the India
first pass at K=80). Thresholds are tuned on the same held-out groups they are reported on, so
0.89696 is optimistic. The model is trained on India only. The final submission covers
1,618,421 of 1,732,544 entities with at least one predicted match; 114,123 are predicted empty
against a true singleton rate of 5.58%, i.e. slightly conservative — the correct direction under a
precision-weighted metric.
