# EDA — Business Entity Resolution (Amazon ML Challenge 2026)

Grain: one row = one business record from one source. Task: for each Source 1 record,
find all Source 2/3 records describing the same real-world business.

## 1. Structure & completeness — data is CLEAN

| File | Rows | Dup IDs | Empty name | Empty address | Malformed |
|---|---|---|---|---|---|
| train_source1 | 2,206,821 | 0 | 0 | 0 | 0 |
| train_source2 | 5,034,616 | 0 | 0 | 3.36% | 0 |
| train_source3 | 5,285,603 | 0 | 0 | 3.33% | 0 |
| test_source1 | 1,732,544 | — | — | — | — |
| test_source2 | 4,887,273 | — | — | 2.65% | — |
| test_source3 | 5,082,316 | — | — | — | — |

No nulls, no dup IDs, no ragged rows. **The difficulty is not dirtiness — it is scale + ambiguity.**
Only real gap: ~3.3% of S2/S3 records have an empty address (name-only matching required).

## 2. Ground truth shape

- 7,638,365 matched (S1 → S2/S3) slots across 2,206,821 S1 entities
- Match count: 0=5.58%, 1=5.40%, 2=17.0%, 3=24.1%, 4=21.9%, 5=14.6%, 6+=11.4%; max 11
- Split: 3,693,619 S2 / 3,944,746 S3 — near balanced
- Per entity: 80.6% match BOTH sources, 7.4% S3-only, 6.5% S2-only, 5.5% singleton

### FINDING 1 (critical) — the matching is ONE-TO-ONE
7,638,365 matched-id slots, **7,638,365 distinct ids. Zero reuse.**
No S2/S3 record is ever claimed by two different S1 entities. Ground truth is a
*partition*, not an arbitrary bipartite graph.

**Exploit:** solve as a global assignment problem, not independent per-pair thresholds.
When two S1 entities both claim one S2 record, at most one is right — resolve by score and
drop the loser. Pure precision gain, which is exactly what F_0.5 rewards. Most teams will
score pairs independently and miss this.

Corollary: ~26% of S2/S3 records match nothing and are pure distractors.

### FINDING 2 (critical) — matches NEVER cross country
0 of 138,120 sampled true pairs cross a country boundary.

**Exploit:** partition by country first. Free ~3x reduction, zero recall cost. This is also
how France (15% of test S1, absent from training) is handled safely — it becomes its own
partition instead of a distribution-shift problem.

## 3. Where the signal lives (138,120 true pairs sampled)

| Similarity on true pairs | p10 | p50 | p90 | mean |
|---|---|---|---|---|
| name token Jaccard | 0.000 | 0.667 | 1.000 | 0.656 |
| name char-3gram Jaccard | 0.174 | 0.700 | 1.000 | 0.654 |
| address token Jaccard | 0.300 | 0.636 | 1.000 | 0.623 |

- exact normalized-name match: **21.9%** (the easy fifth)
- share >=1 name token: **85.19%**
- share >=2 name char-3grams: **90.98%**
- **zero name-token overlap: 14.81%**

### FINDING 3 (critical) — name-only blocking caps recall at ~85%
14.81% of true matches share **no** name token. Char-3grams lift the ceiling to ~91%,
still losing 9% permanently.

**Of those 20,449 zero-name-signal pairs, 19,684 (96.3%) have address Jaccard > 0.2.**

**Exploit:** blocking MUST be a union of a name-keyed blocker and an address-keyed blocker.
Name-only is the default approach and it silently forfeits ~15% of achievable recall.

## 4. Ambiguity — why precision is the hard half

- 39.2% of S1 rows share a normalized name with >=1 other S1 row
- Worst collisions: `primary care group` x253, `ear nose and throat group` x251,
  `pediatric group` x222, `womens health group` x220
- S2: 27.7% of rows share a name; `physical therapy` x526, `primary care` x518

Generic names are everywhere, so **name similarity alone cannot decide a match** — address
must break ties. This is the false-merge risk F_0.5 punishes double.

Token document-frequency in S2 (810,495 distinct tokens):
`com` 4.00% of rows, `center` 3.85%, `partners` 3.03%, `services` 2.88%, `group` 2.77%.
640,329 tokens appear exactly once; p90 token df=6, p99 df=113.

**Exploit:** key blocks on rare (high-IDF) tokens only. Blocking on `group` means a
139,390-record block. Blocking on rare tokens gives blocks of <10.

## 5. Script handling

S1 is **100% ASCII**. Devanagari appears only in S2 (5.35%) and S3 (2.99%).
4.17% of true pairs are S1-latin vs S2/S3-Devanagari — genuine cross-script matching.

### FINDING 4 (bug trap) — `[^\w\s]` destroys Devanagari
The standard punctuation-strip regex shatters Hindi names into single consonants:

```
राम मार्केटिंग प्राइवेट लिमिटेड
  -> ['र','म','म','र','क','ट','ग','प','र','इव','ट','ल','म','ट','ड']
```

Cause: Devanagari vowel signs and virama (U+093E, U+094D, ...) are Unicode categories
Mc/Mn, which `\w` does **not** match. Correct approach — strip by Unicode category,
keeping marks:

```python
''.join(' ' if unicodedata.category(c)[0] in 'PSZ' and not c.isspace() else c
        for c in unicodedata.normalize('NFKC', s.lower()))
  -> ['राम','मार्केटिंग','प्राइवेट','लिमिटेड']
```

Any team using the naive regex loses all Devanagari signal without an error.

## 6. Implications for the pipeline

1. Partition by country (free, lossless).
2. Blocking = union of rare-token name index + address index. Measure recall ceiling per
   blocker and for the union before building anything downstream.
3. Features must include address similarity — 15% of matches have no name overlap.
4. Normalize by Unicode category, not `\w`. Strip legal suffixes incl. FR (sarl/sas/sa/eurl).
5. Score pairs, then resolve to a global one-to-one assignment.
6. Tune threshold directly against macro-F_0.5, keeping singletons (5.5%, worth 1.0 each) in.

## Not applicable
Outlier detection, distribution summaries, correlation/multicollinearity: dataset has
**zero numeric columns**. Substituted with string-similarity, ambiguity and blocking-yield
profiling above.
