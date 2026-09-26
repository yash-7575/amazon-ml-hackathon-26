# Feature Engineering & Entity Matching Engineer — Sources

Curated resources for the 28-section syllabus. Seeded from the two links supplied with the brief,
extended with canonical references. **Start with the Christen book (ch. 5, comparison functions) and
the EMM pipeline docs** — together they cover most of the syllabus.

---

## 1. Pipeline position & reference implementations

- **[Entity Matching Model — Pipeline docs](https://entitymatchingmodel.readthedocs.io/en/latest/pipeline.html)** — explicit blocking → pairwise features → supervised matching structure. The cleanest statement of where this stage sits. *(supplied with the brief)*
- **[shyamsantoki/product-matching-pipeline](https://github.com/shyamsantoki/product-matching-pipeline)** — end-to-end ER: TF-IDF retrieval cutting comparisons 99.8% at 98.6% recall, then XGBoost to F1 0.88. Read the feature-construction step. *(supplied with the brief)*
- **[Data Matching — Peter Christen](https://link.springer.com/book/10.1007/978-3-642-31164-2)** — **ch. 5 is the canonical treatment of comparison functions**; ch. 6 covers classification. The single best source for this stage.
- **[Splink — comparison library docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/comparators.html)** — production-grade comparison levels; excellent for thinking about *graded* rather than continuous similarity.
- **[dedupeio/dedupe — variable types](https://docs.dedupe.io/en/latest/Variable-definition.html)** — how a mature library types and compares fields.
- **[recordlinkage — Comparing docs](https://recordlinkage.readthedocs.io/en/latest/ref-compare.html)** — practical Python API for pairwise comparison vectors.

## 2. String & set similarity

- **[RapidFuzz docs](https://rapidfuzz.github.io/RapidFuzz/)** — the library to use. **Read the `process.cpdist` page** (vectorized element-wise) and the `score_cutoff` semantics.
- **[RapidFuzz — fuzz module](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html)** — `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio` and their differing costs.
- **[Deep Dive into String Similarity (Medium)](https://medium.com/data-science-collective/deep-dive-into-string-similarity-from-edit-distance-to-fuzzy-matching-theory-and-practice-in-68e214c0cb1d)** — theory plus Python practice across the metric families.
- **[Options for Encoding Names — arXiv 1802.07975](https://arxiv.org/pdf/1802.07975)** — comparative study of name encodings; the evidence that phonetics underperform char n-grams.
- **[Fuzzy Name Matching — Babel Street](https://www.babelstreet.com/blog/fuzzy-name-matching-techniques)** — practitioner-grade multi-attribute name comparison.

## 3. Feature engineering craft

- **[Feature Engineering for Machine Learning (Zheng & Casari)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)** — the standard practical text.
- **[Kaggle — Feature Engineering course](https://www.kaggle.com/learn/feature-engineering)** — free, hands-on, includes target-encoding pitfalls.
- **[scikit-learn — Preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html)** — reference for when scaling matters (and, for trees, when it does not).

## 4. Missing data

- **[scikit-learn — Imputation](https://scikit-learn.org/stable/modules/impute.html)** — techniques and, more usefully, their assumptions.
- **[LightGBM — missing value handling](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#missing-value-handle)** — **why NaN is the correct encoding for trees**; read this before choosing a sentinel.
- **[XGBoost — sparsity-aware split finding](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)** — the same mechanism from the other major library.

## 5. Leakage

- **[Leakage in Data Mining (Kaufman et al.)](https://dl.acm.org/doi/10.1145/2020408.2020496)** — the canonical taxonomy and formulation.
- **[Kaggle — Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)** — short, practical, covers target leakage and train-test contamination.
- **[A Critical Re-evaluation of Benchmark Datasets — arXiv 2307.01231](https://arxiv.org/pdf/2307.01231)** — ER-specific: how leaky benchmarks inflate reported results.
- **[Out-of-fold target encoding (category_encoders)](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)** — if you must target-encode, this is the discipline required.

## 6. Importance, ablation & interpretation

- **[SHAP documentation](https://shap.readthedocs.io/en/latest/)** — `TreeExplainer` for LightGBM/XGBoost; sample before running at scale.
- **[Interpretable ML (Molnar) — Permutation Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)** — **read the correlated-features section**; it explains why permutation and gain both mislead there.
- **[sklearn — Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)** — including the explicit warning about correlated features.
- **[Beware Default Random Forest Importances](https://explained.ai/rf-importance/)** — why impurity/gain importance is biased toward high-cardinality features.

## 7. Imbalanced evaluation

- **[The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)** — **why PR-AUC, not ROC-AUC, under class imbalance.** Short and decisive.
- **[sklearn — average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)** — the PR-AUC implementation to use.
- **[Progressive ER: Design Space Exploration — arXiv 2503.08298](https://arxiv.org/html/2503.08298v1)** — ER-specific metric trade-offs.

## 8. Learned pair representations (know the alternative before dismissing it)

- **[megagonlabs/ditto](https://github.com/megagonlabs/ditto)** — PLM matcher that serializes a pair as text, replacing hand-built features.
- **[anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher)** — neural pair scoring.
- **[epfl-dlab/entity-matchers](https://github.com/epfl-dlab/entity-matchers)** — critical re-evaluation; the evidence that **tuned gradient boosting over good hand-built features is a genuinely competitive baseline**, often at a fraction of the cost.

## 9. Multilingual

- **[indic-transliteration](https://github.com/indic-transliteration/indic_transliteration_py)** — ISO 15919 for Indic scripts.
- **[Unidecode](https://pypi.org/project/Unidecode/)** — coarse but fast romanization.
- **[Unicode categories (UAX #44)](https://www.unicode.org/reports/tr44/#General_Category_Values)** — the reference behind the `Mn`/`Mc` trap that `\w` causes.
- **[Bytes Speak All Languages (TDS)](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)** — modern cross-script retrieval.

---

## Suggested reading order
1. **Christen ch. 5** — comparison functions. Foundation for the whole stage.
2. **EMM pipeline docs + product-matching-pipeline repo** — see the stage in context.
3. **RapidFuzz `process.cpdist` + `score_cutoff`** — the implementation that makes 1e8 pairs feasible.
4. **LightGBM missing-value docs** — settles the NaN question permanently.
5. **Saito & Rehmsmeier** — settles the PR-AUC vs ROC-AUC question permanently.
6. **Molnar on permutation importance** (correlated features) + explained.ai on RF importances.
7. **Kaufman on leakage** + the ER re-evaluation paper.
8. **Splink comparison levels** — for graded comparison design.
9. **Ditto / entity-matchers** — to know what you are choosing against.

## Source-evaluation notes
- **Feature importance claims are dataset-specific.** A feature ranking from a product-matching paper
  tells you little about business-name matching. Treat published rankings as hypotheses to ablate.
- **Check whether the baseline was tuned** before believing a neural-vs-classical margin.
- **Check the metric.** "F1 0.88" is not comparable to a PR-AUC or to a macro-averaged F₀.₅.
- **Competitive-ML write-ups** are strong on feature ideas and weak on leakage discipline and
  production cost. Mine them for ideas; verify independently.
