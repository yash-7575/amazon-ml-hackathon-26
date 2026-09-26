# Candidate Generation Research Engineer — Sources

Curated learning resources and code repos for the 16-section syllabus.
Each link has a one-line summary. Read `Christen — Data Matching` first if you have to pick one.

---

## 1. Entity Resolution Foundations

- **[Data Matching — Peter Christen (Springer)](https://link.springer.com/book/10.1007/978-3-642-31164-2)** — the canonical textbook for ER/record linkage/dedup; maps almost 1:1 to the whole syllabus.
- **[(Almost) All of Entity Resolution — arXiv 2008.04443](https://arxiv.org/pdf/2008.04443)** — modern end-to-end survey of the full ER pipeline.
- **[End-to-End Entity Resolution for Big Data: A Survey](https://arxiv.org/pdf/1905.06397)** — pipeline view (blocking → matching → clustering) with big-data focus.
- **[Basics of Entity Resolution — District Data Labs](https://districtdatalabs.com/basics-of-entity-resolution)** — approachable Python intro.
- **[Implementing ER with Python Record Linkage — Fivetran](https://www.fivetran.com/learn/implementing-entity-resolution-with-python-record-linkage)** — practical starter.

## 2. Blocking & Indexing

- **[A Survey of Blocking and Filtering Techniques — arXiv 1905.06167](https://arxiv.org/pdf/1905.06167)** — canonical blocking reference: exact, sorted-neighborhood, canopy, multi-pass, learned.
- **[Comparative Analysis of Approximate Blocking Techniques (VLDB)](http://www.vldb.org/pvldb/vol9/p684-papadakis.pdf)** — head-to-head benchmark of major blocking families.
- **[Resource-Efficient Blocking (ScienceDirect, 2026)](https://www.sciencedirect.com/science/article/pii/S0306437926000992)** — effectiveness/scalability trade-off in blocking.
- **[Dynamic Sorted Neighborhood Indexing](https://www.researchgate.net/publication/267024541_Dynamic_Sorted_Neighborhood_Indexing_for_Real-Time_Entity_Resolution)** — real-time sorted-neighborhood variant.
- **[Spectral Neighborhood Blocking (SPAN)](https://www.researchgate.net/publication/220966724_Efficient_SPectrAl_Neighborhood_blocking_for_entity_resolution)** — spectral method for candidate blocking.
- **[Noise-Tolerant Approximate Blocking](https://centaur.reading.ac.uk/82137/1/237_Liang.pdf)** — robust blocking under noisy data.

## 3. Information Retrieval (TF-IDF, BM25, indexes)

- **[Northeastern CS6200 — TF-IDF & BM25 slides](https://www.khoury.northeastern.edu/home/vip/teach/IRcourse/1_retrieval_models/slides/m03.s03%20-%20TF-IDF%20and%20Okapi%20BM25.pdf)** — clean derivation, best free lecture notes.
- **[BM25 — Arpit Bhayani](https://arpitbhayani.me/blogs/bm25/)** — very readable intuition + math.
- **[ethen8181 BM25 intro notebook](https://ethen8181.github.io/machine-learning/search/bm25_intro.html)** — code walk-through.
- **[GeeksforGeeks — BM25 Algorithm](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)** — quick reference.
- **[Boot.dev — Learn RAG: BM25](https://www.boot.dev/lessons/dea77599-fce9-4039-a632-45f7fed7685e)** — interactive tutorial.

## 4. String & Fuzzy Retrieval

- **[RapidFuzz (GitHub org)](https://github.com/rapidfuzz)** — C++-backed Python string metrics; fastest option in Python.
- **[rapidfuzz/JaroWinkler](https://github.com/rapidfuzz/JaroWinkler)** — dedicated Jaro / Jaro-Winkler library.
- **[fuzzy-match (PyPI)](https://pypi.org/project/fuzzy-match/)** — Cosine, Levenshtein, Jaro-Winkler on trigrams.
- **[darwinagain/fuzzy-match](https://github.com/darwinagain/fuzzy-match)** — simple reference impl.
- **[Deep Dive into String Similarity — Medium](https://medium.com/data-science-collective/deep-dive-into-string-similarity-from-edit-distance-to-fuzzy-matching-theory-and-practice-in-68e214c0cb1d)** — theory + Python practice.
- **[Best Libraries for Fuzzy Matching — CodeX](https://medium.com/codex/best-libraries-for-fuzzy-matching-in-python-cbb3e0ef87dd)** — comparison of the top Python fuzzy libs.

## 5. Modern Vector Retrieval (HNSW, IVF, PQ, ANN)

- **[Pinecone — HNSW deep dive](https://www.pinecone.io/learn/series/faiss/hnsw/)** — best free HNSW explanation with diagrams.
- **[Vector index internals: HNSW, IVF, ScaNN, FAISS](https://www.kunwar.page/chapter/059-vector-index-internals-hnsw-ivf-scann-faiss)** — comparative internals.
- **[PyImageSearch — Vector Search with FAISS (2026)](https://pyimagesearch.com/2026/02/16/vector-search-with-faiss-approximate-nearest-neighbor-ann-explained/)** — hands-on FAISS tutorial.
- **[Vector Search Algorithms 2026 — MyEngineeringPath](https://myengineeringpath.dev/genai-engineer/vector-search/)** — current landscape.
- **[Vector Search & Indexing (Miraftab blog)](https://alimiraftab.github.io/blog/2026/03/30/rrr-16-vector-search-indexing-hnsw-faiss-scann-annoy/)** — HNSW / FAISS / ScaNN / Annoy in one post.
- **[Managing Millions of Vectors — Medium](https://bhargavaparv.medium.com/managing-millions-of-high-dimensional-vectors-in-modern-vector-database-cbad318068fe)** — production scale considerations.
- **[Distribution-Aware HNSW (arXiv 2512.06636)](https://arxiv.org/pdf/2512.06636)** — 2026 research on adaptive HNSW.

## 6. Multi-Attribute Retrieval

- **[Fuzzy Name Matching — Babel Street](https://www.babelstreet.com/blog/fuzzy-name-matching-techniques)** — practitioner-grade multi-attribute name matching.
- **[Adaptive multi-cultural name matching (patent)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8041560)** — engineering deep-dive on composite keys.

## 7. Multilingual & Dirty Data

- **[Bytes Speak All Languages — cross-script names (TDS)](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)** — modern byte-level contrastive approach.
- **[iliaal/phonetic](https://github.com/iliaal/phonetic)** — Double Metaphone, Beider-Morse, Daitch-Mokotoff, NYSIIS reference (port to Python via `jellyfish`/`abydos`).
- **[Options for Encoding Names — ABS (arXiv 1802.07975)](https://arxiv.org/pdf/1802.07975)** — comparative study of name encodings in production.
- **[Happy Birthday, Soundex — Babel Street](https://www.babelstreet.com/blog/happy-birthday-soundex)** — 100-year retrospective on phonetic name matching.
- **[Double Metaphone applied to Amharic (arXiv cs/0408052)](https://arxiv.org/pdf/cs/0408052)** — non-Latin-script adaptation example.

## 8. Graph-Based Methods

- **[GNNs for Inconsistent Cluster Detection — arXiv 2105.05957](https://arxiv.org/pdf/2105.05957)** — GNNs for collective ER.
- **[Graph-based ER in MapReduce (UNR)](https://www.cse.unr.edu/~hkardes/pdfs/organizationEntityResolution.pdf)** — meta-blocking + transitive closure at scale.
- **[TigerGraph — Entity Resolution guide](https://www.tigergraph.com/glossary/entity-resolution/)** — graph-DB angle, connected components, Louvain.
- **[Unsupervised ER with Blocking + Graph Algorithms](https://www.researchgate.net/publication/340989589_Unsupervised_Entity_Resolution_with_Blocking_and_Graph_Algorithms)** — unsupervised pipeline.
- **[Adaptive Graph Refinement + LLMs (arXiv 2605.25814)](https://arxiv.org/pdf/2605.25814)** — LLM-in-the-loop clustering.

## 9. Learned / Adaptive Blocking

- **[qcri/DeepBlocker](https://github.com/qcri/deepblocker)** — deep learning for blocking; design-space exploration.
- **[Towards Universal Dense Blocking — arXiv 2404.14831](https://arxiv.org/pdf/2404.14831)** — SOTA dense blocking.
- **[anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher)** — DL matcher (pair-scoring after blocking).
- **[megagonlabs/ditto](https://github.com/megagonlabs/ditto)** — PLM-based matcher + blocker.
- **[Skyblocking for ER — arXiv 1805.12319](https://arxiv.org/pdf/1805.12319)** — active/skyline blocking-scheme learning.
- **[SAREM — semi-supervised active ER](https://link.springer.com/content/pdf/10.1007/978-3-031-20309-1_7.pdf)** — active + semi-sup framework.
- **[Active Learning Benchmark for ER — arXiv 2003.13114](https://arxiv.org/pdf/2003.13114)** — apples-to-apples AL comparison.
- **[Active Blocking Scheme Learning (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-93037-4_28)** — active sampling + active branching.

## 10. Candidate Quality (metrics)

- **[Progressive ER: Design Space Exploration — arXiv 2503.08298](https://arxiv.org/html/2503.08298v1)** — deep on RR / PC / PQ trade-offs.
- **[ER Benchmarks Beyond F1 — Minimalist Innovation](https://www.minimalistinnovation.com/post/benchmarking-datasets-metrics-entity-resolution)** — practical metric selection.
- **[Low-cost Relevance Generation & Eval Metrics — arXiv 2205.10298](https://arxiv.org/pdf/2205.10298)** — modern evaluation methodology.
- **[Entity Matching Analysis — Emergent Mind](https://www.emergentmind.com/topics/entity-matching-analysis)** — glossary + error taxonomy.

## 11–12. Efficiency & Data Structures

- **[BK-Trees — Nick's Blog](http://blog.notdot.net/2007/4/Damn-Cool-Algorithms-Part-1-BK-Trees)** — classic explainer for edit-distance indexes.
- **[BK-Tree — Wikipedia](https://en.wikipedia.org/wiki/BK-tree)** — reference.
- **[talhasaruhan/fuzzy-search](https://github.com/talhasaruhan/fuzzy-search)** — trie / BK-tree data-structure code to read.
- **[String B-tree (ACM JACM)](https://dl.acm.org/doi/10.1145/301970.301973)** — external-memory string index.
- **[MongoDB B-tree vs Inverted Index](https://medium.com/mongodb/mongodb-text-search-b-tree-vs-inverted-index-use-search-instead-part-1-501560cb0d59)** — practical comparison for text search.

## 13. Scalability Engineering

- **[Gaglia88/sparker](https://github.com/Gaglia88/sparker)** — production-quality distributed ER on Spark.
- **[SparkER paper (EDBT 2019)](https://openproceedings.org/2019/conf/edbt/EDBT19_paper_347.pdf)** — companion paper.
- **[Scaling Identity Resolution — Salesforce Engineering](https://engineering.salesforce.com/scaling-identity-resolution-in-data-cloud-with-lucene-spark-and-fuzzy-matching/)** — 50M → 2B records; excellent bottleneck breakdown.
- **[SparkDWM (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11416992/)** — scalable data-washing machine on Spark.
- **[Real-time ER with Kafka + Spark](https://medium.com/data-surge/real-time-entity-resolution-with-kafka-and-spark-7ab46dfaaefd)** — streaming architecture.
- **[Scaling ER with BERT/MLflow/Databricks](https://medium.com/@debusinha2009/scaling-entity-resolution-in-the-modern-data-ecosystem-using-bert-mlflow-and-databricks-7524923ae268)** — modern stack.

## 14. Experimentation

- **[A Critical Re-evaluation of Benchmark Datasets — arXiv 2307.01231](https://arxiv.org/pdf/2307.01231)** — how to spot leakage & set honest baselines.
- **[epfl-dlab/entity-matchers](https://github.com/epfl-dlab/entity-matchers)** — critical re-eval of neural ER methods.
- **[Papers-with-Code: entity-resolution](https://github.com/topics/entity-resolution)** — reproducible codebases to replicate.

## 15. Research Mindset (dense/sparse/hybrid retrieval reading)

- **[Dense Text Retrieval Survey — arXiv 2211.14876](https://arxiv.org/pdf/2211.14876)** — full survey of dense retrieval with PLMs.
- **[Hybrid Search for RAG: BM25 + SPLADE + Vector — Prem.ai](https://www.premai.io/blog/hybrid-search-for-rag-bm25-splade-and-vector-search-combined/)** — practical hybrid setup.
- **[Qdrant — Hybrid Search + Reranking](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)** — end-to-end hybrid pipeline.
- **[SPLADE vs BM25 — Zilliz](https://zilliz.com/learn/comparing-splade-sparse-vectors-with-bm25)** — learned-sparse vs classical.
- **[Sparse vs Dense vs Hybrid — abhik.ai](https://www.abhik.ai/concepts/embeddings/sparse-vs-dense)** — clear head-to-head with rerankers.
- **[Elastic — Sparse Embeddings](https://www.elastic.co/search-labs/blog/sparse-vector-embedding)** — engineering perspective.

## 16. Engineering Judgment — production OSS to actually use

- **[Splink](https://moj-analytical-services.github.io/splink/index.html)** — probabilistic record linkage; ~1M records/min on a laptop; used by Australia's ABS.
- **[dedupeio/dedupe](https://github.com/dedupeio/dedupe)** — ML fuzzy matching + dedup + record linkage.
- **[Tilores comparison — Splink / Zingg / dedupe](https://tilores.io/content/best-open-source-entity-resolution-and-record-linkage-libraries-splink-zingg-dedupe-and-when-to-move-beyond-them/)** — when to pick which, and when they stop being enough.
- **[FAISS](https://github.com/facebookresearch/faiss)** · **[hnswlib](https://github.com/nmslib/hnswlib)** · **[ScaNN](https://github.com/google-research/google-research/tree/master/scann)** · **[Annoy](https://github.com/spotify/annoy)** — the four ANN libraries to have hands-on experience with.

---

## Suggested reading order

1. Christen's book — foundations (§§1–2, 10–15).
2. Papadakis blocking survey (arXiv 1905.06167) — §§2, 10.
3. Pinecone HNSW + PyImageSearch FAISS tutorial — §5.
4. Splink docs + dedupe README — §16 engineering intuition.
5. DeepBlocker + Ditto papers/repos — §9.
6. Salesforce identity-resolution post + SparkER — §13.
