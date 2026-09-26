# DSA & Algorithm Engineering Research Architect — Sources

Curated learning resources and code repos for the 17-section syllabus.
If you have to pick one starting point: **MIT 6.006 (lecture videos + notes)** for foundations, then **CLRS** as reference, then **cp-algorithms.com** for hands-on.

---

## 1. Foundations of Computer Science

- **[MIT 6.006 — Introduction to Algorithms (OCW)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/)** — the definitive intro course; full videos + lecture notes + problem sets. Start here.
- **[MIT 6.006 Lecture Notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/)** — clean, structured notes covering ADTs, hashing, recursion, sorting.
- **[Sedgewick — Algorithms, 4th Edition (companion site)](https://algs4.cs.princeton.edu/home/)** — best "learn-by-doing" book; Java code, animations, Coursera companion.
- **[Introduction to Algorithms (CLRS) — Wikipedia overview](https://en.wikipedia.org/wiki/Introduction_to_Algorithms)** — reference index to the standard textbook.

## 2. Complexity & Algorithm Analysis

- **[Stanford CS161 Lecture 3 — Solving Recurrences (Jessica Su)](https://web.stanford.edu/class/archive/cs/cs161/cs161.1168/lecture3.pdf)** — the cleanest recurrence-solving notes online.
- **[AlgoMaster — Master Theorem](https://algomaster.io/learn/dsa/master-theorem)** — practical worked examples.
- **[AlgoMaster — Amortized Analysis](https://algomaster.io/learn/dsa/amortized-analysis)** — aggregate / accounting / potential methods.
- **[Programiz — Master Theorem (with examples)](https://www.programiz.com/dsa/master-theorem)** — quick reference.
- **[Dev.to — Solving Recurrences with the Master Theorem](https://dev.to/downey/using-the-master-theorem-to-solve-recurrences-4pdb)** — walkthrough.

## 3. Core Data Structures

- **[MIT 6.046 — Design & Analysis of Algorithms (OCW)](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/pages/lecture-notes/)** — heaps, BSTs, balanced trees, DSU, hashing at graduate depth.
- **[6.046 Complete Lecture Notes PDF](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/b1e5ac3c8f7d60db9b8d5b66c40bc55e_MIT6_046JS12_Notes.pdf)** — single file, print-ready.
- **[cp-algorithms — Data Structures index](https://cp-algorithms.com/)** — implementation-focused reference for every structure below.
- **[Competitive Programmer's Handbook (Laaksonen, PDF)](https://cses.fi/book/book.pdf)** — free, covers every core structure; pair with the CSES problem set.

## 4. Searching & Sorting

- **[Sedgewick Algs4 — Sorting chapter](https://algs4.cs.princeton.edu/home/)** — merge/quick/heap/counting/radix with performance discussion.
- **[External Sorting — see §12 Vitter survey](https://www.ittc.ku.edu/~jsv/Papers/Vit.IO_surveyComputingSurveys.pdf)** — external merge-sort formal treatment.
- **[MIT 6.006 sorting lectures](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/)** — comparison-based lower bounds + non-comparison sorts.

## 5. Algorithm Design Paradigms

- **[Skiena — Algorithm Design Manual (companion)](https://www3.cs.stonybrook.edu/~algorith/book/programs/)** — every technique with real war stories.
- **[Skiena Lecture 11 — Dynamic Programming (PDF)](https://www3.cs.stonybrook.edu/~skiena/392/newlectures/week11.pdf)** — clean DP intro slides.
- **[Bird & de Moor — From Dynamic Programming to Greedy](http://www.cs.ox.ac.uk/people/richard.bird/online/BirdDeMoor93From.pdf)** — Oxford paper connecting DP and greedy formally.
- **[Competitive Programmer's Handbook (Laaksonen)](https://cses.fi/book/book.pdf)** — DP, greedy, backtracking, complete search chapters.

## 6. Graph Algorithms

- **[cp-algorithms — Graphs section](https://cp-algorithms.com/)** — BFS/DFS/SCC/MST/Dijkstra/Bellman-Ford/Floyd-Warshall implementations.
- **[cp-algorithms — Min-Cost Flow](https://cp-algorithms.com/graph/min_cost_flow.html)** — one of the best free flow tutorials.
- **[cp-algorithms — Assignment / Min-Flow](https://cp-algorithms.com/graph/Assignment-problem-min-flow.html)** — matching + flow reductions.
- **[Codeforces — Graph Potentials, Johnson's, MCMF](https://codeforces.com/blog/entry/95823)** — advanced flow tricks.
- **[HackerEarth — Shortest Path tutorials](https://www.hackerearth.com/practice/algorithms/graphs/shortest-path-algorithms/tutorial/)** — practice-oriented refresher.

## 7. String & Text Algorithms

- **[Stanford CS97SI — String Algorithms (Jaehyun Park)](https://web.stanford.edu/class/cs97si/10-string-algorithms.pdf)** — KMP, Z, suffix array in one lecture.
- **[Stanford CS97SI — Suffix Arrays](https://web.stanford.edu/class/cs97si/suffix-array.pdf)** — separate deep dive.
- **[String Algorithms & Data Structures — arXiv 0801.2378](https://arxiv.org/pdf/0801.2378)** — survey.
- **[Suffix Trees — Bioinformatics lecture notes](https://mpop.gitbook.io/bioinformatics-lecture-notes/string-indexing/introduction-to-suffix-trees)** — very readable intro.
- **[USACO Guide — String Searching](https://usaco.guide/adv/string-search)** — modern, code-first walkthrough.
- **[HackerEarth — String Searching](https://www.hackerearth.com/practice/algorithms/string-algorithm/string-searching/tutorial/)** — problem-focused.

## 8. Advanced Data Structures

- **[cp-algorithms — Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html)** — the canonical BIT tutorial online.
- **[Codeforces — Segment / Fenwick Range Queries (JS)](https://codeforces.com/blog/entry/139128)** — modern walkthrough with code.
- **[GeeksforGeeks — Binary Indexed Tree](https://www.geeksforgeeks.org/dsa/binary-indexed-tree-or-fenwick-tree-2/)** — quick reference.
- **[Dev.to — Segment Tree & Fenwick BIT](https://dev.to/rock_win_c053fa5fb2399067/blog-3-segment-tree-fenwick-tree-bit-range-queries-made-easy-3m62)** — friendly explainer.
- **[Erik Demaine — Advanced Data Structures 6.851 (MIT)](https://courses.csail.mit.edu/6.851/)** — persistent, succinct, cache-oblivious; the graduate reference.

## 9. Retrieval & Indexing Structures

- **[BK-Trees — Nick's Blog](http://blog.notdot.net/2007/4/Damn-Cool-Algorithms-Part-1-BK-Trees)** — classic explainer for edit-distance indexes.
- **[BK-Tree — Wikipedia](https://en.wikipedia.org/wiki/BK-tree)** — reference.
- **[talhasaruhan/fuzzy-search](https://github.com/talhasaruhan/fuzzy-search)** — trie / BK-tree code you can read.
- **[ANN indexes — Pinecone HNSW guide](https://www.pinecone.io/learn/series/faiss/hnsw/)** — best free HNSW explanation.
- **[FAISS](https://github.com/facebookresearch/faiss)** · **[hnswlib](https://github.com/nmslib/hnswlib)** · **[ScaNN](https://github.com/google-research/google-research/tree/master/scann)** — the four ANN libraries to know.

## 10. Optimization & Pruning

- **[Skiena — Algorithm Design Manual chapters on backtracking/pruning](https://www3.cs.stonybrook.edu/~algorith/book/programs/)** — practical pruning patterns.
- **[Competitive Programmer's Handbook — Complete search / pruning chapters](https://cses.fi/book/book.pdf)** — bitmask DP, early termination.
- **[Algorithmica — Optimization](https://en.algorithmica.org/hpc/)** — modern hand-optimized algorithms (branchless, SIMD, cache).

## 11. Memory & Cache Engineering

- **[Algorithmica — HPC book (online)](https://en.algorithmica.org/hpc/)** — best free modern resource for cache locality, SIMD, and data-oriented design.
- **[Erik Demaine — Cache-Oblivious Algorithms (BRICS)](https://erikdemaine.org/papers/BRICS2002/paper.pdf)** — foundational lecture notes.
- **[Brodal — Cache-Oblivious Algorithms & Data Structures](https://cs.au.dk/~gerth/papers/swat04invited.pdf)** — comprehensive survey.
- **[Frigo–Leiserson–Prokop–Ramachandran — Cache-Oblivious Algorithms (paper)](https://www.researchgate.net/publication/220390382_Cache-Oblivious_Algorithms)** — the original 1999 paper.

## 12. External-Memory & Massive-Data Algorithms

- **[Vitter — External Memory Algorithms & Data Structures (survey, ACM CS 2001)](https://www.ittc.ku.edu/~jsv/Papers/Vit.IO_surveyComputingSurveys.pdf)** — the canonical reference on out-of-core algorithms.
- **[Vitter survey (ResearchGate)](https://www.researchgate.net/publication/277551128_Vitter_JS_External_Memory_Algorithms_and_Data_Structures_Dealing_with_Massive_Data_ACM_Computing_Surveys_332_209-271)** — mirror.
- **[MIT 6.851 — Advanced Data Structures (external memory lectures)](https://courses.csail.mit.edu/6.851/)** — I/O model, B-trees, cache-oblivious.
- **[Andrew McGregor — Graph Stream Algorithms Survey](https://people.cs.umass.edu/~mcgregor/papers/13-graphsurvey.pdf)** — for graphs that don't fit in memory.

## 13. Parallel & Distributed Algorithms

- **[Guy Blelloch — Introduction to Parallel Algorithms (draft book, CMU)](https://www.cs.cmu.edu/~guyb/paralg/paralg/parallel.pdf)** — the modern free textbook.
- **[Blelloch & Maggs — Parallel Algorithms (CMU chapter)](https://www.cs.cmu.edu/~guyb/papers/BM04.pdf)** — concise reference.
- **[Blelloch — homepage & 15-210 course pointers](http://www.cs.cmu.edu/~guyb/)** — CMU parallel & sequential DS/algorithms.
- **[Simulating Parallel Algorithms in MapReduce (arXiv 1004.4708)](https://arxiv.org/pdf/1004.4708)** — bridges PRAM / BSP / MapReduce.
- **[Peter Sanders — KIT Algorithm Engineering group](https://ae.iti.kit.edu/english/sanders.php)** — for parallel/distributed algorithm engineering papers.

## 14. Approximation & Randomized Algorithms

- **[Pinecone — LSH Illustrated Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)** — best free LSH explainer.
- **[Datasketch — MinHash LSH docs](https://ekzhu.com/datasketch/lsh.html)** — Python library + underlying math.
- **[textreuse — MinHash & LSH (CRAN vignette)](https://cran.r-project.org/web/packages/textreuse/vignettes/textreuse-minhash.html)** — rigorous walk-through.
- **[Callidon/bloom-filters (GitHub)](https://github.com/Callidon/bloom-filters)** — clean reference impls of Bloom, HyperLogLog, Count-Min, MinHash.
- **[LSH & Bloom filters — Uni-Szeged notes (PDF)](https://www.inf.u-szeged.hu/~berendg/docs/dm/DM_lsh_en_pf.pdf)** — course-quality slides.
- **[Williamson & Shmoys — The Design of Approximation Algorithms (free PDF)](https://www.designofapproxalgs.com/book.pdf)** — the standard graduate textbook.
- **[Hochbaum — Approximation Algorithms for NP-Hard Problems (Berkeley PDF)](https://hochbaum.ieor.berkeley.edu/html/pub/Book:Approximation-Algorithms-for-NP-Hard-Problems.pdf)** — classic edited volume.

## 15. Algorithm Engineering & Benchmarking

- **[Algorithmica — Benchmarking chapter](https://en.algorithmica.org/hpc/profiling/benchmarking/)** — how to actually measure runtime honestly.
- **[Sanders (KIT) — Algorithm Engineering group](https://ae.iti.kit.edu/english/sanders.php)** — canonical papers on the AE methodology.
- **[MPI Benchmarking Revisited — arXiv 1505.07734](https://arxiv.org/pdf/1505.07734)** — experimental design + reproducibility for parallel systems.
- **[Algorithmic Profiling — PLDI 2012 (ACM)](https://dl.acm.org/doi/10.1145/2254064.2254074)** — how to profile at the algorithmic level, not just cycles.

## 16. Research & Theoretical Thinking

- **[Williamson–Shmoys — Design of Approximation Algorithms (free)](https://www.designofapproxalgs.com/book.pdf)** — reductions + approximation guarantees end-to-end.
- **[CMU 15-854B — Advanced Approximation Algorithms](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15854-f21/www/)** — graduate lecture notes.
- **[Dartmouth — Hardness of Approximation lecture](https://www.cs.dartmouth.edu/~deepc/Courses/S15/Notes/lec14-scribe.pdf)** — clean reduction examples.
- **[MPI-INF — Approximation Algorithms course](https://www.mpi-inf.mpg.de/departments/algorithms-complexity/teaching/winter15/approx)** — solid alt lecture set.
- **[Andrew McGregor — Streaming Algorithms course](https://people.cs.umass.edu/~mcgregor/book/book.html)** — sketching / AMS / streaming lower bounds.

## 17. Engineering Judgment — production references

These are what you actually reach for when the answer is "measure, don't guess":

- **[Algorithmica HPC book](https://en.algorithmica.org/hpc/)** — end-to-end: from Big-O to SIMD to cache to profiling.
- **[MIT 6.851 Advanced Data Structures](https://courses.csail.mit.edu/6.851/)** — trains the "can an index solve this?" instinct.
- **[Skiena — The Algorithm Design Manual](https://www3.cs.stonybrook.edu/~algorith/book/programs/)** — companion war stories: "which algorithm actually works in practice?"
- **[cp-algorithms.com](https://cp-algorithms.com/)** — implementation-ready, correct-by-default reference.
- **[Competitive Programmer's Handbook (Laaksonen)](https://cses.fi/book/book.pdf)** + **[CSES Problem Set](https://cses.fi/problemset/)** — drills judgment through problems.

---

## Suggested reading order

1. **MIT 6.006 videos + notes** — get foundations solid (§§1–5).
2. **CLRS + Skiena** as reference books alongside 6.006.
3. **cp-algorithms.com + CSES problem set** — build implementation fluency (§§3–8).
4. **MIT 6.046 lecture notes** — DP, greedy, randomization, NP (§§5, 14, 16).
5. **Algorithmica HPC book** — memory, cache, SIMD (§§11, 15).
6. **Vitter survey + Demaine cache-oblivious notes** — massive-data thinking (§12).
7. **Blelloch parallel-algorithms book** — parallelism (§13).
8. **Williamson–Shmoys** — approximation & reductions (§§14, 16).

## Naive → engineered — the DSA architect loop

Every problem walks through this loop; use these sources to shore up each step:

```
Naive → Correctness → Time → Space → Bottleneck →
Data Structure → Better Algorithm → Prune → Cache/IO →
Parallelism → Benchmark → Compare → Ship
```

- Steps 1–4: MIT 6.006 / CLRS / Skiena.
- Steps 5–7: cp-algorithms + MIT 6.046 + Algorithmica.
- Step 8: Demaine cache-oblivious notes, Vitter survey.
- Step 9: Blelloch parallel-algorithms book.
- Steps 10–13: Algorithmica benchmarking chapter + Sanders KIT papers.
