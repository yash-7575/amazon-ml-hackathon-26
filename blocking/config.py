"""Config for blocker_v1. All parameters live here — no scattered literals.

Rationale is inline. Every default is justified by either:
- a specific measured finding in eda/FINDINGS.md, or
- a source from subagents/candidate-generation-research-engineer/sources/README.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class BlockerConfig:
    # --- data ---
    data_root: str = '/home/sagemaker-user/amazon-ml-hackathon-26/dataset/'
    split: str = 'train'  # 'train' or 'test'
    output_path: str = '/home/yash/dev/clg/amlc-2026/blocking/candidate_pairs.tsv'

    # --- partitioning ---
    # FINDING 2: matches never cross country. Free 3x reduction.
    partition_field: str = 'country'
    # Process partitions smallest-first so we surface bugs before hitting the big ones.
    partition_order: str = 'ascending_size'  # or 'descending_size' | 'natural'
    only_partitions: Optional[tuple[str, ...]] = None  # e.g. ('ireland',) for dry runs

    # --- P1: normalized-exact on name core ---
    # ~22% of true pairs (FINDING 3). Near-zero CPU cost.
    p1_enabled: bool = True

    # --- P2: char-3gram TF-IDF on name ---
    # Primary engine. Ceiling per FINDING 3 is 90.98% (share >=2 name char-3grams).
    # No max_df: v0 failed at 45.53% recall precisely because of a hard df cut
    # (PROJECT_CONTEXT "Known negative result"). IDF weighting is the correct answer.
    p2_enabled: bool = True
    p2_analyzer: str = 'char_wb'
    p2_ngram_lo: int = 3
    p2_ngram_hi: int = 3
    p2_min_df: int = 2         # drops pure typos / hapax n-grams; safe.
    p2_max_df: float = 1.0     # NO CAP. IDF handles common n-grams naturally.
    p2_sublinear_tf: bool = True
    p2_top_k: int = 50         # per S1 per source (S2 and S3 kept separate here).
    # Chunked matmul: how many S1 rows to score against the full S2 matrix at once.
    # Peak memory of a chunk = chunk_rows * n_S2 * 8 bytes (dense score) BEFORE top-K.
    # For n_S2 = 200k we want chunk_rows ~ 1024 -> ~1.6GB float64 temp.
    p2_chunk_rows: int = 1024
    p2_score_dtype: str = 'float32'  # halves the temp matrix

    # --- P3: address TF-IDF (word 1-2 grams) ---
    # Recovers the ~15% zero-name-overlap pairs (96.3% of them have addr Jaccard > 0.2).
    # Word n-grams are more meaningful on addresses than char n-grams (numbers matter).
    p3_enabled: bool = True
    p3_analyzer: str = 'word'
    p3_ngram_lo: int = 1
    p3_ngram_hi: int = 2
    p3_min_df: int = 2
    p3_max_df: float = 1.0
    p3_sublinear_tf: bool = True
    p3_top_k: int = 30
    p3_chunk_rows: int = 1024
    p3_score_dtype: str = 'float32'
    # Records with empty address contribute NO document to the P3 index but still
    # emit a candidate list from P1/P2/P4 — this is why P3 is a UNION, not a filter.

    # --- P4: transliterated-name key for Devanagari records ---
    # 4.17% of true pairs are cross-script (FINDING / §5). We add a SECOND document
    # per Devanagari S2/S3 record: the unidecode/ITRANS romanization. That row now
    # lives in both the Devanagari char-3gram vector space AND the Latin one, so
    # an ASCII S1 query can retrieve it via P2 without a special code path at query.
    # We ALSO add these romanized names to the P1 exact-match index.
    p4_enabled: bool = True

    # --- Rescore + union ---
    # After union we rescore every candidate by w_name*cos_name + w_addr*cos_addr
    # and keep the global top-K. Weights favour name because addresses have more
    # missing / dirty data (3.3% empty; addr Jaccard p10=0.30 vs name p10=0.174).
    rescore_w_name: float = 0.6
    rescore_w_addr: float = 0.4
    # K=50 measured 94.30% on India smoke (98.27% union) — 3.97pp gap from
    # rescore ranking, not blocking. K sweep {50,100,150} on the same smoke:
    # K=100 -> 95.37%, K=150 -> 98.19% (within 0.08pp of union). Picked K=150
    # as the smallest K within ~1pp of union. Cost: 3x feature-stage pairs vs
    # K=50. Union caps around ~200/S1 so K=150 is nearly all of it. See
    # results/smoke_test_india_2000_60k_ksweep.log.
    final_top_k: int = 150   # per S1 entity, after union + rescore

    # --- Determinism / I/O ---
    seed: int = 20260926
    write_header: bool = True
    # Score threshold below which we drop a candidate BEFORE writing.
    # 0.0 = keep everything the top-K included. Do NOT tune against ground truth.
    min_output_score: float = 0.0

    # --- Measurement (only used by measure.py) ---
    measure_sample_s1: int = 20000        # stratified S1 sample size for --measure
    measure_k_sweep: tuple[int, ...] = (10, 20, 50, 100, 200)

    def to_dict(self) -> dict:
        return asdict(self)
