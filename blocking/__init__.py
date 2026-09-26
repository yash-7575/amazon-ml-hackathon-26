"""Candidate-generation stage for the Amazon ML Challenge 2026 BER task.

Modules:
- normalize:    Unicode-safe text normalization + Devanagari-safe punctuation strip.
- config:       Frozen BlockerConfig dataclass with per-stage parameters.
- io_utils:     Streaming TSV I/O, country partitioning, deterministic writer.
- index:        TF-IDF index construction and chunked top-K search.
- blocker_v1:   The pipeline orchestrator (P1..P4 + rescore + write).
- measure:      Ground-truth-aware evaluator (recall, RR, per-stage, per-country).
"""
