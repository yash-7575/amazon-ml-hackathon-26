"""Convert the long candidate_pairs.tsv (one row per pair) to the wide format
the submission scorer requires (one row per S1 entity, comma-joined candidates).

Long input (written by blocker_v1.CandidateWriter):
    s1_entity_id\tcandidate_entity_id\tsource\tscore

Wide output (per PROJECT_OVERVIEW.md §0.5 and validate_submission.py):
    source1_entity_id\tcandidate_entity_ids
    S1-...\tS2-...,S3-...,...
    S1-...\t                          <- required for S1 with zero candidates

Rules the official validator enforces (rejection, not score penalty):
- Exact header.
- Every S1 in test_source1.tsv present, exactly once.
- No S1- ids in the candidate column; only S2-/S3-.
- No duplicate ids within a list; no duplicate S1 rows.
- Tab-separated, UTF-8.

Usage:
    PYTHONPATH=. python -m blocking.to_wide \\
        --long blocking/candidate_pairs.tsv \\
        --s1   /path/to/test_source1.tsv \\
        --out  output/candidate_pairs.tsv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import defaultdict

csv.field_size_limit(10**9)


def read_required_s1(path: str) -> list[str]:
    """Return S1 ids from source1.tsv, preserving file order. Deterministic."""
    ids: list[str] = []
    with open(path, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        try:
            id_col = header.index('entity_id')
        except ValueError:
            print(f'ERROR: {path} header {header!r} has no "entity_id" column',
                  file=sys.stderr)
            sys.exit(2)
        for row in r:
            if len(row) <= id_col:
                continue
            ids.append(row[id_col])
    return ids


def group_long(path: str) -> dict[str, list[str]]:
    """Read long TSV; return {s1_id: [cand_id, ...]} preserving input order and
    deduping within each S1's list."""
    grouped: dict[str, list[str]] = defaultdict(list)
    seen_per_s1: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        try:
            s1_col = header.index('s1_entity_id')
            cand_col = header.index('candidate_entity_id')
        except ValueError:
            print(f'ERROR: {path} header {header!r} missing s1_entity_id / '
                  f'candidate_entity_id columns', file=sys.stderr)
            sys.exit(2)
        for row in r:
            if len(row) <= max(s1_col, cand_col):
                continue
            s1 = row[s1_col]
            cand = row[cand_col]
            if not s1 or not cand:
                continue
            if cand in seen_per_s1[s1]:
                continue
            seen_per_s1[s1].add(cand)
            grouped[s1].append(cand)
    return grouped


def write_wide(out_path: str, required_s1: list[str],
               grouped: dict[str, list[str]]) -> tuple[int, int, int]:
    """Write wide TSV. Return (n_rows, n_empty, n_extra_in_long)."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    required_set = set(required_s1)
    n_empty = 0
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n',
                       quoting=csv.QUOTE_NONE, escapechar=None)
        w.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for s1 in required_s1:
            cands = grouped.get(s1, [])
            if not cands:
                n_empty += 1
            w.writerow([s1, ','.join(cands)])
    extra = [s for s in grouped.keys() if s not in required_set]
    return len(required_s1), n_empty, len(extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--long', required=True,
                    help='Long-format candidate_pairs.tsv from blocker_v1.')
    ap.add_argument('--s1', required=True,
                    help='source1.tsv for the split being submitted (defines '
                         'the required S1 id set and their row order).')
    ap.add_argument('--out', required=True,
                    help='Wide-format output path.')
    args = ap.parse_args()

    print(f'[to_wide] reading required S1 ids from {args.s1} ...', flush=True)
    required = read_required_s1(args.s1)
    print(f'[to_wide]   {len(required):,} required S1 entities')

    print(f'[to_wide] grouping long file {args.long} ...', flush=True)
    grouped = group_long(args.long)
    print(f'[to_wide]   {len(grouped):,} distinct S1 entities in long file')

    n_rows, n_empty, n_extra = write_wide(args.out, required, grouped)
    print(f'[to_wide] wrote {n_rows:,} rows to {args.out}')
    print(f'[to_wide]   {n_empty:,} with empty candidate list')
    print(f'[to_wide]   {n_extra:,} S1 in long file but NOT in test set '
          f'(should be 0 for a proper run; non-zero means the blocker '
          f'wrote pairs for S1 entities that will be silently dropped)')
    if n_extra > 0:
        print(f'[to_wide] WARNING: {n_extra} extra S1 dropped', file=sys.stderr)


if __name__ == '__main__':
    main()
