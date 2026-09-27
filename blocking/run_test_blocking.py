"""TEST blocking run: country-partitioned union -> LONG candidate file.

Same frozen code path as training (blocker_v1.run_partition); only the config
differs (split='test', final_top_k, backend). Wide conversion is a separate
step (blocking.to_wide) over the REQUIRED S1 list so empty entities survive.

Usage (from repo root):
    python -m blocking.run_test_blocking --out output/candidates_test.long.tsv
    python -m blocking.run_test_blocking --out ... --only france
    python -m blocking.run_test_blocking --out ... --final-top-k 50 --backend topn --threads 10
"""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace

from .config import BlockerConfig
from .io_utils import CandidateWriter, country_row_counts, peak_rss_mb
from .blocker_v1 import _partition_order, run_partition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Long-format output TSV path')
    ap.add_argument('--final-top-k', type=int, default=50)
    ap.add_argument('--backend', choices=('topn', 'dense'), default='topn')
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--only', default=None, help='Single country_key (e.g. france)')
    ap.add_argument('--order', default='ascending_size',
                    choices=('ascending_size', 'descending_size', 'natural'))
    args = ap.parse_args()

    cfg = BlockerConfig(split='test', output_path=args.out,
                        partition_order=args.order)
    cfg = replace(cfg, final_top_k=args.final_top_k,
                  topk_backend=args.backend, topk_threads=args.threads)

    d = cfg.data_root + 'test/'
    s1_path = d + 'test_source1.tsv'
    print(f'[run_test] scanning S1 country counts...', flush=True)
    counts = country_row_counts(s1_path)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'    {v:>10,}  {k}')
    order = _partition_order(counts, cfg.partition_order)
    if args.only is not None:
        order = [args.only] if args.only in counts else []
        if not order:
            print(f'[run_test] partition {args.only!r} not found', file=sys.stderr)
            sys.exit(2)
    print(f'[run_test] order={order} final_top_k={cfg.final_top_k} '
          f'backend={cfg.topk_backend} threads={cfg.topk_threads}', flush=True)

    t_all = time.time()
    writer = CandidateWriter(cfg.output_path, write_header=cfg.write_header)
    total_pairs = 0
    try:
        for ck in order:
            t0 = time.time()
            print(f'\n[run_test] === partition {ck!r} '
                  f'(S1 rows = {counts[ck]:,}) ===', flush=True)
            stats = run_partition(ck, cfg, writer)
            total_pairs += stats.get('pairs_written', 0)
            for k, v in stats.items():
                print(f'    {k}: {v}')
            print(f'    partition_wall_sec: {time.time()-t0:.1f}', flush=True)
    finally:
        writer.close()
    print(f'\n[run_test] DONE rows_written={writer.rows_written:,} '
          f'total_wall_sec={time.time()-t_all:.1f} '
          f'peak_rss_mib={peak_rss_mb():.1f}', flush=True)


if __name__ == '__main__':
    main()
