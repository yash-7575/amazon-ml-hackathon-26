"""P1-only fast-track TEST submission for Amazon ML Challenge 2026.

Why this file exists: full-fidelity blocking (P2+P3 sparse top-K over
1.73M S1 x 10M S2/S3) was MEASURED at ~36h wall on this 12-core box
(memory-bandwidth bound; see D:/Amazon_challenge_ML/submission_work/ETA_REPORT.md).
This track runs ONLY the P1 normalized-exact pass (~25 min), then the SAME
Stage-2 pipeline as training: identical 54-feature `compute_pair_features`,
the trained XGBoost model at threshold 0.90, and the assignment rules
(conflict resolution, count cap, singleton cutoff, argmax-always).

P1 matching semantics are IDENTICAL to `blocking.blocker_v1.block_p1_exact`:
exact match on `name_core` plus transliterated-alias keys built with the same
`normalize` functions. The container is slimmer (inv index only, no
PartitionData) for memory; `verify_slim_vs_real()` proves row-identical output
on a sample before launch. `blocking/` itself is NOT modified.

Stages (each resumable; outputs under WORK_DIR):
  block  : per-country parallel P1 -> long shards -> <ck>.long.tsv
  score  : feeder joins raw strings -> pool computes features + XGBoost proba
           -> per-country scored_top11.csv (s1,cand,prob,blocker_score)
  assign : per-country conflict resolution + rules -> wide per-country files
  final  : concat wides in fixed country order -> validate

Usage (from repo root):
  python -m training.xgboost.submit_p1 block  --country india
  python -m training.xgboost.submit_p1 score  --country india
  python -m training.xgboost.submit_p1 assign --country india
  python -m training.xgboost.submit_p1 final
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import json
from collections import defaultdict

csv.field_size_limit(10 ** 9)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WORK_DIR = r"D:\Amazon_challenge_ML\submission_work"
OUT_DIR = os.path.join(WORK_DIR, "output")
TEST_DIR = r"D:\Amazon_challenge_ML\Amazon-ML-dataset\student_resource\dataset\test"
MODEL_PKL = os.path.join(REPO, "training", "xgboost", "outputs", "models",
                         "xgboost_model.pkl")

THRESHOLD = 0.90          # trained best threshold (best_threshold.json)
SINGLETON_CUTOFF = 0.50   # max_prob below this -> emit EMPTY (see docstring)
MAX_PER_ENTITY = 11       # train ground-truth range 0-11
COUNTRIES = ("france", "us", "india")  # fixed concat order (ascending size)

N_BLOCK_WORKERS = 10
N_SCORE_WORKERS = 10
BLOCK_SHARDS = 40
SCORE_S1_PER_TASK = 2000


def _wpath(name: str) -> str:
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(WORK_DIR, name)


def log(msg: str) -> None:
    print(f"[submit_p1] {msg}", flush=True)


# ---------------------------------------------------------------------------
# BLOCK stage: slim P1 index (same VALUES as PartitionData/block_p1_exact)
# ---------------------------------------------------------------------------
def build_slim_inv(s2_path: str, s3_path: str, ck: str):
    """Build {name_core_key: [(src, cand_id), ...]} exactly like
    block_p1_exact's inverse index (same normalize calls, same alias rule)."""
    from blocking.normalize import country_key, name_core, name_tokens, \
        base_normalize, has_devanagari, transliterate_deva
    from blocking.io_utils import read_tsv

    inv: dict[str, list] = defaultdict(list)
    n_t = 0
    for path, src in ((s2_path, "S2"), (s3_path, "S3")):
        for row in read_tsv(path):
            if country_key(row.get("country", "")) != ck:
                continue
            eid = row["entity_id"]
            name = row.get("business_name", "") or ""
            nc = name_core(name)
            if nc:
                inv[nc].append((src, eid))
            if has_devanagari(name):
                t = base_normalize(transliterate_deva(name))
                nct = " ".join(sorted(name_tokens(t, drop_suffix=True))) \
                    if t else ""
                if nct and nct != nc:
                    inv[nct].append((src, eid))
            n_t += 1
    return dict(inv), n_t


def s1_cores_for_country(s1_path: str, ck: str):
    """Return ([s1_ids in file order], {s1_id: name_core}) for one country."""
    from blocking.normalize import country_key, name_core
    from blocking.io_utils import read_tsv

    ids: list[str] = []
    cores: dict[str, str] = {}
    for row in read_tsv(s1_path):
        if country_key(row.get("country", "")) != ck:
            continue
        eid = row["entity_id"]
        ids.append(eid)
        cores[eid] = name_core(row.get("business_name", "") or "")
    return ids, cores


_BLOCK_INV = None


def _block_init(s2_path: str, s3_path: str, ck: str) -> None:
    global _BLOCK_INV
    inv, n_t = build_slim_inv(s2_path, s3_path, ck)
    _BLOCK_INV = inv
    print(f"[block-worker pid={os.getpid()}] inv keys={len(inv):,} "
          f"target_rows={n_t:,}", flush=True)


def _block_task(args) -> str:
    """Process one S1 shard -> shard long file. Returns shard path."""
    from blocking.io_utils import CandidateWriter

    shard_idx, s1_items, out_path = args
    w = CandidateWriter(out_path, write_header=True)
    n_hit = 0
    for s1_id, nc in s1_items:
        if not nc:
            continue
        hits = _BLOCK_INV.get(nc, ())
        if not hits:
            continue
        rows = [(cid, src, 1.0) for (src, cid) in hits]
        w.write_s1(s1_id, rows)
        n_hit += 1
    w.close()
    return f"{out_path}|{n_hit}|{w.rows_written}"


def cmd_block(country: str) -> None:
    import multiprocessing as mp

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")
    t_all = time.time()
    log(f"country={country}: streaming S1 ids + name_cores ...")
    ids, cores = s1_cores_for_country(s1_path, country)
    log(f"country={country}: S1={len(ids):,}")
    chunks = [ids[i::BLOCK_SHARDS] for i in range(BLOCK_SHARDS)]
    # NOTE: round-robin sharding balances generic-name hotspots better than
    # contiguous ranges; final long file is re-sorted to S1 file order below.
    tasks = []
    for i, ch in enumerate(chunks):
        out = _wpath(f"p1_{country}_shard{i:02d}.tsv")
        if os.path.exists(out):
            log(f"shard {i:02d} exists, skipping")
            continue
        tasks.append((i, [(sid, cores[sid]) for sid in ch], out))
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(N_BLOCK_WORKERS,
                      initializer=_block_init,
                      initargs=(s2_path, s3_path, country)) as pool:
            for res in pool.imap_unordered(_block_task, tasks):
                log(f"shard done: {res}")
    # concat shards in S1 file order
    order = {sid: k for k, sid in enumerate(ids)}
    rows: dict[str, list] = {}
    for i in range(BLOCK_SHARDS):
        p = _wpath(f"p1_{country}_shard{i:02d}.tsv")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if len(row) < 4:
                    continue
                rows.setdefault(row[0], []).append(
                    (row[1], row[2], row[3]))
    long_path = _wpath(f"p1_{country}.long.tsv")
    n_pairs = 0
    with open(long_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["s1_entity_id", "candidate_entity_id", "source",
                    "score"])
        for sid in ids:
            for (cid, src, sc) in rows.get(sid, []):
                w.writerow([sid, cid, src, sc])
                n_pairs += 1
    log(f"country={country}: long={long_path} pairs={n_pairs:,} "
        f"per_S1={n_pairs/max(len(ids),1):.2f} wall={time.time()-t_all:.0f}s")


# ---------------------------------------------------------------------------
# Verification: slim-P1 vs real block_p1_exact on a sample
# ---------------------------------------------------------------------------
def verify_slim_vs_real(country: str = "france", sample: int = 3000) -> None:
    from blocking.io_utils import stream_country_rows
    from blocking.blocker_v1 import PartitionData, load_partition, \
        block_p1_exact

    t0 = time.time()
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")
    s1 = PartitionData("s1", "S1")
    n = 0
    for row in stream_country_rows(s1_path, country):
        s1.add(row)
        n += 1
        if n >= sample:
            break
    s2 = load_partition(s2_path, country, "S2", "s2")
    s3 = load_partition(s3_path, country, "S3", "s3")
    real = block_p1_exact(s1, [s2, s3])
    real_sets = {s1.ids[i]: {(c, s) for (s, c, _sc, _t) in lst}
                 for i, lst in real.items()}
    inv, _ = build_slim_inv(s2_path, s3_path, country)
    from blocking.normalize import name_core
    need = {r["entity_id"] for r in stream_country_rows(s1_path, country)}
    # recompute S1 cores for sample only
    match = mismatch = 0
    checked = 0
    for i, sid in enumerate(s1.ids):
        nc = s1.name_core[i]
        slim = {(c, s) for (s, c) in inv.get(nc, ())} if nc else set()
        if slim == real_sets.get(sid, set()):
            match += 1
        else:
            mismatch += 1
            if mismatch <= 3:
                log(f"MISMATCH {sid}: slim={len(slim)} real="
                    f"{len(real_sets.get(sid, set()))}")
        checked += 1
    log(f"VERIFY slim-vs-real: checked={checked:,} match={match:,} "
        f"mismatch={mismatch} wall={time.time()-t0:.0f}s")
    if mismatch:
        raise SystemExit(f"VERIFY FAILED: {mismatch} mismatches")


# ---------------------------------------------------------------------------
# SCORE stage: feeder + pool (SAME compute_pair_features + trained model)
# ---------------------------------------------------------------------------
_SCORE_MODEL = None
_SCORE_FEATS = None


def _score_init(model_pkl: str) -> None:
    global _SCORE_MODEL, _SCORE_FEATS
    import pickle
    with open(model_pkl, "rb") as f:
        md = pickle.load(f)
    _SCORE_MODEL = md["model"]
    _SCORE_FEATS = list(md["feature_names"])


def _score_task(task) -> str:
    """task = (rows, out_path): rows = [(s1_id, s1n, s1a, s1c, cand_id,
    cn, ca, cc, bscore), ...]. Compute SAME 54 features, predict, write
    per-S1 top-11 lines. Returns summary string."""
    import numpy as np
    import xgboost as xgb
    from training.xgboost.features import compute_pair_features
    from training.xgboost.config import get_config

    rows, out_path = task
    cfg = get_config()
    feats = _SCORE_FEATS
    mat = np.empty((len(rows), len(feats)), dtype=np.float32)
    for i, r in enumerate(rows):
        d = compute_pair_features(r[1], r[2], r[3], r[5], r[6], r[7], cfg)
        mat[i, :] = [d[k] for k in feats]
    dm = xgb.DMatrix(mat, feature_names=feats)
    prob = _SCORE_MODEL.predict(
        dm, iteration_range=(0, _SCORE_MODEL.best_iteration + 1))
    # group rows by S1 (input order is S1-contiguous)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        i = 0
        n_s1 = 0
        while i < len(rows):
            j = i
            while j < len(rows) and rows[j][0] == rows[i][0]:
                j += 1
            seg_p = prob[i:j]
            order = np.argsort(-seg_p, kind="stable")[:11]
            for k in order:
                r = rows[i + int(k)]
                w.writerow([r[0], r[4], f"{float(seg_p[int(k)]):.6f}",
                            f"{r[8]:.6f}"])
            n_s1 += 1
            i = j
    return f"{out_path}|s1={n_s1}|pairs={len(rows)}"


def _load_str_dicts(country: str):
    """Raw (name, addr, country) dicts for one country (feeder-side)."""
    from blocking.normalize import country_key
    from blocking.io_utils import read_tsv

    s1d: dict[str, tuple] = {}
    for row in read_tsv(os.path.join(TEST_DIR, "test_source1.tsv")):
        if country_key(row.get("country", "")) != country:
            continue
        s1d[row["entity_id"]] = (row.get("business_name", "") or "",
                                 row.get("business_address", "") or "",
                                 row.get("country", "") or "")
    candd: dict[str, tuple] = {}
    for name in ("test_source2.tsv", "test_source3.tsv"):
        for row in read_tsv(os.path.join(TEST_DIR, name)):
            if country_key(row.get("country", "")) != country:
                continue
            candd[row["entity_id"]] = (row.get("business_name", "") or "",
                                      row.get("business_address", "") or "",
                                      row.get("country", "") or "")
    return s1d, candd


def cmd_score(country: str) -> None:
    import multiprocessing as mp

    t_all = time.time()
    long_path = _wpath(f"p1_{country}.long.tsv")
    scored_path = _wpath(f"scored_{country}.tsv")
    if os.path.exists(scored_path):
        log(f"{scored_path} exists, skipping score (delete to redo)")
        return
    log(f"country={country}: loading string dicts ...")
    s1d, candd = _load_str_dicts(country)
    log(f"country={country}: s1_dict={len(s1d):,} cand_dict={len(candd):,}")

    # stream long file, group by S1, batch into tasks.
    # NOTE: at most MAX_OUTSTANDING tasks are queued at any time. Without
    # backpressure the feeder races ahead and the pool task queue holds GBs
    # of pickled rows (observed as a ~10x slowdown on `us` from swapping).
    from collections import deque
    MAX_OUTSTANDING = 2 * N_SCORE_WORKERS
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(N_SCORE_WORKERS, initializer=_score_init,
                    initargs=(MODEL_PKL,))
    pending: list = []
    cur_id = None
    cur_rows: list = []
    task_idx = 0
    outstanding = deque()

    def flush_task():
        nonlocal task_idx
        if not pending:
            return
        out = _wpath(f"score_{country}_part{task_idx:04d}.tsv")
        task_idx += 1
        if os.path.exists(out):
            pending.clear()
            return
        outstanding.append(
            pool.apply_async(_score_task, ((pending.copy(), out),)))
        pending.clear()
        while len(outstanding) > MAX_OUTSTANDING:
            log(f"score task done: {outstanding.popleft().get()}")

    n_pairs = n_miss = 0
    with open(long_path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for row in r:
            if len(row) < 4:
                continue
            s1_id, cid = row[0], row[1]
            try:
                bscore = float(row[3])
            except ValueError:
                bscore = 1.0
            s = s1d.get(s1_id)
            c = candd.get(cid)
            if s is None or c is None:
                n_miss += 1
                continue
            if cur_id is None:
                cur_id = s1_id
            if s1_id != cur_id:
                pending.extend(cur_rows)
                cur_rows = []
                cur_id = s1_id
                # count S1s in pending
                if len(pending) >= SCORE_S1_PER_TASK * 8:
                    flush_task()
            cur_rows.append((s1_id, s[0], s[1], s[2], cid, c[0], c[1],
                             c[2], bscore))
            n_pairs += 1
    pending.extend(cur_rows)
    flush_task()
    pool.close()
    while outstanding:
        log(f"score task done: {outstanding.popleft().get()}")
    pool.join()
    # concat parts in order
    with open(scored_path, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["s1_entity_id", "cand_entity_id", "prob",
                    "blocker_score"])
        for i in range(task_idx):
            p = _wpath(f"score_{country}_part{i:04d}.tsv")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8", newline="") as f:
                for line in f:
                    out.write(line)
    log(f"country={country}: scored={scored_path} pairs={n_pairs:,} "
        f"missing_join={n_miss:,} wall={time.time()-t_all:.0f}s")


# ---------------------------------------------------------------------------
# ASSIGN stage: conflict resolution + count + singleton + argmax
# ---------------------------------------------------------------------------
def cmd_assign(country: str) -> dict:
    import pandas as pd
    import numpy as np

    t_all = time.time()
    scored_path = _wpath(f"scored_{country}.tsv")
    df = pd.read_csv(scored_path, sep="\t", dtype={"s1_entity_id": str,
                                                   "cand_entity_id": str})
    df["prob"] = df["prob"].astype(float)
    log(f"country={country}: scored rows={len(df):,} "
        f"s1={df['s1_entity_id'].nunique():,}")

    # per-S1 shortlist: empty if max < SINGLETON_CUTOFF else
    # argmax + {prob >= THRESHOLD}, cap MAX_PER_ENTITY by prob desc
    df = df.sort_values(["s1_entity_id", "prob"], ascending=[True, False])
    keep_rows = []
    maxprob = df.groupby("s1_entity_id")["prob"].max()
    eligible = set(maxprob[maxprob >= SINGLETON_CUTOFF].index)
    g = df[df["s1_entity_id"].isin(eligible)].groupby("s1_entity_id",
                                                      sort=False)
    for s1_id, grp in g:
        top = grp.iloc[0]  # argmax (prob desc)
        picks = [top]
        rest = grp.iloc[1:]
        rest = rest[rest["prob"] >= THRESHOLD]
        picks.extend([r for _, r in rest.iterrows()])
        picks = sorted(picks, key=lambda r: -r["prob"])[:MAX_PER_ENTITY]
        for r in picks:
            keep_rows.append((s1_id, r["cand_entity_id"], float(r["prob"]),
                              float(r["blocker_score"])))
    short = pd.DataFrame(keep_rows, columns=["s1_entity_id",
                                             "cand_entity_id", "prob",
                                             "blocker_score"])
    log(f"country={country}: shortlisted={len(short):,}")

    # (a) conflict resolution: one candidate -> highest-prob claimant
    # (tie: prob desc, blocker_score desc, s1 asc -> deterministic)
    short = short.sort_values(["prob", "blocker_score", "s1_entity_id"],
                              ascending=[False, False, True])
    winners = short.drop_duplicates(subset=["cand_entity_id"], keep="first")
    won = set(zip(winners["s1_entity_id"], winners["cand_entity_id"]))
    log(f"country={country}: after_conflict={len(winners):,} "
        f"dropped={(len(short)-len(winners)):,}")

    # entities left with nothing after conflict -> single-level backfill from
    # their top-11 pool (unclaimed, prob >= SINGLETON_CUTOFF), else empty
    claimed = set(winners["cand_entity_id"])
    have = set(winners["s1_entity_id"])
    pool = df[~df["cand_entity_id"].isin(claimed)]
    pool = pool[pool["prob"] >= SINGLETON_CUTOFF]
    pool = pool.sort_values(["s1_entity_id", "prob"],
                            ascending=[True, False])
    backfill = pool.drop_duplicates(subset=["s1_entity_id"], keep="first")
    backfill = backfill[~backfill["s1_entity_id"].isin(have)]
    log(f"country={country}: backfilled={len(backfill):,}")
    final = pd.concat([winners[["s1_entity_id", "cand_entity_id"]],
                       backfill[["s1_entity_id", "cand_entity_id"]]],
                      ignore_index=True)

    # required S1 list (file order) for this country
    from blocking.normalize import country_key
    from blocking.io_utils import read_tsv
    req: list[str] = []
    for row in read_tsv(os.path.join(TEST_DIR, "test_source1.tsv")):
        if country_key(row.get("country", "")) == country:
            req.append(row["entity_id"])
    match_map: dict[str, list] = defaultdict(list)
    for s1_id, cid in zip(final["s1_entity_id"], final["cand_entity_id"]):
        if cid not in match_map[s1_id]:
            match_map[s1_id].append(cid)

    m_path = _wpath(f"matching_{country}.tsv")
    n_empty = 0
    with open(m_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in req:
            cands = match_map.get(sid, [])
            if not cands:
                n_empty += 1
            w.writerow([sid, ",".join(cands)])

    # wide candidates from long file (same required order)
    cand_map: dict[str, list] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    with open(_wpath(f"p1_{country}.long.tsv"), encoding="utf-8",
              newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            if row[1] not in seen[row[0]]:
                seen[row[0]].add(row[1])
                cand_map[row[0]].append(row[1])
    c_path = _wpath(f"candidates_{country}.tsv")
    n_cempty = 0
    with open(c_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for sid in req:
            cands = cand_map.get(sid, [])
            if not cands:
                n_cempty += 1
            w.writerow([sid, ",".join(cands)])
    stats = {"country": country, "required": len(req),
             "matched_empty": n_empty, "cand_empty": n_cempty,
             "matched_nonempty": len(req) - n_empty,
             "wall_s": round(time.time() - t_all, 1)}
    with open(_wpath(f"assign_{country}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log(f"country={country}: {stats}")
    return stats


# ---------------------------------------------------------------------------
# FINAL stage: concat + validate
# ---------------------------------------------------------------------------
def cmd_final() -> None:
    import subprocess

    t_all = time.time()
    m_out = os.path.join(OUT_DIR, "matching_results.tsv")
    c_out = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    with open(m_out, "w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for ck in COUNTRIES:
            p = _wpath(f"matching_{ck}.tsv")
            with open(p, encoding="utf-8", newline="") as f:
                next(f, None)
                for line in f:
                    out.write(line)
    with open(c_out, "w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for ck in COUNTRIES:
            p = _wpath(f"candidates_{ck}.tsv")
            with open(p, encoding="utf-8", newline="") as f:
                next(f, None)
                for line in f:
                    out.write(line)
    log(f"concat done wall={time.time()-t_all:.0f}s")
    cmd = [sys.executable,
           os.path.join(REPO, "..", "Amazon-ML-dataset", "student_resource",
                        "utils", "validate_submission.py")]
    # resolve validator path robustly
    v = os.path.join(r"D:\Amazon_challenge_ML\Amazon-ML-dataset",
                     "student_resource", "utils", "validate_submission.py")
    r = subprocess.run([sys.executable, v, "--matching", m_out,
                        "--candidate", c_out, "--test-dir", TEST_DIR],
                       capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    with open(os.path.join(WORK_DIR, "validator_output.txt"), "w",
              encoding="utf-8") as f:
        f.write(r.stdout + "\n" + r.stderr + f"\nreturncode={r.returncode}\n")
    log(f"validator returncode={r.returncode}")
    if r.returncode != 0:
        raise SystemExit("VALIDATION FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("block", "score", "assign", "final",
                                      "verify"))
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    if args.stage == "verify":
        verify_slim_vs_real(args.country or "france")
    elif args.stage == "block":
        cmd_block(args.country or "france")
    elif args.stage == "score":
        cmd_score(args.country or "france")
    elif args.stage == "assign":
        cmd_assign(args.country or "france")
    elif args.stage == "final":
        cmd_final()


if __name__ == "__main__":
    main()
