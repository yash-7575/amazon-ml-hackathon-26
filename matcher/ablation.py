"""Ablation and per-feature diagnostics.

No feature is kept without a number. Feature intuition is wrong often enough that this
is not optional.

marginal(f) = metric(all) - metric(all - f)

Standalone performance is misleading: two features can each look strong alone and be
mutually redundant, or each look weak and be jointly essential. Only the marginal number
distinguishes those cases.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np


def single_feature_auc(col: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC of one feature, NaN-safe. Used for SEPARATION screening and as a
    LEAKAGE alarm (anything above ~0.95 alone is a suspect)."""
    m = np.isfinite(col)
    if m.sum() < 10:
        return float("nan")
    c, yy = col[m], y[m]
    if len(np.unique(yy)) < 2:
        return float("nan")
    order = np.argsort(c, kind="stable")
    ranks = np.empty(len(c), dtype=np.float64)
    ranks[order] = np.arange(1, len(c) + 1)
    n_pos = float((yy == 1).sum()); n_neg = float((yy == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks_statistic(col: np.ndarray, y: np.ndarray) -> float:
    """Max gap between positive and negative CDFs -- a number for distribution overlap."""
    m = np.isfinite(col)
    a, b = np.sort(col[m & (y == 1)]), np.sort(col[m & (y == 0)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / a.size
                               - np.searchsorted(b, grid, "right") / b.size)))


def feature_report(M: np.ndarray, cols: Sequence[str], y: np.ndarray,
                   hard_mask: np.ndarray | None = None) -> List[dict]:
    """Per-feature health. Run this BEFORE fitting any model -- it is fast and it catches
    bad features for free.

    hard_mask restricts to the hard cases (e.g. name_sim > 0.9). A feature that separates
    overall but NOT among near-misses is not earning its place where a precision-weighted
    metric is decided -- this is the most informative version of the analysis.
    """
    out = []
    for k, c in enumerate(cols):
        col = M[:, k].astype(np.float64)
        finite = col[np.isfinite(col)]
        row = {
            "feature": c,
            "variance": round(float(np.var(finite)), 6) if finite.size else 0.0,
            "pct_nan": round(100.0 * (1 - finite.size / max(col.size, 1)), 2),
            "auc": round(single_feature_auc(col, y), 4),
            "ks": round(ks_statistic(col, y), 4),
            "constant": bool(finite.size == 0 or np.ptp(finite) == 0),
        }
        if hard_mask is not None and hard_mask.sum() > 20:
            row["auc_hard"] = round(single_feature_auc(col[hard_mask], y[hard_mask]), 4)
        out.append(row)
    return out


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision. The right default under class imbalance -- ROC-AUC flatters
    models by rewarding performance on the abundant negative class."""
    order = np.argsort(-score, kind="stable")
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    prec = tp / np.maximum(tp + fp, 1)
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    rec = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def ablate(fit_predict, M: np.ndarray, cols: Sequence[str], y: np.ndarray,
           groups: Dict[str, Sequence[str]] | None = None,
           metric=pr_auc) -> List[dict]:
    """Subtractive ablation. `fit_predict(X_train, y_train, X_valid) -> scores`.

    Ablate by FAMILY first (6-8 fits), then drill into the family that matters.
    Fifty individual ablations is fifty fits and usually not the decision you need.
    """
    idx = {c: k for k, c in enumerate(cols)}
    n = M.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(n * 0.7)
    tr, va = perm[:cut], perm[cut:]

    base = metric(y[va], fit_predict(M[tr], y[tr], M[va]))
    results = [{"removed": "<none>", "metric": round(base, 5), "marginal": 0.0}]

    targets = groups or {c: [c] for c in cols}
    for name, members in targets.items():
        drop = [idx[m] for m in members if m in idx]
        if not drop:
            continue
        keep = [k for k in range(M.shape[1]) if k not in set(drop)]
        if not keep:
            continue
        s = metric(y[va], fit_predict(M[tr][:, keep], y[tr], M[va][:, keep]))
        results.append({"removed": name, "metric": round(s, 5),
                        "marginal": round(base - s, 5)})
    return sorted(results, key=lambda r: -r["marginal"])


def report_features(rows: List[dict]) -> str:
    has_hard = any("auc_hard" in r for r in rows)
    head = f"{'feature':<28}{'var':>10}{'%nan':>8}{'AUC':>8}{'KS':>8}"
    if has_hard:
        head += f"{'AUC_hard':>10}"
    L = ["PER-FEATURE DIAGNOSTICS", "-" * len(head), head]
    for r in sorted(rows, key=lambda x: -(x["auc"] if np.isfinite(x["auc"]) else 0)):
        line = (f"{r['feature']:<28}{r['variance']:>10.4f}{r['pct_nan']:>8.1f}"
                f"{r['auc']:>8.3f}{r['ks']:>8.3f}")
        if has_hard:
            ah = r.get("auc_hard", float('nan'))
            line += f"{ah:>10.3f}" if np.isfinite(ah) else f"{'-':>10}"
        if r["constant"]:
            line += "   <- CONSTANT: drop (zero information)"
        elif np.isfinite(r["auc"]) and r["auc"] > 0.95:
            line += "   <- LEAKAGE SUSPECT: investigate"
        L.append(line)
    return "\n".join(L)


def report_ablation(rows: List[dict], noise: float = 0.002) -> str:
    L = ["ABLATION (subtractive)", "-" * 58,
         f"{'removed':<28}{'metric':>10}{'marginal':>12}"]
    for r in rows:
        line = f"{r['removed']:<28}{r['metric']:>10.4f}{r['marginal']:>+12.4f}"
        if r["removed"] != "<none>" and abs(r["marginal"]) < noise:
            line += "   <- within noise: candidate to drop"
        L.append(line)
    L.append(f"\nnoise threshold {noise}: report fold/seed variation and do not claim")
    L.append("gains inside it. That is how feature sets accumulate junk.")
    return "\n".join(L)
