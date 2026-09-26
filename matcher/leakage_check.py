"""Automated leakage detection.

The master test:
    Can this feature be computed, identically, for a pair whose label I do not know?

The sharper version:
    If I deleted the ground-truth file, would the feature matrix be byte-identical?

Leakage produces excellent validation numbers and a collapse on the real test set. These
checks are cheap; run all of them before believing any result.
"""
from __future__ import annotations
import ast
import os
import re
from typing import Dict, List, Sequence
import numpy as np

from .ablation import single_feature_auc

LABELY = re.compile(r"(ground.?truth|_label|labels?\b|is_match|y_true|train_gt|target)", re.I)


def check_no_label_imports(module_path: str,
                           forbidden: Sequence[str] = ("label", "ground_truth", "gt", "truth")) -> dict:
    """STRUCTURAL check: the extractor must not be able to see labels.

    A rule in a comment gets forgotten; an import boundary does not.
    """
    src = open(module_path, encoding="utf-8").read()
    tree = ast.parse(src)
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    bad = [m for m in imported for f in forbidden if f in m.lower()]

    # also flag label-ish identifiers appearing outside comments/docstrings
    code_lines = [l for l in src.splitlines()
                  if not l.strip().startswith("#")]
    hits = sorted({m.group(0) for l in code_lines for m in [LABELY.search(l)] if m})
    return {"module": os.path.basename(module_path), "imports": imported,
            "forbidden_imports": bad, "labelish_identifiers": hits,
            "pass": not bad}


def check_suspicious_auc(M: np.ndarray, cols: Sequence[str], y: np.ndarray,
                         threshold: float = 0.95) -> dict:
    """A single feature above ~0.95 AUC alone is a leak suspect.

    Legitimate features are informative, not decisive. When one feature dramatically
    outranks everything else on an ER task, suspect leakage before celebrating.
    """
    sus = []
    for k, c in enumerate(cols):
        a = single_feature_auc(M[:, k].astype(np.float64), y)
        if np.isfinite(a) and (a > threshold or a < 1 - threshold):
            sus.append({"feature": c, "auc": round(a, 4)})
    return {"suspects": sus, "pass": not sus}


def check_split_distributions(M_tr: np.ndarray, M_te: np.ndarray,
                              cols: Sequence[str], max_shift: float = 0.15) -> dict:
    """Compare per-feature distributions across splits.

    A large shift means leakage OR distribution shift -- both need explaining. This is the
    check that catches GROUP-FORMATION MISMATCH: train with K=50 and test with K=100 and
    rank/group_size/ratio all shift, with no label involved, so no other check flags it.
    """
    shifted = []
    for k, c in enumerate(cols):
        a = M_tr[:, k].astype(np.float64); b = M_te[:, k].astype(np.float64)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 20 or b.size < 20:
            continue
        qs = [10, 25, 50, 75, 90]
        pa, pb = np.percentile(a, qs), np.percentile(b, qs)
        scale = max(np.std(np.concatenate([a, b])), 1e-9)
        d = float(np.max(np.abs(pa - pb)) / scale)
        if d > max_shift:
            shifted.append({"feature": c, "normalized_shift": round(d, 3)})
    return {"shifted": shifted, "pass": not shifted}


def check_shuffle(fit_predict, M: np.ndarray, y: np.ndarray, metric,
                  seed: int = 0, tol: float = 0.05) -> dict:
    """Permute the labels and refit. Performance MUST collapse to the base rate.

    If it does not, some feature encodes the label through another route. The most
    thorough of these checks.
    """
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    perm = rng.permutation(n)
    cut = int(n * 0.7)
    tr, va = perm[:cut], perm[cut:]
    real = metric(y[va], fit_predict(M[tr], y[tr], M[va]))
    y_shuf = rng.permutation(y)
    fake = metric(y_shuf[va], fit_predict(M[tr], y_shuf[tr], M[va]))
    base = float((y == 1).mean())
    return {"real": round(real, 4), "shuffled": round(fake, 4),
            "base_rate": round(base, 4),
            "pass": bool(fake <= base + tol)}


def run_all(module_path: str, M: np.ndarray, cols: Sequence[str], y: np.ndarray,
            M_te: np.ndarray | None = None) -> str:
    L = ["LEAKAGE AUDIT", "=" * 58]
    r1 = check_no_label_imports(module_path)
    L.append(f"[{'PASS' if r1['pass'] else 'FAIL'}] no label imports in {r1['module']}")
    if r1["forbidden_imports"]:
        L.append(f"       forbidden: {r1['forbidden_imports']}")
    if r1["labelish_identifiers"]:
        L.append(f"       label-ish identifiers to review: {r1['labelish_identifiers']}")

    r2 = check_suspicious_auc(M, cols, y)
    L.append(f"[{'PASS' if r2['pass'] else 'WARN'}] no single feature above 0.95 AUC")
    for s in r2["suspects"]:
        L.append(f"       SUSPECT {s['feature']}  AUC={s['auc']}")

    if M_te is not None:
        r3 = check_split_distributions(M, M_te, cols)
        L.append(f"[{'PASS' if r3['pass'] else 'WARN'}] feature distributions stable across splits")
        for s in r3["shifted"]:
            L.append(f"       SHIFTED {s['feature']}  {s['normalized_shift']}")

    L += ["", "Still to do by hand:",
          "  - delete the ground-truth file and confirm the extractor produces identical bytes",
          "  - confirm ONE frozen extractor is called by both train and test paths",
          "  - confirm retrieval config (K, min_df, analyzer) is frozen across splits",
          "  - run check_shuffle() with your model's fit_predict"]
    return "\n".join(L)


if __name__ == "__main__":
    print(run_all.__doc__ or "")
    r = check_no_label_imports(os.path.join(os.path.dirname(__file__), "features.py"))
    print(f"features.py label-import check: {'PASS' if r['pass'] else 'FAIL'}")
    print(f"  imports: {r['imports']}")
    # synthetic demo: one honest feature, one leaked feature
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.15).astype(int)
    honest = y * 0.3 + rng.random(4000) * 0.7
    leaked = y.astype(float) + rng.random(4000) * 0.01
    M = np.vstack([honest, leaked]).T.astype(np.float32)
    out = check_suspicious_auc(M, ["honest_sim", "leaked_from_label"], y)
    print(f"\nAUC scan: {'PASS' if out['pass'] else 'FAIL (as designed)'}")
    for s in out["suspects"]:
        print(f"  SUSPECT {s['feature']}  AUC={s['auc']}")
    assert any(s["feature"] == "leaked_from_label" for s in out["suspects"])
    assert not any(s["feature"] == "honest_sim" for s in out["suspects"])
    print("\nleakage_check self-test: PASS")
