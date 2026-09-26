"""Single source of truth for feature-stage parameters."""
from dataclasses import dataclass, asdict, field
from typing import Tuple
import json


@dataclass
class FeatureConfig:
    # ---- which families ----
    exact: bool = True
    token: bool = True
    ngram: bool = True
    address: bool = True
    cross_field: bool = True
    missingness: bool = True
    group_relative: bool = True
    edit_distance: bool = True           # measured free: cpdist does 120k pairs in 0.02s
    multilingual: bool = True

    # ---- parameters ----
    ngram_n: int = 3
    high_sim_threshold: float = 0.85     # for cross-field "high/low" state flags
    score_cutoff: int = 30               # rapidfuzz early exit -- multiples of speedup
    workers: int = -1                    # cpdist releases the GIL: real parallelism
    chunk_pairs: int = 1_000_000
    dtype: str = "float32"               # never float64 -- halves memory for free
    seed: int = 20260926

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def fingerprint(self) -> str:
        import hashlib
        return hashlib.sha1(self.to_json().encode()).hexdigest()[:12]
