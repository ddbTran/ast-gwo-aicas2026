from dataclasses import dataclass, field
from typing import Optional, List


# ==========================================================
# Default baseline genome (10 macros × 4 bits)
# ==========================================================
DEFAULT_BASELINE_BITS: List[int] = [
    0, 1, 1, 0,
    0, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 1, 0,
    0, 0, 0, 0,
    0, 0, 0, 1,
    0, 1, 1, 0,
    0, 0, 1, 1,
    0, 0, 0, 1,
    0, 1, 1, 0,
]


@dataclass
class GWOConfig:
    # ----------------------------------------------------------
    # Required
    # ----------------------------------------------------------
    design_name: str
    pop_size: int
    dim: int
    iters: int

    # ----------------------------------------------------------
    # Optional with defaults
    # ----------------------------------------------------------
    lut_k: int = 6
    seed: int = 42
    k_min: float = 5.0
    k_max: float = 15.0
    n_workers: Optional[int] = None
    abc_bin: str = "yosys-abc"
    benchmarks_dir: str = "./benchmarks/epfl/arithmetic"
    results_dir: str = "./results/ast_gwo"
    cache_max_size: int = 200_000

    # ----------------------------------------------------------
    # Baseline genome for reference evaluation
    # ----------------------------------------------------------
    baseline_bits: List[int] = field(
        default_factory=lambda: list(DEFAULT_BASELINE_BITS)
    )

    def __post_init__(self):
        if self.dim % 4 != 0:
            raise ValueError(f"dim must be a multiple of 4, got {self.dim}")
        if self.pop_size < 3:
            raise ValueError("pop_size must be >= 3 (need alpha, beta, delta)")
        if self.k_min <= 0 or self.k_max < self.k_min:
            raise ValueError("Require 0 < k_min < k_max")

    @property
    def design_path(self) -> str:
        return f"{self.benchmarks_dir}/{self.design_name}.blif"

    @property
    def run_tag(self) -> str:
        return (
            f"lut_k_{self.lut_k}"
            f"-pop_{self.pop_size}"
            f"-dim_{self.dim}"
            f"-iters_{self.iters}"
            f"-k_range_{self.k_min}_{self.k_max}"
        )
