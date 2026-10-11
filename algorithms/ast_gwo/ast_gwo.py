"""
Binary AST-GWO (Adaptive Sigmoid Transfer Grey Wolf Optimizer) for
logic-synthesis script optimisation.

Contains:
    - Logging helpers (console/file logger, CSV run logger)
    - Leaders   – tracks the alpha/beta/delta trio
    - AstGWO    – the optimizer itself
    - CLI entrypoint (``python -m algorithms.ast_gwo.ast_gwo ...``)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

from .config import GWOConfig
from .evaluator import EvalCache, Evaluator
from .utils import evaluate_design


# ==========================================================
# Logging
# ==========================================================
_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)


def build_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger that writes to both
    the console and *log_file*.  Safe to call multiple times
    with the same *name*.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(_FORMATTER)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(_FORMATTER)
    logger.addHandler(fh)

    return logger


class RunLogger:
    """
    Writes two CSV files per run:
        iter.log  – one row per iteration (summary)
        pop.log   – one row per individual per iteration
    """

    ITER_HEADER = "iter,best_qor,best_improve,cache_hit,cache_miss\n"
    POP_HEADER  = "iter,idx,bits,qor\n"

    def __init__(self, logdir: str) -> None:
        self.iter_path = os.path.join(logdir, "iter.log")
        self.pop_path  = os.path.join(logdir, "pop.log")

        with open(self.iter_path, "w") as f:
            f.write(self.ITER_HEADER)

        with open(self.pop_path, "w") as f:
            f.write(self.POP_HEADER)

    # ----------------------------------------------------------
    def log_iter(
        self,
        t: int,
        best_qor: float,
        best_improve: float,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        with open(self.iter_path, "a") as f:
            f.write(
                f"{t},{best_qor},{best_improve},"
                f"{cache_hits},{cache_misses}\n"
            )

    # ----------------------------------------------------------
    def log_pop(
        self,
        t: int,
        pop,          # np.ndarray uint8 [N, dim]
        pop_qor,      # np.ndarray float [N]
    ) -> None:
        lines = []
        for i in range(len(pop)):
            bits_str = "".join(pop[i].astype(str))
            lines.append(f"{t},{i},{bits_str},{pop_qor[i]}\n")
        with open(self.pop_path, "a") as f:
            f.writelines(lines)


# ==========================================================
# Leader trio
# ==========================================================
class Leaders:
    """Tracks the three best individuals (alpha, beta, delta)."""

    def __init__(self) -> None:
        self.bits = [None, None, None]   # np.ndarray | None
        self.fit  = [np.inf, np.inf, np.inf]
        self.impr = [0.0, 0.0, 0.0]

    @property
    def alpha(self) -> Optional[np.ndarray]:
        return self.bits[0]

    @property
    def alpha_fit(self) -> float:
        return self.fit[0]

    @property
    def alpha_improve(self) -> float:
        return self.impr[0]

    def update(
        self,
        pop: np.ndarray,
        pop_fit: np.ndarray,
        pop_improve: np.ndarray,
    ) -> None:
        top_idx = np.argsort(pop_fit)[:3]

        if self.bits[0] is None:
            for rank, i in enumerate(top_idx):
                self.bits[rank] = pop[i].copy()
                self.fit[rank]  = pop_fit[i]
                self.impr[rank] = pop_improve[i]
            return

        pool_bits = np.vstack([np.array(self.bits), pop[top_idx]])
        pool_fit  = np.array(self.fit  + [pop_fit[i]     for i in top_idx])
        pool_impr = np.array(self.impr + [pop_improve[i] for i in top_idx])

        order = np.argsort(pool_fit)
        for rank in range(3):
            o = order[rank]
            self.bits[rank] = pool_bits[o].copy()
            self.fit[rank]  = pool_fit[o]
            self.impr[rank] = pool_impr[o]


# ==========================================================
# Binary GWO with adaptive sigmoid transfer (AST-GWO)
# ==========================================================
class AstGWO:
    """
    Binary Grey Wolf Optimizer with Adaptive Sigmoid Transfer.

    Usage::

        cfg = GWOConfig(design_name="adder", pop_size=50, dim=80, iters=100)
        gwo = AstGWO(cfg)
        best_bits, best_qor = gwo.run()
    """

    def __init__(self, cfg: GWOConfig) -> None:
        self.cfg = cfg

        if not os.path.exists(cfg.design_path):
            raise FileNotFoundError(f"Design not found: {cfg.design_path}")

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        logdir = os.path.join(
            cfg.results_dir,
            cfg.run_tag,
            cfg.design_name,
            str(cfg.seed),
        )
        os.makedirs(logdir, exist_ok=True)

        self._logger  = build_logger(
            f"AstGWO_{cfg.seed}",
            os.path.join(logdir, "run.log"),
        )
        self._run_log = RunLogger(logdir)

        # --------------------------------------------------
        # Reference evaluation
        # --------------------------------------------------
        self._logger.info("Evaluating reference design ...")
        self.ref_lut, self.ref_level = evaluate_design(
            design_file=cfg.design_path,
            bits=cfg.baseline_bits,
            lut_k=cfg.lut_k,
            abc_bin=cfg.abc_bin,
        )
        self._logger.info(f"REF  lut={self.ref_lut}  level={self.ref_level}")

        # --------------------------------------------------
        # Evaluator + cache
        # --------------------------------------------------
        self._cache = EvalCache(max_size=cfg.cache_max_size)
        self._evaluator = Evaluator(
            design=cfg.design_path,
            lut_k=cfg.lut_k,
            ref_lut=self.ref_lut,
            ref_level=self.ref_level,
            cache=self._cache,
            n_workers=cfg.n_workers,
            abc_bin=cfg.abc_bin,
        )

        # --------------------------------------------------
        # RNG + population
        # --------------------------------------------------
        self._rng = np.random.default_rng(cfg.seed)

        self.pop = self._rng.integers(
            0, 2,
            size=(cfg.pop_size, cfg.dim),
            dtype=np.uint8,
        )
        self.pop_fit     = np.full(cfg.pop_size, np.inf)
        self.pop_improve = np.zeros(cfg.pop_size)

        # --------------------------------------------------
        # Leaders + GWO state
        # --------------------------------------------------
        self._leaders = Leaders()
        self.a = 2.0
        self.k = cfg.k_min

        self._logger.info(
            f"CONFIG  design={cfg.design_name}  lut_k={cfg.lut_k}  "
            f"pop={cfg.pop_size}  dim={cfg.dim}  iters={cfg.iters}  "
            f"seed={cfg.seed}  k_min={cfg.k_min}  k_max={cfg.k_max}"
        )

    # =====================================================
    # Properties
    # =====================================================
    @property
    def best_bits(self) -> Optional[np.ndarray]:
        return self._leaders.alpha

    @property
    def best_fit(self) -> float:
        return self._leaders.alpha_fit

    @property
    def best_improve(self) -> float:
        return self._leaders.alpha_improve

    # =====================================================
    # Coefficient update
    # =====================================================
    def _update_coefficients(self, t: int) -> None:
        """Linearly decay a; advance adaptive sigmoid slope k."""
        self.a = 2.0 - 2.0 * t / self.cfg.iters
        self.k = self.cfg.k_min + (t / self.cfg.iters) * (self.cfg.k_max - self.cfg.k_min)

    # =====================================================
    # Position update  — mirrors original bGWO2.py exactly
    # =====================================================
    def _step(self) -> None:
        """Move the whole population one step toward the leaders."""

        x_alpha = self._leaders.bits[0]
        x_beta  = self._leaders.bits[1]
        x_delta = self._leaders.bits[2]

        r1_alpha = self._rng.random(self.pop.shape)
        r1_beta  = self._rng.random(self.pop.shape)
        r1_delta = self._rng.random(self.pop.shape)
        r2_alpha = self._rng.random(self.pop.shape)
        r2_beta  = self._rng.random(self.pop.shape)
        r2_delta = self._rng.random(self.pop.shape)

        A_alpha = 2 * self.a * r1_alpha - self.a
        A_beta  = 2 * self.a * r1_beta  - self.a
        A_delta = 2 * self.a * r1_delta - self.a
        C_alpha = 2 * r2_alpha
        C_beta  = 2 * r2_beta
        C_delta = 2 * r2_delta

        # Distance vectors  Eq. (3.1.4) - (3.1.6)
        D_alpha = np.abs(C_alpha * x_alpha - self.pop)
        D_beta  = np.abs(C_beta  * x_beta  - self.pop)
        D_delta = np.abs(C_delta * x_delta - self.pop)

        # Candidate positions  Eq. (3.1.7) - (3.1.9)
        X1 = x_alpha - A_alpha * D_alpha
        X2 = x_beta  - A_beta  * D_beta
        X3 = x_delta - A_delta * D_delta

        # Mean position  Eq. (3.1.10)
        X_mean = (X1 + X2 + X3) / 3.0

        # Adaptive sigmoid transfer  Eq. (3.4.1)
        S = 1.0 / (1.0 + np.exp(-self.k * (X_mean - 0.5)))

        # Binary update  Eq. (3.4.3)
        r = self._rng.random(self.pop.shape)
        self.pop = (r < S).astype(np.uint8)

    # =====================================================
    # Main loop
    # =====================================================
    def run(self) -> Tuple[np.ndarray, float]:
        """Run the optimisation and return (best_bits, best_qor)."""

        # Initial evaluation
        self.pop_fit, self.pop_improve, hits, misses = (
            self._evaluator.evaluate(self.pop)
        )
        self._leaders.update(self.pop, self.pop_fit, self.pop_improve)
        self._log(0, hits, misses)

        for t in range(1, self.cfg.iters + 1):

            self._step()
            self._update_coefficients(t)

            self.pop_fit, self.pop_improve, hits, misses = (
                self._evaluator.evaluate(self.pop)
            )
            self._leaders.update(self.pop, self.pop_fit, self.pop_improve)
            self._log(t, hits, misses)

            total    = hits + misses
            hit_rate = hits / total if total else 0.0

            self._logger.info(
                f"[Iter {t:03d}]  a={self.a:.3f}  k={self.k:6.2f}  "
                f"qor={self.best_fit:.6f}  improve={self.best_improve:.4f}%  "
                f"alpha={''.join(self._leaders.bits[0].astype(str))}  "
                f"cache hit={hits:02d} miss={misses:02d} rate={hit_rate:7.2%}"
            )

        return self._leaders.bits[0].copy(), self._leaders.alpha_fit

    # ----------------------------------------------------------
    def _log(self, t: int, hits: int, misses: int) -> None:
        self._run_log.log_iter(t, self.best_fit, self.best_improve, hits, misses)
        self._run_log.log_pop(t, self.pop, self.pop_fit)


# ==========================================================
# CLI entrypoint
# ==========================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary AST-GWO for logic synthesis optimisation"
    )

    # Required
    p.add_argument(
        "--design",
        required=True,
        metavar="NAME",
        help="Benchmark name, e.g. adder, bar, hyp …",
    )

    # Population / search
    p.add_argument("--pop",   type=int,   default=50)
    p.add_argument("--dim",   type=int,   default=80)
    p.add_argument("--iters", type=int,   default=200)

    # ABC / LUT
    p.add_argument("--lut_k", type=int,   default=6)
    p.add_argument("--abc",   type=str,   default="yosys-abc",
                   dest="abc_bin", metavar="ABC_BIN")

    # Algorithm hyper-parameters
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--k_min", type=float, default=6.0)
    p.add_argument("--k_max", type=float, default=14.0)

    # Parallelism
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Worker processes for parallel eval (default: auto)",
    )

    # Paths
    p.add_argument(
        "--benchmarks_dir",
        default="./benchmarks/epfl/arithmetic",
    )
    p.add_argument(
        "--results_dir",
        default="./results/runs/ast_gwo",
    )

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    cfg = GWOConfig(
        design_name=args.design,
        pop_size=args.pop,
        dim=args.dim,
        iters=args.iters,
        lut_k=args.lut_k,
        seed=args.seed,
        k_min=args.k_min,
        k_max=args.k_max,
        n_workers=args.workers,
        abc_bin=args.abc_bin,
        benchmarks_dir=args.benchmarks_dir,
        results_dir=args.results_dir,
    )

    gwo = AstGWO(cfg)
    best_bits, best_qor = gwo.run()

    print("\n===== FINAL RESULT =====")
    print("BEST BITS :", "".join(best_bits.astype(str)))
    print(f"BEST QoR  : {best_qor:.6f}")
    print(f"IMPROVE   : {gwo.best_improve:.2f} %")


if __name__ == "__main__":
    main(sys.argv[1:])
