from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from .utils import evaluate_design, compute_fitness


# ==========================================================
# Shared worker payload (must be picklable for multiprocessing)
# ==========================================================
class EvalTask(NamedTuple):
    design: str
    lut_k: int
    ref_lut: float
    ref_level: float
    bits: List[int]
    abc_bin: str


def _worker(task: EvalTask) -> Tuple[float, float]:
    """Top-level function so ProcessPoolExecutor can pickle it."""
    lut, level = evaluate_design(
        design_file=task.design,
        bits=task.bits,
        lut_k=task.lut_k,
        abc_bin=task.abc_bin,
    )
    return compute_fitness(task.ref_lut, task.ref_level, lut, level)


# ==========================================================
# LRU cache
# ==========================================================
class EvalCache:
    """
    Fixed-capacity LRU cache keyed on bytes(bits).
    Evicts the least-recently-used entry when full.
    """

    def __init__(self, max_size: int = 200_000) -> None:
        self._max_size = max_size
        self._store: OrderedDict[bytes, Tuple[float, float]] = OrderedDict()

    # ----------------------------------------------------------
    def get(self, key: bytes) -> Optional[Tuple[float, float]]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)          # mark as recently used
        return self._store[key]

    def put(self, key: bytes, value: Tuple[float, float]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)   # evict LRU entry
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ==========================================================
# Evaluator
# ==========================================================
class Evaluator:
    """
    Evaluates a population matrix against an ABC design.

    Attributes exposed after each call to ``evaluate``:
        pop_qor      – (pop_size,) QoR scores
        pop_improve  – (pop_size,) improvement percentages
    """

    # Minimum tasks before spawning a process pool
    _PARALLEL_THRESHOLD = 4

    def __init__(
        self,
        design: str,
        lut_k: int,
        ref_lut: float,
        ref_level: float,
        cache: EvalCache,
        n_workers: Optional[int] = None,
        abc_bin: str = "yosys-abc",
    ) -> None:
        self.design = design
        self.lut_k = lut_k
        self.ref_lut = ref_lut
        self.ref_level = ref_level
        self.cache = cache
        self.n_workers = n_workers
        self.abc_bin = abc_bin

    # ----------------------------------------------------------
    def evaluate(
        self, pop: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Evaluate every row of *pop* (uint8, shape [N, dim]).

        Returns:
            pop_qor      – shape (N,)  QoR per individual
            pop_improve  – shape (N,)  improve % per individual
            cache_hits   – int
            cache_misses – int
        """
        n = pop.shape[0]
        pop_qor = np.full(n, np.inf)
        pop_improve = np.zeros(n)

        tasks: List[EvalTask] = []
        task_indices: List[int] = []
        task_keys: List[bytes] = []

        cache_hits = 0
        cache_misses = 0

        # --------------------------------------------------
        # Cache lookup
        # --------------------------------------------------
        for i in range(n):
            key = bytes(pop[i])
            cached = self.cache.get(key)
            if cached is not None:
                pop_qor[i], pop_improve[i] = cached
                cache_hits += 1
            else:
                tasks.append(
                    EvalTask(
                        design=self.design,
                        lut_k=self.lut_k,
                        ref_lut=self.ref_lut,
                        ref_level=self.ref_level,
                        bits=pop[i].tolist(),
                        abc_bin=self.abc_bin,
                    )
                )
                task_indices.append(i)
                task_keys.append(key)
                cache_misses += 1

        if not tasks:
            return pop_qor, pop_improve, cache_hits, cache_misses

        # --------------------------------------------------
        # Parallel or serial evaluation
        # --------------------------------------------------
        if (
            len(tasks) < self._PARALLEL_THRESHOLD
            or self.n_workers == 1
        ):
            results = [_worker(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                results = list(pool.map(_worker, tasks))

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------
        for idx, key, (qor, improve) in zip(task_indices, task_keys, results):
            self.cache.put(key, (qor, improve))
            pop_qor[idx] = qor
            pop_improve[idx] = improve

        return pop_qor, pop_improve, cache_hits, cache_misses
