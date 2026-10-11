# AST-GWO — Adaptive Sigmoid Transfer Binary Grey Wolf Optimizer

This repository contains the artifact for our AST-GWO paper presented at IEEE AICAS 2026.

## Repository layout

```text
ast-gwo-aicas2026/
├── README.md
├── algorithms/
│   └── ast_gwo/
│       ├── __init__.py
│       ├── ast_gwo.py      # Leaders, AstGWO, logging, CLI (main)
│       ├── config.py       # GWOConfig dataclass
│       ├── evaluator.py    # Evaluator, LRU cache, multiprocessing
│       ├── utils.py        # bit<->script codec, ABC invocation, fitness
│       └── test/
│           └── test_utils.py
├── benchmarks/
│   └── epfl/                # place EPFL arithmetic/random_control .blif here
├── scripts/
│   ├── run_single.sh        # one (design, seed) run
│   └── run_sweep.sh         # grid sweep over designs × seeds
└── requirements.txt
```

## Requirements

- Python 3.9+
- [`yosys-abc`](https://github.com/YosysHQ/yosys) (or any ABC build) on your
  `PATH`, or pass `--abc /path/to/abc`
- EPFL benchmark suite `.blif` files under `benchmarks/epfl/...`

```bash
pip install -r requirements.txt
```

## Quick start

Single run:

```bash
bash scripts/run_single.sh --design adder
bash scripts/run_single.sh --design bar --pop 50 --dim 80 --iters 50 --seed 0
```

Equivalent direct invocation:

```bash
python -m algorithms.ast_gwo.ast_gwo --design adder --pop 50 --dim 80 --iters 200
```

Sweep over multiple designs and seeds:

```bash
# default sweep (all designs, seeds 0-4)
bash scripts/run_sweep.sh

# custom designs/seeds, 4 parallel jobs
bash scripts/run_sweep.sh --designs "adder bar hyp" --seeds "0 1 2" --jobs 4

# dry run — print the commands without executing
bash scripts/run_sweep.sh --dry-run
```

Results (per-iteration and per-individual CSV logs, plus a run log) are
written under `results/ast_gwo/<run_tag>/<design>/<seed>/`.

## Tests

```bash
pytest algorithms/ast_gwo/test/
```

The ABC-dependent test is skipped automatically if `yosys-abc` or the
benchmark files aren't available.

## Citation

If you use this code, please cite the accompanying AICAS 2026 paper.
