# AST-GWO — Adaptive Sigmoid Transfer Binary Grey Wolf Optimizer

This repository contains the artifact for our AST-GWO paper presented at IEEE AICAS 2026.

## Citation

If you use this code, please cite the accompanying IEEE AICAS 2026 paper.

## Repository layout

```text
algorithms/ast_gwo/   # Optimizer, configuration, ABC evaluation, and CLI
benchmarks/epfl/     # EPFL benchmarks (Git submodule)
scripts/
  run_single.sh     # Run one design and seed
  run_sweep.sh      # Sweep designs and seeds
  visualizer.py     # Plot convergence and population diversity
requirements.txt   # Python dependencies
results/           # Generated logs (ignored by Git)
```

## Usage

Requires Python 3.11+, Git, Bash, and `yosys-abc` on `PATH`.
Run commands from the repository root:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
git submodule update --init --recursive

# Single run (200 iterations, seed 42)
bash scripts/run_single.sh --design adder

# Custom run
bash scripts/run_single.sh --design bar --pop 50 --dim 80 --iters 50 --seed 0

# Default sweep (adder, seeds 0–4, 50 iterations)
bash scripts/run_sweep.sh

# Custom sweep
bash scripts/run_sweep.sh --designs "adder bar hyp" --seeds "0 1 2" --jobs 4

# Preview sweep commands (requires ABC and benchmarks)
bash scripts/run_sweep.sh --dry-run

# Generate the dataset expected by the visualizer, then plot
bash scripts/run_sweep.sh --designs "adder multiplier"
python scripts/visualizer.py
```

Both run scripts default to slope range 5–15; override with `--k_min` and
`--k_max`. Use `--abc /path/to/abc` for another ABC executable,
`--workers N` for evaluation processes per run, and `--results_dir PATH`
for a custom output directory. For random/control designs, pass
`--benchmarks_dir benchmarks/epfl/random_control`.

Single-run logs default to `results/ast_gwo`; sweep logs default to
`results/runs/ast_gwo/conference`. Each run contains `iter.log`, `pop.log`,
and `run.log` under `<results_dir>/<run_tag>/<design>/<seed>/`.
