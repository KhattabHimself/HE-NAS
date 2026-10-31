# HE-NAS

Evolutionary co-optimization of CKKS homomorphic-encryption parameters and CNN
architectures using Grammatical Evolution (GE).

Private inference under CKKS forces two coupled choices: the network (depth,
activations) and the encryption parameters (ring dimension N, modulus chain).
A deeper network needs a longer modulus chain, and a longer chain is only
128-bit secure at a larger N. HE-NAS searches both at once from a single integer
genome, so every candidate it returns is accurate, fits its own level budget,
and meets the 128-bit security bound.

## How it works

A BNF grammar encodes the CKKS parameters (N, and a modulus chain of 2 to 12
primes) together with a CNN of 2 to 5 conv layers (filters, kernel size,
polynomial activation, optional pooling). Each conv is followed by BatchNorm and
an HE-friendly polynomial activation, since CKKS cannot evaluate ReLU:

- `square`: x², the CryptoNets activation. Cheap in depth.
- `relu4`: the degree-4 least-squares ReLU fit over [-6, 6] from Ishiyama et al.
  (2020). More accurate, twice the depth.

Fitness is test accuracy, penalized when the network's depth exceeds the chain's
level budget (`soft` exponential penalty or `hard` death penalty). Configurations
below 128-bit security always score 0.

### Depth model (measured, not derived)

Level consumption was measured against TenSEAL rather than taken from a
circuit-depth count:

```
levels consumed = sum of activation polynomial degrees
budget          = len(modulus_chain) - 2
```

So `square` costs 2 levels per layer and `relu4` costs 4. Per layer this splits
into 1 level for the convolution (a ciphertext x plaintext product still raises
the scale and forces a rescale) plus d-1 for a degree-d activation, because
TenSEAL's `polyval` evaluates naively. A conv-free count underestimates the
requirement by a factor of two at these degrees. The full write-up is in
[docs/DEPTH_MEASUREMENT.md](docs/DEPTH_MEASUREMENT.md) and the raw measurements
are in [measurements/](measurements/).

## Repository layout

```
HE-NAS/
├── main.py                  entry point: CLI and experiment drivers
├── config.py                paths, experiment settings, baseline definitions
├── grammar.py               BNF grammar, genome decoding, genome freezing
├── ckks.py                  depth budget and security estimation
├── fitness.py               datasets, CNN builder, training, fitness function
├── evolution.py             GE engine and NSGA-II (self-contained)
├── analysis.py              statistics (Mann-Whitney, Cohen's d, A12) and selftest
├── profiling.py             runtime / GPU profiling helpers
├── champions.py             retrain the best config of each experiment at full scale
├── validate_ckks.py         run evolved configs under real CKKS (TenSEAL)
├── validate_depth_model.py  measure whole-network level consumption (TenSEAL)
├── probe_decomposition.py   measure per-primitive level cost (conv vs polyval)
├── probe_ops.py             measure level cost of individual TenSEAL ops
├── submit.sh                SLURM job script (Digital Research Alliance of Canada)
├── requirements.txt         torch, torchvision, numpy, scipy
├── requirements-he.txt      adds tenseal for the validation scripts
├── docs/DEPTH_MEASUREMENT.md
└── measurements/            TenSEAL depth measurements the model is calibrated on
```

`main.py` builds a config from the CLI flags and calls a driver. The driver runs
`evolution.py`, which decodes genomes with `grammar.py` and scores them with
`fitness.py`, which in turn checks the CKKS constraints in `ckks.py`. Results are
written as JSON and compared with `analysis.py`.

These directories are created at runtime and are not tracked:

```
data/          downloaded datasets (MNIST / Fashion-MNIST / CIFAR-10)
experiments/   per-run logs: experiments/<name>/run_seed<seed>/results.json
results/       per-experiment summaries and results/stats/ comparisons
```

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt        # search only
pip install -r requirements-he.txt     # plus TenSEAL for the CKKS validation scripts
```

For a CUDA GPU, install the CUDA build of PyTorch first (see pytorch.org).

## Quick start

```bash
python main.py selftest      # seconds, no GPU: grammar, decoder, depth, security, NSGA-II
python main.py smoke         # tiny end-to-end GE run, a few minutes
```

## Experiments

Every experiment is a subcommand of `main.py`:

| Command | What it runs |
|---|---|
| `ge` | Single-objective GE (the main method) |
| `random` | Random-search baseline, same budget and fitness |
| `sequential --which ckks_first` | B1: CKKS parameters frozen, architecture evolves |
| `sequential --which arch_first` | B2: architecture frozen, CKKS parameters evolve |
| `expert` | B3 (properly provisioned) and B4 (naive) hand-designed configs |
| `ablation` | 2x2 ablation of penalty type x elitism |
| `nsga2` | Multi-objective NSGA-II (accuracy, security margin, depth headroom) |
| `stats` | Mann-Whitney U, Cohen's d and Vargha-Delaney A12 between two summaries |
| `profile` | Runtime / GPU profiling report |
| `smoke` | Tiny end-to-end run to check the pipeline |
| `selftest` | Unit checks, no GPU |
| `download` | Pre-download datasets |

Each command accepts `--help` for its full flag list, and `--smoke` to shrink any
experiment to a quick sanity check.

```bash
python main.py ge     --dataset cifar10 --seeds 30
python main.py expert --dataset cifar10
python main.py nsga2  --dataset cifar10 --seeds 5
python main.py stats  results/ge_cifar10_summary.json results/b3_expert_provisioned_summary.json
```

All runs resume: re-running skips seeds that already wrote `results.json`
(`--fresh` forces a recompute). A long sweep can be split with `--seed-base`,
for example `--seeds 10 --seed-base 42` followed by `--seeds 10 --seed-base 52`.

### Key defaults

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `mnist` | `mnist` / `fashion_mnist` / `cifar10` |
| `--seeds` / `--seed-base` | 30 / 42 | seeds 42..71 |
| `--population` / `--generations` | 20 / 15 | NSGA-II defaults to 40 / 25 |
| `--train-samples` / `--test-samples` | 10000 / 500 | |
| `--epochs` | 30 | |
| `--penalty` | `soft` | `hard` = death penalty |
| `--smoke` | off | 2 seeds, pop 4, 2 gens, 200 samples, 2 epochs |

## After the search

```bash
python champions.py                          # retrain each experiment's best config on the full dataset
python validate_ckks.py --demo               # check the TenSEAL path
python validate_ckks.py experiments/ge_cifar10/run_seed42/results.json --top-k 1
python validate_depth_model.py               # ~10 min, CPU: whole-network level consumption
python probe_decomposition.py                # ~1 min, CPU: conv vs activation level cost
```

`validate_ckks.py` runs the evolved activation chain on encrypted data under the
config's own CKKS parameters and reports feasibility, approximation error and
latency. It does not perform full encrypted image classification, which TenSEAL
cannot do cleanly for multi-layer CNNs.

## Running on a SLURM cluster (DRAC)

Compute nodes on DRAC have no internet access, so set up on a login node first:

```bash
module load StdEnv/2023 python/3.11
virtualenv --no-download venv && source venv/bin/activate
pip install --no-index torch torchvision numpy scipy
python main.py selftest
python main.py download --dataset cifar10
```

Set `--account` (and optionally `--mail-user`) at the top of `submit.sh`, then:

```bash
sbatch submit.sh                                   # the full study, cheapest experiments first
sbatch --time=01:00:00 submit.sh smoke             # or a single experiment
sbatch --time=72:00:00 submit.sh ge --dataset cifar10
```

`submit.sh` runs each seed as its own process, spread across the allocated GPUs,
then makes one aggregation pass to write the summary. Resubmitting after a
timeout continues where the job stopped.

Rough cost on one modern GPU, 30 seeds, MNIST: GE, random and each sequential
baseline take about 75 GPU-hours; the ablation about 4x GE; expert about 1
GPU-hour; NSGA-II (5 seeds) about 55 GPU-hours. CIFAR-10 is roughly 10x MNIST per
evaluation.

## Reference

Ishiyama, T., Suzuki, T., and Yamana, H. (2020). Highly accurate CNN inference
using approximate activation functions over homomorphic encryption. IEEE
International Conference on Big Data.
