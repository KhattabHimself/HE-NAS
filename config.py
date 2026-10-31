"""Paths and experiment configuration.

Everything is relative to this folder, so the project relocates cleanly
(laptop, cloud VM, DRAC scratch space). Cfg replaces the old YAML config
system: one dataclass, filled from CLI flags in main.py.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"


def ensure_dirs():
    for d in (DATA_DIR, EXPERIMENTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    """Settings for one experiment (defaults match the original production runs)."""
    name: str = ""
    dataset: str = "cifar10"           # mnist | fashion_mnist | cifar10
    n_seeds: int = 30
    seed_base: int = 42
    population: int = 20
    generations: int = 15
    crossover_prob: float = 0.7
    mutation_prob: float = 0.05
    tournament_size: int = 3
    elitism: bool = True
    penalty: str = "soft"            # soft | hard
    train_samples: int = 10000
    test_samples: int = 500
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64
    resume: bool = True              # skip seeds that already wrote results.json
    write_summary: bool = True       # off for parallel single-seed workers (see submit.sh)

    @property
    def seeds(self):
        return [self.seed_base + i for i in range(self.n_seeds)]


# Applied when --smoke is passed: shrink everything for a fast sanity run.
SMOKE_OVERRIDES = dict(n_seeds=2, population=4, generations=2,
                       train_samples=200, test_samples=100, epochs=2)


# --------------------------------------------------------------------------- #
# Fixed baseline definitions (B1-B4).
# --------------------------------------------------------------------------- #
# Level costs are the CALIBRATED ones (ckks.py): a layer costs its activation's
# polynomial degree, so square = 2 and relu4 = 4. Budget = len(chain) - 2. Every
# chain below is checked against the 128-bit ceiling for its own N -- note that
# 14 primes is 600 bits, which is insecure at N=16384 (438) and forces N=32768.
# That coupling is the point of the study, and it binds the baselines too.
#
# B1/B2 are the sequential (one-sided) baselines about search *ordering*; they
# stay activation-agnostic (square) so they isolate the co-design effect.
#
# B1 pins the encryption half the way a cryptographer would if asked first,
# before seeing any architecture: a middle-of-the-road 10-prime chain (budget 8)
# with generous security headroom. It admits up to 4 square layers or 2 relu4 --
# a real but arbitrary commitment, which is exactly what one-sided design is.
B1_CKKS = {"poly_degree": 32768,
           "modulus_chain": [60] + [40] * 8 + [60]}     # 10 primes, budget 8, 256-bit
B2_ARCH = {"filters": [64, 128, 128], "activation": "square"}         # depth 6
_B_ARCH = ["Conv64_3_square", "Conv128_3_square", "Conv128_3_square"] # depth 6
#
# B3/B4 are the *expert* baseline: the hand-designed net a human copies from the
# literature. We use Ishiyama et al. (2020) Table IV verbatim -- their CIFAR-10
# CNN (3 conv 32/64/128, 3x3, relu4, avg-pool after each). Depth = 3 x relu4 =
# 12 levels, so it needs a 14-prime chain (budget 12) at N=32768. This is
# exactly the point of the study: the expert must hand-match a long chain to a
# relu4-heavy net, and B4 shows the naive failure of not doing so. The GE
# searches a *broader* space (2-5 layers, up to 256 filters, 5x5 kernels,
# per-layer square/relu4 mixing, co-evolved N and chain length) and is judged
# against this fixed expert design.
#
# B4's chain is not arbitrary: 8 primes (budget 6) is precisely what the
# superseded log2-degree depth model prescribed for this architecture. B4 is
# therefore the concrete cost of trusting an uncalibrated analytic depth model,
# which is the provisioning mistake this project actually made.
_ISHIYAMA_ARCH = ["Conv32_3_relu4+Pool", "Conv64_3_relu4+Pool",
                  "Conv128_3_relu4+Pool"]                            # depth 12
B3_CONFIG = {"ckks": {"poly_degree": 32768,
                      "modulus_chain": [60] + [40] * 12 + [60]},     # 14 primes, budget 12
             "architecture": _ISHIYAMA_ARCH}                        # provisioned, 188-bit
B4_CONFIG = {"ckks": {"poly_degree": 32768,
                      "modulus_chain": [60] + [40] * 6 + [60]},      # 8 primes, budget 6
             "architecture": _ISHIYAMA_ARCH}                        # naive: the log2 model

# The four ablation conditions (penalty type x elitism).
ABLATION_CONDITIONS = [
    {"name": "soft_elit", "penalty": "soft", "elitism": True},
    {"name": "soft_noelit", "penalty": "soft", "elitism": False},
    {"name": "hard_elit", "penalty": "hard", "elitism": True},
    {"name": "hard_noelit", "penalty": "hard", "elitism": False},
]
