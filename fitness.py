"""Datasets, CNN builder, training, and the fitness function.

fitness = test accuracy, scaled by a depth-budget penalty:
  penalty 'soft': fitness = accuracy * exp(-(depth - budget)) on violation.
  penalty 'hard': fitness = 0 on violation (death penalty).
Insecure CKKS parameters (< 128-bit) always give fitness 0.

torch/torchvision are imported lazily so the rest of the project (selftest,
stats) works on machines without them.
"""
import math
import os
import random
import warnings

# torchvision ships CIFAR-10/100 as pickled NumPy arrays; unpickling them under
# NumPy 2.4+ emits a cosmetic VisibleDeprecationWarning ("dtype(): align ...")
# that does not affect training or results. Silence just that one message.
warnings.filterwarnings("ignore", message=r"dtype\(\): align")

from ckks import check_depth_constraint, estimate_security_bits, is_secure
from config import DATA_DIR
from grammar import DEPTH_BUDGETS, parse_layer_token

# --------------------------------------------------------------------------- #
# Fitness shaping constants.
# --------------------------------------------------------------------------- #
# Soft-penalty decay, in levels. Under the calibrated depth model every layer
# costs its activation's polynomial degree (2 for square, 4 for relu4), so one
# extra layer overruns the budget by 2-4 levels rather than 1-2. Dividing by the
# cheapest layer cost keeps the penalty's meaning fixed at roughly e^-1 per
# surplus LAYER, which is what the soft regime was designed to express. Without
# it, exp(-(depth-budget)) decays twice as fast as intended and the soft penalty
# collapses into the hard one -- which would leave the soft/hard ablation
# comparing two nearly identical regimes.
PENALTY_SCALE = 2.0

# Among FEASIBLE solutions, prefer the one that wastes fewest levels.
#
# With fitness = accuracy alone, a budget-19 chain and a budget-10 chain score
# identically for a depth-10 network, so nothing pushes the search toward exact
# provisioning -- which is the concrete claim this project makes ("provisions
# exactly what is required, no more"). Encrypted latency grows with chain length
# (measured: 57 ms at 8 primes to 966 ms at 20), so slack is a real cost the
# search was blind to.
#
# The cost is proportional to SLACK (budget - depth), not to absolute chain
# length. Penalising length would punish a deep architecture for legitimately
# needing a long chain; penalising slack is depth-neutral, since exact
# provisioning scores the same whether the network needs 4 levels or 18.
#
# This is a tie-break, not a driver: at 0.05 across the full slack range it
# stays under the accuracy gaps that separate methods (0.03-0.15), and it is
# smaller than the +-2% sampling noise of the 500-image search proxy. Champions
# therefore still land with some slack; champions.py reports it, and the
# encrypted validation uses the exactly-provisioned chain (which is free -- see
# the B3/B4 pair, identical accuracy at p=1.00, because plaintext training never
# observes the encryption parameters).
EFFICIENCY_WEIGHT = 0.05
MAX_SLACK = max(DEPTH_BUDGETS) - 4          # deepest budget minus shallowest depth


def set_global_seed(seed):
    """Seed Python random, NumPy, and torch (CPU + CUDA, deterministic cuDNN)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


_DEVICE = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"[he-nas] device: {_DEVICE} ({name})")
    return _DEVICE


# --------------------------------------------------------------------------- #
# Datasets — MNIST / Fashion-MNIST / CIFAR-10 behind one loader.
# --------------------------------------------------------------------------- #
DATASET_SPECS = {
    # name: (input_channels, num_classes, (mean,), (std,))
    'mnist': (1, 10, (0.1307,), (0.3081,)),
    'fashion_mnist': (1, 10, (0.2860,), (0.3530,)),
    'cifar10': (3, 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def _dataset_class(name):
    from torchvision import datasets
    return {'mnist': datasets.MNIST,
            'fashion_mnist': datasets.FashionMNIST,
            'cifar10': datasets.CIFAR10}[name]


def load_dataset(name, train_samples, test_samples, seed, batch_size=64,
                 download=True):
    """Return (train_loader, test_loader, input_channels, num_classes)."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {sorted(DATASET_SPECS)}")
    in_ch, n_classes, mean, std = DATASET_SPECS[name]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    data_dir = DATA_DIR / name
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_cls = _dataset_class(name)
    train_ds = ds_cls(str(data_dir), train=True, download=download, transform=tfm)
    test_ds = ds_cls(str(data_dir), train=False, download=download, transform=tfm)

    n_train = min(train_samples, len(train_ds))
    n_test = min(test_samples, len(test_ds))
    loader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    gen = torch.Generator()
    gen.manual_seed(int(seed) % (2 ** 31))
    train_loader = DataLoader(Subset(train_ds, range(n_train)), batch_size=batch_size,
                              shuffle=True, generator=gen, **loader_kwargs)
    test_loader = DataLoader(Subset(test_ds, range(n_test)), batch_size=batch_size,
                             shuffle=False, **loader_kwargs)
    return train_loader, test_loader, in_ch, n_classes


# --------------------------------------------------------------------------- #
# GPU-resident datasets.
#
# The models here are tiny, so the DataLoader's CPU-side batching and the
# per-batch host->device copy dominate wall-clock time -- the GPU finishes
# each batch faster than it can be fed. MNIST/CIFAR subsets are a few hundred
# MB at most, so we materialize the whole subset onto the device *once* and
# slice tensors in-place during training: no workers, no per-batch transfer.
# The cache key omits the seed because range(n) picks a deterministic subset;
# the seed only drives shuffle order, which stays per-run inside training.
# --------------------------------------------------------------------------- #
_GPU_DATA_CACHE = {}


def load_dataset_gpu(name, train_samples, test_samples):
    """Materialize the subset as device-resident tensors, cached across calls.

    Returns (X_train, y_train, X_test, y_test, in_ch, n_classes) with all four
    tensors already on the training device."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    device = get_device()
    key = (name, int(train_samples), int(test_samples), str(device))
    if key in _GPU_DATA_CACHE:
        return _GPU_DATA_CACHE[key]

    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {sorted(DATASET_SPECS)}")
    in_ch, n_classes, mean, std = DATASET_SPECS[name]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    data_dir = DATA_DIR / name
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_cls = _dataset_class(name)
    train_ds = ds_cls(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = ds_cls(str(data_dir), train=False, download=True, transform=tfm)

    def _materialize(ds, n):
        n = min(n, len(ds))
        # One giant batch runs the transform pipeline once, then we move it all.
        loader = DataLoader(Subset(ds, range(n)), batch_size=n, shuffle=False)
        x, y = next(iter(loader))
        return x.to(device), y.to(device)

    x_train, y_train = _materialize(train_ds, train_samples)
    x_test, y_test = _materialize(test_ds, test_samples)
    result = (x_train, y_train, x_test, y_test, in_ch, n_classes)
    _GPU_DATA_CACHE[key] = result
    return result


# --------------------------------------------------------------------------- #
# Model builder.
# --------------------------------------------------------------------------- #
# HE-friendly polynomial activations (no ReLU: comparisons are not native to
# CKKS). BatchNorm before each one normalises the input to roughly a unit
# Gaussian, which is exactly the range the polynomials below were fitted over
# (Ishiyama et al. 2020, "Highly Accurate CNN Inference ...", Table II). The
# fits are least-squares approximations of ReLU; relu4 turns down outside its
# [-6,6] range (the -0.002 x^4 tail), so BN + gradient clipping keep training
# inside the fitted region.
#   square : the CryptoNets/Badawi baseline x^2       (degree 2, 2 levels)
#   relu4  : degree-4 ReLU fit over [-6,6], best on   (degree 4, 4 levels)
#            CIFAR-10 in Ishiyama's Table V (81.02%)
# Level costs are the MEASURED ones (= polynomial degree), not the retired
# log2(d) estimate; see ckks.py and probe_decomposition.py.
# The degree-2 ReLU fit is intentionally omitted: it costs the same two levels
# as square but is nearly linear over the normalised range and underperforms
# it, so it is dominated and adds nothing to the search.
POLY_ACTIVATIONS = {
    'square': lambda x: x ** 2,
    'relu4': lambda x: (0.119782 + 0.5 * x + 0.147298 * x ** 2
                        - 0.002015 * x ** 4),
}


def build_model(architecture, input_channels=1, num_classes=10):
    """Torch CNN from decoded tokens. Each token: Conv(same-pad) -> BatchNorm
    -> polynomial activation -> [AvgPool2d(2)].
    Global AdaptiveAvgPool + Linear head adapts to any input size."""
    import torch.nn as nn

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, filters, kernel, activation, use_pool):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, filters, kernel_size=kernel,
                                  padding=kernel // 2)
            self.norm = nn.BatchNorm2d(filters)
            if activation not in POLY_ACTIVATIONS:
                raise ValueError(f"Unknown activation '{activation}'. "
                                 f"Choices: {sorted(POLY_ACTIVATIONS)}")
            self.act = POLY_ACTIVATIONS[activation]
            self.pool = nn.AvgPool2d(2, 2) if use_pool else None

        def forward(self, x):
            out = self.act(self.norm(self.conv(x)))
            if self.pool is not None:
                out = self.pool(out)
            return out

    blocks, in_ch = [], input_channels
    for token in architecture:
        if 'Conv' not in token:
            continue
        filters, kernel, activation, use_pool = parse_layer_token(token)
        blocks.append(_ConvBlock(in_ch, filters, kernel, activation, use_pool))
        in_ch = filters

    class _GrammarCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(in_ch, num_classes))

        def forward(self, x):
            return self.head(self.features(x))

    return _GrammarCNN()


# --------------------------------------------------------------------------- #
# Training + fitness.
# --------------------------------------------------------------------------- #
def train_and_evaluate(architecture, cfg):
    """Train the decoded CNN on cfg.dataset and return test accuracy."""
    import torch
    import torch.nn as nn

    device = get_device()
    seed = torch.initial_seed()
    x_train, y_train, x_test, y_test, in_ch, n_classes = load_dataset_gpu(
        cfg.dataset, cfg.train_samples, cfg.test_samples)

    model = build_model(architecture, in_ch, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    n_train, bs = x_train.size(0), cfg.batch_size
    # Shuffle on-device; a device generator keeps runs reproducible per seed.
    shuffle_gen = torch.Generator(device=device)
    shuffle_gen.manual_seed(int(seed) % (2 ** 31))

    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n_train, generator=shuffle_gen, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            loss_fn(model(x_train[idx]), y_train[idx]).backward()
            # Polynomial activations grow super-linearly and can explode the
            # gradient (documented failure mode of square nets on deeper models,
            # Njungle et al. 2025); clip to norm 1, matching the benchmark
            # paper's training recipe (Khattab & Eaman 2026, S4.1).
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, x_test.size(0), bs):
            data, target = x_test[i:i + bs], y_test[i:i + bs]
            correct += (model(data).argmax(dim=1) == target).sum().item()
            total += target.size(0)
    return correct / total


def evaluate_fitness(ckks_params, architecture, cfg):
    """Return (fitness, metadata) for one decoded (CKKS, architecture) pair."""
    sec_bits = estimate_security_bits(ckks_params['poly_degree'],
                                      ckks_params['modulus_chain'])
    if not is_secure(ckks_params['poly_degree'], ckks_params['modulus_chain']):
        return 0.0, {'error': 'insecure', 'fitness': 0.0, 'security_bits': sec_bits,
                     'penalty_type': cfg.penalty, 'dataset': cfg.dataset}

    depth, budget, is_valid = check_depth_constraint(
        architecture, ckks_params['modulus_chain'])
    try:
        accuracy = train_and_evaluate(architecture, cfg)
    except Exception as e:
        return 0.0, {'error': f'training_failed: {e}', 'fitness': 0.0,
                     'penalty_type': cfg.penalty, 'dataset': cfg.dataset}

    n_primes = len(ckks_params['modulus_chain'])
    slack = max(0, budget - depth)
    efficiency_cost = EFFICIENCY_WEIGHT * slack / MAX_SLACK
    if depth <= budget:
        penalty = 1.0
        fitness = accuracy - efficiency_cost
    elif cfg.penalty == 'soft':
        penalty = math.exp(-(depth - budget) / PENALTY_SCALE)
        fitness = accuracy * penalty
    else:  # hard
        penalty, fitness = 0.0, 0.0

    return fitness, {
        'fitness': fitness, 'accuracy': accuracy, 'depth': depth,
        'budget': budget, 'depth_valid': is_valid,
        'penalty_applied': depth > budget, 'penalty_factor': penalty,
        'efficiency_cost': efficiency_cost if depth <= budget else 0.0,
        'n_primes': n_primes, 'slack': slack if depth <= budget else None,
        'penalty_type': cfg.penalty, 'dataset': cfg.dataset,
        'poly_degree': ckks_params['poly_degree'],
        'modulus_chain': ckks_params['modulus_chain'],
        'security_bits': sec_bits, 'num_layers': len(architecture),
    }
