"""CKKS multiplicative-depth accounting and security estimation.

The level costs here are CALIBRATED AGAINST MEASUREMENT, not derived. They were
obtained by growing a well-formed modulus chain under TenSEAL 0.3.16 (Microsoft
SEAL) until each architecture first executed, and they reproduce the measured
minimum on 6/6 architectures (validate_depth_model.py).

  - Convolutions cost ONE level. Inference runs the MLaaS way, so a conv is
    ciphertext x plaintext -- but that still multiplies the ciphertext scale by
    the plaintext scale and requires a rescale, which consumes a prime. Counting
    only ciphertext x ciphertext products is correct for *multiplicative depth*
    in the theoretical sense and wrong for *level budgeting*, where the operative
    question is how many rescalings occur. BatchNorm folds into the conv weights
    and is free; a real conv's rotations are also free, and its many ct x pt
    products all sit at one level, so one rescale per conv layer is right.
  - Activations are polynomials (Ishiyama et al. 2020, Table II) and cost
    DEGREE - 1 levels: SEAL's polyval evaluates naively, not via
    Paterson-Stockmeyer, so a degree-d polynomial consumes d-1. Measured
    directly by probe_decomposition.py, which isolates each primitive:
    ct x pt = 1 level, polyval(2) = 1, polyval(4) = 3, and they add. The
    calibration in validate_depth_model.py fixes only the TOTAL
    (= sum of activation degrees); the split needs the separate probe.
    square (x^2)
    costs 1; relu4 (degree-4 ReLU fit) costs 3.
  - Average pooling is linear (0 levels).

Net effect: a layer with a degree-d activation costs 1 + (d-1) = d levels, so
total levels = sum of activation polynomial degrees. square -> 2, relu4 -> 4.

Budget = len(modulus_chain) - 2: one prime maintains the base level and one
provides output scale headroom. This half was confirmed by the same calibration.

HISTORICAL NOTE. Earlier versions of this project charged CONV_LEVEL_COST = 0
and {square: 1, relu4: 2} (= log2 of the polynomial degree), an uncalibrated
circuit-depth count that charges only ct x ct products and prices a degree-d
activation at the Paterson-Stockmeyer bound. Measured, it is optimistic by
exactly a factor of two for these two activations -- a coincidence of
d = 2*log2(d) at d in {2,4}, not a general law. A degree-8 activation would be
underestimated in the ratio 8:3. Configurations searched under it do not
execute; see docs/DEPTH_MEASUREMENT.md and measurements/depth_validation.json.

Security: approximation from the HomomorphicEncryption.org standard tables --
security ~ 128 * (max_modulus_bits[N] / total_modulus_bits). Production
systems should use the lattice-estimator library instead.
"""

from grammar import parse_layer_token

# Max total modulus bits still giving 128-bit security, per polynomial degree.
_MAX_BITS_128 = {8192: 218, 16384: 438, 32768: 881}

# Levels consumed by one convolution, and by each polynomial activation.
# Measured under TenSEAL 0.3.16 -- see module docstring before changing these.
CONV_LEVEL_COST = 1                                 # ct x pt still forces a rescale
ACTIVATION_LEVEL_COST = {'square': 1, 'relu4': 3}   # degree - 1

# Polynomial degree of each activation. Kept alongside the level costs because
# the calibrated rule is "levels = sum of degrees"; the selftest asserts the two
# tables stay consistent.
ACTIVATION_DEGREE = {'square': 2, 'relu4': 4}

# Primes reserved for the base level and for output scale headroom.
RESERVED_PRIMES = 2


def activation_cost(activation):
    """Multiplicative levels consumed by one polynomial activation."""
    try:
        return ACTIVATION_LEVEL_COST[activation]
    except KeyError:
        raise ValueError(f"Unknown activation '{activation}'. "
                         f"Choices: {sorted(ACTIVATION_LEVEL_COST)}")


def calculate_depth(architecture):
    """Total levels consumed: one per conv, plus each activation's cost."""
    depth = 0
    for layer in architecture:
        if 'Conv' not in layer:
            continue
        _, _, act, _ = parse_layer_token(layer)
        depth += CONV_LEVEL_COST + activation_cost(act)
    return depth


def depth_budget(modulus_chain):
    """Levels a chain supplies: len - 2 (base level + output scale headroom)."""
    return len(modulus_chain) - RESERVED_PRIMES


def check_depth_constraint(architecture, modulus_chain):
    """Return (depth, budget, depth <= budget)."""
    depth = calculate_depth(architecture)
    budget = depth_budget(modulus_chain)
    return depth, budget, depth <= budget


def estimate_security_bits(poly_degree, modulus_chain):
    max_bits = _MAX_BITS_128.get(poly_degree, 0)
    if max_bits == 0:
        return 0.0
    return 128.0 * max_bits / sum(modulus_chain)


def is_secure(poly_degree, modulus_chain, min_security=128):
    return estimate_security_bits(poly_degree, modulus_chain) >= min_security
