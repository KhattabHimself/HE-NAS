"""BNF grammar and genome handling.

The grammar encodes CKKS parameters and a CNN architecture in one integer
genome. Solution format: "N=<degree>,B=<budget>|<conv_layer>,..."
e.g. "N=16384,B=6|Conv64_3_square,Conv128_5_relu4+Pool"

No production is recursive (explicit layer counts), so derivation always
terminates. Codons 0-255 select productions via modulo (standard GE).

Also provides partial-genome freezing for the sequential baselines: the
grammar expands the left-most non-terminal first, so the whole CKKS half
decodes from a fixed-length codon prefix before <arch> is reached. One half
can therefore be pinned deterministically while the other half evolves.

WHY THE CHAIN IS BUILT, NOT EVOLVED
-----------------------------------
An earlier version let the grammar emit each prime independently (2-22 primes,
each 40 or 60 bits). That expressed ~25 million distinct chains, of which
almost none were deployable, because CKKS scale management is not free-form:
rescaling divides the ciphertext scale by the prime being dropped, so with
global_scale = 2^40 the middle primes MUST be 40-bit. Drop a 60-bit prime
against a 2^40 scale and the scale collapses to 2^20, then to nothing, and SEAL
raises "scale out of bounds". Chains like [60,60,60,40,60,40,60,60,60,40,40,40]
-- which that grammar readily produced -- cannot be used at any length.

So the genome now chooses a single number, the depth budget, and build_chain()
constructs the one canonical well-formed chain for it: a 60-bit prime to carry
the base level, `budget` 40-bit primes matching the scale, and a 60-bit prime
for output precision. This is the standard SEAL/TenSEAL idiom and the shape
every configuration in measurements/depth_validation.json was measured with.

The effect is to delete ~25 million unusable configurations while losing no
real freedom: 3 ring dimensions x 20 budgets = 60 CKKS configurations, all of
them deployable. It also shortens the worst-case derivation from 49 codons to
26, which cuts genetic drift.

Note that N is still evolved, NOT derived. A large budget needs a large ring to
stay 128-bit secure, and letting the search discover that coupling -- rather
than hard-wiring it -- is the point of the study. Genomes that get it wrong
score zero, which costs no GPU time (the security check precedes training).
"""
import random

# Depth budgets the grammar can express. Budget = len(chain) - 2 by
# construction, so budget B yields a chain of B+2 primes.
#
# The upper end is load-bearing. Under the calibrated depth model (ckks.py) a
# layer costs its activation's polynomial degree, so the deepest architecture
# (5 layers x relu4) consumes 20 levels. An earlier version capped chains at 12
# primes (budget 10) under a depth model that was optimistic by 2x; the two
# errors cancelled, and the cap silently made over half the discovered
# architectures INEXPRESSIBLE -- no genome anywhere in the search space could
# provision them. A permissive constraint does not merely admit invalid
# solutions, it suppresses the feedback that would reveal the ceiling.
# analysis.run_selftest now asserts every architecture has a secure feasible
# chain, which is the guard that would have caught it.
#
# 20 is the true ceiling, not a round number: a budget-B canonical chain totals
# 120 + 40B bits, and 128-bit security caps the total at 881 bits for N=32768,
# so B <= 19 is secure and B = 20 is not. Exactly one architecture (5 layers of
# relu4, depth 20) is therefore unreachable at 128-bit security -- 1 of the 60
# distinct activation patterns. B = 20 is kept in the grammar deliberately so
# that limit is something the search runs into and reports, not something the
# encoding hides. Low budgets are kept for the same reason: they are infeasible
# for every architecture and give the soft penalty a gradient to push against.
DEPTH_BUDGETS = tuple(range(1, 21))

# Canonical chain shape. The endpoints are wider than the scale primes: the
# first carries the base level, the last provides output-scale headroom.
SPECIAL_PRIME_BITS = 60
SCALE_PRIME_BITS = 40


def build_chain(budget):
    """The one well-formed modulus chain supplying `budget` usable levels."""
    return ([SPECIAL_PRIME_BITS] + [SCALE_PRIME_BITS] * int(budget)
            + [SPECIAL_PRIME_BITS])


GRAMMAR = {
    '<solution>': ['<ckks>|<arch>'],
    '<ckks>': ['N=<poly>,B=<budget>'],
    '<poly>': ['8192', '16384', '32768'],
    '<budget>': [str(b) for b in DEPTH_BUDGETS],
    '<arch>': ['<l2>', '<l3>', '<l4>', '<l5>'],
    '<l2>': ['<conv_layer>,<conv_layer>'],
    '<l3>': ['<conv_layer>,<conv_layer>,<conv_layer>'],
    '<l4>': ['<conv_layer>,<conv_layer>,<conv_layer>,<conv_layer>'],
    '<l5>': ['<conv_layer>,<conv_layer>,<conv_layer>,<conv_layer>,<conv_layer>'],
    '<conv_layer>': [
        'Conv<f>_<kernel>_<act>',
        'Conv<f>_<kernel>_<act>+Pool',
    ],
    '<f>': ['32', '64', '128', '256'],
    '<kernel>': ['3', '5'],
    # HE-friendly polynomial activations (Ishiyama et al. 2020, Table II) --
    # two non-dominated points on the depth/accuracy trade-off. The dominated
    # degree-2 ReLU fit (same 1-level cost as square, lower accuracy) is left
    # out on purpose. The [-6,6] fit range is the one that wins at degree 4 in
    # their Table V, so it is baked in rather than searched.
    #   square = x^2                       (CryptoNets/Badawi baseline, 2 levels)
    #   relu4  = degree-4 ReLU fit [-6,6]  (Ishiyama best CIFAR-10, 4 levels)
    '<act>': ['square', 'relu4'],
}

# Worst-case derivation: 4 CKKS codons (<solution>, <ckks>, <poly>, <budget>),
# then 2 arch codons + 5 layers x 4 (<conv_layer>, <f>, <kernel>, <act>) = 26.
# GENOME_MIN is set above that so every genome decodes in a single pass without
# wrapping -- wrapping re-reads codons in a different grammatical position,
# which couples unrelated genes and is pure noise here. A genome that fails to
# decode does not error, it silently scores 0, so this bound must hold with room
# to spare.
GENOME_MIN, GENOME_MAX, CODON_MAX = 30, 60, 255


def random_genome(rng=random):
    length = rng.randint(GENOME_MIN, GENOME_MAX)
    return [rng.randint(0, CODON_MAX) for _ in range(length)]


def decode_genome(genome, grammar=None, max_wraps=2):
    """Decode an integer genome into (ckks_params, architecture)."""
    if grammar is None:
        grammar = GRAMMAR
    derivation = '<solution>'
    codon_idx, wraps = 0, 0
    while '<' in derivation:
        if codon_idx >= len(genome):
            codon_idx = 0
            wraps += 1
            if wraps >= max_wraps:
                raise ValueError("Genome exhausted: incomplete derivation")
        start = derivation.find('<')
        end = derivation.find('>', start) + 1
        non_terminal = derivation[start:end]
        if non_terminal not in grammar:
            raise ValueError(f"Unknown non-terminal: {non_terminal}")
        productions = grammar[non_terminal]
        production = productions[genome[codon_idx] % len(productions)]
        derivation = derivation[:start] + production + derivation[end:]
        codon_idx += 1

    ckks_str, arch_str = derivation.split('|')
    n_part, b_part = ckks_str.split(',B=')
    poly_degree = int(n_part.replace('N=', ''))
    modulus_chain = build_chain(int(b_part))
    architecture = [layer.strip() for layer in arch_str.split(',')]
    return {'poly_degree': poly_degree, 'modulus_chain': modulus_chain}, architecture


DEFAULT_ACT = 'square'   # assumed for legacy tokens that carry no activation


def parse_layer_token(token):
    """'Conv64_3_relu4+Pool' -> (filters=64, kernel=3, activation='relu4', use_pool)."""
    parts = token.split('+')
    use_pool = 'Pool' in parts
    body = parts[0].replace('Conv', '')
    fields = body.split('_')   # activation names carry no '_', so a plain split works
    filters = int(fields[0])
    kernel = int(fields[1]) if len(fields) > 1 else 3   # legacy 'Conv64' tokens
    activation = fields[2] if len(fields) > 2 else DEFAULT_ACT
    return filters, kernel, activation, use_pool


# --------------------------------------------------------------------------- #
# Partial-genome freezing (sequential baselines B1 / B2).
# --------------------------------------------------------------------------- #
_POLY_INDEX = {8192: 0, 16384: 1, 32768: 2}
_FILTER_INDEX = {32: 0, 64: 1, 128: 2, 256: 3}
_KERNEL_INDEX = {3: 0, 5: 1}
_ACT_INDEX = {'square': 0, 'relu4': 1}
_MIN_ARCH_CODONS = 22   # <arch> + <lN> + 5 layers x (<conv_layer>+<f>+<kernel>+<act>)


def _ckks_codons(ckks_config):
    """Codon prefix that decodes to exactly this CKKS configuration."""
    poly = ckks_config["poly_degree"]
    budget = len(ckks_config["modulus_chain"]) - 2
    if budget not in DEPTH_BUDGETS:
        raise ValueError(f"Budget {budget} is outside the grammar "
                         f"({min(DEPTH_BUDGETS)}..{max(DEPTH_BUDGETS)}).")
    # <solution>, <ckks>, <poly>, <budget>
    return [0, 0, _POLY_INDEX[poly], DEPTH_BUDGETS.index(budget)]


def _arch_codons(filters, kernel=3, activation=DEFAULT_ACT):
    codons = [len(filters) - 2, 0]
    for f in filters:
        # 0 -> plain conv (no pool).
        codons += [0, _FILTER_INDEX[f], _KERNEL_INDEX[kernel], _ACT_INDEX[activation]]
    return codons


def _codons_before_arch(genome):
    """Number of leading codons consumed to fully derive the CKKS half."""
    derivation, codon_idx, wraps = "<solution>", 0, 0
    while "<" in derivation:
        start = derivation.find("<")
        end = derivation.find(">", start) + 1
        nt = derivation[start:end]
        if nt == "<arch>":
            return codon_idx
        if codon_idx >= len(genome):
            codon_idx = 0
            wraps += 1
            if wraps >= 2:
                raise ValueError("Genome exhausted before reaching <arch>.")
        prods = GRAMMAR[nt]
        derivation = (derivation[:start] + prods[genome[codon_idx] % len(prods)]
                      + derivation[end:])
        codon_idx += 1
    raise ValueError("Derivation completed without encountering <arch>.")


def freeze_ckks(genome_template, ckks_config):
    """Pin the CKKS prefix; keep architecture codons evolvable."""
    prefix = _ckks_codons(ckks_config)
    tail = list(genome_template)[len(prefix):]
    filler = list(genome_template) or [0]
    while len(tail) < _MIN_ARCH_CODONS:
        tail.append(filler[len(tail) % len(filler)])
    return prefix + tail


def freeze_arch(genome_template, arch_config):
    """Pin the architecture suffix; keep the CKKS prefix evolvable."""
    filters = [int(f) for f in arch_config["filters"]]
    activation = arch_config.get("activation", DEFAULT_ACT)
    k = _codons_before_arch(genome_template)
    return list(genome_template)[:k] + _arch_codons(filters, activation=activation)
