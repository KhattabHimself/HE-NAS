#!/usr/bin/env python3
"""Does Eq. (1) of the MobiSPC paper actually fail, or does it just not transfer?

The MobiSPC benchmark uses MLPs:  Linear(128) -> x^2, repeated d times,
implemented as TenSEAL `mm` (matrix-vector with plaintext weights) followed by
a square. Its Eq. (1) says a depth-d net needs d+2 primes, i.e. ONE level per
layer, and its 24-run matrix confirms that boundary empirically.

HE-NAS uses a different pair of primitives: element-wise `mul` against plaintext
weights, then `polyval`. probe_decomposition.py measured that at TWO levels per
layer for square.

Both cannot be "the cost of a layer" unless the primitives differ. This probe
measures each operation on its own so the question is settled by measurement.

    python probe_ops.py
"""
import numpy as np
import tenseal as ts

SCALE, N, SLOTS = 40, 8192, 16
RNG = np.random.default_rng(0)
VEC = RNG.standard_normal(SLOTS).tolist()
MAT = RNG.standard_normal((SLOTS, SLOTS)).tolist()      # plaintext weights
WVEC = (0.6 + 0.2 * RNG.standard_normal(SLOTS)).tolist()


def ctx(usable):
    c = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N,
                   coeff_mod_bit_sizes=[60] + [SCALE] * usable + [60])
    c.global_scale = 2 ** SCALE
    c.generate_galois_keys()
    return c


def min_levels(op, hi=10):
    """Smallest usable-level count at which `op` runs and decrypts finite."""
    for u in range(0, hi + 1):
        try:
            out = np.array(op(ctx(u)).decrypt())
            if np.all(np.isfinite(out)):
                return u, out
        except Exception:
            continue
    return None, None


OPS = {
    "encrypt only (baseline)":     lambda c: ts.ckks_vector(c, VEC),
    "mul  by plaintext vector":    lambda c: ts.ckks_vector(c, VEC) * WVEC,
    "mm   by plaintext matrix":    lambda c: ts.ckks_vector(c, VEC).mm(MAT),
    "square()":                    lambda c: ts.ckks_vector(c, VEC).square(),
    "polyval([0,0,1])  (= x^2)":   lambda c: ts.ckks_vector(c, VEC).polyval([0, 0, 1]),
    "mm then square()":            lambda c: ts.ckks_vector(c, VEC).mm(MAT).square(),
    "mul then polyval(x^2)":       lambda c: (ts.ckks_vector(c, VEC) * WVEC).polyval([0, 0, 1]),
    "mm then square, x2 layers":   lambda c: ts.ckks_vector(c, VEC).mm(MAT).square().mm(MAT).square(),
    "mul then polyval, x2 layers": lambda c: ((ts.ckks_vector(c, VEC) * WVEC)
                                              .polyval([0, 0, 1]) * WVEC).polyval([0, 0, 1]),
}

print("=" * 68)
print("Level cost per operation.  TenSEAL %s" % ts.__version__)
print("=" * 68)
base = None
res = {}
for label, op in OPS.items():
    u, _ = min_levels(op)
    res[label] = u
    if base is None:
        base = u
    delta = "" if u is None or base is None else "   (+%d over baseline)" % (u - base)
    print("  %-30s -> %s usable level(s)%s" % (label, u, delta))

print("\n" + "=" * 68)
print("VERDICT")
print("=" * 68)
mlp1 = res.get("mm then square()")
hen1 = res.get("mul then polyval(x^2)")
mlp2 = res.get("mm then square, x2 layers")
hen2 = res.get("mul then polyval, x2 layers")
print("  MobiSPC-style layer (mm + square)      : %s level(s)/layer  "
      "[1 layer=%s, 2 layers=%s]" % (
          None if (mlp2 is None or mlp1 is None) else mlp2 - mlp1, mlp1, mlp2))
print("  HE-NAS-style layer  (mul + polyval)    : %s level(s)/layer  "
      "[1 layer=%s, 2 layers=%s]" % (
          None if (hen2 is None or hen1 is None) else hen2 - hen1, hen1, hen2))
print("""
If the MobiSPC-style layer costs 1 and the HE-NAS-style layer costs 2, then
Eq. (1) (d+2 primes) is CORRECT for the primitives it was derived and tested
on, and the HE-NAS failure was carrying it across to different primitives,
not an error in the proposition itself.""")
