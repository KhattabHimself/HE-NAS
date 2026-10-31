#!/usr/bin/env python3
"""Settle the conv/polyval split by direct measurement.

The calibration in validate_depth_model.py measures the TOTAL levels a network
consumes and establishes: total = sum of activation polynomial degrees. That is
solid. But every calibration architecture pairs exactly one conv with exactly
one activation, so the total does NOT determine how the cost splits between
them:

    conv=1, polyval(d)=d-1  ->  total per layer = d   (what ckks.py claims)
    conv=0, polyval(d)=d    ->  total per layer = d   (identical prediction)

This probe isolates each primitive so the split is measured, not inferred.

    python probe_decomposition.py
"""
import sys
import numpy as np
import tenseal as ts

SCALE = 40
N = 16384
SLOTS = 1024
RNG = np.random.default_rng(0)
BASE = RNG.standard_normal(SLOTS).tolist()

# Same coefficient sets the project uses, so the degrees match the paper.
COEFFS = {
    2: [0.0, 0.0, 1.0],                       # x^2, degree 2  ("square")
    4: [0.119, 0.5, 0.318, 0.0, -0.0128],     # degree-4 ReLU fit ("relu4")
}


def ctx_with(usable):
    """A well-formed chain [60, 40*usable, 60]; `usable` = spendable levels."""
    chain = [60] + [SCALE] * usable + [60]
    c = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N,
                   coeff_mod_bit_sizes=chain)
    c.global_scale = 2 ** SCALE
    return c


def min_levels(op, lo=0, hi=12):
    """Smallest usable-level count at which `op` runs and decrypts."""
    for usable in range(lo, hi + 1):
        try:
            v = op(ctx_with(usable))
            out = np.array(v.decrypt())
            if np.all(np.isfinite(out)):
                return usable, out
        except Exception:
            continue
    return None, None


def rel_err(got, want):
    want = np.asarray(want, dtype=float)
    return float(np.max(np.abs(got[:len(want)] - want)) / (np.max(np.abs(want)) + 1e-12))


print("=" * 72)
print("Isolating the primitives.  TenSEAL %s" % ts.__version__)
print("=" * 72)

# ---- 1. ciphertext x plaintext multiply, on its own -------------------------
w = (0.6 + 0.2 * RNG.standard_normal(SLOTS)).tolist()

def just_encrypt(c):
    return ts.ckks_vector(c, BASE)

def one_ctpt(c):
    return ts.ckks_vector(c, BASE) * w

def two_ctpt(c):
    return (ts.ckks_vector(c, BASE) * w) * w

n0, _ = min_levels(just_encrypt)
n1, o1 = min_levels(one_ctpt)
n2, o2 = min_levels(two_ctpt)

print("\n1.  ciphertext x plaintext  (this is what a conv is)")
print("    encrypt only, no ops        -> %s usable levels" % n0)
print("    1 ct x pt multiply          -> %s usable levels   (rel err %.1e)"
      % (n1, rel_err(o1, np.array(BASE) * np.array(w))))
print("    2 ct x pt multiplies        -> %s usable levels   (rel err %.1e)"
      % (n2, rel_err(o2, np.array(BASE) * np.array(w) * np.array(w))))
conv_cost = None if (n1 is None or n0 is None) else n1 - n0
inc = None if (n2 is None or n1 is None) else n2 - n1
print("    => cost of one ct x pt multiply = %s level(s)  (increment %s)"
      % (conv_cost, inc))

# ---- 2. polyval, on its own -------------------------------------------------
print("\n2.  polyval, with no convolution anywhere")
poly_cost = {}
for d, co in COEFFS.items():
    def op(c, co=co):
        return ts.ckks_vector(c, BASE).polyval(co)
    nd, od = min_levels(op)
    poly_cost[d] = nd
    want = np.polyval(list(reversed(co)), np.array(BASE))
    print("    polyval degree %d              -> %s usable levels   (rel err %.1e)"
          % (d, nd, rel_err(od, want) if od is not None else float("nan")))

# ---- 3. the two composed ----------------------------------------------------
print("\n3.  one conv then one activation  (a full layer)")
for d, co in COEFFS.items():
    def op(c, co=co):
        return (ts.ckks_vector(c, BASE) * w).polyval(co)
    nd, _ = min_levels(op)
    parts = "%s + %s" % (conv_cost, poly_cost[d])
    print("    conv + polyval degree %d       -> %s usable levels   (parts: %s)"
          % (d, nd, parts))

# ---- verdict ----------------------------------------------------------------
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
print("  conv (ct x pt)      costs %s level(s)" % conv_cost)
for d in sorted(poly_cost):
    claim = d - 1
    got = poly_cost[d]
    mark = "MATCHES" if got == claim else "DOES NOT MATCH"
    print("  polyval degree %d    costs %s level(s);  'd-1' predicts %s  -> %s"
          % (d, got, claim, mark))
print("""
Read this against the claim in ckks.py:29 and slide 8 of the deck, namely
'conv = 1 level, polyval(d) = d-1 levels'. The TOTAL (= sum of activation
degrees) is separately established by validate_depth_model.py and does not
depend on this split.""")
