#!/usr/bin/env python3
"""Stage 1 of the paper pipeline: measure real CKKS level consumption.

Runs every architecture of interest against TenSEAL, growing a well-formed
modulus chain until it first executes, and records the measured minimum. This
is what the corrected depth model is calibrated against, and it is the only
stage that needs TenSEAL.

    python validate_depth_model.py            # ~10 min, CPU only
    python validate_depth_model.py --quick    # calibration set only

Writes measurements/depth_validation.json. Nothing in the
search path is imported for writing -- validate_ckks.validate_config is reused
read-only so the encryption path is identical to the project's own validator.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT
sys.path.insert(0, str(PROJ))

OUT = ROOT / "measurements" / "depth_validation.json"

# Level cost per activation. OLD is the model the search optimised against
# (ckks.py); NEW is the corrected rule this script tests: cost = poly degree.
OLD = {"square": 1, "relu4": 2}
NEW = {"square": 2, "relu4": 4}
MAX_BITS_128 = {8192: 218, 16384: 438, 32768: 881}

# Architectures used to derive the rule, then to test it on held-out cases.
CALIBRATION = [
    ("1x square",            ["Conv8_3_square"]),
    ("1x relu4",             ["Conv8_3_relu4"]),
    ("2x square",            ["Conv8_3_square", "Conv8_3_square"]),
    ("3x square",            ["Conv8_3_square"] * 3),
    ("relu4, square",        ["Conv8_3_relu4", "Conv8_3_square"]),
    ("square, 2x relu4",     ["Conv8_3_square", "Conv8_3_relu4", "Conv8_3_relu4"]),
]


def acts_of(architecture):
    return [t.split("_")[-1].replace("+Pool", "") for t in architecture]


def predict(architecture, table):
    return sum(table[a] for a in acts_of(architecture))


def min_usable_levels(ts, validate, arch, base, lo=1, hi=24, scale=40):
    """Smallest usable-level count (len(chain)-2) at which arch runs.

    Tries the smallest ring dimension whose 128-bit modulus ceiling admits the
    chain, so a failure means depth exhaustion rather than an invalid context.
    """
    for usable in range(lo, hi + 1):
        chain = [60] + [40] * usable + [60]
        bits = sum(chain)
        N = next((n for n in (8192, 16384, 32768) if bits <= MAX_BITS_128[n]), None)
        if N is None:
            break
        cfg = {"architecture": arch, "poly_degree": N,
               "modulus_chain": chain, "score": 0.0}
        r = validate(ts, cfg, base, scale)
        if r.get("feasible"):
            return {"usable_levels": usable, "n_primes": len(chain),
                    "poly_degree": N, "total_bits": bits,
                    "security_bits": 128.0 * MAX_BITS_128[N] / bits,
                    "rel_error": r["rel_error"],
                    "latency_ms": r["latency_total_s"] * 1e3}
    return None


def champion_architectures():
    """(label, architecture, own chain, accuracy) for every real champion."""
    path = PROJ / "results" / "champions.json"
    if not path.exists():
        return []
    out = []
    for name, v in json.load(open(path)).items():
        if "smoke" in name or not v.get("architecture"):
            continue
        out.append((name, v["architecture"], v.get("modulus_chain") or [],
                    v.get("full_scale_accuracy"), v.get("poly_degree")))
    return sorted(out, key=lambda r: -(r[3] or 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="calibration architectures only (skip champions)")
    ap.add_argument("--slots", type=int, default=1024)
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import tenseal as ts
        from validate_ckks import validate_config
    except ImportError as e:
        print(f"Need tenseal + numpy in this interpreter: {e}")
        print("  pip install tenseal")
        return 1

    base = np.random.default_rng(0).standard_normal(args.slots).tolist()
    results = {"calibration": [], "champions": [], "slots": args.slots,
               "tenseal_version": getattr(ts, "__version__", "unknown")}

    print("=" * 78)
    print("CALIBRATION -- measured minimum usable levels vs the two models")
    print("=" * 78)
    print(f"{'architecture':<26}{'measured':>9}{'corrected':>10}{'original':>9}   match")
    print("-" * 78)
    for label, arch in CALIBRATION:
        m = min_usable_levels(ts, validate_config, arch, base)
        new, old = predict(arch, NEW), predict(arch, OLD)
        meas = m["usable_levels"] if m else None
        ok = meas == new
        results["calibration"].append(
            {"label": label, "architecture": arch, "measured": meas,
             "corrected": new, "original": old, "match": ok,
             **({k: m[k] for k in ("n_primes", "poly_degree", "total_bits",
                                   "security_bits", "rel_error", "latency_ms")}
                if m else {})})
        print(f"{label:<26}{str(meas):>9}{new:>10}{old:>9}   "
              f"{'YES' if ok else 'NO'}")

    n_ok = sum(r["match"] for r in results["calibration"])
    print("-" * 78)
    print(f"corrected model matches measurement on {n_ok}/"
          f"{len(results['calibration'])} architectures")

    if not args.quick:
        print()
        print("=" * 78)
        print("CHAMPIONS -- as evolved, and after post-hoc re-provisioning")
        print("=" * 78)
        print(f"{'experiment':<30}{'need':>5}{'own':>5}{'as-evolved':>12}{'repaired':>10}")
        print("-" * 78)
        for name, arch, own, acc, own_N in champion_architectures():
            need = predict(arch, NEW)
            own_budget = max(len(own) - 2, 0)
            # As evolved, with its own chain.
            asis = None
            if own:
                r = validate_config(ts, {"architecture": arch,
                                         "poly_degree": own_N or 32768,
                                         "modulus_chain": own, "score": 0.0},
                                    base, 40)
                asis = bool(r.get("feasible"))
            # Repaired: shortest well-formed chain carrying the true depth.
            rep = min_usable_levels(ts, validate_config, arch, base,
                                    lo=need, hi=need)
            results["champions"].append(
                {"experiment": name, "architecture": arch, "accuracy": acc,
                 "corrected_need": need, "original_need": predict(arch, OLD),
                 "own_chain_len": len(own), "own_budget": own_budget,
                 "feasible_as_evolved": asis,
                 "repaired": rep})
            print(f"{name:<30}{need:>5}{own_budget:>5}"
                  f"{('PASS' if asis else 'FAIL'):>12}"
                  f"{('PASS' if rep else 'none'):>10}")

        n_rep = sum(1 for c in results["champions"] if c["repaired"])
        n_as = sum(1 for c in results["champions"] if c["feasible_as_evolved"])
        print("-" * 78)
        print(f"as evolved: {n_as}/{len(results['champions'])} run under real CKKS")
        print(f"repaired:   {n_rep}/{len(results['champions'])} run under real CKKS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
