"""Real-CKKS validation of the top-K evolved configurations (standalone).

This is the post-hoc "validate the winners under real encryption" step. It is
deliberately a *single, self-contained* file: drop it next to the other modules,
`pip install tenseal`, and run it against a finished results file. It imports
only `grammar.parse_layer_token` (read-only) and touches nothing in the search
path -- main.py / evolution.py / fitness.py / ckks.py are unchanged.

WHAT IT VALIDATES (and what it does not)
----------------------------------------
For each top-K config it builds a genuine TenSEAL CKKS context from that
config's OWN (poly_degree, modulus_chain) and runs the config's real
multiplicative-depth sequence -- the evolved per-layer polynomial activations
(square / relu4), each preceded by a plaintext scaling that stands in for the
convolution's ciphertext x plaintext multiply -- on encrypted data. It reports:

  * FEASIBLE : did the config's own modulus chain actually carry its
               architecture's depth, or did CKKS run out of levels? This
               empirically tests the depth accounting in ckks.py against a real
               library (where a linear op may cost a level that our
               conv-free model prices at zero).
  * PRECISION: CKKS approximation error of the polynomial-activation chain,
               encrypted vs plaintext (CKKS is approximate arithmetic).
  * LATENCY  : wall-clock per homomorphic layer and total.

It does NOT perform full encrypted image classification. Multi-conv CNN
inference is not tractable in TenSEAL (no clean re-packing between conv layers;
hidden tensors exceed a single ciphertext's slot count). That needs a
CNN-capable FHE backend and is scoped as future work -- see the SLR's
"activation approximation" / packing discussion.

USAGE
-----
    pip install tenseal                     # CPU only; login node or laptop
    python validate_ckks.py --demo          # smoke-test the TenSEAL path, no data
    python validate_ckks.py experiments/nsga2_cifar10/run_seed42/pareto_front_final.json --top-k 3
    python validate_ckks.py experiments/ge_cifar10/run_seed42/results.json --top-k 1

Point it at a `pareto_front_final.json` (NSGA-II) or a `results.json` (ge /
expert); both carry the architecture and CKKS parameters per solution.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from ckks import calculate_depth, depth_budget
from grammar import parse_layer_token

# Ishiyama et al. (2020) Table II coefficients, low-order -> high-order, kept in
# sync with fitness.POLY_ACTIVATIONS. square is x^2; relu4 is the degree-4 fit.
ACT_COEFFS = {
    "square": [0.0, 0.0, 1.0],
    "relu4": [0.119782, 0.5, 0.147298, 0.0, -0.002015],
}
# Level accounting comes from ckks.py -- one source of truth. This file used to
# keep its own table, which drifted from the search path's; the encrypted run
# below (one ct x pt multiply then one polyval per layer) IS what ckks.py's
# costs are calibrated against, so they must not be duplicated here.


# --------------------------------------------------------------------------- #
# Pull candidate configs out of whatever results JSON we were handed.
# --------------------------------------------------------------------------- #
def _score(d):
    """Best available quality signal for ranking a candidate."""
    for k in ("accuracy", "best_fitness", "fitness"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    objs = d.get("objectives")
    if isinstance(objs, (list, tuple)) and objs:
        return float(objs[0])          # NSGA-II: first objective is accuracy
    return float("-inf")


def _as_config(d):
    """Normalise one dict to {architecture, poly_degree, modulus_chain, score} or None."""
    arch = d.get("architecture")
    poly = d.get("poly_degree")
    chain = d.get("modulus_chain")
    if poly is None or chain is None:            # ge/expert nest these under 'ckks_params'
        ck = d.get("ckks_params") or d.get("ckks") or {}
        poly = poly if poly is not None else ck.get("poly_degree")
        chain = chain if chain is not None else ck.get("modulus_chain")
    if not (isinstance(arch, list) and poly and isinstance(chain, list)):
        return None
    return {"architecture": arch, "poly_degree": int(poly),
            "modulus_chain": [int(x) for x in chain], "score": _score(d)}


def extract_configs(obj):
    """Walk any of our results shapes and collect every (arch, CKKS) candidate."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            c = _as_config(o)
            if c is not None:
                found.append(c)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    # De-duplicate identical (arch, chain, N) configs, keep the best score.
    uniq = {}
    for c in found:
        key = (tuple(c["architecture"]), c["poly_degree"], tuple(c["modulus_chain"]))
        if key not in uniq or c["score"] > uniq[key]["score"]:
            uniq[key] = c
    return sorted(uniq.values(), key=lambda c: c["score"], reverse=True)


# --------------------------------------------------------------------------- #
# The real-CKKS run for one config.
# --------------------------------------------------------------------------- #
def _activations(architecture):
    acts = []
    for tok in architecture:
        if "Conv" in tok:
            acts.append(parse_layer_token(tok)[2])
    return acts


def _plaintext_reference(vec, weights, acts):
    """Same layer sequence as the encrypted run, in plain floats."""
    x = list(vec)
    for w, act in zip(weights, acts):
        coeffs = ACT_COEFFS[act]
        x = [xi * wi for xi, wi in zip(x, w)]
        x = [sum(c * (xi ** p) for p, c in enumerate(coeffs)) for xi in x]
    return x


def validate_config(ts, cfg, base_vec, scale_bits):
    """Run cfg's depth sequence under real CKKS. Returns a result dict."""
    import numpy as np

    arch, poly, chain = cfg["architecture"], cfg["poly_degree"], cfg["modulus_chain"]
    acts = _activations(arch)
    need = calculate_depth(arch)                     # calibrated: sum of activation degrees
    have = depth_budget(chain)                       # len(chain) - 2
    rng = np.random.default_rng(0)
    slots = len(base_vec)
    weights = [(0.6 + 0.2 * rng.standard_normal(slots)).tolist() for _ in acts]

    result = {"architecture": arch, "poly_degree": poly, "modulus_chain": chain,
              "score": cfg["score"], "n_layers": len(acts),
              "levels_needed": need, "levels_available": have}
    try:
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly,
                         coeff_mod_bit_sizes=chain)
        ctx.global_scale = 2 ** scale_bits
        ctx.generate_relin_keys()

        enc = ts.ckks_vector(ctx, list(base_vec))
        per_layer = []
        for w, act in zip(weights, acts):
            t0 = time.perf_counter()
            enc = enc.mul(w)                          # ct x pt  (models the conv/BN scaling)
            enc = enc.polyval(ACT_COEFFS[act])        # polynomial activation
            per_layer.append(time.perf_counter() - t0)

        dec = np.asarray(enc.decrypt(), dtype=float)
        ref = np.asarray(_plaintext_reference(base_vec, weights, acts), dtype=float)
        denom = float(np.mean(np.abs(ref))) or 1e-12
        result.update({
            "feasible": True,
            "rel_error": float(np.mean(np.abs(dec - ref)) / denom),
            "max_abs_error": float(np.max(np.abs(dec - ref))),
            "latency_total_s": float(sum(per_layer)),
            "latency_per_layer_s": float(statistics.mean(per_layer)) if per_layer else 0.0,
        })
    except Exception as e:                            # level exhaustion, scale-out-of-bounds, ...
        result.update({"feasible": False, "error": f"{type(e).__name__}: {e}"})
    return result


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def _print_report(results):
    print("\n" + "=" * 78)
    print("REAL-CKKS VALIDATION (TenSEAL)  --  depth feasibility / precision / latency")
    print("=" * 78)
    for i, r in enumerate(results, 1):
        arch = ",".join(r["architecture"])
        print(f"\n[{i}] score={r['score']:.4f}  N={r['poly_degree']}  "
              f"chain={r['modulus_chain']}")
        print(f"    arch: {arch}")
        print(f"    depth: needs {r['levels_needed']} levels, chain provides "
              f"{r['levels_available']}")
        if r.get("feasible"):
            print(f"    [OK]   ran under real CKKS")
            print(f"           rel-error {r['rel_error']:.2e}   "
                  f"max-abs {r['max_abs_error']:.2e}")
            print(f"           latency  {r['latency_total_s']*1e3:.1f} ms total  "
                  f"({r['latency_per_layer_s']*1e3:.1f} ms/layer)")
        else:
            print(f"    [FAIL] {r['error']}")
            print(f"           -> this config's chain did NOT carry its depth under a "
                  f"naive\n                 ct x pt cost model; see ckks.py conv-cost note.")
    feas = sum(1 for r in results if r.get("feasible"))
    print("\n" + "-" * 78)
    print(f"{feas}/{len(results)} configs feasible under real CKKS with their own chain.")
    print("Note: this validates CKKS depth + precision, not full encrypted "
          "classification accuracy.")


def _demo_config():
    return {"architecture": ["Conv32_3_square+Pool", "Conv64_3_relu4"],
            "poly_degree": 16384,
            "modulus_chain": [60, 40, 40, 40, 40, 40, 40, 40],
            "score": 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-CKKS validation of top-K evolved configs.")
    ap.add_argument("results", nargs="?", help="pareto_front_final.json or results.json")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--slots", type=int, default=2048, help="packed vector width")
    ap.add_argument("--scale-bits", type=int, default=40, help="CKKS global scale (2^bits)")
    ap.add_argument("--demo", action="store_true",
                    help="run one synthetic config to smoke-test the TenSEAL path")
    args = ap.parse_args(argv)

    try:
        import tenseal as ts
    except ImportError:
        print("TenSEAL is not installed. Run `pip install tenseal` "
              "(CPU-only; a login node or laptop is fine).", file=sys.stderr)
        return 2

    import numpy as np
    # BN normalises activations to ~unit Gaussian; use that as the encrypted input
    # so the polynomials operate in the range they were fitted over.
    base_vec = np.random.default_rng(0).standard_normal(args.slots).tolist()

    if args.demo:
        configs = [_demo_config()]
    else:
        if not args.results:
            ap.error("pass a results JSON, or use --demo")
        obj = json.loads(Path(args.results).read_text())
        configs = extract_configs(obj)[:args.top_k]
        if not configs:
            print("No (architecture, CKKS) configs found in that file. Point me at a "
                  "pareto_front_final.json or a run's results.json.", file=sys.stderr)
            return 1

    results = [validate_config(ts, c, base_vec, args.scale_bits) for c in configs]
    _print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
