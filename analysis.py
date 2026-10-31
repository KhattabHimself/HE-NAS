"""Statistics for method comparison, plus the selftest.

Non-parametric comparison suitable for stochastic search (Arcuri & Briand
2014): Mann-Whitney U with Cohen's d and Vargha-Delaney A12 effect sizes.
"""
import itertools
import json
import math
import random
from pathlib import Path

from ckks import (ACTIVATION_DEGREE, ACTIVATION_LEVEL_COST, CONV_LEVEL_COST,
                  calculate_depth, check_depth_constraint, depth_budget,
                  is_secure)
from config import B1_CKKS, B2_ARCH, RESULTS_DIR
from grammar import (DEPTH_BUDGETS, GENOME_MAX, GENOME_MIN, GRAMMAR,
                     build_chain, decode_genome, freeze_arch, freeze_ckks,
                     parse_layer_token, random_genome)


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #
def compare_methods(values_a, values_b):
    """Compare two per-seed samples; returns test + effect sizes + summaries."""
    import numpy as np
    from scipy import stats as sps

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size == 0 or b.size == 0 or not (np.all(np.isfinite(a))
                                          and np.all(np.isfinite(b))):
        raise ValueError("Both samples must be non-empty and finite.")

    mw = sps.mannwhitneyu(a, b, alternative="two-sided")

    # Cohen's d (pooled SD).
    var_a = a.var(ddof=1) if a.size > 1 else 0.0
    var_b = b.var(ddof=1) if b.size > 1 else 0.0
    pooled = math.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b)
                       / max(a.size + b.size - 2, 1))
    diff = a.mean() - b.mean()
    d = (0.0 if pooled == 0.0 and diff == 0.0
         else (math.inf * (1 if diff > 0 else -1) if pooled == 0.0
               else diff / pooled))

    # Vargha-Delaney A12 from the rank sum.
    ranks = sps.rankdata(np.concatenate([a, b]))
    a12 = (ranks[:a.size].sum() / a.size - (a.size + 1) / 2.0) / b.size

    def summarize(x):
        q1, q3 = np.percentile(x, [25, 75])
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
                "iqr": float(q3 - q1), "min": float(x.min()),
                "max": float(x.max()), "n": int(x.size)}

    ad, adev = abs(d), abs(a12 - 0.5)
    d_mag = ("negligible" if ad < 0.2 else "small" if ad < 0.5
             else "medium" if ad < 0.8 else "large")
    a_mag = ("negligible" if adev < 0.06 else "small" if adev < 0.14
             else "medium" if adev < 0.21 else "large")

    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "summary_a": summarize(a), "summary_b": summarize(b),
        "mann_whitney_U": float(mw.statistic), "p_value": float(mw.pvalue),
        "significant_at_0.05": bool(mw.pvalue < 0.05),
        "cohens_d": float(d), "cohens_d_magnitude": d_mag,
        "vargha_delaney_a12": float(a12), "vargha_delaney_magnitude": a_mag,
    }


def _extract_metric(results, metric):
    for key in (metric, f"{metric}_per_seed"):
        if isinstance(results.get(key), list):
            return results[key]
    per_seed = results.get("per_seed")
    if isinstance(per_seed, dict) and metric in per_seed:
        return per_seed[metric]
    raise ValueError(f"Could not find a per-seed list for metric '{metric}'.")


def run_stats(path_a, path_b, metric="best_fitness"):
    """Compare two summary JSONs, print a table, save under results/stats/."""
    with open(path_a) as f:
        results_a = json.load(f)
    with open(path_b) as f:
        results_b = json.load(f)
    comparison = compare_methods(_extract_metric(results_a, metric),
                                 _extract_metric(results_b, metric))

    name_a, name_b = Path(path_a).stem, Path(path_b).stem
    comparison.update({"metric": metric, "name_a": name_a, "name_b": name_b})

    sa, sb = comparison["summary_a"], comparison["summary_b"]
    print("=" * 64)
    print(f"Statistical comparison on metric: {metric}")
    print("=" * 64)
    print(f"{'':<10}{name_a:>26}{name_b:>26}")
    for label in ("n", "mean", "median", "std", "iqr", "min", "max"):
        fmt = (lambda v: f"{v:d}" if isinstance(v, int) else f"{v:.6f}")
        print(f"{label:<10}{fmt(sa[label]):>26}{fmt(sb[label]):>26}")
    print("-" * 64)
    print(f"Mann-Whitney U      : {comparison['mann_whitney_U']:.4f}")
    print(f"p-value (two-sided) : {comparison['p_value']:.3e}  "
          f"(significant @0.05: {'YES' if comparison['significant_at_0.05'] else 'no'})")
    print(f"Cohen's d           : {comparison['cohens_d']:.4f} "
          f"({comparison['cohens_d_magnitude']})")
    print(f"Vargha-Delaney A12  : {comparison['vargha_delaney_a12']:.4f} "
          f"({comparison['vargha_delaney_magnitude']})")

    out_dir = RESULTS_DIR / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name_a}_vs_{name_b}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return comparison


# --------------------------------------------------------------------------- #
# Selftest: fast unit checks that need no GPU and no datasets.
# --------------------------------------------------------------------------- #
def run_selftest():
    from evolution import fast_nondominated_sort, hypervolume3d

    failures = []

    def check(name, ok):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    print("Grammar / decoder:")
    for genome, name in [([0, 1, 0, 1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1], "genome A"),
                         ([2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 0], "genome B"),
                         ([1, 0, 1, 0, 0, 1, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2], "genome C")]:
        try:
            ckks, arch = decode_genome(genome)
            chain = ckks['modulus_chain']
            ok = (ckks['poly_degree'] in (8192, 16384, 32768)
                  and depth_budget(chain) in DEPTH_BUDGETS
                  and chain == build_chain(depth_budget(chain))
                  and 2 <= len(arch) <= 5
                  and all(parse_layer_token(t)[2] in ('square', 'relu4')
                          for t in arch))
        except Exception:
            ok = False
        check(f"decode {name}", ok)

    # Every chain the grammar can produce must be canonical [60, 40*B, 60].
    # Free-form prime composition was removed because rescaling divides the
    # scale by the dropped prime: with global_scale = 2^40 a 60-bit middle
    # prime collapses the scale and SEAL raises "scale out of bounds", so such
    # chains are undeployable at ANY length. See grammar.py.
    rng = random.Random(3)
    canonical = True
    for _ in range(500):
        chain = decode_genome(random_genome(rng))[0]['modulus_chain']
        canonical &= (chain[0] == 60 and chain[-1] == 60
                      and all(p == 40 for p in chain[1:-1])
                      and depth_budget(chain) in DEPTH_BUDGETS)
    check("every evolved chain is canonical [60, 40*B, 60]", canonical)

    print("Random genomes always decode:")
    rng = random.Random(1)
    ok_decode = True
    for _ in range(500):
        try:
            decode_genome(random_genome(rng))
        except Exception:
            ok_decode = False
            break
    check("random_genome x500 decodes", ok_decode)

    print("Depth tracker (1 level per conv + degree-1 per activation):")
    # (architecture, modulus chain, expected depth, expected validity)
    cases = [(['Conv32_3_square', 'Conv64_3_square'], [60] * 6, 4, True),
             (['Conv32_3_square', 'Conv64_3_square'], [60] * 5, 4, False),
             (['Conv32_3_square+Pool', 'Conv64_5_relu4'], [60, 60, 60], 6, False),
             (['Conv32_3_relu4', 'Conv64_5_relu4'], [60] * 10, 8, True),
             (['Conv32_3_relu4', 'Conv64_5_relu4'], [60] * 9, 8, False),
             (['Conv64', 'Conv128'], [60] * 6, 4, True)]  # legacy tokens -> square
    for arch, mod, exp_depth, expected in cases:
        depth, _, valid = check_depth_constraint(arch, mod)
        check(f"depth {arch} vs {len(mod)} primes", depth == exp_depth and valid == expected)

    # The calibrated rule: a layer costs its activation's polynomial degree,
    # because the conv's ct x pt rescale costs 1 and a degree-d polyval costs
    # d-1. Assert the two constant tables in ckks.py cannot drift apart.
    check("level costs equal polynomial degree",
          all(CONV_LEVEL_COST + ACTIVATION_LEVEL_COST[a] == ACTIVATION_DEGREE[a]
              for a in ACTIVATION_DEGREE))

    # Ground truth from TenSEAL 0.3.16 (validate_depth_model.py): the
    # measured minimum usable levels at which each architecture first executes.
    # These are MEASUREMENTS, not predictions -- if calculate_depth stops
    # reproducing them, the search is optimising a constraint that the library
    # does not enforce, which is exactly the failure this project already made
    # once. Do not "fix" this test by editing the expected values.
    measured = [(['Conv8_3_square'], 2),
                (['Conv8_3_relu4'], 4),
                (['Conv8_3_square'] * 2, 4),
                (['Conv8_3_square'] * 3, 6),
                (['Conv8_3_relu4', 'Conv8_3_square'], 6),
                (['Conv8_3_square', 'Conv8_3_relu4', 'Conv8_3_relu4'], 10)]
    check("matches TenSEAL-measured minimum levels (6 architectures)",
          all(calculate_depth(a) == m for a, m in measured))

    # A depth-d square net consumes 2d levels and therefore needs exactly
    # 2d+2 primes. Anchors both halves of the model at once.
    anchored = True
    for d in range(1, 6):
        arch = ['Conv32_3_square'] * d
        anchored &= check_depth_constraint(arch, [60] * (2 * d + 2))[2]
        anchored &= not check_depth_constraint(arch, [60] * (2 * d + 1))[2]
    check("depth-d square net needs exactly 2d+2 primes", anchored)

    print("Depth budget is satisfiable:")
    # THE guard. This is the invariant whose absence let a 2x-optimistic depth
    # model hide an encoding ceiling: the grammar capped chains at 12 primes
    # (budget 10) while real depths ran to 20, so over half the architectures
    # the search could express had NO valid chain anywhere in the search space.
    # Evolution cannot find what the grammar cannot write down. If this fails,
    # raise MODULUS_SIZES in grammar.py -- do not relax the depth model.
    acts = GRAMMAR['<act>']
    chains = [build_chain(b) for b in DEPTH_BUDGETS]
    secure_budgets = {depth_budget(c) for N in (8192, 16384, 32768) for c in chains
                      if is_secure(N, c)}
    max_encodable = max(DEPTH_BUDGETS)
    max_secure = max(secure_budgets)

    patterns = [[f'Conv32_3_{a}' for a in combo]
                for n in (2, 3, 4, 5)
                for combo in itertools.product(acts, repeat=n)]
    inexpressible = [a for a in patterns if calculate_depth(a) > max_encodable]
    insecure_only = [a for a in patterns
                     if max_secure < calculate_depth(a) <= max_encodable]

    # (a) HARD FAILURE. An architecture the grammar can build but cannot
    # provision is a defect in the ENCODING -- evolution cannot find what it
    # cannot write down. Fix by raising DEPTH_BUDGETS, never by relaxing the
    # depth model.
    check(f"every architecture is encodable (max budget {max_encodable}, "
          f"{len(inexpressible)} inexpressible)", not inexpressible)
    for a in inexpressible[:5]:
        print(f"      !! needs {calculate_depth(a)} levels, grammar tops out "
              f"at {max_encodable}: {','.join(a)}")

    # (b) EXPECTED, NOT A BUG. A canonical budget-B chain totals 120+40B bits
    # against an 881-bit ceiling at N=32768, so B<=19 is 128-bit secure and
    # B=20 is not. Exactly one of the 60 activation patterns (5 layers of
    # relu4, depth 20) is therefore ruled out by cryptography rather than by
    # the encoding. That is a genuine security/depth limit and belongs in the
    # results; pinning the count here means a change in it gets noticed.
    check(f"exactly 1 of {len(patterns)} architectures excluded by 128-bit "
          f"security (max secure budget {max_secure})", len(insecure_only) == 1)
    for a in insecure_only:
        print(f"      -- security-limited (expected): needs "
              f"{calculate_depth(a)} levels: {','.join(a)}")

    print("Security estimation:")
    check("N=16384 M=[60,60,60] secure", is_secure(16384, [60, 60, 60]))
    check("N=8192 M=[60,60,60,60] insecure", not is_secure(8192, [60, 60, 60, 60]))

    print("Partial-genome freezing:")
    rng = random.Random(0)
    ok_ckks = ok_arch = True
    for _ in range(200):
        template = [rng.randint(0, 255)
                    for _ in range(rng.randint(GENOME_MIN, GENOME_MAX))]
        ckks, arch = decode_genome(freeze_ckks(template, B1_CKKS))
        ok_ckks &= (ckks == B1_CKKS and 2 <= len(arch) <= 5)
        ckks, arch = decode_genome(freeze_arch(template, B2_ARCH))
        ok_arch &= (arch == ["Conv64_3_square", "Conv128_3_square", "Conv128_3_square"])
    check("freeze_ckks x200", ok_ckks)
    check("freeze_arch x200", ok_arch)

    print("Hypervolume:")
    check("HV single point", abs(hypervolume3d([(1, 1, 1)]) - 1.0) < 1e-9)
    check("HV two boxes", abs(hypervolume3d([(2, 1, 1), (1, 2, 2)]) - 5.0) < 1e-9)
    check("HV dominated point ignored",
          abs(hypervolume3d([(2, 2, 2), (1, 1, 1)]) - 8.0) < 1e-9)

    print("Non-dominated sort:")
    objs = [(1, 1, 1), (2, 2, 2), (0.5, 3, 1), (0.1, 0.1, 0.1)]
    fronts = fast_nondominated_sort(objs)
    check("sort fronts", fronts[0] == [1, 2] and 0 in fronts[1] and 3 in fronts[-1])

    print("Model builder (torch):")
    try:
        import torch
        from fitness import build_model
        for act in ('square', 'relu4'):
            model = build_model([f'Conv32_3_{act}+Pool', f'Conv64_5_{act}'], 1, 10)
            out = model(torch.randn(2, 1, 28, 28))
            check(f"forward pass shape ({act})", tuple(out.shape) == (2, 10))
    except ImportError:
        print("  [SKIP] torch not installed")

    print("Stats (scipy):")
    try:
        c = compare_methods([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        check("compare_methods runs", 0.0 <= c["p_value"] <= 1.0)
    except ImportError:
        print("  [SKIP] scipy/numpy not installed")

    print(f"\nSelftest: {'ALL PASSED' if not failures else 'FAILED: ' + str(failures)}")
    return 1 if failures else 0
