"""champions.py — retrain the best config from every experiment at full scale.

Standalone reporting helper (drop-in, single file). It scans `experiments/`,
pulls the single best (architecture, CKKS) config out of each experiment, then
retrains that champion on the FULL dataset -- the canonical train/test split,
full epochs -- and reports final PLAINTEXT test accuracy next to its depth and
security. This is the headline "champion numbers" table for the writeup,
distinct from the fast, sub-sampled fitness the search used to rank candidates.

It reuses the project's own training path (`fitness.train_and_evaluate`), so
the numbers are produced exactly as in the search, only at full data + epochs.
All imports are read-only; nothing in the search path changes.

This one DOES need the GPU (it retrains a handful of CNNs) -- minutes on an
H100. It is plaintext accuracy, not encrypted; encrypted validation is the
separate `validate_ckks.py`.

    python champions.py                              # every experiment, full data, 30 epochs
    python champions.py --epochs 50 --batch-size 256
    python champions.py --experiments-dir experiments --out results/champions.json
    python champions.py --only ge_cifar10 nsga2_cifar10     # just these experiments
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from ckks import check_depth_constraint, estimate_security_bits, is_secure
from config import Cfg, EXPERIMENTS_DIR, RESULTS_DIR
from fitness import set_global_seed, train_and_evaluate

_DATASETS = ("fashion_mnist", "cifar10", "mnist")   # longest-match-first for name inference


def _infer_dataset(explicit, name):
    if explicit:
        return explicit
    for ds in _DATASETS:
        if ds in name:
            return ds
    return None


def _score(d):
    for k in ("accuracy", "best_fitness", "fitness"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    objs = d.get("objectives")
    if isinstance(objs, (list, tuple)) and objs:
        return float(objs[0])
    return float("-inf")


def _as_config(d):
    """Normalise one dict to (arch, poly, chain, score) or None."""
    arch, poly, chain = d.get("architecture"), d.get("poly_degree"), d.get("modulus_chain")
    if poly is None or chain is None:
        ck = d.get("ckks_params") or d.get("ckks") or {}
        poly = poly if poly is not None else ck.get("poly_degree")
        chain = chain if chain is not None else ck.get("modulus_chain")
    if not (isinstance(arch, list) and poly and isinstance(chain, list)):
        return None
    return {"architecture": arch, "poly_degree": int(poly),
            "modulus_chain": [int(x) for x in chain], "score": _score(d)}


def _configs_in(obj):
    """Every (arch, CKKS) candidate anywhere in one loaded JSON."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            c = _as_config(o)
            if c:
                out.append(c)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def _deployable(c):
    """Is this config actually shippable: 128-bit secure AND depth-feasible?

    NSGA-II stores a whole Pareto front, and _score falls back to the first
    objective (accuracy), so an unfiltered argmax returns the accuracy-extreme
    corner of the front -- which is routinely depth-infeasible. Reporting that
    as "the champion" is how a configuration that cannot be encrypted at all
    ends up in the results table. Rank deployable configs above the rest, and
    only fall back to an infeasible one if an experiment produced nothing else.
    """
    _, _, valid = check_depth_constraint(c["architecture"], c["modulus_chain"])
    return valid and is_secure(c["poly_degree"], c["modulus_chain"])


def find_champions(experiments_dir, only=None):
    """Best deployable config per experiment. Returns {experiment_name: champion}."""
    experiments_dir = Path(experiments_dir)
    groups = {}          # experiment name -> (best_config, dataset)
    for path in experiments_dir.rglob("*.json"):
        if path.name not in ("results.json", "pareto_front_final.json"):
            continue
        # experiment = the dir that holds the run_seed* folders.
        try:
            name = str(path.parent.parent.relative_to(experiments_dir)).replace("\\", "/")
        except ValueError:
            continue
        if only and name not in only:
            continue
        try:
            obj = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        dataset = obj.get("dataset") if isinstance(obj, dict) else None
        for c in _configs_in(obj):
            c["dataset"] = _infer_dataset(dataset, name)
            c["deployable"] = _deployable(c)
            best, _ = groups.get(name, (None, None))
            # Lexicographic: deployable first, then score. A deployable config
            # always beats an infeasible one regardless of accuracy.
            key = (c["deployable"], c["score"])
            if best is None or key > (best["deployable"], best["score"]):
                groups[name] = (c, c["dataset"])
    return {k: v[0] for k, v in groups.items()}


def retrain_champion(champ, args):
    """Full-scale plaintext retrain. Returns the champion dict + measured fields."""
    dataset = champ.get("dataset") or args.dataset
    if dataset is None:
        return {**champ, "error": "unknown dataset (pass --dataset)"}
    cfg = Cfg(dataset=dataset, train_samples=args.train_samples,
              test_samples=args.test_samples, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)
    set_global_seed(args.seed)
    depth, budget, valid = check_depth_constraint(champ["architecture"],
                                                  champ["modulus_chain"])
    sec = estimate_security_bits(champ["poly_degree"], champ["modulus_chain"])
    try:
        acc = train_and_evaluate(champ["architecture"], cfg)
        return {**champ, "dataset": dataset, "full_scale_accuracy": acc,
                "depth": depth, "budget": budget, "depth_valid": valid,
                "security_bits": sec, "epochs": args.epochs}
    except Exception as e:
        return {**champ, "dataset": dataset, "error": f"{type(e).__name__}: {e}",
                "depth": depth, "budget": budget, "security_bits": sec}


def _print_table(rows):
    print("\n" + "=" * 96)
    print("CHAMPION CONFIGS -- full-scale plaintext test accuracy")
    print("=" * 96)
    print(f"{'experiment':<26}{'dataset':<14}{'acc':>7}  {'depth/bud':>10}  "
          f"{'sec':>7}  arch")
    print("-" * 96)
    for name, r in rows:
        if "full_scale_accuracy" in r:
            acc = f"{r['full_scale_accuracy']:.4f}"
            db = f"{r['depth']}/{r['budget']}" + ("" if r["depth_valid"] else "!")
            sec = f"{r['security_bits']:.0f}"
        else:
            acc, db, sec = "ERR", "-", "-"
        arch = ",".join(r["architecture"])
        if len(arch) > 40:
            arch = arch[:37] + "..."
        print(f"{name:<26}{str(r.get('dataset','?')):<14}{acc:>7}  {db:>10}  "
              f"{sec:>7}  {arch}")
        if "error" in r:
            print(f"{'':<26}-> {r['error']}")
    print("-" * 96)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrain every experiment's champion at full scale.")
    ap.add_argument("--experiments-dir", default=str(EXPERIMENTS_DIR))
    ap.add_argument("--only", nargs="*", help="restrict to these experiment names")
    ap.add_argument("--dataset", default=None, help="override / fallback dataset")
    ap.add_argument("--train-samples", type=int, default=100000, help="caps to full split")
    ap.add_argument("--test-samples", type=int, default=100000, help="caps to full split")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS_DIR / "champions.json"))
    args = ap.parse_args(argv)

    champs = find_champions(args.experiments_dir, set(args.only) if args.only else None)
    if not champs:
        print(f"No finished experiments found under {args.experiments_dir}.")
        return 1

    print(f"Found {len(champs)} experiment champion(s); retraining at full scale "
          f"({args.epochs} epochs)...")
    rows = []
    for name in sorted(champs):
        print(f"  [{name}] retraining ...", flush=True)
        rows.append((name, retrain_champion(champs[name], args)))

    _print_table(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({n: r for n, r in rows}, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
