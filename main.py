#!/usr/bin/env python3
"""
HE-NAS entry point: experiment drivers + CLI.

    python main.py selftest                 # fast unit checks, no GPU needed
    python main.py download --dataset mnist # pre-download data (DRAC login node)
    python main.py smoke                    # tiny end-to-end pipeline check
    python main.py ge         [flags]       # single-objective GE (main result)
    python main.py random     [flags]       # random-search baseline
    python main.py sequential --which ckks_first|arch_first   # B1 / B2
    python main.py expert     [flags]       # B3 + B4 expert baselines
    python main.py ablation   [flags]       # 2x2 penalty x elitism ablation
    python main.py nsga2      [flags]       # multi-objective NSGA-II
    python main.py stats A.json B.json [metric]   # Mann-Whitney + effect sizes

Common flags (see `python main.py <cmd> --help`):
    --dataset mnist|fashion_mnist|cifar10   --seeds N --seed-base 42
    --population 20 --generations 15 --train-samples 10000 --test-samples 500
    --epochs 30 --penalty soft|hard --no-elitism --name <experiment_name>
    --smoke  (shrink everything for a fast sanity run)




Outputs:
    experiments/<name>/run_seed<seed>/results.json  (+ per-generation logs)
    results/<name>_summary.json   (per-seed best_fitness list, feeds `stats`)
"""
import argparse
import json
import random
from dataclasses import replace
from datetime import datetime

from config import (ABLATION_CONDITIONS, B1_CKKS, B2_ARCH, B3_CONFIG, B4_CONFIG,
                    Cfg, EXPERIMENTS_DIR, RESULTS_DIR, SMOKE_OVERRIDES,
                    ensure_dirs)
from grammar import CODON_MAX, decode_genome, freeze_arch, freeze_ckks
from profiling import (Timer, aggregate_profiles, device_info, format_report,
                       profile_entry)


# --------------------------------------------------------------------------- #
# Results plumbing shared by all drivers.
# --------------------------------------------------------------------------- #
def build_results(best, history, seed, cfg, method):
    meta = best.get("metadata", {}) if isinstance(best, dict) else {}
    return {
        "best": best,
        "best_solution": best,
        "history": history,
        "seed": seed,
        "method": method,
        "dataset": cfg.dataset,
        "experiment_name": cfg.name,
        "best_fitness": best.get("fitness") if isinstance(best, dict) else None,
        "best_accuracy": meta.get("accuracy"),
        "depth_valid": meta.get("depth_valid"),
        "population_size": cfg.population,
        "generations": cfg.generations,
        "timestamp": datetime.now().isoformat(),
    }


def resume_seed(cfg, run_dir):
    """Seed-level resume: if this seed already produced a valid results.json,
    return its per-seed summary entry so a re-launched (e.g. SLURM-preempted)
    run skips it instead of recomputing. Returns None when the seed must run
    (resume disabled, no file, or a truncated/corrupt file from a killed write).
    """
    if not cfg.resume:
        return None
    path = run_dir / "results.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            r = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None   # partial write from an interrupted run -> recompute
    return {"seed": r.get("seed"), "best_fitness": r.get("best_fitness"),
            "best_accuracy": r.get("best_accuracy"),
            "depth_valid": r.get("depth_valid"),
            "profile": r.get("profile")}


def write_summary(name, method, per_seed):
    """results/<name>_summary.json with the per-seed best_fitness list, in the
    shape the `stats` subcommand consumes directly. Also folds in a profiling
    roll-up and an environment stamp when per-seed timings are present."""
    summary = {
        "experiment_name": name,
        "method": method,
        "seeds": [p["seed"] for p in per_seed],
        "best_fitness": [p["best_fitness"] for p in per_seed],
        "best_accuracy": [p.get("best_accuracy") for p in per_seed],
        "per_seed_detail": per_seed,
    }
    profiles = [p["profile"] for p in per_seed if p.get("profile")]
    if profiles:
        summary["environment"] = device_info()
        summary["profiling"] = aggregate_profiles(profiles)
    path = RESULTS_DIR / f"{name.replace('/', '_')}_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {path}")
    return summary


# --------------------------------------------------------------------------- #
# Experiment drivers.
# --------------------------------------------------------------------------- #
def run_ge(cfg, method="ge", ge_factory=None):
    """Run n_seeds independent GE runs; ge_factory allows FrozenGE injection."""
    from evolution import GrammaticalEvolution
    from fitness import set_global_seed

    per_seed = []
    for seed in cfg.seeds:
        run_dir = EXPERIMENTS_DIR / cfg.name / f"run_seed{seed}"
        done = resume_seed(cfg, run_dir)
        if done is not None:
            print(f"[resume] {cfg.name} seed={seed} already complete, skipping")
            per_seed.append(done)
            continue
        set_global_seed(seed)
        ge = (ge_factory(cfg, seed) if ge_factory
              else GrammaticalEvolution(cfg, seed))
        with Timer() as t:
            best, history = ge.evolve(run_dir)
        prof = profile_entry(t.seconds, cfg.population * cfg.generations,
                             t.peak_gpu_mb)
        results = build_results(best, history, seed, cfg, method)
        results["profile"] = prof
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        per_seed.append({"seed": seed, "best_fitness": results["best_fitness"],
                         "best_accuracy": results["best_accuracy"],
                         "depth_valid": results["depth_valid"], "profile": prof})
        print(f"[{cfg.name} seed={seed}] best_fitness={results['best_fitness']:.4f}"
              f"  ({prof['wall_clock_seconds']}s, "
              f"{prof['evals_per_second']} evals/s)")
    if cfg.write_summary:
        write_summary(cfg.name, method, per_seed)
    return per_seed


def run_random_search(cfg, n_evaluations, genome_length=40):
    """Random-search baseline over the same search space and fitness."""
    from fitness import evaluate_fitness, set_global_seed

    per_seed = []
    for seed in cfg.seeds:
        run_dir = EXPERIMENTS_DIR / cfg.name / f"run_seed{seed}"
        done = resume_seed(cfg, run_dir)
        if done is not None:
            print(f"[resume] {cfg.name} seed={seed} already complete, skipping")
            per_seed.append(done)
            continue
        set_global_seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        fitnesses, history = [], []
        best = {"fitness": -1.0, "evaluation": -1}
        print(f"\n{'=' * 60}\nRandom search seed={seed}: {n_evaluations} "
              f"evaluations\n{'=' * 60}")
        with Timer() as t:
            for i in range(n_evaluations):
                genome = [random.randint(0, CODON_MAX) for _ in range(genome_length)]
                try:
                    ckks_params, architecture = decode_genome(genome)
                    fitness, metadata = evaluate_fitness(ckks_params, architecture, cfg)
                except Exception as e:
                    fitness, metadata = 0.0, {"error": str(e), "fitness": 0.0}
                    ckks_params = architecture = None
                fitnesses.append(fitness)
                if fitness > best["fitness"]:
                    best = {"fitness": fitness, "evaluation": i, "genome": genome,
                            "ckks_params": ckks_params, "architecture": architecture,
                            "metadata": metadata}
                valid = [f for f in fitnesses if f > 0]
                stats = {"best_fitness": max(fitnesses),
                         "mean_fitness": sum(fitnesses) / len(fitnesses),
                         "valid_count": len(valid),
                         "invalid_count": len(fitnesses) - len(valid),
                         "best_valid_mean": sum(valid) / len(valid) if valid else 0.0}
                history.append({"generation": i, **stats})
                with open(run_dir / f"eval_{i:03d}.json", "w") as f:
                    json.dump({"generation": i, "statistics": stats,
                               "metadata": [metadata]}, f, indent=2)
                print(f"Eval {i:3d}: fitness={fitness:.4f} "
                      f"best_so_far={stats['best_fitness']:.4f}")

        prof = profile_entry(t.seconds, n_evaluations, t.peak_gpu_mb)
        results = build_results(best, history, seed, cfg, "random_search")
        results["n_evaluations"] = n_evaluations
        results["wall_clock_seconds"] = prof["wall_clock_seconds"]
        results["profile"] = prof
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        per_seed.append({"seed": seed, "best_fitness": best["fitness"],
                         "best_accuracy": best.get("metadata", {}).get("accuracy"),
                         "depth_valid": best.get("metadata", {}).get("depth_valid"),
                         "profile": prof})
    if cfg.write_summary:
        write_summary(cfg.name, "random_search", per_seed)
    return per_seed


def run_expert_baselines(cfg):
    """B3 (properly provisioned) and B4 (naive) expert configs, many seeds."""
    from fitness import evaluate_fitness, set_global_seed

    out = {}
    for name, config in (("b3_expert_provisioned", B3_CONFIG),
                         ("b4_expert_naive", B4_CONFIG)):
        per_seed = []
        for seed in cfg.seeds:
            run_dir = EXPERIMENTS_DIR / name / f"run_seed{seed}"
            done = resume_seed(cfg, run_dir)
            if done is not None:
                print(f"[resume] {name} seed={seed} already complete, skipping")
                per_seed.append(done)
                continue
            set_global_seed(seed)
            with Timer() as t:
                fitness, metadata = evaluate_fitness(config["ckks"],
                                                     config["architecture"], cfg)
            prof = profile_entry(t.seconds, 1, t.peak_gpu_mb)
            best = {"fitness": fitness, "generation": 0, "genome": None,
                    "ckks_params": config["ckks"],
                    "architecture": config["architecture"], "metadata": metadata}
            history = [{"generation": 0, "best_fitness": fitness,
                        "mean_fitness": fitness,
                        "valid_count": 1 if fitness > 0 else 0,
                        "invalid_count": 0 if fitness > 0 else 1,
                        "best_valid_mean": fitness if fitness > 0 else 0.0}]
            results = build_results(best, history, seed, cfg, name)
            results["profile"] = prof
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            per_seed.append({"seed": seed, "best_fitness": fitness,
                             "best_accuracy": metadata.get("accuracy"),
                             "depth_valid": metadata.get("depth_valid"),
                             "profile": prof})
            print(f"  [{name} seed={seed}] accuracy="
                  f"{metadata.get('accuracy', 0.0):.4f} fitness={fitness:.4f}")
        if cfg.write_summary:
            out[name] = write_summary(name, name, per_seed)
    return out


def run_ablation(cfg):
    """2x2 penalty x elitism ablation; each condition is a full GE study."""
    summary = {}
    for cond in ABLATION_CONDITIONS:
        cond_cfg = replace(cfg, name=f"{cfg.name}/{cond['name']}",
                           penalty=cond["penalty"], elitism=cond["elitism"])
        per_seed = run_ge(cond_cfg, method=cond["name"])
        summary[cond["name"]] = {"penalty_type": cond["penalty"],
                                 "elitism": cond["elitism"],
                                 "per_seed": per_seed}
    with open(RESULTS_DIR / f"{cfg.name.replace('/', '_')}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def add_common_flags(p, pop=20, gens=15, seeds=30):
    from fitness import DATASET_SPECS
    # Must match config.Cfg.dataset -- the CLI default wins over the dataclass,
    # so a mismatch silently runs the whole study on the wrong dataset.
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_SPECS))
    p.add_argument("--name", default=None,
                   help="Experiment name (default: <command>_<dataset>)")
    p.add_argument("--seeds", type=int, default=seeds, dest="n_seeds",
                   help=f"Number of seeds (default {seeds})")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--population", type=int, default=pop)
    p.add_argument("--generations", type=int, default=gens)
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--test-samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--penalty", default="soft", choices=["soft", "hard"])
    p.add_argument("--no-elitism", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="Recompute all seeds even if results.json exists "
                        "(default: resume by skipping completed seeds)")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip writing results/<name>_summary.json. Used by the "
                        "parallel single-seed workers in submit.sh so they do "
                        "not race on one file; the aggregation pass writes it.")
    p.add_argument("--smoke", action="store_true",
                   help="Shrink everything for a fast sanity run")


def cfg_from_args(args, default_name):
    cfg = Cfg(name=args.name or f"{default_name}_{args.dataset}",
              dataset=args.dataset, n_seeds=args.n_seeds,
              seed_base=args.seed_base, population=args.population,
              generations=args.generations, elitism=not args.no_elitism,
              penalty=args.penalty, train_samples=args.train_samples,
              test_samples=args.test_samples, epochs=args.epochs,
              lr=args.lr, batch_size=args.batch_size,
              resume=not args.fresh,
              write_summary=not args.no_summary)
    if args.smoke:
        cfg = replace(cfg, **SMOKE_OVERRIDES)
    return cfg


def banner(title, cfg):
    print("\n" + "=" * 70)
    print(f"{title} | experiment={cfg.name} dataset={cfg.dataset}")
    print(f"seeds={cfg.seeds} pop={cfg.population} gens={cfg.generations} "
          f"epochs={cfg.epochs} penalty={cfg.penalty} elitism={cfg.elitism}")
    print("=" * 70)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selftest", help="Fast unit checks (no GPU needed)")

    p = sub.add_parser("download", help="Pre-download datasets (DRAC login node)")
    p.add_argument("--dataset", default="all",
                   choices=["all", "mnist", "fashion_mnist", "cifar10"])

    p = sub.add_parser("smoke", help="Tiny end-to-end GE run (~minutes)")
    add_common_flags(p)

    p = sub.add_parser("ge", help="Single-objective GE (the main result)")
    add_common_flags(p)

    p = sub.add_parser("random", help="Random-search baseline")
    add_common_flags(p)
    p.add_argument("--evaluations", type=int, default=None,
                   help="Budget per seed (default population x generations)")

    p = sub.add_parser("sequential", help="Sequential one-sided baselines B1/B2")
    p.add_argument("--which", required=True, choices=["ckks_first", "arch_first"])
    add_common_flags(p)

    p = sub.add_parser("expert", help="Expert baselines B3 (provisioned) + B4 (naive)")
    add_common_flags(p)

    p = sub.add_parser("ablation", help="2x2 penalty x elitism ablation")
    add_common_flags(p)

    p = sub.add_parser("nsga2", help="Multi-objective NSGA-II")
    add_common_flags(p, pop=40, gens=25, seeds=5)

    p = sub.add_parser("stats", help="Compare two summary JSONs (Mann-Whitney etc.)")
    p.add_argument("results_a")
    p.add_argument("results_b")
    p.add_argument("metric", nargs="?", default="best_fitness")

    p = sub.add_parser("profile", help="Print the runtime/GPU profiling report "
                                       "for a results/<name>_summary.json")
    p.add_argument("summary_json")

    args = parser.parse_args(argv)

    if args.command == "selftest":
        from analysis import run_selftest
        return run_selftest()
    if args.command == "profile":
        with open(args.summary_json) as f:
            print(format_report(json.load(f)))
        return 0
    if args.command == "stats":
        from analysis import run_stats
        ensure_dirs()
        run_stats(args.results_a, args.results_b, args.metric)
        return 0

    ensure_dirs()

    if args.command == "download":
        from fitness import DATASET_SPECS, load_dataset
        names = (sorted(DATASET_SPECS) if args.dataset == "all"
                 else [args.dataset])
        for name in names:
            print(f"Downloading {name} ...")
            load_dataset(name, 1, 1, seed=0)
        print("Done. Datasets are cached under data/.")
        return 0

    if args.command == "smoke":
        args.smoke = True
        cfg = cfg_from_args(args, "smoke_test")
        banner("SMOKE TEST", cfg)
        run_ge(cfg, method="ge")
        print("Smoke test complete: check experiments/ and results/.")
        return 0

    if args.command == "ge":
        cfg = cfg_from_args(args, "ge")
        banner("GE single-objective", cfg)
        run_ge(cfg, method="ge")
    elif args.command == "random":
        cfg = cfg_from_args(args, "random_search")
        n_eval = args.evaluations or cfg.population * cfg.generations
        banner(f"RANDOM SEARCH ({n_eval} evals/seed)", cfg)
        run_random_search(cfg, n_eval)
    elif args.command == "sequential":
        from evolution import FrozenGE
        if args.which == "ckks_first":
            cfg = cfg_from_args(args, "seq_ckks_first")
            banner("SEQUENTIAL B1 (CKKS frozen, arch evolves)", cfg)
            run_ge(cfg, method="sequential_ckks_first",
                   ge_factory=lambda c, s: FrozenGE(c, s, freeze_ckks, B1_CKKS))
        else:
            cfg = cfg_from_args(args, "seq_arch_first")
            banner("SEQUENTIAL B2 (arch frozen, CKKS evolves)", cfg)
            run_ge(cfg, method="sequential_arch_first",
                   ge_factory=lambda c, s: FrozenGE(c, s, freeze_arch, B2_ARCH))
    elif args.command == "expert":
        cfg = cfg_from_args(args, "expert")
        banner("EXPERT BASELINES B3 + B4", cfg)
        run_expert_baselines(cfg)
    elif args.command == "ablation":
        cfg = cfg_from_args(args, "ablation")
        banner("ABLATION (penalty x elitism)", cfg)
        run_ablation(cfg)
    elif args.command == "nsga2":
        from evolution import run_nsga2
        cfg = cfg_from_args(args, "nsga2")
        if args.smoke:
            cfg = replace(cfg, population=8, generations=3)
        banner("NSGA-II multi-objective", cfg)
        run_nsga2(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
