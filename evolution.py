"""Evolutionary algorithms: single-objective GE and multi-objective NSGA-II.

GrammaticalEvolution: tournament selection, two-point crossover, point
mutation, optional elitism. FrozenGE re-imposes a genome freeze before every
evaluation (used by the sequential baselines).

NSGA-II is self-contained (no DEAP / moocore). Three MAXIMIZE objectives:
  accuracy in [0,1], security_margin = max(0, sec_bits-128),
  depth_headroom = max(0, budget-depth).
Hypervolume vs. reference point (0,0,0) via z-sweep over 2D staircase areas.
"""
import json
import math
import random
from pathlib import Path

from ckks import check_depth_constraint, estimate_security_bits
from config import EXPERIMENTS_DIR, RESULTS_DIR
from fitness import evaluate_fitness, set_global_seed
from grammar import CODON_MAX, decode_genome, random_genome
from profiling import Timer, device_info, profile_entry


class GrammaticalEvolution:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.seed = seed
        self.population = []
        self.fitnesses = []
        self.metadata_list = []

    def initialize_population(self):
        self.population = [random_genome() for _ in range(self.cfg.population)]

    def tournament_selection(self):
        candidates = random.sample(range(self.cfg.population),
                                   self.cfg.tournament_size)
        winner = max(candidates, key=lambda i: self.fitnesses[i])
        return self.population[winner][:]

    def two_point_crossover(self, parent1, parent2):
        if random.random() > self.cfg.crossover_prob:
            return parent1[:], parent2[:]
        min_len = min(len(parent1), len(parent2))
        if min_len < 3:
            return parent1[:], parent2[:]
        p1 = random.randint(1, min_len - 2)
        p2 = random.randint(p1 + 1, min_len - 1)
        return (parent1[:p1] + parent2[p1:p2] + parent1[p2:],
                parent2[:p1] + parent1[p1:p2] + parent2[p2:])

    def point_mutation(self, genome):
        mutated = genome[:]
        for i in range(len(mutated)):
            if random.random() < self.cfg.mutation_prob:
                mutated[i] = random.randint(0, CODON_MAX)
        return mutated

    def evaluate_population(self):
        self.fitnesses, self.metadata_list = [], []
        for genome in self.population:
            try:
                ckks_params, architecture = decode_genome(genome)
                fitness, metadata = evaluate_fitness(ckks_params, architecture,
                                                     self.cfg)
            except Exception as e:
                fitness, metadata = 0.0, {'error': str(e), 'fitness': 0.0}
            self.fitnesses.append(fitness)
            self.metadata_list.append(metadata)

    def get_statistics(self):
        valid = [f for f in self.fitnesses if f > 0]
        n = len(self.fitnesses)
        return {
            'best_fitness': max(self.fitnesses) if self.fitnesses else 0.0,
            'mean_fitness': sum(self.fitnesses) / n if n else 0.0,
            'valid_count': len(valid),
            'invalid_count': n - len(valid),
            'best_valid_mean': sum(valid) / len(valid) if valid else 0.0,
        }

    def evolve(self, run_dir):
        if self.seed is not None:
            random.seed(self.seed)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}\nGE run -> {run_dir} (seed={self.seed}, "
              f"pop={self.cfg.population}, gens={self.cfg.generations})\n{'=' * 60}")

        self.initialize_population()
        best_ever = {'fitness': -1, 'generation': -1}
        history = []

        for gen in range(self.cfg.generations):
            print(f"Gen {gen:2d}: evaluating {self.cfg.population} individuals...",
                  end=' ', flush=True)
            self.evaluate_population()
            stats = self.get_statistics()
            print(f"best={stats['best_fitness']:.4f} mean={stats['mean_fitness']:.4f} "
                  f"valid={stats['valid_count']}/{self.cfg.population}")

            best_idx = self.fitnesses.index(max(self.fitnesses))
            if self.fitnesses[best_idx] > best_ever['fitness']:
                try:
                    ckks_params, architecture = decode_genome(self.population[best_idx])
                    best_ever = {
                        'fitness': self.fitnesses[best_idx], 'generation': gen,
                        'genome': self.population[best_idx],
                        'ckks_params': ckks_params, 'architecture': architecture,
                        'metadata': self.metadata_list[best_idx],
                    }
                except Exception:
                    pass

            history.append({'generation': gen, **stats})
            with open(run_dir / f'gen_{gen:03d}.json', 'w') as f:
                json.dump({'generation': gen, 'statistics': stats,
                           'metadata': self.metadata_list}, f, indent=2)

            if gen < self.cfg.generations - 1:
                new_population = []
                if self.cfg.elitism:
                    new_population.append(self.population[best_idx][:])
                while len(new_population) < self.cfg.population:
                    child1, child2 = self.two_point_crossover(
                        self.tournament_selection(), self.tournament_selection())
                    new_population.append(self.point_mutation(child1))
                    new_population.append(self.point_mutation(child2))
                self.population = new_population[:self.cfg.population]

        print(f"Run complete. Best fitness: {best_ever['fitness']:.4f}\n")
        return best_ever, history


class FrozenGE(GrammaticalEvolution):
    """GE that re-imposes a freeze on the population before every evaluation,
    so one half of the genome stays constant while the other evolves."""

    def __init__(self, cfg, seed, freeze_fn, freeze_config):
        super().__init__(cfg, seed)
        self._freeze_fn = freeze_fn
        self._freeze_config = freeze_config

    def evaluate_population(self):
        self.population = [self._freeze_fn(ind, self._freeze_config)
                           for ind in self.population]
        super().evaluate_population()


# --------------------------------------------------------------------------- #
# NSGA-II building blocks.
# --------------------------------------------------------------------------- #
# Worst possible signed headroom: deepest architecture (5 x relu4 = 20 levels)
# against the shortest chain (2 primes, budget 0). Used to shift the headroom
# objective positive so hypervolume can see it -- see evaluate_multiobjective.
_HEADROOM_FLOOR = -20.0


def evaluate_multiobjective(genome, cfg):
    """Return (accuracy, security_margin, depth_headroom) — all MAXIMIZE.

    depth_headroom is SIGNED (shifted positive by _HEADROOM_FLOOR). It used to
    be max(0, budget-depth), which clamped every infeasible configuration to
    exactly the same value as a tight-but-valid one, so NSGA-II could not
    distinguish them on that axis at all and the selected champion was routinely
    the accuracy-extreme, depth-infeasible corner of the front. Signing it
    restores the gradient; the shift keeps every coordinate positive, which
    hypervolume3d requires (it drops non-positive points against ref (0,0,0)).
    """
    try:
        ckks, arch = decode_genome(list(genome))
    except Exception:
        return (0.0, 0.0, 0.0)
    sec_bits = estimate_security_bits(ckks['poly_degree'], ckks['modulus_chain'])
    depth, budget, _ = check_depth_constraint(arch, ckks['modulus_chain'])
    security_margin = max(0.0, float(sec_bits) - 128.0)
    depth_headroom = float(budget - depth) - _HEADROOM_FLOOR
    _, meta = evaluate_fitness(ckks, arch, cfg)
    return (float(meta.get('accuracy', 0.0)), security_margin, depth_headroom)


def dominates(a, b):
    """Maximization dominance: a >= b everywhere and > somewhere."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def fast_nondominated_sort(objs):
    """Return list of fronts (lists of indices), best front first."""
    n = len(objs)
    dominated_by = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                dominated_by[i].append(j)
            elif dominates(objs[j], objs[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
        k += 1
    return fronts[:-1]


def crowding_distance(objs, front):
    """Crowding distance for the indices in `front`."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    n_obj = len(objs[front[0]])
    for m in range(n_obj):
        ordered = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[ordered[0]][m], objs[ordered[-1]][m]
        dist[ordered[0]] = dist[ordered[-1]] = math.inf
        if hi == lo:
            continue
        for k in range(1, len(ordered) - 1):
            dist[ordered[k]] += ((objs[ordered[k + 1]][m] - objs[ordered[k - 1]][m])
                                 / (hi - lo))
    return dist


def _staircase_area(points):
    """Area of the union of anchored rectangles [0,x] x [0,y]."""
    area, best_y = 0.0, 0.0
    for x, y in sorted(set(points), key=lambda p: -p[0]):
        if y > best_y:
            area += x * (y - best_y)
            best_y = y
    return area


def hypervolume3d(points, ref=(0.0, 0.0, 0.0)):
    """Hypervolume (maximization) of 3D points vs. reference point `ref`."""
    pts = [tuple(p[i] - ref[i] for i in range(3)) for p in points]
    pts = [p for p in pts if all(v > 0 for v in p)]
    if not pts:
        return 0.0
    zs = sorted({p[2] for p in pts}, reverse=True) + [0.0]
    vol = 0.0
    for k in range(len(zs) - 1):
        active = [(p[0], p[1]) for p in pts if p[2] >= zs[k]]
        vol += _staircase_area(active) * (zs[k] - zs[k + 1])
    return vol


def update_archive(archive, new_entries):
    """Merge new (genome, objs) entries into a non-dominated archive."""
    merged = archive + list(new_entries)
    keep = []
    for i, (_, oi) in enumerate(merged):
        is_dominated = any(dominates(oj, oi) for j, (_, oj) in enumerate(merged)
                           if i != j)
        duplicate = any(oj == oi for _, oj in keep)
        if not is_dominated and not duplicate:
            keep.append(merged[i])
    return keep


def _front_to_dicts(entries):
    out = []
    for genome, objs in entries:
        acc, sec, dep = objs
        try:
            ckks, arch = decode_genome(list(genome))
            poly, mod = ckks['poly_degree'], ckks['modulus_chain']
        except Exception:
            arch = poly = mod = None
        out.append({'genome': list(genome), 'objectives': list(objs),
                    'accuracy': acc, 'security_margin': sec,
                    'depth_headroom': dep, 'architecture': arch,
                    'poly_degree': poly, 'modulus_chain': mod})
    return out


# --------------------------------------------------------------------------- #
# NSGA-II driver.
# --------------------------------------------------------------------------- #
def _resume_nsga2(cfg, run_dir, seed):
    """Seed-level resume: rebuild a per-seed summary from disk if this NSGA-II
    run already finished (pareto_front_final.json is written last). Returns None
    when the seed must run (resume off, unfinished, or corrupt/partial files)."""
    if not cfg.resume:
        return None
    if not (run_dir / "pareto_front_final.json").exists():
        return None
    try:
        with open(run_dir / "pareto_front_final.json") as f:
            pareto = json.load(f)
        with open(run_dir / "hypervolume_history.json") as f:
            hv_history = json.load(f)
        with open(run_dir / "all_evaluations.json") as f:
            n_eval = len(json.load(f))
        with open(run_dir / "profile.json") as f:
            prof = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None   # partial write from an interrupted run -> recompute
    if not hv_history:
        return None
    return {
        'seed': seed, 'dataset': cfg.dataset, 'population': cfg.population,
        'generations': cfg.generations, 'pareto_size': len(pareto),
        'hypervolume_final': hv_history[-1]['hypervolume'],
        'hypervolume_history': [h['hypervolume'] for h in hv_history],
        'n_evaluations': n_eval, 'profile': prof,
    }


def run_nsga2(cfg):
    """NSGA-II for each seed; writes Pareto/hypervolume logs per run."""
    summaries = {}
    for seed in cfg.seeds:
        run_dir = EXPERIMENTS_DIR / cfg.name / f"run_seed{seed}"
        done = _resume_nsga2(cfg, run_dir, seed)
        if done is not None:
            print(f"[resume] NSGA-II seed={seed} already complete, skipping")
            summaries[f'run_seed{seed}'] = done
            continue
        set_global_seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        mu = cfg.population
        print(f"\n{'=' * 60}\nNSGA-II seed={seed} dataset={cfg.dataset} "
              f"(pop={mu}, gens={cfg.generations}, epochs={cfg.epochs})\n{'=' * 60}")

        all_evaluations, hv_history, per_generation = [], [], []
        archive = []
        ge_ops = GrammaticalEvolution(cfg, seed)   # reuse variation operators

        def evaluate_all(genomes):
            objs = []
            for g in genomes:
                o = evaluate_multiobjective(g, cfg)
                objs.append(o)
                all_evaluations.append({'genome': list(g), 'objectives': list(o)})
            return objs

        def log_generation(gen):
            hv = hypervolume3d([o for _, o in archive])
            hv_history.append({'generation': gen, 'hypervolume': hv,
                               'archive_size': len(archive)})
            first = fast_nondominated_sort(objs)[0]
            per_generation.append({'generation': gen, 'front': _front_to_dicts(
                [(pop[i], objs[i]) for i in first])})
            print(f"Gen {gen:2d}: archive={len(archive)}, HV={hv:.4f}")

        with Timer() as t:
            pop = [random_genome() for _ in range(mu)]
            objs = evaluate_all(pop)
            archive = update_archive(archive, zip(pop, objs))
            log_generation(0)

            for gen in range(1, cfg.generations + 1):
                fronts = fast_nondominated_sort(objs)
                rank, crowd = {}, {}
                for r, front in enumerate(fronts):
                    cd = crowding_distance(objs, front)
                    for i in front:
                        rank[i], crowd[i] = r, cd[i]

                def tournament():
                    i, j = random.randrange(mu), random.randrange(mu)
                    if rank[i] != rank[j]:
                        return pop[i if rank[i] < rank[j] else j][:]
                    return pop[i if crowd[i] >= crowd[j] else j][:]

                offspring = []
                while len(offspring) < mu:
                    c1, c2 = ge_ops.two_point_crossover(tournament(), tournament())
                    offspring.append(ge_ops.point_mutation(c1))
                    offspring.append(ge_ops.point_mutation(c2))
                offspring = offspring[:mu]
                off_objs = evaluate_all(offspring)
                archive = update_archive(archive, zip(offspring, off_objs))

                # (mu + lambda) environmental selection.
                comb, comb_objs = pop + offspring, objs + off_objs
                new_pop, new_objs = [], []
                for front in fast_nondominated_sort(comb_objs):
                    if len(new_pop) + len(front) <= mu:
                        chosen = front
                    else:
                        cd = crowding_distance(comb_objs, front)
                        chosen = sorted(front, key=lambda i: -cd[i])[:mu - len(new_pop)]
                    new_pop += [comb[i] for i in chosen]
                    new_objs += [comb_objs[i] for i in chosen]
                    if len(new_pop) >= mu:
                        break
                pop, objs = new_pop, new_objs
                log_generation(gen)

        prof = profile_entry(t.seconds, len(all_evaluations), t.peak_gpu_mb)
        final_front = _front_to_dicts(archive)
        # profile.json is written before pareto_front_final.json (the resume
        # marker), so a completed run is guaranteed to have both.
        for fname, obj in (('profile.json', prof),
                           ('pareto_front_per_generation.json', per_generation),
                           ('hypervolume_history.json', hv_history),
                           ('all_evaluations.json', all_evaluations),
                           ('pareto_front_final.json', final_front)):
            with open(run_dir / fname, 'w') as f:
                json.dump(obj, f, indent=2)

        summaries[f'run_seed{seed}'] = {
            'seed': seed, 'dataset': cfg.dataset, 'population': mu,
            'generations': cfg.generations, 'pareto_size': len(final_front),
            'hypervolume_final': hv_history[-1]['hypervolume'],
            'hypervolume_history': [h['hypervolume'] for h in hv_history],
            'n_evaluations': len(all_evaluations), 'profile': prof,
        }
        print(f"NSGA-II seed={seed} done: |PF|={len(final_front)}, "
              f"HV={hv_history[-1]['hypervolume']:.4f}  "
              f"({prof['wall_clock_seconds']}s, {prof['evals_per_second']} evals/s)")

    summaries['_environment'] = device_info()
    # Gated for the same reason as main.write_summary: parallel single-seed
    # workers would otherwise all write this one path at once. The aggregation
    # pass (full seed range, everything resumed from disk) writes the real one.
    if getattr(cfg, 'write_summary', True):
        with open(RESULTS_DIR / f'{cfg.name}_summary.json', 'w') as f:
            json.dump(summaries, f, indent=2)
    return summaries
