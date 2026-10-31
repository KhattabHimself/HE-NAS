"""Runtime + GPU profiling helpers.

Records wall-clock time, throughput (evaluations/second), and peak GPU memory
per seed, plus a one-shot environment stamp (GPU model, CUDA/torch versions) so
a results file is self-describing. Intended for reporting GPU speedups, e.g.
this machine vs. Colab, in a grant proposal.

torch is imported lazily so the rest of the project keeps working where torch
is absent (the stamp then reports only what it can, timing still works).
"""
import os
import platform
import time


def _cuda_torch():
    """Return the torch module iff CUDA is usable, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def device_info():
    """Environment stamp: GPU model, CUDA/torch versions, host/CPU details.

    Written once per experiment summary so two logs (e.g. laptop GPU vs. Colab)
    are directly comparable without guessing which hardware produced them.
    """
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
    except ImportError:
        info["torch"] = None
        info["device"] = "cpu"
        return info
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_total_memory_mb": round(props.total_memory / 1024 ** 2, 1),
        })
    else:
        info["device"] = "cpu"
    return info


class Timer:
    """Context manager: wall-clock seconds and peak GPU memory (MB) for a block.

    Synchronizes CUDA on entry and exit so asynchronous kernels are fully
    counted (otherwise GPU work can be timed as near-instant).
    """

    def __init__(self):
        self.seconds = 0.0
        self.peak_gpu_mb = None

    def __enter__(self):
        t = _cuda_torch()
        if t is not None:
            t.cuda.reset_peak_memory_stats()
            t.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        t = _cuda_torch()
        if t is not None:
            t.cuda.synchronize()
            self.peak_gpu_mb = round(t.cuda.max_memory_allocated() / 1024 ** 2, 1)
        self.seconds = time.perf_counter() - self._t0
        return False


def profile_entry(seconds, n_evaluations, peak_gpu_mb=None):
    """Standard per-seed profiling block with derived throughput fields."""
    return {
        "wall_clock_seconds": round(seconds, 3),
        "n_evaluations": n_evaluations,
        "seconds_per_eval": (round(seconds / n_evaluations, 4)
                             if n_evaluations else None),
        "evals_per_second": (round(n_evaluations / seconds, 4)
                             if seconds > 0 else None),
        "peak_gpu_mb": peak_gpu_mb,
    }


def aggregate_profiles(profiles):
    """Roll up a list of per-seed profile_entry dicts into run-level totals."""
    secs = [p["wall_clock_seconds"] for p in profiles
            if p.get("wall_clock_seconds") is not None]
    evals = [p["n_evaluations"] for p in profiles if p.get("n_evaluations")]
    peaks = [p["peak_gpu_mb"] for p in profiles if p.get("peak_gpu_mb") is not None]
    total_secs, total_evals = sum(secs), sum(evals)
    return {
        "n_seeds": len(profiles),
        "total_wall_clock_seconds": round(total_secs, 3),
        "mean_seconds_per_seed": round(total_secs / len(secs), 3) if secs else None,
        "total_evaluations": total_evals,
        "mean_seconds_per_eval": (round(total_secs / total_evals, 4)
                                  if total_evals else None),
        "mean_evals_per_second": (round(total_evals / total_secs, 4)
                                  if total_secs > 0 else None),
        "peak_gpu_mb": max(peaks) if peaks else None,
    }


def _fmt_hms(seconds):
    if seconds is None:
        return "n/a"
    s = int(round(seconds))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 \
        else f"{s // 60:d}m{s % 60:02d}s" if s >= 60 else f"{seconds:.2f}s"


def format_report(summary):
    """Human-readable profiling table from a summary dict (for the CLI)."""
    env = summary.get("environment") or {}
    prof = summary.get("profiling")
    lines = ["=" * 62,
             f"Profiling report: {summary.get('experiment_name', '?')}",
             "=" * 62]
    dev = env.get("device", "?")
    if dev == "cuda":
        lines.append(f"Device      : {env.get('gpu_name')} "
                     f"({env.get('gpu_total_memory_mb')} MB)")
        lines.append(f"CUDA / torch: {env.get('cuda_version')} / {env.get('torch')}"
                     f"  (cuDNN {env.get('cudnn_version')})")
    else:
        lines.append(f"Device      : CPU  (torch {env.get('torch')})")
    lines.append(f"Host        : {env.get('platform')}  "
                 f"[{env.get('cpu_count')} CPUs]")
    lines.append("-" * 62)
    if not prof:
        lines.append("No profiling data recorded in this summary.")
        lines.append("=" * 62)
        return "\n".join(lines)
    lines += [
        f"Seeds run              : {prof.get('n_seeds')}",
        f"Total wall-clock       : {_fmt_hms(prof.get('total_wall_clock_seconds'))}"
        f"  ({prof.get('total_wall_clock_seconds')} s)",
        f"Mean time / seed       : {_fmt_hms(prof.get('mean_seconds_per_seed'))}",
        f"Total evaluations      : {prof.get('total_evaluations')}",
        f"Mean time / evaluation : {prof.get('mean_seconds_per_eval')} s",
        f"Throughput             : {prof.get('mean_evals_per_second')} evals/s",
        f"Peak GPU memory        : {prof.get('peak_gpu_mb')} MB",
        "=" * 62,
    ]
    return "\n".join(lines)
