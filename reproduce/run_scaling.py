"""Run Table 2 of the GraCo paper: every coarsener on each of the four tiers.

    python -m graco.reproduce.run_scaling                  # the four tiers
    python -m graco.reproduce.run_scaling --tiers Cora     # one tier
    python -m graco.reproduce.scaling --dataset Cora --r 0.9 --algo metis   # one cell

Each method runs in its own process through scaling.py; this loop stops one
that overruns its time or its host memory and records why. The memory columns
are absolute peaks, host and device apart; every process starts from the same
floor, the interpreter and the loaded graph.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time

from graco.algorithms import COARSENING_ALGOS
from graco.reproduce.scaling import (TIERS, VARIANT_TIERS, VARIANTS,
                                        default_results_dir, pick_device,
                                        write_json)



def host_mem_limit_MB() -> float | None:
    """Host memory this process may use, in MB: the cgroup limit, else MemTotal, else None."""
    for path in ("/sys/fs/cgroup/memory.max",                    # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            break
        try:
            v = int(raw)
        except ValueError:
            continue
        # v1 writes a sentinel near 2**63 when unlimited.
        if 0 < v < 2 ** 60:
            return v / (1024 ** 2)
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None

def child_rss_MB(pid: int) -> float:
    """Resident set of a process in MB, 0 once it is gone.

    Polled so the runner stops a runaway child before the kernel kills it without
    a row. An RLIMIT_AS would do it too, but it breaks CUDA context creation.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0.0


def run_watched(cmd: list[str], timeout_s: int, rss_cap_MB: float | None,
                poll_s: float = 1.0):
    """Run `cmd` watching wall time and resident set.

    Returns (returncode, stdout, stderr, sampled_peak_MB, reason), `reason`
    being None on a normal exit, "timeout" or "oom_host".
    """
    # The output goes to files: a pipe holds 64 KB, and a child that prints more
    # (a progress bar over hours) would block until the pipe is read, here only
    # after it exits.
    with tempfile.TemporaryFile("w+") as out_f, tempfile.TemporaryFile("w+") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
        t0 = time.perf_counter()
        peak = 0.0
        reason: str | None = None

        while proc.poll() is None:
            peak = max(peak, child_rss_MB(proc.pid))
            if rss_cap_MB is not None and peak > rss_cap_MB:
                reason = "oom_host"
                break
            if time.perf_counter() - t0 > timeout_s:
                reason = "timeout"
                break
            time.sleep(poll_s)

        if reason is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        out_f.seek(0)
        err_f.seek(0)
        return proc.returncode, out_f.read(), err_f.read(), peak, reason


def run_dataset(dataset: str, r: float, *, rsa_k: int | None = None,
                timeout_s: int = 3600, seed: int = 14,
                algos: list[str] | None = None,
                exclude: set[str] | None = None,
                device: str = "cuda",
                rss_cap_MB: float | None = None,
                skip_done: bool = False,
                skip_timeout: bool = False) -> list[dict]:
    """Run every coarsener on dataset at r, then merge and return the rows.

    device defaults to cuda, where Table 2 was measured, and falls back to the
    host when no GPU is visible.
    """
    resolved = str(pick_device(device))
    if resolved != device:
        print(f"# {device} not visible, falling back to {resolved}", flush=True)
    device = resolved

    if algos is None:
        algos = sorted(COARSENING_ALGOS) + sorted(VARIANTS)
        algos = [a for a in algos
                 if a not in VARIANT_TIERS or dataset in VARIANT_TIERS[a]]
    if exclude:
        algos = [a for a in algos if a not in exclude]
    out_dir = default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# {dataset}  r={r}  {len(algos)} methods  device={device}  "
          f"timeout {timeout_s / 3600:.1f} h/method  "
          f"host cap {str(rss_cap_MB) + ' MB' if rss_cap_MB else 'none'}  "
          f"RSA {'K=' + str(rsa_k) if rsa_k else 'skipped'}", flush=True)

    rows: list[dict] = []
    for algo in algos:
        # Under --skip-done a row already on disk is kept; delete its file to
        # rerun it after a change of default.
        done = out_dir / "raw" / f"{dataset}_r{r}_{algo}.json"
        if skip_done and done.exists():
            try:
                prev = json.loads(done.read_text())
            except json.JSONDecodeError:
                prev = None
            keep = prev and (prev.get("status") == "ok"
                             or (skip_timeout and prev.get("status") == "timeout"))
            if keep:
                print(f"  {algo:<14} already {prev['status']}, skipped", flush=True)
                rows.append(prev)
                continue

        cmd = [sys.executable, "-m", "graco.reproduce.scaling",
               "--dataset", dataset, "--r", str(r), "--algo", algo,
               "--seed", str(seed), "--out-dir", str(out_dir),
               "--device", device]
        if rsa_k:
            cmd += ["--rsa-k", str(rsa_k)]
        # A coarsener that takes a budget gets 80% of the timeout, the rest
        # going to max_acc and the RSA.
        cmd += ["--time-budget", str(timeout_s * 0.8)]

        before = done.read_bytes() if done.exists() else None
        t0 = time.perf_counter()
        rc, out, err, sampled, reason = run_watched(cmd, timeout_s, rss_cap_MB)

        if reason == "timeout":
            row = {"dataset": dataset, "r": r, "algo": algo, "status": "timeout",
                   "error": f"exceeded {timeout_s} s"}
        elif reason == "oom_host":
            row = {"dataset": dataset, "r": r, "algo": algo, "status": "oom_host",
                   "error": f"resident set passed {rss_cap_MB} MB"}
        elif rc == 0 and out.strip():
            row = json.loads(out.strip().splitlines()[-1])
        else:
            # Non-zero exit with no JSON: the worker was killed before it
            # reported.
            row = {"dataset": dataset, "r": r, "algo": algo, "status": "killed",
                   "error": err[-400:] or f"exit code {rc}, no output"}

        row["wall_s"] = time.perf_counter() - t0
        row["sampled_rss_peak_MB"] = sampled
        rows.append(row)
        # Write the row here when the worker wrote none, so merge_tier sees it.
        if (done.read_bytes() if done.exists() else None) == before:
            write_json(done, row)

        if row["status"] == "ok":
            gpu = row.get("gpu_peak_MB")
            print(f"  {algo:<14} n={row['n_clusters']:>8}  "
                  f"t={row['time_s']:>9.1f}s  host={row['rss_peak_MB']:>8.1f}MB  "
                  + (f"gpu={gpu:>8.1f}MB  " if gpu else "")
                  + f"max_acc={row['max_acc']:.4f}"
                  + (f"  rsa={row['rsa']:.3f}" if row.get("rsa") is not None else ""),
                  flush=True)
        else:
            print(f"  {algo:<14} {row['status'].upper()}  "
                  f"({row['wall_s']:.0f}s, sampled peak {sampled:.0f}MB)", flush=True)

    # Merge every row on disk for this tier, including those of other runs.
    from graco.reproduce.collect_scaling import merge_tier
    summary = merge_tier(dataset, r, out_dir)
    print(f"\n# wrote {summary}", flush=True)
    return rows




def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Scalability table runner.")
    ap.add_argument("--tiers", default=None,
                    help="Comma-separated dataset names. Omit for all four.")
    ap.add_argument("--algo", default=None,
                    help="Comma-separated methods. Omit to run every method "
                         "of each tier.")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated methods to skip on every tier.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rss-cap-mb", type=float, default=None,
                    help="Host resident set at which a method is stopped and "
                         "recorded as oom_host. Default: a fraction of what "
                         "the cgroup allows, so the runner gets there before "
                         "the kernel does.")
    ap.add_argument("--rss-cap-frac", type=float, default=0.85,
                    help="Fraction of the host limit used when --rss-cap-mb "
                         "is not given.")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip a method whose row is already on disk with "
                         "status ok. Delete the file under results/raw/ to "
                         "force a re-run, which a changed default needs.")
    ap.add_argument("--skip-timeout", action="store_true",
                    help="With --skip-done, also skip a method whose row says "
                         "timeout. A second submission then spends its hours "
                         "on what has never run rather than on what already "
                         "ate a full budget.")
    ap.add_argument("--seed", type=int, default=14)
    args = ap.parse_args()

    wanted = set(args.tiers.split(",")) if args.tiers else None

    cap = args.rss_cap_mb
    if cap is None:
        limit = host_mem_limit_MB()
        cap = limit * args.rss_cap_frac if limit else None
        print(f"# host limit {limit:.0f} MB, cap {cap:.0f} MB "
              f"({args.rss_cap_frac:.0%})" if limit else
              "# host limit unknown, no memory cap", flush=True)

    t_start = time.perf_counter()

    for dataset, r, _tier, rsa_k, timeout_s in TIERS:
        if wanted and dataset not in wanted:
            continue
        spent = time.perf_counter() - t_start
        print(f"\n{'=' * 72}\n# tier {dataset}  r={r}  "
              f"{spent / 3600:.1f} h in\n{'=' * 72}", flush=True)
        run_dataset(dataset, r, rsa_k=rsa_k, timeout_s=timeout_s, seed=args.seed,
                    algos=args.algo.split(",") if args.algo else None,
                    exclude=set(args.exclude.split(",")) if args.exclude else None,
                    device=args.device, rss_cap_MB=cap,
                    skip_done=args.skip_done, skip_timeout=args.skip_timeout)

    print(f"\n# done in {(time.perf_counter() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
