"""Merge the per-method rows of Table 2 into one file per tier, and print them.

    python -m graco.reproduce.collect_scaling

Each worker writes results/raw/<dataset>_r<r>_<algo>.json; this writes
results/<dataset>_r<r>.json, one entry per method. Reads GRACO_RESULTS_DIR, or
results/ next to the scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

from graco.reproduce.scaling import TIERS, default_results_dir, write_json


def merge_tier(dataset: str, r: float, out_dir: Path | None = None) -> Path:
    """Merge results/raw/<dataset>_r<r>_*.json into results/<dataset>_r<r>.json and return its path.

    An existing merged file is left as is when no raw row is found.
    """
    out_dir = out_dir or default_results_dir()
    raw_dir = out_dir / "raw"
    by_algo: dict[str, dict] = {}
    if raw_dir.is_dir():
        for f in sorted(raw_dir.glob(f"{dataset}_r{r}_*.json")):
            try:
                row = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            by_algo[row["algo"]] = row

    path = out_dir / f"{dataset}_r{r}.json"
    # No row found, e.g. no raw/ in a fresh clone: keep the merged file as is.
    if by_algo or not path.exists():
        write_json(path, {"dataset": dataset, "r": r,
                          "methods": dict(sorted(by_algo.items()))})
    return path


def print_tier(dataset: str, r: float, tier: str, out_dir: Path | None = None) -> None:
    path = merge_tier(dataset, r, out_dir)
    methods = json.loads(path.read_text())["methods"]
    if not methods:
        print(f"\n## {tier}, {dataset} r={r}: nothing yet")
        return

    print(f"\n## {tier}, {dataset} r={r}  ({len(methods)} methods)")
    base = next((r_["rss_baseline_MB"] for r_ in methods.values()
                 if r_.get("rss_baseline_MB")), None)
    if base:
        print(f"host floor (interpreter, torch, loaded graph): {base:.0f} MB")
    print(f"{'method':<14} {'status':<9} {'n':>9} {'time_s':>10} "
          f"{'host_MB':>9} {'gpu_MB':>9} {'gpu_resv':>9} {'max_acc':>8} {'rsa':>7}")
    for algo, row in methods.items():
        if row.get("status") != "ok":
            print(f"{algo:<14} {row.get('status', '?'):<9}")
            continue
        gpu, resv, rsa = (row.get("gpu_peak_MB"), row.get("gpu_reserved_MB"),
                          row.get("rsa"))
        print(f"{algo:<14} {'ok':<9} {row['n_clusters']:>9} "
              f"{row['time_s']:>10.1f} {row['rss_peak_MB']:>9.1f} "
              f"{(f'{gpu:.1f}' if gpu else '--'):>9} "
              f"{(f'{resv:.1f}' if resv else '--'):>9} "
              f"{row['max_acc']:>8.4f} "
              f"{(f'{rsa:.3f}' if rsa is not None else '--'):>7}")


def main() -> None:
    for dataset, r, tier, _rsa_k, _timeout in TIERS:
        print_tier(dataset, r, tier)


if __name__ == "__main__":
    main()
