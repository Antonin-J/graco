"""SGWL on Cora under its two stopping rules: Table 12 of the GraCo paper (App. C.8).

For r = 0.5, 0.7 and 0.9, the number of supernodes asked for and the number
returned under stop_rule="size", the rule of the released code, and
stop_rule="count", the default.

Run:  python -m graco.reproduce.sgwl_stop
Output: results/sgwl_stop_Cora.json
"""

from __future__ import annotations

from graco.algorithms.gw_sgwl import coarsen_gw_sgwl
from graco.loaders import get_dataset
from graco.reproduce.scaling import default_results_dir, write_json

ratios = (0.5, 0.7, 0.9)


def main() -> None:
    data = get_dataset("Cora")
    rows = []
    for r in ratios:
        row = {"r": r}
        for rule in ("size", "count"):
            partition, info = coarsen_gw_sgwl(data, r=r, stop_rule=rule)
            row["n_asked"] = int(info["n_target"])
            row[f"n_{rule}"] = int(partition.n_clusters)
        rows.append(row)
        print(f"r={r}  asked={row['n_asked']}  size={row['n_size']}  "
              f"count={row['n_count']}", flush=True)

    out = default_results_dir() / "sgwl_stop_Cora.json"
    write_json(out, {"N": int(data.num_nodes), "rows": rows})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
