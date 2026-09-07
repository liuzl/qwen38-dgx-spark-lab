#!/usr/bin/env python3
"""Median table across benchmark-serving-matrix.py result files (one per arm)."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--metric",
        default="aggregate_output_tok_s",
        help="row metric; C1 rows also show request_decode_tok_s_mean",
    )
    args = parser.parse_args()
    table = {}
    cases = []
    for path in args.files:
        data = json.loads(path.read_text())
        groups = defaultdict(list)
        spec = defaultdict(lambda: [0.0, 0.0])
        for row in data["rows"]:
            key = (row["model"], row["case"]["name"])
            metric = args.metric
            if row["case"]["concurrency"] == 1:
                metric = "request_decode_tok_s_mean"
            groups[key].append(row[metric])
            delta = row["speculative_counter_delta_including_warmup"]
            spec[key][0] += delta["num_accepted_tokens"]
            spec[key][1] += delta["num_draft_tokens"]
            if key[1] not in cases:
                cases.append(key[1])
        table[data["label"]] = {
            key: (
                statistics.median(values),
                spec[key][0] / spec[key][1] if spec[key][1] else None,
            )
            for key, values in groups.items()
        }
    for model in ("qwen3.8-27b", "qwen3.8-27b-uncensored"):
        print(f"\n{model}  (median tok/s; C1=decode, Cn=aggregate; [acceptance incl. warmup])")
        print("arm".ljust(22) + "".join(c.rjust(22) for c in cases))
        for label, rows in table.items():
            cells = []
            for case in cases:
                med, acc = rows.get((model, case), (None, None))
                if med is None:
                    cells.append("-".rjust(22))
                else:
                    acc_s = f"[{acc:.1%}]" if acc is not None else ""
                    cells.append(f"{med:8.2f} {acc_s}".rjust(22))
            print(label.ljust(22) + "".join(cells))


if __name__ == "__main__":
    main()
