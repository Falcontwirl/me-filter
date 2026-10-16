"""Experiment 4 (RQ2, primary): do the frozen discovery heads transfer to held-out predicates?

Cells (source predicate -> destination predicate), each for B1 and B2:
  seen->seen, unseen->seen, seen->unseen, unseen->unseen
All cells use data/splits/all.json, so every cell draws distractors from the
same pool. No new heads are located. Held-out categories that failed the
behavioral baseline are dropped.

Head sets: the frozen filter heads, plus matched random-head controls.
"""

import pandas as pd

from filtergen import cli, upstream
from filtergen.controls import matched_random_heads
from filtergen.evaluate import run_conditions
from filtergen.pairs import EVAL_N_DISTRACTORS, load_pairs, load_task, sample_pairs, save_pairs
from filtergen.results import RESULTS, read_json, summarize, write_csv, write_json, write_jsonl
from filtergen.splits import DISCOVERY, POOLS, SPLIT_DIR, split_hash

FROZEN = RESULTS / "heads" / "frozen_discovery.json"
N_RANDOM_CONTROLS = 2


def main():
    p = cli.parser(__doc__)
    p.set_defaults(seed=2000)
    p.add_argument("--n_per_cell", type=int, default=256)
    p.add_argument("--attn_n", type=int, default=64)
    p.add_argument("--pools", nargs="+", default=["b1", "b2"])
    p.add_argument("--ignore_gate", action="store_true", help="for debugging only; recorded in output")
    args = p.parse_args()
    cli.setup(args)

    gate = read_json(RESULTS / "gate.json")
    if not gate["pass"] and not args.ignore_gate:
        raise SystemExit("Replication gate failed (results/gate.json). Not running the extension.")
    frozen = read_json(FROZEN)
    assert frozen["split_hash"] == split_hash(), "split changed since the heads were frozen"
    heads = [tuple(h) for h in frozen["heads"]]
    passing = set(read_json(RESULTS / "baseline_pass.json")["passing"])

    mt = upstream.load_model(args.model)
    task_file = SPLIT_DIR / "all.json"
    task, task_json = load_task(task_file), read_json(task_file)
    seen = [c for c in DISCOVERY if c in passing]

    cells = {("all", "seen->seen"): (seen, seen)}
    for pool in args.pools:
        unseen = [c for c in POOLS[pool] if c in passing]
        dropped = sorted(set(POOLS[pool]) - set(unseen))
        if dropped:
            print(f"{pool}: dropping categories that failed the baseline: {dropped}")
        if not unseen:
            continue
        cells[(pool, "unseen->seen")] = (unseen, seen)
        cells[(pool, "seen->unseen")] = (seen, unseen)
        if len(unseen) > 1:
            cells[(pool, "unseen->unseen")] = (unseen, unseen)

    head_sets = {"filter": heads}
    for k in range(N_RANDOM_CONTROLS):
        head_sets[f"random_{k}"] = matched_random_heads(heads, mt.config.num_attention_heads, seed=k)

    records = []
    for (pool, cell), (src_pool, dst_pool) in cells.items():
        pairs_path = RESULTS / "raw" / "heldout_pairs" / f"{pool}_{cell.replace('->', '_to_')}.jsonl"
        if pairs_path.exists():
            pairs = load_pairs(pairs_path)
        else:
            pairs = sample_pairs(mt, task, task_json, src_pool, dst_pool, args.n_per_cell, EVAL_N_DISTRACTORS)
            save_pairs(pairs, pairs_path)
        print(f"evaluating {pool} {cell} ({len(pairs)} pairs)")
        records += run_conditions(mt, pairs, head_sets, attn_n=args.attn_n,
                                  experiment="heldout", pool=pool, cell=cell)
        write_jsonl(records, RESULTS / "raw" / "heldout_records.jsonl")  # checkpoint

    df = pd.DataFrame(records)
    df["condition"] = df.head_set.where(~df.head_set.str.startswith("random"), "random")
    metrics = ["causal", "delta_target_logit", "pref_shift", "delta_attn_target"]
    table = summarize(df, ["pool", "cell", "condition"], metrics)
    write_csv(table, RESULTS / "tables" / "heldout_cells.csv")
    print(table[["pool", "cell", "condition", "n", "causal", "causal_lo", "causal_hi"]].to_string(index=False))

    f = df[df.head_set == "filter"]
    matrix = f.pivot_table(index="src_category", columns="dst_category", values="causal", aggfunc="mean")
    counts = f.pivot_table(index="src_category", columns="dst_category", values="causal", aggfunc="count")
    write_csv(matrix.reset_index(), RESULTS / "tables" / "heldout_transfer_matrix.csv")
    write_csv(counts.reset_index(), RESULTS / "tables" / "heldout_transfer_counts.csv")

    write_json({"ignored_gate": args.ignore_gate, "n_heads": len(heads), "cells": [list(k) for k in cells],
                "meta": upstream.run_metadata(mt, seed=args.seed, split_hash=split_hash())},
               RESULTS / "heldout_run.json")


if __name__ == "__main__":
    main()
