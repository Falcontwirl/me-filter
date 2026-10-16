"""Experiment 6 (RQ4): does each predicate class recruit the same heads?

Prerequisite: head sets localized separately (same lambda as the frozen set)
for each class with experiments/02_locate.py, e.g. runs b1_s0, b2_s0, comp_s0.

Reports
  - Jaccard overlap between every pair of head sets (layer-cutoff versions)
  - cross-evaluation: causality of every head set on every class's
    validation pairs (which were not used to train any mask)
"""

import itertools

import pandas as pd

from filtergen import cli, upstream
from filtergen.controls import matched_random_heads
from filtergen.evaluate import run_conditions
from filtergen.pairs import load_pairs
from filtergen.results import RESULTS, read_json, summarize, write_csv, write_json, write_jsonl

HEADS_DIR = RESULTS / "heads"


def jaccard(a, b) -> float:
    a, b = set(map(tuple, a)), set(map(tuple, b))
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    p = cli.parser(__doc__)
    p.set_defaults(seed=4000)
    p.add_argument("--runs", nargs="+", default=["discovery_s0", "b1_s0", "b2_s0", "comp_s0"],
                   help="results/heads/<run>; the first must be the frozen discovery run")
    p.add_argument("--n_val", type=int, default=128)
    args = p.parse_args()
    cli.setup(args)

    frozen = read_json(HEADS_DIR / "frozen_discovery.json")
    lam = frozen["lam"]
    head_sets = {args.runs[0]: [tuple(h) for h in frozen["heads"]]}
    for run in args.runs[1:]:
        summary = HEADS_DIR / run / f"lam_{lam:g}" / "summary.json"
        if not summary.exists():
            print(f"skipping {run}: no mask trained at lambda={lam:g} ({summary})")
            continue
        head_sets[run] = [tuple(h) for h in read_json(summary)["heads_cutoff"]]

    overlap = pd.DataFrame(
        [{"a": a, "b": b, "n_a": len(head_sets[a]), "n_b": len(head_sets[b]),
          "shared": len(set(head_sets[a]) & set(head_sets[b])), "jaccard": jaccard(head_sets[a], head_sets[b])}
         for a, b in itertools.combinations(head_sets, 2)]
    )
    write_csv(overlap, RESULTS / "tables" / "overlap_jaccard.csv")
    print(overlap.to_string(index=False))

    mt = upstream.load_model(args.model)
    evaluated = dict(head_sets)
    evaluated["random"] = matched_random_heads(head_sets[args.runs[0]], mt.config.num_attention_heads, seed=0)
    records = []
    for data_run in head_sets:
        pairs = load_pairs(HEADS_DIR / data_run / "pairs_val.jsonl")[: args.n_val]
        print(f"cross-evaluating on {data_run} val pairs ({len(pairs)})")
        records += run_conditions(mt, pairs, evaluated, experiment="overlap", data=data_run)
    write_jsonl(records, RESULTS / "raw" / "overlap_records.jsonl")
    table = summarize(pd.DataFrame(records), ["head_set", "data"], ["causal", "delta_target_logit"])
    write_csv(table, RESULTS / "tables" / "overlap_cross_eval.csv")
    print(table.pivot(index="head_set", columns="data", values="causal").round(3).to_string())

    write_json({"lam": lam, "head_sets": {k: [list(h) for h in v] for k, v in head_sets.items()},
                "meta": upstream.run_metadata(mt, seed=args.seed)}, RESULTS / "overlap_run.json")


if __name__ == "__main__":
    main()
