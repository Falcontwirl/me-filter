"""Experiment 3: the replication gate. Nothing in RQ2-RQ4 runs unless this passes.

Fresh discovery pairs (seed differs from localization), evaluated with:
  filter   - the frozen heads
  random_k - 5 random head sets matched on per-layer counts
  none     - no patch

Pass criteria, fixed before running:
  1. filter causality >= 0.50                 (paper's Llama-8B: 0.826)
  2. filter CI lower bound > mean random-control causality
  3. mean random-control causality <= 0.10
"""

import pandas as pd

from filtergen import cli, upstream
from filtergen.controls import matched_random_heads
from filtergen.evaluate import run_conditions
from filtergen.pairs import EVAL_N_DISTRACTORS, load_pairs, load_task, sample_pairs, save_pairs
from filtergen.results import RESULTS, read_json, summarize, write_csv, write_json, write_jsonl
from filtergen.splits import DISCOVERY, SPLIT_DIR, split_hash

FROZEN = RESULTS / "heads" / "frozen_discovery.json"
N_RANDOM_CONTROLS = 5
PAPER_8B_CAUSALITY = 0.826


def main():
    p = cli.parser(__doc__)
    p.set_defaults(seed=1000)
    p.add_argument("--n_pairs", type=int, default=512)
    p.add_argument("--attn_n", type=int, default=64)
    args = p.parse_args()
    cli.setup(args)

    frozen = read_json(FROZEN)
    assert frozen["split_hash"] == split_hash(), "split changed since the heads were frozen"
    heads = [tuple(h) for h in frozen["heads"]]
    mt = upstream.load_model(args.model)

    pairs_path = RESULTS / "raw" / "gate_pairs.jsonl"
    if pairs_path.exists():
        pairs = load_pairs(pairs_path)
    else:
        task_file = SPLIT_DIR / "discovery.json"
        pairs = sample_pairs(mt, load_task(task_file), read_json(task_file),
                             DISCOVERY, DISCOVERY, args.n_pairs, EVAL_N_DISTRACTORS)
        save_pairs(pairs, pairs_path)

    head_sets = {"filter": heads, "none": []}
    for k in range(N_RANDOM_CONTROLS):
        head_sets[f"random_{k}"] = matched_random_heads(heads, mt.config.num_attention_heads, seed=k)
    records = run_conditions(mt, pairs, head_sets, attn_n=args.attn_n,
                             attn_on=("filter", "random_0"), experiment="gate")
    write_jsonl(records, RESULTS / "raw" / "gate_records.jsonl")

    df = pd.DataFrame(records)
    df["condition"] = df.head_set.where(~df.head_set.str.startswith("random"), "random")
    metrics = ["causal", "delta_target_logit", "pref_shift", "delta_attn_target"]
    table = summarize(df, ["condition"], metrics)
    write_csv(table, RESULTS / "tables" / "gate.csv")
    print(table.to_string(index=False))

    f = table.set_index("condition")
    random_mean = f.loc["random", "causal"]
    criteria = {
        "causality_ge_0.50": bool(f.loc["filter", "causal"] >= 0.50),
        "ci_above_random": bool(f.loc["filter", "causal_lo"] > random_mean),
        "random_le_0.10": bool(random_mean <= 0.10),
    }
    verdict = {
        "pass": all(criteria.values()),
        "criteria": criteria,
        "filter_causality": float(f.loc["filter", "causal"]),
        "filter_ci": [float(f.loc["filter", "causal_lo"]), float(f.loc["filter", "causal_hi"])],
        "random_causality": float(random_mean),
        "paper_llama8b_causality": PAPER_8B_CAUSALITY,
        "n_heads": len(heads),
        "n_pairs": len(pairs),
        "meta": upstream.run_metadata(mt, seed=args.seed, split_hash=split_hash()),
    }
    write_json(verdict, RESULTS / "gate.json")
    print(f"\nGATE {'PASSED' if verdict['pass'] else 'FAILED'}: {criteria}")


if __name__ == "__main__":
    main()
