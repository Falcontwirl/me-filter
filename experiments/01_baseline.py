"""Experiment 1: behavioral baseline per category (upstream SelectOne prompts).

A category is kept for mechanistic analysis only if accuracy >= 0.90.
Writes results/tables/baseline.csv and results/baseline_pass.json.
"""

import random

import pandas as pd

from filtergen import cli, upstream
from filtergen.pairs import LOCATE_N_DISTRACTORS, OPTION_STYLE, PROMPT_TEMPLATE_IDX, load_task
from filtergen.results import RESULTS, summarize, write_csv, write_json, write_jsonl
from filtergen.splits import POOLS, SPLIT_DIR, split_hash, write_split_files
from src.selection.utils import verify_correct_option

PASS_THRESHOLD = 0.90


def main():
    p = cli.parser(__doc__)
    p.add_argument("--n_per_category", type=int, default=100)
    args = p.parse_args()
    cli.setup(args)

    write_split_files()
    mt = upstream.load_model(args.model)
    task = load_task(SPLIT_DIR / "all.json")
    pool_of = {c: name for name, cats in POOLS.items() for c in cats}

    records = []
    for cat in task.categories:
        for _ in range(args.n_per_category):
            s = task.get_random_sample(
                mt=mt,
                category=cat,
                n_distractors=random.choice(LOCATE_N_DISTRACTORS),
                prompt_template_idx=PROMPT_TEMPLATE_IDX,
                option_style=OPTION_STYLE,
                filter_by_lm_prediction=False,
            )
            ok, _, _ = verify_correct_option(
                mt=mt, target=s.ans_token_id, options=s.options, input=s.prompt()
            )
            records.append({"category": cat, "pool": pool_of[cat], "correct": int(ok),
                            "n_options": len(s.options), "answer": s.obj, "options": s.options})
        print(f"{cat}: {sum(r['correct'] for r in records if r['category'] == cat)}/{args.n_per_category}")

    write_jsonl(records, RESULTS / "raw" / "baseline.jsonl")
    df = pd.DataFrame(records)
    table = summarize(df, ["pool", "category"], ["correct"]).rename(columns={"correct": "accuracy"})
    table["pass"] = table["accuracy"] >= PASS_THRESHOLD
    write_csv(table, RESULTS / "tables" / "baseline.csv")
    print(table.to_string(index=False))

    write_json({
        "threshold": PASS_THRESHOLD,
        "passing": table.loc[table["pass"], "category"].tolist(),
        "failing": table.loc[~table["pass"], "category"].tolist(),
        "split_hash": split_hash(),
        "meta": upstream.run_metadata(mt, seed=args.seed, n_per_category=args.n_per_category),
    }, RESULTS / "baseline_pass.json")


if __name__ == "__main__":
    main()
