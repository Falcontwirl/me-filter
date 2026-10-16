"""Experiment 2: locate filter heads with upstream's learned query mask.

Two steps, so a lambda sweep (SLURM array) trains on identical data:

  build: sample train/val pairs for a category pool
      python experiments/02_locate.py build --run discovery_s0 --pool seen --seed 0
  train: train the mask for one lambda, select heads, score them on val pairs
      python experiments/02_locate.py train --run discovery_s0 --lam 0.01

Training settings follow scripts/locate_selection_heads.py (legacy interface):
AdamW lr 1e-2, batch 16, promote_suppress loss, query at the last token only,
heads = mask > 0.5. Epochs default to 10, which the authors' notebooks load
(epoch_10.npz) and their job scripts use.

Pools: "seen" uses data/splits/discovery.json (discovery categories only, for
predicates and distractors). "b1"/"b2" (RQ4) draw predicates from that pool
and distractors from all categories.
"""

import numpy as np
import pandas as pd
import torch

from filtergen import cli, upstream
from filtergen.evaluate import UPSTREAM_8B_LAYER_CUTOFF, apply_layer_cutoff, run_conditions
from filtergen.pairs import LOCATE_N_DISTRACTORS, load_pairs, load_task, sample_pairs, save_pairs
from filtergen.results import RESULTS, read_json, write_json, write_jsonl
from filtergen.splits import POOLS, SPLIT_DIR, split_hash, write_split_files
from src.selection.optimization import get_optimal_head_mask_prev

HEADS_DIR = RESULTS / "heads"


def build_compositional(args, mt):
    """RQ4: comp->simple pairs over the given predicate ids (default: all)."""
    import random

    from filtergen.compositional import CompositionalBuilder

    builder = CompositionalBuilder(mt)
    pids = args.categories or list(builder.spec["predicates"])
    run_dir = HEADS_DIR / args.run
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        pairs = [builder.pair_comp_to_simple(random.choice(pids)) for _ in range(n)]
        save_pairs(pairs, run_dir / f"pairs_{split}.jsonl")
    write_json({"pool": "compositional", "categories": pids, "split_hash": split_hash(),
                "n_train": args.n_train, "n_val": args.n_val,
                "meta": upstream.run_metadata(mt, seed=args.seed)}, run_dir / "build.json")


def build(args, mt):
    write_split_files()
    if args.pool == "compositional":
        return build_compositional(args, mt)
    task_file = SPLIT_DIR / ("discovery.json" if args.pool == "seen" else "all.json")
    task, task_json = load_task(task_file), read_json(task_file)
    pool = POOLS[args.pool]
    if args.categories:
        pool = [c for c in pool if c in args.categories]
    run_dir = HEADS_DIR / args.run
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        pairs = sample_pairs(mt, task, task_json, pool, pool, n, LOCATE_N_DISTRACTORS)
        save_pairs(pairs, run_dir / f"pairs_{split}.jsonl")
    write_json({"pool": args.pool, "categories": pool, "task_file": str(task_file.name),
                "split_hash": split_hash(), "n_train": args.n_train, "n_val": args.n_val,
                "meta": upstream.run_metadata(mt, seed=args.seed)}, run_dir / "build.json")


def train(args, mt):
    run_dir = HEADS_DIR / args.run
    out_dir = run_dir / f"lam_{args.lam:g}"
    train_set = load_pairs(run_dir / "pairs_train.jsonl")
    val_set = load_pairs(run_dir / "pairs_val.jsonl")

    mask, losses = get_optimal_head_mask_prev(
        mt=mt,
        train_set=train_set,
        learning_rate=1e-2,
        n_epochs=args.n_epochs,
        lamb=args.lam,
        batch_size=args.batch_size,
        query_indices=[-1],
        save_path=str(out_dir),
        save_step=5,
        loss_fn="promote_suppress",
    )
    heads = [tuple(h) for h in torch.nonzero(mask > 0.5).tolist()]
    head_sets = {
        "all_layers": heads,
        "cutoff": apply_layer_cutoff(heads, UPSTREAM_8B_LAYER_CUTOFF),
    }
    records = run_conditions(mt, val_set, head_sets)
    write_jsonl(records, out_dir / "val_records.jsonl")
    df = pd.DataFrame(records)
    summary = {
        "run": args.run,
        "lam": args.lam,
        "n_epochs": args.n_epochs,
        "heads": heads,
        "heads_cutoff": head_sets["cutoff"],
        "layer_cutoff": UPSTREAM_8B_LAYER_CUTOFF,
        "val": {
            name: {"n_heads": len(h), "causality": float(df[df.head_set == name].causal.mean()),
                   "delta_target_logit": float(df[df.head_set == name].delta_target_logit.mean())}
            for name, h in head_sets.items()
        },
        "final_loss": float(np.mean(losses[-len(range(0, len(train_set), args.batch_size)):])),
        "meta": upstream.run_metadata(mt, seed=args.seed, split_hash=split_hash()),
    }
    write_json(summary, out_dir / "summary.json")
    print(summary["val"])


def main():
    p = cli.parser(__doc__)
    p.add_argument("step", choices=["build", "train"])
    p.add_argument("--run", required=True, help="results/heads/<run>")
    p.add_argument("--pool", choices=[*POOLS, "compositional"], default="seen")
    p.add_argument("--categories", nargs="*",
                   help="restrict the pool (baseline-passing categories, or predicate ids for compositional)")
    p.add_argument("--n_train", type=int, default=512)
    p.add_argument("--n_val", type=int, default=256)
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    cli.setup(args)
    mt = upstream.load_model(args.model)
    {"build": build, "train": train}[args.step](args, mt)


if __name__ == "__main__":
    main()
