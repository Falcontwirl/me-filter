"""Pick lambda from the sweep and freeze the head set. CPU only.

Rule, fixed before seeing the sweep: take the best validation causality of the
layer-cutoff head set; among lambdas within 0.02 of it, pick the one with the
fewest heads. With --stability_run, also report Jaccard overlap with the same
lambda trained on an independently sampled pair set.

    python experiments/02b_freeze.py --run discovery_s0 --stability_run discovery_s1
"""

import argparse

from filtergen.results import RESULTS, read_json, write_json
from filtergen.splits import split_hash

HEADS_DIR = RESULTS / "heads"
FROZEN = HEADS_DIR / "frozen_discovery.json"
TOLERANCE = 0.02


def jaccard(a, b) -> float:
    a, b = set(map(tuple, a)), set(map(tuple, b))
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="discovery_s0")
    p.add_argument("--stability_run", default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing frozen set")
    args = p.parse_args()

    if FROZEN.exists() and not args.force:
        # heads already frozen: only attach the stability comparison
        frozen = read_json(FROZEN)
        assert args.stability_run, f"{FROZEN} exists; pass --stability_run, or --force with a documented reason"
        add_stability(frozen, args.stability_run)
        write_json(frozen, FROZEN)
        return

    sweep = [read_json(f) for f in sorted((HEADS_DIR / args.run).glob("lam_*/summary.json"))]
    assert sweep, f"no sweep results under {HEADS_DIR / args.run}"
    for s in sweep:
        v = s["val"]["cutoff"]
        print(f"lam={s['lam']:<7g} heads={v['n_heads']:<4} causality={v['causality']:.3f}")
    best = max(s["val"]["cutoff"]["causality"] for s in sweep)
    eligible = [s for s in sweep if s["val"]["cutoff"]["causality"] >= best - TOLERANCE]
    chosen = min(eligible, key=lambda s: s["val"]["cutoff"]["n_heads"])
    print(f"chosen lambda: {chosen['lam']:g}")

    frozen = {
        "heads": chosen["heads_cutoff"],
        "heads_all_layers": chosen["heads"],
        "lam": chosen["lam"],
        "layer_cutoff": chosen["layer_cutoff"],
        "val": chosen["val"],
        "selection_rule": f"max val causality (cutoff set), then fewest heads within {TOLERANCE}",
        "source_run": args.run,
        "split_hash": split_hash(),
        "meta": chosen["meta"],
    }

    if args.stability_run:
        add_stability(frozen, args.stability_run)
    write_json(frozen, FROZEN)
    print(f"froze {len(frozen['heads'])} heads -> {FROZEN}")


def add_stability(frozen: dict, stability_run: str) -> None:
    other = HEADS_DIR / stability_run / f"lam_{frozen['lam']:g}" / "summary.json"
    if not other.exists():
        print(f"stability run not trained yet: {other}")
        return
    o = read_json(other)
    frozen["stability"] = {
        "run": stability_run,
        "jaccard_cutoff": jaccard(frozen["heads"], o["heads_cutoff"]),
        "jaccard_all_layers": jaccard(frozen["heads_all_layers"], o["heads"]),
        "other_val": o["val"],
    }
    print(f"stability: {frozen['stability']}")


if __name__ == "__main__":
    main()
