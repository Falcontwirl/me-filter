"""Experiment 5 (RQ3): compositional predicates with the frozen discovery heads.

1. behavior   accuracy on lists containing lures; predicates under 0.80 are
              reported but excluded from the causal analysis
2. patching   comp->simple and simple->comp pairs; the patched choice is
              classified as target / lure_a / lure_b / dest_answer / neutral
3. q-sum      (exploratory) patch q(component A) + q(component B) instead of
              q(composite), mirroring the paper's App. D vector addition
4. cosine     (exploratory) cosine of q(composite) with q(A), q(B), q(A)+q(B)
              per frozen head, on identical lists
"""

import copy

import pandas as pd
import torch

from filtergen import cli, upstream
from filtergen.compositional import CompositionalBuilder
from filtergen.controls import matched_random_heads
from filtergen.evaluate import build_q_patches, run_conditions
from filtergen.pairs import load_pairs, save_pairs
from filtergen.results import RESULTS, read_json, summarize, write_csv, write_json, write_jsonl
from filtergen.splits import split_hash
from src.functional import PatchSpec, patch_with_baukit
from src.selection.functional import cache_q_projections
from src.selection.utils import get_first_token_id
from src.tokens import prepare_input

FROZEN = RESULTS / "heads" / "frozen_discovery.json"
BEHAVIOR_THRESHOLD = 0.80
OUT = RESULTS / "raw" / "compositional"


def top_role(mt, sample, logits) -> str:
    toks = {get_first_token_id(o, mt.tokenizer, prefix=" "): o for o in sample.options}
    top = max(toks, key=lambda t: logits[t].item())
    return sample.metadata["roles"][str(top)]


@torch.inference_mode()
def behavior(mt, builder, pid, n):
    rows = []
    for _ in range(n):
        s = builder.compositional_list(pid)
        logits = mt._model(**prepare_input(prompts=s.prompt(), tokenizer=mt)).logits[0, -1]
        role = top_role(mt, s, logits)
        rows.append({"predicate": pid, "correct": int(role == "answer"), "chosen_role": role})
    return rows


@torch.inference_mode()
def q_sum_patch(mt, builder, clean, patch, heads) -> str:
    comps = builder.predicate(clean.metadata["predicate"])["components"]
    summed = {}
    for comp in comps:
        src = copy.deepcopy(patch)
        src.category = comp
        clean_tok, patches = build_q_patches(mt, clean, src, heads)
        for p in patches:
            summed[p.location] = summed.get(p.location, 0) + p.patch
    patches = [PatchSpec(location=loc, patch=v) for loc, v in summed.items()]
    logits = patch_with_baukit(mt=mt, inputs=clean_tok, patches=patches).logits[0, -1]
    return top_role(mt, clean, logits)


@torch.inference_mode()
def cosine_rows(mt, builder, pid, heads, n):
    p = builder.predicate(pid)
    rows = []
    for i in range(n):
        s = builder.compositional_list(pid)
        prompts = []
        for q in [p["phrase"], *p["components"]]:
            s.category = q
            prompts.append(s.prompt())
        batch = prepare_input(prompts=prompts, tokenizer=mt)
        qs = cache_q_projections(mt=mt, input=batch, heads=heads, token_indices=[[-1]] * 3)
        for l, h in heads:
            comp, a, b = (qs[k][(l, h, -1)].float() for k in range(3))
            cos = torch.nn.functional.cosine_similarity
            rows.append({"predicate": pid, "list": i, "layer": l, "head": h,
                         "cos_comp_a": cos(comp, a, dim=0).item(), "cos_comp_b": cos(comp, b, dim=0).item(),
                         "cos_a_b": cos(a, b, dim=0).item(), "cos_comp_sum": cos(comp, a + b, dim=0).item()})
    return rows


def main():
    p = cli.parser(__doc__)
    p.set_defaults(seed=3000)
    p.add_argument("--n_behavior", type=int, default=100)
    p.add_argument("--n_pairs", type=int, default=128, help="per predicate and direction")
    p.add_argument("--n_qsum", type=int, default=64)
    p.add_argument("--n_cosine", type=int, default=32)
    p.add_argument("--attn_n", type=int, default=32)
    p.add_argument("--ignore_gate", action="store_true")
    args = p.parse_args()
    cli.setup(args)

    if not read_json(RESULTS / "gate.json")["pass"] and not args.ignore_gate:
        raise SystemExit("Replication gate failed (results/gate.json). Not running the extension.")
    frozen = read_json(FROZEN)
    assert frozen["split_hash"] == split_hash(), "split changed since the heads were frozen"
    heads = [tuple(h) for h in frozen["heads"]]

    mt = upstream.load_model(args.model)
    builder = CompositionalBuilder(mt)
    pids = list(builder.spec["predicates"])

    # 1. behavior
    beh = [r for pid in pids for r in behavior(mt, builder, pid, args.n_behavior)]
    write_jsonl(beh, OUT / "behavior.jsonl")
    beh_df = pd.DataFrame(beh)
    beh_table = summarize(beh_df, ["predicate"], ["correct"]).rename(columns={"correct": "accuracy"})
    roles = beh_df.groupby("predicate").chosen_role.value_counts(normalize=True).unstack(fill_value=0)
    beh_table = beh_table.merge(roles.reset_index(), on="predicate")
    write_csv(beh_table, RESULTS / "tables" / "compositional_behavior.csv")
    print(beh_table.to_string(index=False))
    kept = beh_table.loc[beh_table.accuracy >= BEHAVIOR_THRESHOLD, "predicate"].tolist()
    print(f"kept for causal analysis: {kept}")

    # 2. patching
    head_sets = {"filter": heads, "random_0": matched_random_heads(heads, mt.config.num_attention_heads, seed=0)}
    records, pairs_by_key = [], {}
    for pid in kept:
        for direction in ("comp_to_simple", "simple_to_comp"):
            path = OUT / "pairs" / f"{pid}_{direction}.jsonl"
            if path.exists():
                pairs = load_pairs(path)
            else:
                make = builder.pair_comp_to_simple if direction == "comp_to_simple" else builder.pair_simple_to_comp
                pairs = [make(pid) for _ in range(args.n_pairs)]
                save_pairs(pairs, path)
            pairs_by_key[(pid, direction)] = pairs
            print(f"evaluating {pid} {direction}")
            records += run_conditions(mt, pairs, head_sets, attn_n=args.attn_n, experiment="compositional",
                                      predicate=pid, ptype=builder.predicate(pid)["type"], direction=direction)
            write_jsonl(records, OUT / "records.jsonl")
    df = pd.DataFrame(records)
    table = summarize(df, ["predicate", "ptype", "direction", "head_set"],
                      ["causal", "delta_target_logit", "pref_shift", "delta_attn_target"])
    choice = (df.groupby(["predicate", "direction", "head_set"]).top_after_role
              .value_counts(normalize=True).unstack(fill_value=0).reset_index())
    table = table.merge(choice, on=["predicate", "direction", "head_set"])
    write_csv(table, RESULTS / "tables" / "compositional_patching.csv")
    print(table.to_string(index=False))

    # 3. q-sum (exploratory)
    qsum = []
    for pid in kept:
        for clean, patch in pairs_by_key[(pid, "comp_to_simple")][: args.n_qsum]:
            qsum.append({"predicate": pid, "role": q_sum_patch(mt, builder, clean, patch, heads)})
    if qsum:
        q_df = pd.DataFrame(qsum)
        q_table = q_df.groupby("predicate").role.value_counts(normalize=True).unstack(fill_value=0).reset_index()
        write_csv(q_table, RESULTS / "tables" / "compositional_qsum.csv")
        print(q_table.to_string(index=False))

    # 4. cosine (exploratory)
    cos = [r for pid in kept for r in cosine_rows(mt, builder, pid, heads, args.n_cosine)]
    if cos:
        cos_df = pd.DataFrame(cos)
        write_csv(cos_df, OUT / "cosine.csv")
        c_table = cos_df.groupby("predicate")[["cos_comp_a", "cos_comp_b", "cos_a_b", "cos_comp_sum"]].mean()
        write_csv(c_table.reset_index(), RESULTS / "tables" / "compositional_cosine.csv")
        print(c_table.to_string())

    write_json({"kept": kept, "behavior_threshold": BEHAVIOR_THRESHOLD, "ignored_gate": args.ignore_gate,
                "meta": upstream.run_metadata(mt, seed=args.seed, split_hash=split_hash())},
               RESULTS / "compositional_run.json")


if __name__ == "__main__":
    main()
