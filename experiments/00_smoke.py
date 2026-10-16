"""Smoke test on one GPU before any real run.

Checks:
1. the model answers a SelectOne prompt
2. patching one head's query at the last token changes only that head:
   earlier layers and the other heads in the same layer keep identical attention
3. our attention-path patches give the same logits as upstream's patching
4. one evaluated pair, plus timings for pair sampling, evaluation, and one
   mask-training step (to size the SLURM jobs)
"""

import random
import time

import numpy as np
import torch

from filtergen import cli, upstream
from filtergen.evaluate import attention_shift, build_q_patches, evaluate_pair
from filtergen.pairs import EVAL_N_DISTRACTORS, PROMPT_TEMPLATE_IDX, load_task, sample_pair
from filtergen.results import RESULTS, write_json
from filtergen.splits import SPLIT_DIR, write_split_files
from src.attention import get_attention_matrices
from src.functional import PatchSpec, patch_with_baukit
from src.selection.optimization import get_optimal_head_mask_prev
from src.selection.utils import verify_correct_option
from src.tokens import prepare_input


def main():
    args = cli.parser(__doc__).parse_args()
    cli.setup(args)
    report = {}

    write_split_files()
    mt = upstream.load_model(args.model)
    report["meta"] = upstream.run_metadata(mt)
    print(report["meta"])
    task = load_task(SPLIT_DIR / "discovery.json")

    # 1. plain answer
    s = task.get_random_sample(mt=mt, category="fruit", n_distractors=4,
                               prompt_template_idx=PROMPT_TEMPLATE_IDX)
    ok, preds, _ = verify_correct_option(mt=mt, target=s.ans_token_id, options=s.options, input=s.prompt())
    print(f"\n{s.prompt()} -> top: {preds[0]} | expected {s.obj} | correct={ok}")
    report["single_prompt_correct"] = bool(ok)

    # 2. intervention isolation
    layer, head = 15, 7
    tok = prepare_input(prompts=s.prompt(), tokenizer=mt)
    other = task.get_random_sample(mt=mt, category="vehicle", n_distractors=4,
                                   prompt_template_idx=PROMPT_TEMPLATE_IDX)
    other_tok = prepare_input(prompts=other.prompt(), tokenizer=mt)
    q_name = mt.attn_module_name_format.format(layer) + ".q_proj"
    head_dim = mt.config.hidden_size // mt.config.num_attention_heads
    with torch.no_grad():
        q_src = {}
        def grab(module, inp, out):
            q_src["q"] = out[0, -1].view(-1, head_dim)[head].clone()
        h = mt._model.get_submodule(q_name).register_forward_hook(grab)
        mt._model(**other_tok)
        h.remove()
    spec = PatchSpec(location=(q_name, head, -1), patch=q_src["q"])
    a0 = get_attention_matrices(input=tok, mt=mt, patches=[]).attention_matrices
    a1 = get_attention_matrices(input=tok, mt=mt, patches=[spec]).attention_matrices
    others = [x for x in range(a0.shape[1]) if x != head]
    iso = {
        "earlier_layers_max_diff": float(np.abs(a0[:layer] - a1[:layer]).max()),
        "same_layer_other_heads_max_diff": float(np.abs(a0[layer, others] - a1[layer, others]).max()),
        "patched_head_last_row_diff": float(np.abs(a0[layer, head, -1] - a1[layer, head, -1]).max()),
        "patched_head_other_rows_max_diff": float(np.abs(a0[layer, head, :-1] - a1[layer, head, :-1]).max()),
    }
    tol = 1e-5
    iso["pass"] = (iso["earlier_layers_max_diff"] <= tol and iso["same_layer_other_heads_max_diff"] <= tol
                   and iso["patched_head_other_rows_max_diff"] <= tol and iso["patched_head_last_row_diff"] > tol)
    print(f"\nisolation check: {iso}")
    report["isolation"] = iso

    # 3 + 4. one pair; random heads stand in for filter heads (none located yet)
    t0 = time.time()
    clean, patch = sample_pair(mt, task, "fruit", "vehicle", EVAL_N_DISTRACTORS)
    report["sec_sample_pair"] = time.time() - t0
    print(f"\nsource: {patch.prompt()!r} -> {patch.obj}")
    print(f"dest:   {clean.prompt()!r} -> {clean.obj} | target after patch: {clean.metadata['track_type_obj']}")

    rng = random.Random(args.seed)
    heads = sorted(rng.sample([(l, h) for l in range(8, 24) for h in range(32)], 40))
    t0 = time.time()
    rec = evaluate_pair(mt, clean, patch, heads)
    report["sec_evaluate_pair"] = time.time() - t0
    print(f"evaluate_pair (40 random heads): {rec}")
    report["example_record"] = rec

    clean_tok, patches = build_q_patches(mt, clean, patch, heads)
    ours = patch_with_baukit(mt=mt, inputs=clean_tok, patches=patches).logits[0, -1].float()
    tgt = clean.metadata["track_type_obj_token_id"]
    report["patch_consistency_abs_diff"] = abs(ours[tgt].item() - rec["target_logit_after"])
    print(f"our patches vs upstream, target logit |diff| = {report['patch_consistency_abs_diff']:.4f}")

    t0 = time.time()
    report["example_attention"] = attention_shift(mt, clean, patch, heads)
    report["sec_attention_shift"] = time.time() - t0

    # mask-training step timing on a 16-pair batch
    train = [sample_pair(mt, task, *rng.sample(["fruit", "vehicle", "animal", "clothing"], 2),
                         EVAL_N_DISTRACTORS) for _ in range(16)]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    get_optimal_head_mask_prev(mt=mt, train_set=train, learning_rate=1e-2, n_epochs=1,
                               lamb=1e-2, batch_size=16, query_indices=[-1])
    report["sec_mask_train_step_bs16"] = time.time() - t0
    report["peak_gpu_gb_mask_step"] = torch.cuda.max_memory_allocated() / 1e9

    write_json(report, RESULTS / "smoke.json")
    print(f"\nwrote {RESULTS / 'smoke.json'}")
    for k in ("sec_sample_pair", "sec_evaluate_pair", "sec_attention_shift",
              "sec_mask_train_step_bs16", "peak_gpu_gb_mask_step"):
        print(f"{k}: {report[k]:.2f}")


if __name__ == "__main__":
    main()
