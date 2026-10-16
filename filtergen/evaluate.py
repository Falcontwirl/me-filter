"""Query-patching evaluation. All causal numbers come from upstream's
`validate_q_proj_ie_on_sample_pair`, called with the settings the authors use
in notebooks/checking_head_generalization/select_one.ipynb:

- patch the last two tokens ("\\nAnswer", ":") plus the "?" position
- average the source query over copies of the source prompt with the answer
  moved to every position (removes order-ID information, paper App. F)
- causality = target has the highest probability among the destination
  options + the source answer (not the full vocabulary)

The attention metric is ours; `build_q_patches` mirrors upstream's patch
construction so the attention run sees the same intervention. The smoke test
checks that both give identical logits.
"""

import copy
import logging

import numpy as np
import torch

from filtergen import upstream  # noqa: F401
from filtergen.spans import option_token_spans
from src.attention import get_attention_matrices
from src.functional import PatchSpec
from src.selection.functional import cache_q_projections, find_quesmark_pos
from src.selection.optimization import validate_q_proj_ie_on_sample_pair
from src.tokens import prepare_input
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)

# Heads at layer >= 23 are dropped for Llama-3.1-8B in the authors' notebooks.
# We report with and without this cutoff; see experiments/02_locate.py.
UPSTREAM_8B_LAYER_CUTOFF = 23


def apply_layer_cutoff(heads, cutoff: int | None):
    return [h for h in heads if cutoff is None or h[0] < cutoff]


def _rotated_copies(patch):
    """Upstream's distinct_options=False averaging: the same source prompt with
    its answer swapped into each position."""
    copies = []
    for obj_idx in range(len(patch.options)):
        s = copy.deepcopy(patch)
        s.options[obj_idx], s.options[s.obj_idx] = s.options[s.obj_idx], s.options[obj_idx]
        s.obj_idx = obj_idx
        copies.append(s)
    return copies


def evaluate_pair(mt, clean, patch, heads) -> dict:
    """One causal patching run. Returns a flat record for the results table."""
    result = validate_q_proj_ie_on_sample_pair(
        mt=mt,
        clean_sample=clean,
        patch_sample=patch,
        heads=[tuple(h) for h in heads],
        query_indices={-2: -2, -1: -1},  # fresh dict: upstream mutates it
        add_ques_pos_to_query_indices=True,
        patch_args={"batch_size": len(patch.options), "distinct_options": False},
    )
    before, after = result["clean_track"], result["int_track"]
    target = clean.metadata["track_type_obj_token_id"]
    dest_ans = clean.ans_token_id
    src_ans = patch.ans_token_id
    top_after = next(iter(after))

    def logit(track, tok):
        return track[tok][1].logit

    pref_before = logit(before, target) - logit(before, dest_ans)
    pref_after = logit(after, target) - logit(after, dest_ans)
    extra = {}
    if "roles" in clean.metadata:  # compositional lists label every option (target, lure_a, ...)
        extra["top_after_role"] = clean.metadata["roles"].get(str(top_after), "source_answer")
    return {
        **extra,
        "src_category": patch.category,
        "dst_category": clean.category,
        "n_options": len(clean.options),
        "causal": int(top_after == target),
        "top_after": (
            "target" if top_after == target
            else "dest_answer" if top_after == dest_ans
            else "source_answer" if top_after == src_ans
            else "other_option"
        ),
        "target_logit_before": logit(before, target),
        "target_logit_after": logit(after, target),
        "delta_target_logit": logit(after, target) - logit(before, target),
        "pref_shift": pref_after - pref_before,
        "target_rank_after": after[target][0],
        "dest_answer_rank_after": after[dest_ans][0],
    }


def run_conditions(mt, pairs, head_sets: dict, attn_n: int = 0, attn_on=("filter",), **tags) -> list[dict]:
    """Evaluate every pair under every head set (e.g. filter heads, random
    controls, an empty set for "no patch"). Attention is measured for the
    first `attn_n` pairs of the head sets named in `attn_on`."""
    records = []
    for i, (clean, patch) in enumerate(pairs):
        for name, heads in head_sets.items():
            rec = {"pair_idx": i, "head_set": name, "n_heads": len(heads), **tags}
            rec.update(evaluate_pair(mt, clean, patch, heads))
            if i < attn_n and name in attn_on and heads:
                rec.update(attention_shift(mt, clean, patch, heads))
            records.append(rec)
        if (i + 1) % 32 == 0:
            logger.info(f"evaluated {i + 1}/{len(pairs)} pairs")
    return records


@torch.inference_mode()
def build_q_patches(mt, clean, patch, heads):
    """Same patches upstream builds inside validate_q_proj_ie_on_sample_pair."""
    clean_tok = prepare_input(prompts=clean.prompt(), tokenizer=mt, return_offsets_mapping=True)
    clean_off = clean_tok.pop("offset_mapping")[0]
    clean_q = find_quesmark_pos(
        prompt=clean.prompt(), tokenizer=mt.tokenizer, tokenized=clean_tok, offset_mapping=clean_off
    )
    patch_tok = prepare_input(prompts=patch.prompt(), tokenizer=mt, return_offsets_mapping=True)
    patch_off = patch_tok.pop("offset_mapping")[0]
    patch_q = find_quesmark_pos(
        prompt=patch.prompt(), tokenizer=mt.tokenizer, tokenized=patch_tok, offset_mapping=patch_off
    )
    query_indices = {-2: -2, -1: -1, patch_q: clean_q}

    copies = _rotated_copies(patch)
    batch = prepare_input(
        prompts=[s.prompt() for s in copies], tokenizer=mt, return_offsets_mapping=True
    )
    batch_off = batch.pop("offset_mapping")
    token_indices = []
    for i, s in enumerate(copies):
        cur = {k: k for k in query_indices}
        q = find_quesmark_pos(
            prompt=s.prompt(),
            tokenizer=mt.tokenizer,
            tokenized=TokenizerOutput(data={k: v[i : i + 1, :] for k, v in batch.items()}),
            offset_mapping=batch_off[i],
        )
        cur[q] = clean_q
        token_indices.append(list(cur.keys()))

    cached = cache_q_projections(
        mt=mt, input=batch, heads=[tuple(h) for h in heads], token_indices=token_indices
    )
    patches = []
    for loc in cached[0]:
        layer_idx, head_idx, patch_idx = loc
        avg = torch.stack([c[loc] for c in cached]).mean(dim=0)
        patches.append(
            PatchSpec(
                location=(
                    mt.attn_module_name_format.format(layer_idx) + ".q_proj",
                    head_idx,
                    query_indices[patch_idx],
                ),
                patch=avg,
            )
        )
    return clean_tok, patches


@torch.inference_mode()
def attention_shift(mt, clean, patch, heads) -> dict:
    """Mean attention (over `heads`) from the last token to the target item and
    to the destination answer, before and after query patching."""
    heads = [tuple(h) for h in heads]
    clean_tok, patches = build_q_patches(mt, clean, patch, heads)
    offsets = prepare_input(
        prompts=clean.prompt(), tokenizer=mt, return_offsets_mapping=True
    )["offset_mapping"][0]
    spans = option_token_spans(clean.prompt(), clean.options, offsets.tolist())
    target_span = spans[clean.metadata["track_type_obj_idx"]]
    answer_span = spans[clean.obj_idx]

    out = {}
    for label, p in (("before", []), ("after", patches)):
        attn = get_attention_matrices(input=clean_tok, mt=mt, patches=p).attention_matrices
        rows = np.stack([attn[l, h, -1] for l, h in heads])  # (n_heads, seq)
        out[f"attn_target_{label}"] = float(rows[:, slice(*target_span)].sum(1).mean())
        out[f"attn_answer_{label}"] = float(rows[:, slice(*answer_span)].sum(1).mean())
    out["delta_attn_target"] = out["attn_target_after"] - out["attn_target_before"]
    return out
