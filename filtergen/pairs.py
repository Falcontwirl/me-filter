"""Counterfactual (source, destination) pairs, built with upstream's sampler.

Upstream naming: `patch_sample` is the source prompt (whose query we cache),
`clean_sample` is the destination prompt we patch into. Pairs are stored
as (clean, patch) tuples, matching upstream's train/validation sets.
"""

import json
import logging
import random
from pathlib import Path

from filtergen import upstream  # noqa: F401  (puts upstream on sys.path)
from filtergen.splits import valid_pairs
from src.selection.data import (
    CounterFactualSamplePair,
    SelectOneTask,
    get_counterfactual_samples_within_task,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_IDX = 3  # "<options>\nWhich among these objects mentioned above is a <category>?\nAnswer:"
OPTION_STYLE = "single_line"
# Upstream uses different distractor counts for localization and evaluation:
LOCATE_N_DISTRACTORS = range(2, 7)  # scripts/locate_selection_heads.prepare_dataset
EVAL_N_DISTRACTORS = range(2, 6)  # notebooks/checking_head_generalization/select_one.ipynb


def load_task(path: Path) -> SelectOneTask:
    return SelectOneTask.load(path=str(path))


def sample_pair(mt, task: SelectOneTask, src: str, dst: str, n_distractors: range):
    """One pair the model solves in all three runs (source, destination, gold).

    Upstream re-samples recursively until the model gets all three right. A
    category the model can't handle will exhaust Python's recursion limit
    (RecursionError), which we leave to the caller.
    """
    patch, clean = get_counterfactual_samples_within_task(
        task=task,
        mt=mt,
        patch_category=src,
        clean_category=dst,
        filter_by_lm_prediction=True,
        distinct_options=True,
        patch_n_distractors=random.choice(n_distractors),
        clean_n_distractors=random.choice(n_distractors),
        prompt_template_idx=PROMPT_TEMPLATE_IDX,
        option_style=OPTION_STYLE,
    )
    return clean, patch


def sample_pairs(
    mt,
    task: SelectOneTask,
    task_json: dict,
    src_pool: list[str],
    dst_pool: list[str],
    n: int,
    n_distractors: range,
) -> list[tuple]:
    """n pairs, with (src, dst) categories drawn uniformly from the valid combinations."""
    combos = list(valid_pairs(task_json, src_pool, dst_pool))
    assert combos, f"no valid (src, dst) pairs for {src_pool} x {dst_pool}"
    pairs = []
    while len(pairs) < n:
        src, dst = random.choice(combos)
        try:
            pairs.append(sample_pair(mt, task, src, dst, n_distractors))
        except RecursionError:
            logger.error(f"model could not solve {src} -> {dst}; skipping this draw")
            continue
        if len(pairs) % 64 == 0:
            logger.info(f"sampled {len(pairs)}/{n} pairs")
    return pairs


def save_pairs(pairs: list[tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for clean, patch in pairs:
            pair = CounterFactualSamplePair(patch_sample=patch, clean_sample=clean)
            pair.detensorize()
            d = pair.to_dict()
            for side in ("patch_sample", "clean_sample"):
                d[side]["metadata"].pop("tokenized", None)
                d[side]["prediction"] = None
            f.write(json.dumps(d) + "\n")


def load_pairs(path: Path) -> list[tuple]:
    pairs = []
    with open(path) as f:
        for line in f:
            pair = CounterFactualSamplePair.from_dict(json.loads(line))
            pairs.append((pair.clean_sample, pair.patch_sample))
    return pairs
