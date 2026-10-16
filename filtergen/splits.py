"""Category split for RQ2. Fixed before any head localization.

Upstream's `exclude_categories` groups confusable categories:
{fruit, vegetable, flower, tree} and {electronics, kitchen appliance, office supply}.
Each group stays on one side of the split, so no held-out category is a
near-synonym of a discovery category.

Usage: python -m filtergen.splits   (writes data/splits/*.json and prints the hash)
"""

import copy
import hashlib
import json
from pathlib import Path

from filtergen.upstream import OBJECTS_JSON, ROOT

B2_JSON = ROOT / "data" / "b2_categories.json"
SPLIT_DIR = ROOT / "data" / "splits"

# Used for head localization (predicates and distractors).
DISCOVERY = [
    "fruit", "vegetable", "flower", "tree",
    "vehicle", "furniture", "animal", "clothing", "building", "sport equipment",
]
# Upstream categories never seen during localization.
HELDOUT_B1 = [
    "electronics", "kitchen appliance", "office supply",
    "music instrument", "bathroom item", "jewelry",
]
# New categories we authored (data/b2_categories.json).
HELDOUT_B2 = ["medical equipment", "astronomical object", "geological formation"]

POOLS = {"seen": DISCOVERY, "b1": HELDOUT_B1, "b2": HELDOUT_B2}

# Upstream SelectOne pairs use up to 6 distractors (2 must-have + up to 5
# sampled from distinct other categories), so a task must leave >= 5 other
# categories after exclusions for every valid (source, destination) pair.
MAX_SAMPLED_DISTRACTOR_CATEGORIES = 5


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _merge_excludes(*maps: dict) -> dict:
    merged: dict[str, set] = {}
    for m in maps:
        for cat, excl in m.items():
            merged.setdefault(cat, set()).update(excl)
    return {k: sorted(v) for k, v in merged.items()}


def build_task_json(categories: list[str]) -> dict:
    """An objects.json-schema dict restricted to `categories`."""
    objects = _load(OBJECTS_JSON)
    b2 = _load(B2_JSON)
    pool = {**objects["categories"], **b2["categories"]}
    missing = [c for c in categories if c not in pool]
    assert not missing, f"unknown categories: {missing}"

    excludes = _merge_excludes(objects["exclude_categories"], b2["exclude_categories"])
    keep = set(categories)
    task = copy.deepcopy(objects)
    task["name"] = "objects"
    task["categories"] = {c: pool[c] for c in categories}
    task["exclude_categories"] = {
        c: [e for e in excl if e in keep]
        for c, excl in excludes.items()
        if c in keep and any(e in keep for e in excl)
    }
    return task


def valid_pairs(task: dict, src_pool: list[str], dst_pool: list[str]):
    """(src, dst) category pairs upstream's sampler can produce."""
    excl = task["exclude_categories"]
    for s in src_pool:
        for d in dst_pool:
            if d != s and d not in excl.get(s, []):
                yield s, d


def remaining_distractor_categories(task: dict, src: str, dst: str) -> int:
    excl = task["exclude_categories"]
    blocked = {src, dst} | set(excl.get(src, [])) | set(excl.get(dst, []))
    return len(set(task["categories"]) - blocked)


def split_definition() -> dict:
    return {
        "discovery": DISCOVERY,
        "heldout_b1": HELDOUT_B1,
        "heldout_b2": HELDOUT_B2,
        "b2_items": _load(B2_JSON)["categories"],
    }


def split_hash() -> str:
    blob = json.dumps(split_definition(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_split_files(out_dir: Path = SPLIT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        # localization: discovery categories only
        "discovery": build_task_json(DISCOVERY),
        # evaluation: everything, so all RQ2 cells share one distractor pool
        "all": build_task_json(DISCOVERY + HELDOUT_B1 + HELDOUT_B2),
    }
    paths = {}
    for name, task in files.items():
        paths[name] = out_dir / f"{name}.json"
        with open(paths[name], "w") as f:
            json.dump(task, f, indent=2)
    with open(out_dir / "split.json", "w") as f:
        json.dump({"hash": split_hash(), **split_definition()}, f, indent=2)
    return paths


if __name__ == "__main__":
    for name, path in write_split_files().items():
        print(f"{name}: {path}")
    print(f"split hash: {split_hash()}")
