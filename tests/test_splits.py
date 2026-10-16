"""Split validity. Runs without a GPU or model weights."""

import json

import pytest

from filtergen.splits import (
    B2_JSON,
    DISCOVERY,
    HELDOUT_B1,
    HELDOUT_B2,
    MAX_SAMPLED_DISTRACTOR_CATEGORIES,
    POOLS,
    build_task_json,
    remaining_distractor_categories,
    valid_pairs,
)
from filtergen.upstream import OBJECTS_JSON


@pytest.fixture(scope="module")
def objects():
    return json.load(open(OBJECTS_JSON))


@pytest.fixture(scope="module")
def b2():
    return json.load(open(B2_JSON))


def test_pools_disjoint_and_cover_upstream(objects):
    assert not set(DISCOVERY) & set(HELDOUT_B1)
    assert not set(DISCOVERY + HELDOUT_B1) & set(HELDOUT_B2)
    assert sorted(DISCOVERY + HELDOUT_B1) == sorted(objects["categories"])


def test_confusable_groups_not_split(objects):
    for cat, excl in objects["exclude_categories"].items():
        for other in excl:
            assert (cat in DISCOVERY) == (other in DISCOVERY), f"{cat} / {other} straddle the split"


def test_discovery_task_has_no_heldout_categories_or_items(objects, b2):
    task = build_task_json(DISCOVERY)
    assert set(task["categories"]) == set(DISCOVERY)
    discovery_items = {i.lower() for items in task["categories"].values() for i in items}
    heldout_items = {
        i.lower()
        for c in HELDOUT_B1
        for i in objects["categories"][c]
    } | {i.lower() for items in b2["categories"].values() for i in items}
    assert not discovery_items & heldout_items
    for cat, excl in task["exclude_categories"].items():
        assert cat in DISCOVERY and set(excl) <= set(DISCOVERY)


@pytest.mark.parametrize(
    "categories,src,dst",
    [
        (DISCOVERY, "seen", "seen"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "seen", "seen"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "b1", "seen"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "seen", "b1"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "b1", "b1"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "b2", "seen"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "seen", "b2"),
        (DISCOVERY + HELDOUT_B1 + HELDOUT_B2, "b2", "b2"),
    ],
)
def test_upstream_sampler_has_enough_distractor_categories(categories, src, dst):
    """Upstream calls random.sample over the remaining categories with k up to 5;
    fewer would raise ValueError mid-run."""
    task = build_task_json(categories)
    pairs = list(valid_pairs(task, POOLS[src], POOLS[dst]))
    assert pairs
    for s, d in pairs:
        assert remaining_distractor_categories(task, s, d) >= MAX_SAMPLED_DISTRACTOR_CATEGORIES, (s, d)


def test_b2_items(objects, b2):
    upstream_items = {i.lower() for items in objects["categories"].values() for i in items}
    seen = set()
    for cat, items in b2["categories"].items():
        assert len(items) >= 12, cat
        for item in items:
            assert item.lower() not in upstream_items, item
            assert item.lower() not in seen, item
            seen.add(item.lower())
    for cat, excl in b2["exclude_categories"].items():
        for other in excl:
            assert cat in b2["exclude_categories"].get(other, []), f"exclusion {cat}->{other} not symmetric"
