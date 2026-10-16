"""Structure of data/compositional.json (mirrors CompositionalBuilder's pools without loading torch)."""

import json

import pytest

from filtergen.splits import DISCOVERY, build_task_json
from filtergen.upstream import ROOT

SPEC = json.load(open(ROOT / "data" / "compositional.json"))
TASK = build_task_json(DISCOVERY)
PREDICATES = list(SPEC["predicates"].items())


def match_pool(p):
    if "match_from_categories" in p:
        return [i for c in p["match_from_categories"] for i in TASK["categories"][c]]
    return p["match"]


def simple_categories(p):
    return [c for c in SPEC["destination_categories"] if c not in p["dest_avoid"]]


def neutral_categories(p, simple):
    blocked = set(p["avoid"]) | {simple} | set(TASK["exclude_categories"].get(simple, []))
    return [c for c in DISCOVERY if c not in blocked]


@pytest.mark.parametrize("pid,p", PREDICATES)
def test_roles_disjoint(pid, p):
    m, a, b = set(match_pool(p)), set(p["lure_a"]), set(p["lure_b"])
    assert len(m) >= 5
    assert not m & a and not m & b and not a & b


@pytest.mark.parametrize("pid,p", PREDICATES)
def test_avoid_covers_match_categories(pid, p):
    """Neutral distractors may not come from a category that contains a match item."""
    for cat, items in TASK["categories"].items():
        if set(items) & set(match_pool(p)):
            assert cat in p["avoid"], f"{pid}: neutral pool could supply a second match from {cat}"
    assert set(p["avoid"]) <= set(p["dest_avoid"])


@pytest.mark.parametrize("pid,p", PREDICATES)
def test_simple_predicates_unambiguous(pid, p):
    """A simple category paired with this predicate must not contain any match or lure item."""
    sims = simple_categories(p)
    assert sims
    special = set(match_pool(p)) | set(p["lure_a"]) | set(p["lure_b"])
    for s in sims:
        assert not set(TASK["categories"][s]) & special, (pid, s)
        assert len(neutral_categories(p, s)) >= 4, (pid, s)  # builder samples up to 4
