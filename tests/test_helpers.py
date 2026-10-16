from collections import Counter

import numpy as np
import pandas as pd

from filtergen.controls import matched_random_heads
from filtergen.results import bootstrap_ci, summarize
from filtergen.spans import option_token_spans


def test_matched_random_heads_preserves_layer_histogram_and_is_disjoint():
    heads = [(10, 3), (10, 7), (12, 0), (20, 31)]
    for seed in range(5):
        rand = matched_random_heads(heads, n_heads_per_layer=32, seed=seed)
        assert Counter(l for l, _ in rand) == Counter(l for l, _ in heads)
        assert not set(rand) & set(heads)
    assert matched_random_heads(heads, 32, seed=1) == matched_random_heads(heads, 32, seed=1)


def test_bootstrap_ci_brackets_mean():
    x = np.r_[np.ones(80), np.zeros(20)]
    lo, hi = bootstrap_ci(x)
    assert lo < 0.8 < hi and 0.65 < lo and hi < 0.95


def test_summarize_groups():
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "causal": [1] * 10 + [0] * 10})
    t = summarize(df, ["g"], ["causal"]).set_index("g")
    assert t.loc["a", "causal"] == 1 and t.loc["b", "causal"] == 0 and t.loc["a", "n"] == 10


def _whitespace_offsets(prompt):
    """Tokens = runs of non-space chars with their leading space, like BPE."""
    offsets, i = [(0, 0)], 0  # BOS
    while i < len(prompt):
        j = i
        while j < len(prompt) and prompt[j] == " ":
            j += 1
        k = j
        while k < len(prompt) and prompt[k] not in " ,.":
            k += 1
        if k == j:
            k = j + 1
        offsets.append((i, k))
        i = k
    return offsets


def test_option_token_spans():
    options = ["Apple", "Coffee table", "Bus"]
    prompt = f"Options: {', '.join(options)}.\nWhich among these objects mentioned above is a fruit?\nAnswer:"
    offsets = _whitespace_offsets(prompt)
    spans = option_token_spans(prompt, options, offsets)
    for opt, (s, e) in zip(options, spans):
        text = "".join(prompt[a:b] for a, b in offsets[s:e]).strip()
        assert text == opt
