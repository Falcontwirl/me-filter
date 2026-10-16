"""Random head sets matched to a real head set, for controls."""

import random
from collections import Counter


def matched_random_heads(
    heads: list[tuple[int, int]], n_heads_per_layer: int, seed: int
) -> list[tuple[int, int]]:
    """Same number of heads in each layer as `heads`, chosen at random from
    the heads in that layer that are not in `heads`."""
    rng = random.Random(seed)
    chosen = set(map(tuple, heads))
    out = []
    for layer, count in sorted(Counter(l for l, _ in chosen).items()):
        pool = [h for h in range(n_heads_per_layer) if (layer, h) not in chosen]
        assert len(pool) >= count, f"layer {layer}: need {count} non-selected heads, have {len(pool)}"
        out.extend((layer, h) for h in rng.sample(pool, count))
    return sorted(out)
