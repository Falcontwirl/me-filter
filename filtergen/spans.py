"""Locating list items in a tokenized prompt (no model dependencies)."""


def option_token_spans(prompt: str, options: list[str], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Token range [start, end) of each option in a single_line prompt
    ("Options: a, b, c."), given the tokenizer's character offsets."""
    char = prompt.index("Options: ") + len("Options: ")
    char_spans = []
    for opt in options:
        assert prompt[char : char + len(opt)] == opt, (prompt, opt)
        char_spans.append((char, char + len(opt)))
        char += len(opt) + len(", ")
    spans = []
    for cs, ce in char_spans:
        toks = [i for i, (ts, te) in enumerate(offsets) if te > ts and ts < ce and te > cs]
        spans.append((toks[0], toks[-1] + 1))
    return spans
