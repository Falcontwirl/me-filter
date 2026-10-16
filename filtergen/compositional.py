"""RQ3: lists and counterfactual pairs for compositional predicates.

Every list has exactly one item satisfying its question. Lists for a
compositional predicate also contain lures that satisfy only one component
(e.g. a red non-fruit and a non-red fruit for "a red fruit"), so a patched
model's choice tells full composition apart from partial matching.

Samples are upstream `SelectionSample`s, so `filtergen.evaluate` (upstream
patching) runs on them unchanged. The template equals upstream template 3 with
the article moved into the category ("is a <cat>" -> "is <a cat>"), so simple
predicates produce exactly upstream's prompt text.
"""

import json
import random
from dataclasses import dataclass

from filtergen import upstream  # noqa: F401
from filtergen.splits import DISCOVERY, build_task_json
from src.selection.data import SelectionSample
from src.selection.utils import get_first_token_id, verify_correct_option
from src.tokens import prepare_input

COMP_JSON = upstream.ROOT / "data" / "compositional.json"
MAX_TRIES = 50


def load_spec() -> dict:
    with open(COMP_JSON) as f:
        return json.load(f)


@dataclass
class ListSpec:
    question: str  # phrase with article, e.g. "a red fruit"
    answer: str
    others: dict  # item -> role


class CompositionalBuilder:
    def __init__(self, mt):
        self.mt = mt
        self.spec = load_spec()
        self.template = self.spec["template"]
        self.task = build_task_json(DISCOVERY)
        self.items = self.task["categories"]

    # ---------- item pools ----------
    def predicate(self, pid: str) -> dict:
        return self.spec["predicates"][pid]

    def match_pool(self, pid: str) -> list[str]:
        p = self.predicate(pid)
        if "match_from_categories" in p:
            return [i for c in p["match_from_categories"] for i in self.items[c]]
        return p["match"]

    def simple_categories(self, pid: str) -> list[str]:
        """Simple predicates that can be paired with `pid` without ambiguity."""
        avoid = set(self.predicate(pid)["dest_avoid"])
        return [c for c in self.spec["destination_categories"] if c not in avoid]

    def neutral_categories(self, pid: str, simple: str | None) -> list[str]:
        blocked = set(self.predicate(pid)["avoid"])
        if simple is not None:
            blocked |= {simple} | set(self.task["exclude_categories"].get(simple, []))
        return [c for c in DISCOVERY if c not in blocked]

    def _tok(self, item: str) -> int:
        return get_first_token_id(item, self.mt.tokenizer, prefix=" ")

    def _pick(self, pool, used_tokens: set) -> str:
        options = [i for i in pool if self._tok(i) not in used_tokens]
        choice = random.choice(options)
        used_tokens.add(self._tok(choice))
        return choice

    # ---------- lists ----------
    def _fill(self, required: dict, pid: str, simple: str | None, n_neutral: int, used: set) -> dict:
        """required: role -> item pool. Returns item -> role, all first tokens distinct."""
        roles = {}
        for role, pool in required.items():
            if pool:
                roles[self._pick(pool, used)] = role
        neutral_cats = random.sample(self.neutral_categories(pid, simple), n_neutral)
        for c in neutral_cats:
            roles[self._pick(self.items[c], used)] = "neutral"
        return roles

    def _sample(self, question: str, roles: dict, answer: str) -> SelectionSample:
        options = list(roles)
        random.shuffle(options)
        sample = SelectionSample(
            obj=answer,
            obj_idx=options.index(answer),
            prompt_template=self.template,
            options=options,
            answer=answer,
            category=question,
            ans_token_id=self._tok(answer),
            default_option_style="single_line",
        )
        sample.metadata["roles"] = {str(self._tok(o)): r for o, r in roles.items()}
        return sample

    def _correct(self, sample: SelectionSample) -> bool:
        ok, _, _ = verify_correct_option(
            mt=self.mt,
            target=sample.ans_token_id,
            options=sample.options,
            input=prepare_input(prompts=sample.prompt(), tokenizer=self.mt),
        )
        return ok

    def compositional_list(self, pid: str) -> SelectionSample:
        """Behavioral probe: one match, one lure per component, 1-2 neutral items."""
        p = self.predicate(pid)
        used = set()
        roles = self._fill({"answer": self.match_pool(pid), "lure_a": p["lure_a"], "lure_b": p["lure_b"]},
                           pid, None, random.choice([1, 2]), used)
        answer = next(i for i, r in roles.items() if r == "answer")
        return self._sample(p["phrase"], roles, answer)

    # ---------- counterfactual pairs ----------
    def pair_comp_to_simple(self, pid: str, verify: bool = True):
        """Source asks the compositional predicate; destination asks a simple
        category. Target = the destination item satisfying the compositional
        predicate. Returns (clean, patch) like upstream."""
        p = self.predicate(pid)
        for _ in range(MAX_TRIES):
            simple = random.choice(self.simple_categories(pid))
            used = set()
            src_roles = self._fill({"answer": self.match_pool(pid), "lure_a": p["lure_a"], "lure_b": p["lure_b"]},
                                   pid, simple, random.choice([1, 2]), used)
            dst_roles = self._fill({"dest_answer": self.items[simple], "target": self.match_pool(pid),
                                    "lure_a": p["lure_a"], "lure_b": p["lure_b"]},
                                   pid, simple, random.choice([1, 2]), used)
            pair = self._finish(p["phrase"], src_roles, f"a {simple}", dst_roles, gold_question=p["phrase"],
                                verify=verify)
            if pair is not None:
                clean, patch = pair
                clean.metadata.update(predicate=pid, direction="comp_to_simple", simple=simple)
                patch.metadata.update(predicate=pid, direction="comp_to_simple", simple=simple)
                if "match_from_categories" in p:
                    t = clean.metadata["track_type_obj"]
                    clean.metadata["target_component"] = next(
                        c for c in p["match_from_categories"] if t in self.items[c])
                return clean, patch
        raise RuntimeError(f"could not build a verified comp->simple pair for {pid}")

    def pair_simple_to_comp(self, pid: str, verify: bool = True):
        """Source asks a simple category; destination asks the compositional
        predicate. Target = the destination item of the simple category."""
        p = self.predicate(pid)
        for _ in range(MAX_TRIES):
            simple = random.choice(self.simple_categories(pid))
            used = set()
            src_roles = self._fill({"answer": self.items[simple]}, pid, simple, random.choice([2, 3, 4]), used)
            dst_roles = self._fill({"dest_answer": self.match_pool(pid), "target": self.items[simple],
                                    "lure_a": p["lure_a"], "lure_b": p["lure_b"]},
                                   pid, simple, random.choice([1, 2]), used)
            pair = self._finish(f"a {simple}", src_roles, p["phrase"], dst_roles, gold_question=f"a {simple}",
                                verify=verify)
            if pair is not None:
                clean, patch = pair
                clean.metadata.update(predicate=pid, direction="simple_to_comp", simple=simple)
                patch.metadata.update(predicate=pid, direction="simple_to_comp", simple=simple)
                return clean, patch
        raise RuntimeError(f"could not build a verified simple->comp pair for {pid}")

    def _finish(self, src_q, src_roles, dst_q, dst_roles, gold_question, verify):
        src_answer = next(i for i, r in src_roles.items() if r == "answer")
        dst_answer = next(i for i, r in dst_roles.items() if r == "dest_answer")
        target = next(i for i, r in dst_roles.items() if r == "target")
        patch = self._sample(src_q, src_roles, src_answer)
        # as upstream: neither the destination answer nor the target sits at the source answer's index
        for _ in range(MAX_TRIES):
            clean = self._sample(dst_q, dst_roles, dst_answer)
            if patch.obj_idx not in (clean.options.index(dst_answer), clean.options.index(target)):
                break
        else:
            return None
        clean.metadata.update(
            track_type_obj=target,
            track_type_obj_idx=clean.options.index(target),
            track_type_obj_token_id=self._tok(target),
        )
        if verify:
            gold = self._sample(gold_question, dst_roles, target)
            gold.options = clean.options
            gold.obj_idx = clean.options.index(target)
            if not (self._correct(patch) and self._correct(clean) and self._correct(gold)):
                return None
        return clean, patch
