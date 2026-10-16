## Scope Decisions (2026-10-15) — these override anything below that conflicts

- **Reuse the original repo (https://github.com/arnab-api/filter) as much as possible.** It is the default implementation for dataset utilities, model loading, head location (`scripts/locate_selection_heads.py`), query patching, the causality score, and the sparse-mask head search. Import or vendor it; only write new code for the extension's datasets and experiments.
- **Effort goes solely into our extension.** Replication (RQ1, Experiments 2–3) is a *gate*, not a deliverable:
  - run the repo's head location on our model
  - freeze the head set
  - confirm query patching shifts logits and attention on Dataset A above controls
  - if the gate fails, stop and report; do not proceed to the extension
- **In scope:** RQ2 (unseen predicates, Experiment 4 — primary), RQ3 (compositional, Experiment 5), RQ4 (mechanism stability).
- **Out of scope:** RQ5 / Experiment 6 (lazy vs eager, a replication of the paper's own finding) and Dataset E.
- **Figures:** Figure 1 (localization) as a sanity check, Figures 3–4 required, Figure 2 optional, Figure 5 dropped.
- **Implementation:** `README.md` has the layout and run order. Upstream code is fetched into `third_party/filter/` by `third_party/fetch_upstream.sh` (pinned commit, not redistributed; unmodified except our `env.yml`); our code lives in `filtergen/` and `experiments/`. Compute is Brown's Oscar cluster, via `slurm/`.

---

# CLAUDE.md — Generalization of Filtering Heads

## 1. Project Overview

**Project:** Replication and Extension of *LLMs Process Lists With General Filter Heads*

**Original paper:** Arnab Sen Sharma, Giordano Rogers, Natalie Shapira, and David Bau, ICLR 2026.

**Resources:**
- Paper: https://arxiv.org/pdf/2510.26784
- Project: https://filter.baulab.info/
- Code: https://github.com/arnab-api/filter

### Primary objective

Replicate the original finding that certain attention heads encode filtering predicates in their query vectors, then investigate whether these same heads generalize to more challenging, novel, and compositional predicates.

The project should prioritize a rigorous, reproducible experiment over an unnecessarily large implementation.

**Target:** Obtain meaningful experimental results in approximately one day, using a small, accessible open-weight model and available GPU resources.

Do not assume the original results automatically transfer from Llama-70B to smaller models.

---

## 2. Core Mechanistic Hypothesis

Transformers perform list filtering through specialized attention heads.

Consider:

```text
a. Apple
b. Carrot
c. Banana
d. Truck

Find the fruit.
Answer:
```

At the final colon token:

1. The residual stream contains information about the predicate, "is a fruit?"
2. Certain attention heads project that representation into a query vector.
3. The query interacts with key vectors representing the list items.
4. Matching items receive greater attention.
5. Their value vectors contribute to the updated residual stream.
6. Downstream computation uses that information to produce the answer.

The original paper demonstrates that a query vector from one predicate can be patched into a different prompt, causing the head to attend to items satisfying the source predicate.

**Important:** The paper identifies multiple cooperating filtering heads, not one universal head. Individual heads may have weak causal effects.

### Our hypothesis

The same filtering heads may implement a reusable predicate-selection mechanism that extends to new semantic categories, relational predicates, and compositional filtering rules.

We want to determine where this generalization succeeds and where it breaks down.

---

## 3. Research Questions

**RQ1 — Replication**

Can we reproduce filtering-head behavior and causal query portability in a smaller open-weight model?

**RQ2 — Predicate generalization**

Do filtering heads identified using ordinary categories also operate on new predicates that were excluded from head discovery?

**RQ3 — Compositional generalization**

Can the same filtering mechanism handle predicates involving conjunctions, disjunctions, negation, or relationships?

**RQ4 — Mechanism stability**

Does generalization rely on the same heads, or does the model recruit different circuits for different predicate classes?

**RQ5 — Lazy versus eager**

Does moving the predicate before the list reduce the causal influence of the original filtering heads, as reported in the paper?

---

## 4. Experimental Infrastructure

### Models

Start with a model that fits available compute.

Candidate models:
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Smaller models for initial debugging

Use one primary model for all experiments.

Do not claim to reproduce the original Llama-70B result unless that model is actually evaluated.

Record model version, checkpoint, precision, and tokenizer.

### Libraries

- Python 3.11+
- PyTorch
- Hugging Face Transformers
- baukit or TransformerLens, where compatible
- NumPy
- Pandas
- Matplotlib
- pytest

Use forward-pass inference and activation interventions. No model training or fine-tuning is required for the main experiments.

### Existing repository

Inspect the original repository before implementing anything.

Identify:
- Dataset-generation utilities
- Prompt templates
- Model-loading utilities
- Head-location scripts
- Query-patching functions
- Causality-score calculations
- Existing experiments for cross-task and cross-format portability

Use the existing implementation where it saves time, but write our experiments in separate modules.

The original repository exposes `scripts/locate_selection_heads.py` and a demonstration notebook.

Avoid reimplementing tested machinery without a reason.

---

## 5. Dataset Construction

Construct deterministic, automatically labeled filtering examples.

Each example must store:

```json
{
  "id": "example_001",
  "predicate": "fruit",
  "predicate_type": "category",
  "items": ["apple", "carrot", "banana", "truck"],
  "matching_items": ["apple", "banana"],
  "prompt": "...",
  "answer": "...",
  "template_id": "template_01",
  "seed": 42
}
```

For the initial replication, prioritize **exactly one matching item per list**, matching the original SelectOne setting.

Multiple-match examples should be a separate experiment because selecting several items changes the output task.

### Dataset A — Original-style predicates

Examples:
- Fruit
- Vehicle
- Animal
- Clothing
- Furniture

Purpose: Replicate ordinary semantic filtering.

### Dataset B — Held-out semantic predicates

Examples:
- Musical instrument
- Medical equipment
- Geological formation
- Astronomical object
- Kitchen appliance

Purpose: Evaluate predicates not used to identify the heads.

Hold out both predicate categories and test item combinations.

### Dataset C — Compositional predicates

Examples:
- Is a fruit AND is yellow
- Is an animal AND can fly
- Is a vehicle OR a musical instrument
- Is an animal but NOT a mammal

Purpose: Test whether the mechanism handles Boolean compositions.

Ensure that all combinations are unambiguous and contain appropriate positive and negative examples.

### Dataset D — Relational and contextual predicates

Examples:
- Is larger than a mouse
- Is used for cutting
- Is associated with winter
- Is suitable for underwater use

Purpose: Explore whether filtering heads generalize beyond simple taxonomic membership.

These predicates require careful validation because answers may depend on world knowledge or context.

### Dataset E — Prompt ordering

Create paired formats:

**Lazy:**
```text
Apple, carrot, banana, truck.
Find the vehicle.
Answer:
```

**Eager:**
```text
Find the vehicle.
Apple, carrot, banana, truck.
Answer:
```

Purpose: Test whether predicate position changes the mechanism.

### Dataset requirements

- Fix random seeds.
- Randomize item order.
- Balance answer positions.
- Prevent lexical shortcuts.
- Avoid ambiguous examples.
- Ensure source and destination lists are disjoint.
- Include an item matching the source predicate in the destination list.
- Ensure the source and destination predicates select different answers.
- Validate tokenization and answer scoring.

Start with approximately 50–100 examples per condition for pilot experiments. Increase only if computationally feasible.

---

## 6. Experiment 1 — Behavioral Baseline

Before performing mechanistic analysis, determine whether the model can solve each filtering task.

### Procedure

1. Load the model.
2. Generate the filtering prompts.
3. Run ordinary inference.
4. Evaluate the model's selected answer.
5. Record accuracy by predicate class.

### Metrics

- Exact-match accuracy
- Correct-answer logit or sequence log probability
- Accuracy by predicate type
- Accuracy by prompt format

For multi-token answers, calculate complete answer-sequence probabilities or use a validated candidate-scoring method.

### Deliverable

A table reporting baseline performance for every experimental condition.

**Success condition:** The model solves enough examples correctly to support meaningful mechanistic analysis.

If behavioral accuracy is poor, do not interpret weak patching effects as evidence against the filtering mechanism.

---

## 7. Experiment 2 — Locate Filtering Heads

Identify candidate attention heads that participate in ordinary semantic filtering.

### Procedure

1. Select a training/discovery subset from Dataset A.
2. Run each source prompt.
3. Cache query vectors at the answer-position token.
4. Run destination prompts using different predicates and disjoint lists.
5. Patch the source query vectors into destination runs.
6. Measure changes in the target answer's logit.
7. Rank attention heads by their causal influence.

Use the original head-location implementation where possible.

The paper also uses sparse learned masks to identify cooperating heads. If single-head effects are weak, use the existing sparse-mask approach rather than assuming filtering is absent.

### Important technical requirement

Patch the correct query state, at the correct layer and head, before positional encoding as in the original implementation.

Verify that the intervention changes the intended tensor and does not accidentally alter other heads.

### Output

- Ranked filtering-head candidates
- Layer/head heatmap
- Individual-head causal effects
- Joint causal effects of selected heads

**Freeze the identified head set before testing held-out predicates.**

This prevents selecting heads specifically because they perform well on the extension.

---

## 8. Experiment 3 — Replicate Query Patching

This is the central replication experiment.

### Example

Source:
```text
Apple, carrot, banana, truck.
Find the fruit.
Answer:
```

Destination:
```text
Violin, peach, bus, chair.
Find the vehicle.
Answer:
```

Expected unpatched destination: `bus`

Expected patched destination: `peach`

### Procedure

1. Run the source prompt.
2. Cache the filtering-head query vectors at the final answer token.
3. Run the destination prompt normally.
4. Rerun the destination prompt while replacing the selected query vectors with those from the source.
5. Measure the attention redistribution.
6. Measure changes in output logits and predicted answers.

### Metrics

Define the source-matching target and destination-matching target.

Calculate:

**Target logit change**

`logit_patched(source_target) - logit_clean(source_target)`

**Preference shift**

Change in the logit difference between the source-matching and destination-matching answers.

**Patch success rate**

Fraction of examples where patching causes the model to favor the source-matching target.

**Attention shift**

Change in attention mass toward source-predicate-matching items.

Also implement the original paper's causality score for direct comparison.

### Controls

- No patch
- Same-predicate patch
- Random unrelated-head patch
- Random source-query patch
- Query patch from a different predicate
- Original filtering-head patch

Use multiple random seeds and report uncertainty.

### Deliverable

A visualization showing:

- Original attention distribution
- Patched attention distribution
- Clean answer scores
- Patched answer scores

---

## 9. Experiment 4 — Extension: Unseen Predicates

This is our primary extension.

**Question:** Do filtering heads identified using common categories continue to implement filtering for predicates excluded from the discovery dataset?

### Procedure

1. Use the frozen head set from Experiment 2.
2. Construct new source/destination pairs from Dataset B.
3. Perform query patching without identifying new heads.
4. Compare results with ordinary-category patching.
5. Measure changes in attention and answer logits.

### Conditions

- Seen predicate → seen predicate
- Unseen predicate → seen predicate
- Seen predicate → unseen predicate
- Unseen predicate → unseen predicate

Ensure source and destination prompts are behaviorally solvable.

### Key comparison

Does the original head set remain causally effective when the predicate changes?

If yes, this supports broader predicate generality in the tested model.

If no, investigate whether other heads implement the new predicate before concluding that the model lacks a filtering mechanism.

### Deliverable

A predicate-transfer matrix with source predicates on one axis and destination predicates on the other.

---

## 10. Experiment 5 — Extension: Compositional Filtering

**Question:** Can filtering heads represent predicates assembled from multiple conditions?

### Procedure

Use Dataset C.

Compare simple predicates with:

- Conjunction
- Disjunction
- Negation

Example:

```text
Apple, lemon, banana, carrot.
Find the yellow fruit.
Answer:
```

Use examples with a unique correct answer.

Patch a compositional-predicate query into a destination prompt containing a different collection.

### Measurements

- Behavioral accuracy
- Attention mass on correct candidates
- Query-patching causal effect
- Head-set overlap with simple predicates

### Additional exploratory analysis

Compare query representations for:

- Fruit
- Yellow
- Yellow fruit

Measure cosine similarity and test whether the composite query can be approximated by simple combinations of component-query vectors.

Do not assume that query vectors compose linearly.

This analysis is exploratory and should be reported separately from the primary causal results.

---

## 11. Experiment 6 — Lazy vs Eager Filtering

Replicate the original paper's observation that predicate ordering changes how filtering is implemented.

### Procedure

For matched examples:

1. Run the predicate-after-list format.
2. Run the predicate-before-list format.
3. Patch the same filtering heads.
4. Compare causal effects.

### Metrics

- Behavioral accuracy
- Filtering-head causal effect
- Attention selectivity
- Difference between lazy and eager conditions

### Expected result

Filtering-head query patching should have stronger effects when the predicate appears after the list.

A reduction in causal effect does not alone prove the presence of eager flags.

If time permits, reproduce the original flag-swapping intervention to test that alternative mechanism directly.

---

## 12. Statistical Analysis

For every major condition, report:

- Sample size
- Mean effect
- Standard deviation
- Bootstrap 95% confidence interval
- Behavioral accuracy
- Patch success rate

Use paired comparisons when the same underlying examples are evaluated under different interventions.

Do not report results from only a few cherry-picked prompts.

Distinguish:

1. Behavioral correctness
2. Attention selectivity
3. Causal intervention effects

These are different forms of evidence.

### Important failure modes

- Smaller models may not contain comparable filtering heads.
- Tokenization can confound query and answer comparisons.
- Head effects may be distributed.
- Some predicates may be semantically ambiguous.
- Patching may create off-distribution activations.
- A change in attention does not guarantee a change in output.
- Prompt formatting can change the relevant token position.

Document these limitations.

---

## 13. Project Structure

Suggested layout:

```text
filter-generalization/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml
├── data/
│   ├── discovery.jsonl
│   ├── heldout.jsonl
│   └── compositional.jsonl
├── src/
│   ├── model_loader.py
│   ├── dataset_generator.py
│   ├── prompt_templates.py
│   ├── activation_cache.py
│   ├── query_patching.py
│   ├── head_discovery.py
│   ├── metrics.py
│   └── visualization.py
├── experiments/
│   ├── baseline.py
│   ├── locate_heads.py
│   ├── replicate_patching.py
│   ├── unseen_predicates.py
│   ├── compositional.py
│   └── lazy_eager.py
├── tests/
├── results/
│   ├── tables/
│   ├── figures/
│   └── raw/
└── report/
    └── findings.md
```

Adapt this structure to the original repository rather than duplicating existing working modules.

---

## 14. Execution Plan — One-Day Sprint

### Phase 1 — Setup (approximately 1 hour)

- Inspect original repository and paper.
- Identify reusable functions.
- Confirm GPU resources.
- Load a small model.
- Run one successful filtering prompt.

### Phase 2 — Replication (approximately 2–3 hours)

- Generate original-style prompts.
- Establish behavioral accuracy.
- Locate candidate filtering heads.
- Run query-patching interventions.
- Verify that attention and logits change.

### Phase 3 — Primary extension (approximately 2 hours)

- Generate held-out semantic predicates.
- Freeze the discovered heads.
- Run predicate-transfer experiments.
- Compare seen and unseen conditions.

### Phase 4 — Additional extensions (approximately 1–2 hours)

- Run a small compositional-predicate pilot.
- Compare lazy and eager formats.
- Skip expensive analyses if replication remains unstable.

### Phase 5 — Analysis (approximately 1 hour)

- Generate summary tables.
- Plot causal effects.
- Identify strongest and weakest conditions.
- Write preliminary findings.

These are target allocations, not guaranteed runtimes. GPU availability and model compatibility may require reducing scope.

**Priority order:**

1. Working query patching
2. Causal replication
3. Held-out predicate transfer
4. Compositional predicates
5. Lazy/eager analysis

If time is limited, finish the first three rigorously.

---

## 15. Required Figures

Generate at least:

**Figure 1 — Filtering-head localization**

Heatmap of causal effects across layers and heads.

**Figure 2 — Query patching**

Attention distributions before and after patching for one representative prompt pair.

**Figure 3 — Predicate generalization**

Comparison of causal effects for seen versus unseen predicates.

**Figure 4 — Compositional predicates**

Causal effects by predicate type, if completed.

**Figure 5 — Lazy vs eager**

Comparison of filtering-head causal effects across prompt orderings, if completed.

Save all numerical results as CSV or JSON in addition to visualizations.

---

## 16. Research Integrity Requirements

- Never fabricate experimental results.
- Never claim successful replication based only on attention visualization.
- Distinguish original-paper findings from our findings.
- Use held-out predicates for evaluation.
- Keep the discovered head set fixed during the main extension.
- Preserve raw experimental outputs.
- Log failed experiments and their causes.
- Record seeds and model configurations.
- Prefer a smaller complete experiment over a larger incomplete one.
- Avoid changing the hypothesis after observing results without documenting the change.

The original paper already investigates portability across different formats, languages, and tasks. Our contribution must be framed as a more systematic investigation of held-out predicate classes and compositional generalization, not as the first demonstration of predicate portability.

---

## 17. Final Deliverables

Produce:

1. A working reproducible codebase.
2. A behavioral baseline dataset and results.
3. A list of candidate filtering heads.
4. A successful or clearly documented unsuccessful query-patching replication.
5. A held-out predicate generalization experiment.
6. At least three useful figures.
7. A concise research report.

The report should answer:

- Did we identify filtering heads?
- Did query patching causally redirect the model?
- Did the same heads generalize to unseen predicates?
- Did compositional predicates behave differently?
- What evidence supports or contradicts the hypothesis?
- What experiment should come next?

---

## 18. Instructions for Claude Code

You are acting as a research engineer conducting a mechanistic interpretability experiment.

**Before writing code:**

1. Inspect the original repository.
2. Read the relevant sections of the original paper.
3. Identify the exact implementation of query patching.
4. Check available compute.
5. Present a concise implementation plan.

**While implementing:**

- Work incrementally.
- Validate tensor shapes.
- Test interventions on one example before batching.
- Avoid unnecessary abstractions.
- Reuse existing research code when appropriate.
- Keep experiment configurations explicit.
- Save intermediate outputs.
- Use deterministic seeds wherever possible.

**When running experiments:**

- Run a small smoke test first.
- Confirm that query patching actually modifies the intended attention head.
- Verify that source and destination examples are valid.
- Record clean and patched logits.
- Report failures immediately.
- Do not silently substitute another experiment.

**Most important instruction:**

The goal is not merely to make the model answer filtering questions. The goal is to establish whether the **same identifiable attention-head mechanism causally implements filtering across different predicates**.

Always distinguish model behavior from mechanistic evidence.
