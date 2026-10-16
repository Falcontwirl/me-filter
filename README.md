# Generalization of filter heads (Llama-3.1-8B)

This project extends *LLMs Process Lists With General Filter Heads* (Sharma et al., ICLR 2026). The question: do the filter heads located on ordinary categories also implement held-out predicates (RQ2) and compositional ones (RQ3)? And does each class of predicate use the same heads (RQ4)?

`CLAUDE.md` holds the research spec and scope decisions. The authors' code is not redistributed here. `third_party/fetch_upstream.sh` clones it into `third_party/filter/` at a pinned commit (see `third_party/PATCHES.md`). All model loading, pair sampling, mask training and query patching go through it. Our own code lives in `filtergen/` and `experiments/`.

## Layout

| Path | Contents |
|---|---|
| `filtergen/splits.py` | Discovery and held-out category split. Generates `data/splits/*.json`. |
| `filtergen/pairs.py` | Counterfactual pairs, built with upstream's sampler. |
| `filtergen/evaluate.py` | Upstream query patching run with the authors' 8B evaluation settings, plus our attention metric. |
| `filtergen/compositional.py` | Lists and pairs with lures for conjunction, disjunction and negation. |
| `data/b2_categories.json` | Three new held-out categories (B2). |
| `data/compositional.json` | Compositional predicates, match items and lure items. |
| `experiments/00`–`07` | One script per experiment. Each has a matching `slurm/*.sbatch`. |

## Oscar setup (once)

```bash
git clone <this repo> ~/me-filter && cd ~/me-filter
# slurm/config.sh is set for Oscar (anaconda3/2023.09-0-aqbc); edit it if paths differ
bash slurm/setup_env.sh
huggingface-cli login     # account needs access to meta-llama/Llama-3.1-8B-Instruct
```

## Run order

Submit every job from the repo root. Each step reads the previous step's outputs under `results/`.

```bash
sbatch slurm/smoke.sbatch                                    # 00: isolation + consistency checks, timings
sbatch slurm/baseline.sbatch                                 # 01: per-category accuracy -> results/baseline_pass.json

# 02: localize on discovery categories (lambda sweep shares one dataset)
sbatch slurm/locate_build.sbatch discovery_s0 seen 0
sbatch slurm/locate_train.sbatch discovery_s0                # array: lambda in {0.005, 0.01, 0.02, 0.05}
python experiments/02b_freeze.py --run discovery_s0          # CPU; writes results/heads/frozen_discovery.json
sbatch slurm/locate_build.sbatch discovery_s1 seen 1         # stability check on independent data
sbatch --array=0 slurm/locate_train.sbatch discovery_s1 <chosen lambda>
python experiments/02b_freeze.py --stability_run discovery_s1

sbatch slurm/gate.sbatch                                     # 03: replication gate -> results/gate.json
# --- RQ2-RQ4 refuse to run unless the gate passed ---
sbatch slurm/heldout.sbatch                                  # 04: RQ2
sbatch slurm/compositional.sbatch                            # 05: RQ3

# 06: RQ4. Localize per class at the frozen lambda, then compare head sets.
#     Pass only the B2 categories / predicates that passed their behavioral baselines.
sbatch slurm/locate_build.sbatch b1_s0 b1 0
sbatch slurm/locate_build.sbatch b2_s0 b2 0 --categories <passing B2 categories>
sbatch slurm/locate_build.sbatch comp_s0 compositional 0 --categories <kept predicate ids>
sbatch --array=0 slurm/locate_train.sbatch b1_s0 <chosen lambda>   # same for b2_s0, comp_s0
sbatch slurm/overlap.sbatch

python experiments/07_figures.py                             # CPU; results/figures/
```

## Decisions fixed before seeing results

- **Split** (`data/splits/split.json`; every result records its hash). Confusable category groups stay on one side of the split.
- **Head selection.** Use the best validation causality for the layer-cutoff head set (cutoff at layer 23, as in the authors' 8B notebook). Among lambdas within 0.02 of that best, take the one with the fewest heads.
- **Gate.** Filter-head causality must be ≥ 0.50. Its 95% CI must sit above the matched random-head control, and the control must be ≤ 0.10. The paper reports 0.826 for Llama-8B.
- **Behavioral inclusion.** Categories need ≥ 0.90 accuracy and compositional predicates ≥ 0.80. Pairs are only used when the model is correct on the source, destination and gold prompts, which mirrors upstream.

## Local checks (no GPU)

```bash
bash third_party/fetch_upstream.sh
python -m venv .venv-local && .venv-local/bin/pip install pytest pandas numpy pyyaml matplotlib
.venv-local/bin/python -m pytest -q
```
