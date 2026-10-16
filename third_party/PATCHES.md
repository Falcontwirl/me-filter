# Upstream code

The authors' code (https://github.com/arnab-api/filter) is **not** included in this repo, because upstream has no license.
`third_party/fetch_upstream.sh` clones it into `third_party/filter/` (gitignored) at commit
`d4a3010b391eda4b60f824e37c63335bb8940b7f` (2025-11-27). `slurm/setup_env.sh` runs that script.

## What we add to the checkout

| File | Change | Why |
|---|---|---|
| `env.yml` (copied from `third_party/env.yml`) | `DATA_DIR: data_save`, `RESULTS_DIR: ../../results/upstream` | Upstream's demo config points at `data/`, but the data ships in `data_save/`. |

No upstream source files are modified. Workarounds live in `filtergen/` instead:

- **Model weights are frozen** (`filtergen.upstream.load_model`). `get_optimal_head_mask_prev` never freezes the model, so `loss.backward()` would allocate gradients for all 8B weights (+16 GB).
- **Mask training calls `get_optimal_head_mask_prev` directly** (`experiments/02_locate.py`). `scripts/locate_selection_heads.find_optimal_masks` reads `train_set` from module globals, and `opt_interface="updated"` crashes with a TypeError.
- **Evaluation uses the settings from the authors' notebooks** (`notebooks/checking_head_generalization/select_one.ipynb`), not those in the locator's `validate()`. See `filtergen/evaluate.py`.
