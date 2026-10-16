"""Figures from saved tables (CPU only). Each figure's numbers are in the CSV
named next to it, which serves as the table view.

  fig1_localization.png   frozen head mask, layers x heads   (results/heads/frozen_discovery.json)
  fig3_heldout.png        causality per RQ2 cell, filter vs random  (tables/heldout_cells.csv)
  fig3b_transfer.png      per-category transfer matrix        (tables/heldout_transfer_matrix.csv)
  fig4_compositional.png  what the patched model picks         (tables/compositional_patching.csv)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from filtergen.results import RESULTS, read_json  # noqa: E402

FIG = RESULTS / "figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FILTER, CONTROL = "#2a78d6", "#b8b7b0"  # series 1 for filter heads; neutral gray for the random baseline
ROLE_COLORS = {  # fixed categorical order (validated palette, slots 1-5)
    "target": "#2a78d6", "lure_a": "#eb6834", "lure_b": "#1baf7a", "dest_answer": "#eda100", "other": "#e87ba4",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def fig1():
    f = RESULTS / "heads" / "frozen_discovery.json"
    if not f.exists():
        return
    frozen = read_json(f)
    meta = frozen["meta"]
    mask = np.zeros((meta["n_layer"], meta["n_heads"]))
    for l, h in frozen["heads_all_layers"]:
        mask[l, h] = 0.4  # dropped by the layer cutoff
    for l, h in frozen["heads"]:
        mask[l, h] = 1.0
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.imshow(mask.T, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("m", [SURFACE, FILTER]),
              aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(frozen["layer_cutoff"] - 0.5, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="layer", ylabel="head",
           title=f"Filter heads in Llama-3.1-8B: {len(frozen['heads'])} kept (dark), "
                 f"{len(frozen['heads_all_layers']) - len(frozen['heads'])} past layer cutoff (light)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_localization.png", dpi=200)


def fig3():
    f = RESULTS / "tables" / "heldout_cells.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t["label"] = t["pool"].str.upper().replace({"ALL": ""}) + " " + t["cell"]
    labels = list(dict.fromkeys(t["label"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 3.8))
    for i, (cond, color, name) in enumerate([("filter", FILTER, "frozen filter heads"),
                                              ("random", CONTROL, "matched random heads")]):
        d = t[t.condition == cond].set_index("label").reindex(labels)
        ax.bar(x + (i - 0.5) * 0.36, d.causal, width=0.34, color=color, label=name,
               yerr=[d.causal - d.causal_lo, d.causal_hi - d.causal], ecolor=INK2, capsize=2,
               error_kw={"lw": 0.8})
        if cond == "filter":
            for xi, v, hi in zip(x, d.causal, d.causal_hi):
                ax.text(xi - 0.18, hi + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylim=(0, 1.1), ylabel="causality (95% CI)", title="RQ2: query patching with held-out predicates")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_heldout.png", dpi=200)

    m = RESULTS / "tables" / "heldout_transfer_matrix.csv"
    if m.exists():
        mat = pd.read_csv(m).set_index("src_category")
        fig, ax = plt.subplots(figsize=(0.45 * mat.shape[1] + 3, 0.4 * mat.shape[0] + 2))
        im = ax.imshow(mat.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "seq", ["#cde2fb", "#2a78d6", "#0d366b"]), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
        ax.set(xlabel="destination predicate", ylabel="source predicate", title="Causality by category pair")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="causality")
        fig.tight_layout()
        fig.savefig(FIG / "fig3b_transfer.png", dpi=200)


def fig4():
    f = RESULTS / "tables" / "compositional_patching.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t = t[t.head_set == "filter"].copy()
    for role in ROLE_COLORS:
        if role not in t:
            t[role] = 0.0
    t["other"] = 1 - t[["target", "lure_a", "lure_b", "dest_answer"]].sum(axis=1)
    t["label"] = t.predicate.str.replace("_", " ") + "\n" + t.direction.map(
        {"comp_to_simple": "comp → simple", "simple_to_comp": "simple → comp"})
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(t)), 4.2))
    bottom = np.zeros(len(t))
    for role, color in ROLE_COLORS.items():
        ax.bar(t.label, t[role], bottom=bottom, color=color, label=role, width=0.6,
               edgecolor=SURFACE, linewidth=2)
        bottom += t[role].to_numpy()
    ax.set(ylim=(0, 1), ylabel="share of patched predictions",
           title="RQ3: what the model picks after patching compositional queries")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_compositional.png", dpi=200)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for make in (fig1, fig3, fig4):
        make()
    print(f"figures in {FIG}")
