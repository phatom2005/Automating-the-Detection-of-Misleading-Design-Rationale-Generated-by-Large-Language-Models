"""
make_figures.py

Generates the figures for the paper from data/results/verifier_predictions.csv
and data/results/metrics_summary.json (both produced by run_experiment.py +
evaluate.py -- run those first). Pure plotting, no API calls.

Per the project's figure budget (max ~4-5 figures for a 12-15 page LNCS paper):
  1. confusion_matrix_misleading.pdf  - 2x2 confusion matrix heatmap (vector
     PDF, print-ready at final size) for the primary condition
     (verifier_variant=cot, include_context=True), Misleading-presence:
     predicted vs ground truth.
  2. f1_by_category.png              - bar chart of P/R/F1 per IHUM category
     for the primary condition.
  3. ablation_prompt_variant.png     - bar chart comparing Misleading F1 across
     verifier prompt variants (zero_shot / cot / few_shot), primary context=True.
  4. ablation_context.png            - bar chart comparing Misleading F1
     with vs without original context (primary variant=cot).

Every number plotted here is read directly from metrics_summary.json (itself
computed deterministically from verifier_predictions.csv), so figures always
match the tables in the paper -- no hand-typed numbers, no separate re-computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIG_DIR = RESULTS_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "metrics_summary.json"

CATEGORIES = ["Insightful", "Helpful", "Uncertain", "Misleading"]
PRIMARY_KEY = "variant=cot|context=True"


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} not found. Run evaluate.py first.")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def fig_confusion_matrix_misleading(summary: dict) -> None:
    m = summary["ablation_grid"][PRIMARY_KEY]["per_category"]["Misleading"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

    # Print-ready sizing for llncs: 3.6in = 0.75 * 4.82in (\textwidth), so the
    # figure is drawn at its final printed size and LaTeX never rescales it.
    with matplotlib.rc_context({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Misleading", "Misleading"])
        ax.set_yticklabels(["Not Misleading", "Misleading"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth (human experts)")
        # No ax.set_title(): the LaTeX \caption already describes the figure.
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                text_color = "white" if value > thresh else "black"
                ax.text(j, i, str(value), ha="center", va="center",
                         color=text_color, fontsize=9, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        # Constrain everything inside the nominal figsize canvas *before* the
        # tight-bbox save, otherwise long tick/axis labels can poke outside
        # the canvas and bbox_inches="tight" will save a bbox wider than the
        # 3.6in target -- which \includegraphics[width=0.75\textwidth] would
        # then rescale, shrinking the 8pt font below the ~7pt print threshold.
        fig.tight_layout(pad=0.15)
        fig.savefig(FIG_DIR / "confusion_matrix_misleading.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def fig_f1_by_category(summary: dict) -> None:
    per_cat = summary["ablation_grid"][PRIMARY_KEY]["per_category"]
    precisions = [per_cat[c]["precision"] for c in CATEGORIES]
    recalls = [per_cat[c]["recall"] for c in CATEGORIES]
    f1s = [per_cat[c]["f1"] for c in CATEGORIES]

    x = np.arange(len(CATEGORIES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Verifier Agent performance per IHUM category\n(variant=cot, with context)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f1_by_category.png", dpi=200)
    plt.close(fig)


def _extract_ablation_series(summary: dict, group_by: str) -> dict[str, float]:
    """group_by: 'variant' or 'context' -- returns {condition_label: misleading_f1}"""
    out = {}
    for key, val in summary["ablation_grid"].items():
        # key format: "variant=<v>|context=<c>"
        variant_part, context_part = key.split("|")
        variant = variant_part.split("=")[1]
        context = context_part.split("=")[1]
        f1 = val["per_category"]["Misleading"]["f1"]
        if group_by == "variant" and context == "True":
            out[variant] = f1
        elif group_by == "context" and variant == "cot":
            out[context] = f1
    return out


def fig_ablation_prompt_variant(summary: dict) -> None:
    data = _extract_ablation_series(summary, "variant")
    if not data:
        print("Skipping ablation_prompt_variant.png -- no data for multiple verifier variants yet.")
        return
    order = [v for v in ["zero_shot", "cot", "few_shot"] if v in data]
    values = [data[v] for v in order]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(order, values, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Misleading F1")
    ax.set_title("Ablation: Verifier prompt strategy\n(with original context)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_prompt_variant.png", dpi=200)
    plt.close(fig)


def fig_ablation_context(summary: dict) -> None:
    data = _extract_ablation_series(summary, "context")
    if len(data) < 2:
        print("Skipping ablation_context.png -- need both context=True and context=False runs.")
        return
    labels = ["With context" if k == "True" else "No context" for k in data]
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Misleading F1")
    ax.set_title("Ablation: does original context matter?\n(verifier variant = cot)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_context.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    if PRIMARY_KEY not in summary["ablation_grid"]:
        raise KeyError(
            f"Primary condition '{PRIMARY_KEY}' not found in metrics_summary.json. "
            "Make sure run_experiment.py was run with --variants including 'cot' and context=True."
        )

    fig_confusion_matrix_misleading(summary)
    fig_f1_by_category(summary)
    fig_ablation_prompt_variant(summary)
    fig_ablation_context(summary)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
