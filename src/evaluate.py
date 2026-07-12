"""
evaluate.py

Pure data-processing script (NO API calls) that turns
data/results/verifier_predictions.csv into the metrics needed for the paper:

  RQ1 - Can the Verifier Agent detect Misleading DR, and how accurately
        compared to the human-expert ground truth?
  RQ2 - Does adding the Verifier reduce the rate of Misleading DR reaching
        the user, relative to the no-verifier baseline?

Because the public ground truth is a per-DR ARGUMENT COUNT per category
(not a sentence-level label), the primary metric operationalizes each
category as a per-DR binary "is this category present at all in this DR"
label (gt_X > 0 vs pred_X > 0), and computes Precision/Recall/F1/Cohen's
Kappa for that binary presence, per category, per (verifier_variant,
include_context) condition. This directly answers RQ1 at the same
granularity the Verifier makes its predictions.

A secondary count-level view (MAE between predicted and ground-truth
argument counts per category) is also reported, plus confound breakdowns
by architecture-decision type and by generated-DR length quartile
(per the pre-registered confound-analysis plan in the project notes).

Outputs:
  - data/results/metrics_summary.json   (everything below, machine-readable)
  - printed tables to stdout            (for quick inspection)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import build_dataset  # noqa: E402 - reuse dataset loader for length/type confounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PRED_PATH = RESULTS_DIR / "verifier_predictions.csv"
SUMMARY_PATH = RESULTS_DIR / "metrics_summary.json"

CATEGORIES = ["Insightful", "Helpful", "Uncertain", "Misleading"]


KEY_COLUMNS = ["ID", "generation_strategy", "verifier_variant", "include_context"]


def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"{PRED_PATH} not found. Run run_experiment.py first (locally, not in the sandbox)."
        )
    df = pd.read_csv(PRED_PATH)
    # normalize include_context which may load as "True"/"False" strings
    if df["include_context"].dtype == object:
        df["include_context"] = df["include_context"].map({"True": True, "False": False, True: True, False: False})

    # run_experiment.py's resume feature retries rows that previously errored out
    # (e.g. daily-quota 429s) by appending a fresh row for the same
    # (ID, strategy, variant, context) key, WITHOUT removing the old errored row.
    # Dedupe here: for each key, prefer a successful row (empty error) over an
    # errored one; if duplicates remain, keep the last one written (most recent).
    df["_error_flag"] = df["error"].fillna("").astype(str).str.strip().ne("")
    n_before = len(df)
    df = df.sort_values("_error_flag").drop_duplicates(subset=KEY_COLUMNS, keep="first")
    n_after = len(df)
    if n_before != n_after:
        print(f"Deduped {n_before - n_after} duplicate/retried rows in {PRED_PATH.name} "
              f"({n_before} -> {n_after} rows).")
    n_still_error = int(df["_error_flag"].sum())
    if n_still_error:
        print(f"WARNING: {n_still_error} rows still have unresolved errors after dedup "
              f"(no successful retry found for that key). These will still be counted as "
              f"pred_X=0 for all categories -- inspect data/results/api_call_log.jsonl.")
    df = df.drop(columns=["_error_flag"])
    return df


def binary_presence_metrics(df: pd.DataFrame, category: str) -> dict:
    gt = (df[f"gt_{category}"] > 0).astype(int)
    pred = (df[f"pred_{category}"] > 0).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0
    )
    kappa = cohen_kappa_score(gt, pred) if gt.nunique() > 1 or pred.nunique() > 1 else float("nan")
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    return {
        "n": int(len(df)),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "cohen_kappa": round(float(kappa), 3) if not np.isnan(kappa) else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_positive_rate": round(float(gt.mean()), 3),
        "pred_positive_rate": round(float(pred.mean()), 3),
    }


def count_level_mae(df: pd.DataFrame, category: str) -> float:
    return round(float((df[f"pred_{category}"] - df[f"gt_{category}"]).abs().mean()), 3)


def macro_f1(per_category_metrics: dict) -> float:
    f1s = [per_category_metrics[c]["f1"] for c in CATEGORIES]
    return round(float(np.mean(f1s)), 3)


def per_condition_report(df: pd.DataFrame) -> dict:
    """Metrics broken down by (verifier_variant, include_context) -- the ablation grid."""
    report = {}
    for (variant, ctx), group in df.groupby(["verifier_variant", "include_context"]):
        key = f"variant={variant}|context={ctx}"
        per_cat = {cat: binary_presence_metrics(group, cat) for cat in CATEGORIES}
        report[key] = {
            "per_category": per_cat,
            "macro_f1": macro_f1(per_cat),
            "count_mae": {cat: count_level_mae(group, cat) for cat in CATEGORIES},
        }
    return report


def confound_report(df: pd.DataFrame) -> dict:
    """Breaks Misleading-detection performance down by decision type and DR length quartile.

    Requires re-joining the generated_dr text and Type of Architecture Decision
    from the raw dataset (not stored in verifier_predictions.csv to keep it lean).
    """
    strategies = df["generation_strategy"].unique().tolist()
    context_df = build_dataset(strategies, limit=None)[
        ["ID", "generation_strategy", "generated_dr", "Type of Architecture Decision"]
    ]
    merged = df.merge(context_df, on=["ID", "generation_strategy"], how="left")
    merged["dr_length_words"] = merged["generated_dr"].astype(str).str.split().str.len()

    # length-quartile confound (Misleading category, primary condition: cot + context)
    primary = merged[(merged["verifier_variant"] == "cot") & (merged["include_context"] == True)]
    length_report = {}
    if len(primary) > 0:
        primary = primary.copy()
        primary["length_quartile"] = pd.qcut(primary["dr_length_words"], 4, labels=["Q1_shortest", "Q2", "Q3", "Q4_longest"], duplicates="drop")
        for q, group in primary.groupby("length_quartile", observed=True):
            length_report[str(q)] = binary_presence_metrics(group, "Misleading")

    # decision-type confound
    type_report = {}
    if len(primary) > 0:
        for t, group in primary.groupby("Type of Architecture Decision"):
            if len(group) >= 5:  # skip tiny groups, not statistically meaningful
                type_report[str(t)] = binary_presence_metrics(group, "Misleading")

    return {"by_dr_length_quartile": length_report, "by_decision_type": type_report}


def main():
    df = load_predictions()
    print(f"Loaded {len(df)} verifier predictions from {PRED_PATH}")

    ablation = per_condition_report(df)
    confounds = confound_report(df)

    summary = {
        "n_total_predictions": len(df),
        "ablation_grid": ablation,
        "confound_analysis": confounds,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote full summary -> {SUMMARY_PATH}\n")

    print("=== Ablation grid (per verifier_variant x context condition) ===")
    for key, val in ablation.items():
        print(f"\n-- {key} --  (macro F1 = {val['macro_f1']})")
        for cat in CATEGORIES:
            m = val["per_category"][cat]
            print(
                f"  {cat:12s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"kappa={m['cohen_kappa']} (gt_rate={m['gt_positive_rate']:.3f}, pred_rate={m['pred_positive_rate']:.3f}) "
                f"count_MAE={val['count_mae'][cat]}"
            )

    print("\n=== Confound: Misleading-detection by DR length quartile (cot + context condition) ===")
    for q, m in confounds["by_dr_length_quartile"].items():
        print(f"  {q:12s} n={m['n']:3d} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    print("\n=== Confound: Misleading-detection by architecture decision type (cot + context condition) ===")
    for t, m in confounds["by_decision_type"].items():
        print(f"  {t:30s} n={m['n']:3d} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
