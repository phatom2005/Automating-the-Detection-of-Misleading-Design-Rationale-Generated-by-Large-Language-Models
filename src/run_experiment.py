"""
run_experiment.py

Runs the Verifier Agent over the Zhou et al. (LLM4DR) benchmark and writes
predictions to data/results/verifier_predictions.csv, plus the full raw
argument-level JSON to data/results/verifier_raw.jsonl.

Scope (see project decisions):
  - Generation model: gpt-4-0613 (best-quality DR generator in the dataset)
  - Generation strategies: zero-shot, CoT, AI-Agent (all 3 -> 100 problems x 3 = 300 DRs)
  - Verifier prompt variants (ablation): zero_shot, cot, few_shot
  - Context ablation: with_context=True/False

By default this script runs the FULL ablation grid (3 generation strategies
x 3 verifier prompt variants x 2 context conditions x 100 problems = 1800
Verifier calls). That is a lot of free-tier API calls -- use --variants /
--strategies / --no-context-ablation to shrink the run for a quick pass,
then expand once the pipeline is confirmed working.

Usage examples:
    # Quick smoke test: 10 problems, 1 strategy, 1 verifier variant, with context only
    python src/run_experiment.py --limit 10 --strategies zero-shot --variants cot

    # Main experiment for the paper (primary condition only, cheaper):
    python src/run_experiment.py --strategies zero-shot,CoT,AI-Agent --variants cot

    # Full ablation grid (run once main results look right):
    python src/run_experiment.py --full-ablation

NOTE: this script makes real network calls to the Gemini API. It will NOT
work inside the Cowork sandbox (network allowlist blocks
generativelanguage.googleapis.com) -- run it locally / via Antigravity on a
machine with normal internet access. See HANDOFF.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_agent import aggregate_counts, call_verifier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

STRATEGY_FILES = {
    "zero-shot": RAW_DIR / "Results" / "zero-shot.xlsx",
    "CoT": RAW_DIR / "Results" / "CoT.xlsx",
    "AI-Agent": RAW_DIR / "Results" / "AI-Agent.xlsx",
}
GENERATION_MODEL_SHEET = "gpt-4-0613"  # matches STRATEGY_FILES sheet names (some have trailing spaces on other models, gpt-4-0613 is clean)

GT_COLUMNS = ["Insightful", "Helpful", "Uncertain", "Misleading"]


def load_context() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "Collected_Data.xlsx", sheet_name="Selected Data")
    return df[["ID", "Architecture Problem", "Architecture Decision", "Type of Architecture Decision"]]


def load_strategy_rows(strategy: str) -> pd.DataFrame:
    path = STRATEGY_FILES[strategy]
    df = pd.read_excel(path, sheet_name=GENERATION_MODEL_SHEET)
    # the generated-DR column name varies slightly by file, find it dynamically
    dr_col = [c for c in df.columns if c.strip().startswith("Design Rationale") and "gpt-4" in c][0]
    out = df[["ID", dr_col] + GT_COLUMNS].copy()
    out = out.rename(columns={dr_col: "generated_dr"})
    out["generation_strategy"] = strategy
    return out


def build_dataset(strategies: list[str], limit: int | None) -> pd.DataFrame:
    context = load_context()
    frames = []
    for strat in strategies:
        rows = load_strategy_rows(strat)
        merged = rows.merge(context, on="ID", how="left")
        frames.append(merged)
    full = pd.concat(frames, ignore_index=True)
    if limit:
        # take `limit` problems per strategy, not `limit` total rows
        full = full.groupby("generation_strategy", group_keys=False).head(limit)
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", type=str, default="zero-shot,CoT,AI-Agent")
    parser.add_argument("--variants", type=str, default="zero_shot,cot,few_shot")
    parser.add_argument("--full-ablation", action="store_true", help="run with_context and no_context both")
    parser.add_argument("--no-context-ablation", action="store_true", help="skip the no-context condition entirely (faster)")
    parser.add_argument("--limit", type=int, default=None, help="limit number of problems per strategy (for smoke testing)")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "verifier_predictions.csv"))
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    context_conditions = [True, False] if (args.full_ablation and not args.no_context_ablation) else [True]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = RESULTS_DIR / "verifier_raw.jsonl"
    output_path = Path(args.output)

    dataset = build_dataset(strategies, args.limit)
    total_calls = len(dataset) * len(variants) * len(context_conditions)
    print(f"Dataset rows: {len(dataset)} | variants: {variants} | context_conditions: {context_conditions}")
    print(f"Total Verifier calls to make: {total_calls}")

    fieldnames = [
        "ID",
        "generation_strategy",
        "verifier_variant",
        "include_context",
        "pred_Insightful",
        "pred_Helpful",
        "pred_Uncertain",
        "pred_Misleading",
        "gt_Insightful",
        "gt_Helpful",
        "gt_Uncertain",
        "gt_Misleading",
        "num_arguments_predicted",
        "error",
    ]

    import time

    # Load already processed keys to allow resuming without duplicate calls
    processed_keys = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            skipped_error_rows = 0
            for _, r in existing_df.iterrows():
                # Rows that previously errored out (e.g. 429 RESOURCE_EXHAUSTED from a
                # daily-quota cutoff) must NOT be treated as "done" -- otherwise resuming
                # after the quota resets silently bakes empty/garbage predictions into the
                # final dataset for that (ID, strategy, variant, context) combo forever.
                err = r.get("error", "")
                if pd.notna(err) and str(err).strip() != "":
                    skipped_error_rows += 1
                    continue
                # Normalizing include_context to bool
                ctx = r["include_context"]
                if isinstance(ctx, str):
                    ctx = ctx.lower() in ("true", "1")
                processed_keys.add((str(r["ID"]), str(r["generation_strategy"]), str(r["verifier_variant"]), bool(ctx)))
            print(f"Loaded {len(processed_keys)} successfully processed records. These will be skipped.")
            if skipped_error_rows:
                print(f"Found {skipped_error_rows} previously ERRORED rows -- these WILL be retried.")
        except Exception as e:
            print(f"Warning: could not read existing predictions to resume: {e}")

    write_header = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as csv_f, open(
        raw_jsonl_path, "a", encoding="utf-8"
    ) as raw_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="problems"):
            for variant in variants:
                for include_context in context_conditions:
                    key = (str(row["ID"]), str(row["generation_strategy"]), variant, include_context)
                    if key in processed_keys:
                        continue

                    request_id = f"{row['ID']}__{row['generation_strategy']}__{variant}__ctx{include_context}"
                    
                    # Sleep to respect 15 RPM limit on free tier, preventing 429 backoff delays
                    time.sleep(4.2)
                    
                    result = call_verifier(
                        architecture_problem=str(row["Architecture Problem"]),
                        architecture_decision=str(row["Architecture Decision"]),
                        generated_dr=str(row["generated_dr"]),
                        prompt_variant=variant,
                        include_context=include_context,
                        request_id=request_id,
                    )
                    pred_counts = aggregate_counts(result.arguments)

                    writer.writerow(
                        {
                            "ID": row["ID"],
                            "generation_strategy": row["generation_strategy"],
                            "verifier_variant": variant,
                            "include_context": include_context,
                            "pred_Insightful": pred_counts["Insightful"],
                            "pred_Helpful": pred_counts["Helpful"],
                            "pred_Uncertain": pred_counts["Uncertain"],
                            "pred_Misleading": pred_counts["Misleading"],
                            "gt_Insightful": row["Insightful"],
                            "gt_Helpful": row["Helpful"],
                            "gt_Uncertain": row["Uncertain"],
                            "gt_Misleading": row["Misleading"],
                            "num_arguments_predicted": len(result.arguments),
                            "error": result.error or "",
                        }
                    )
                    csv_f.flush()

                    raw_f.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "ID": row["ID"],
                                "generation_strategy": row["generation_strategy"],
                                "verifier_variant": variant,
                                "include_context": include_context,
                                "arguments": result.arguments,
                                "error": result.error,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    raw_f.flush()

    print(f"Done. Predictions -> {output_path}")
    print(f"Raw argument-level output -> {raw_jsonl_path}")
    print(f"API call log -> {RESULTS_DIR / 'api_call_log.jsonl'}")


if __name__ == "__main__":
    main()
