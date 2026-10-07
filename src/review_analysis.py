"""Additional analyses requested in peer review: per-generation-strategy breakdown,
problem-clustered bootstrap CIs, and review-workload quantification.
Reads data/results/verifier_predictions.csv only (no API calls)."""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
d = pd.read_csv(RES / "verifier_predictions.csv")
d["_e"] = d["error"].fillna("").astype(str).str.strip().ne("")
d = d.sort_values("_e").drop_duplicates(["ID","generation_strategy","verifier_variant","include_context"], keep="first")
d["pred"] = d["pred_Misleading"] > 0
d["gt"] = d["gt_Misleading"] > 0
CATS = ["Insightful", "Helpful", "Uncertain", "Misleading"]

def prf(g, cat="Misleading"):
    p, t = g[f"pred_{cat}"] > 0, g[f"gt_{cat}"] > 0
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(n=len(g), gt_pos=int(t.sum()), tp=int(tp), fp=int(fp), fn=int(fn),
                precision=round(pr, 3), recall=round(rc, 3), f1=round(f1, 3))

def macro(g):
    return np.mean([prf(g, c)["f1"] for c in CATS])

out = {"by_generation_strategy": {}, "bootstrap": {}, "workload": {}}
conds = d.groupby(["verifier_variant", "include_context"])
for (v, c), g in conds:
    key = f"{v}|context={c}"
    out["by_generation_strategy"][key] = {s: prf(h) for s, h in g.groupby("generation_strategy")}

# clustered bootstrap (resample the 100 problems, keep their 3 DRs together)
rng = np.random.default_rng(0)
for (v, c), g in conds:
    key = f"{v}|context={c}"
    ids = g["ID"].unique()
    by = {i: h for i, h in g.groupby("ID")}
    mf, ma = [], []
    for _ in range(2000):
        s = pd.concat([by[i] for i in rng.choice(ids, len(ids))])
        mf.append(prf(s)["f1"]); ma.append(macro(s))
    out["bootstrap"][key] = {
        "misleading_f1": [round(float(np.mean(mf)), 3), *map(lambda x: round(float(x), 3), np.percentile(mf, [2.5, 97.5]))],
        "macro_f1": [round(float(np.mean(ma)), 3), *map(lambda x: round(float(x), 3), np.percentile(ma, [2.5, 97.5]))]}

# paired bootstrap: context effect (cot with minus without) on Misleading F1 / macro F1
a = d[(d.verifier_variant == "cot") & d.include_context]
b = d[(d.verifier_variant == "cot") & ~d.include_context]
ids = a["ID"].unique(); A = {i: h for i, h in a.groupby("ID")}; B = {i: h for i, h in b.groupby("ID")}
df, dm = [], []
for _ in range(2000):
    s = rng.choice(ids, len(ids))
    sa, sb = pd.concat([A[i] for i in s]), pd.concat([B[i] for i in s])
    df.append(prf(sa)["f1"] - prf(sb)["f1"]); dm.append(macro(sa) - macro(sb))
out["bootstrap"]["context_effect_cot"] = {
    "delta_misleading_f1_ci95": [round(float(x), 3) for x in np.percentile(df, [2.5, 97.5])],
    "delta_macro_f1_ci95": [round(float(x), 3) for x in np.percentile(dm, [2.5, 97.5])]}

# workload per condition
for (v, c), g in conds:
    r = prf(g); flags = r["tp"] + r["fp"]
    out["workload"][f"{v}|context={c}"] = dict(
        flagged_DRs=flags, of_n=r["n"], flag_rate=round(flags / r["n"], 3),
        true_misleading_DRs=r["gt_pos"], caught=r["tp"],
        DRs_reviewed_per_true_catch=round(flags / r["tp"], 1) if r["tp"] else None)
(RES / "review_analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
