# evaluate_metrics_smoke.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

OUT_ROOT = Path("outputs/privatized")
OUT_SUMMARY = Path("outputs/metrics_summary.csv")

# Which metric per task
TASK_METRIC = {
    "cola": "mcc",
    "sst2": "acc",
    "mrpc": "f1_acc",
    "rte": "acc",
    "yelp10": "acc",
    "trustpilot": "acc",
    "IMDB_reviews": "acc",
    "mini_yelp": "acc",
    "name_redacted_bios": "acc",
}

EPS = [10, 50, 250]
SPLITS = ["validation", "test"]

# Possible label columns per dataset family
LABEL_CANDS_COMMON = ["label", "labels", "y", "target"]
LABEL_CANDS = {
    "default": LABEL_CANDS_COMMON,
    "yelp10": ["author", "author_id", "user", "user_id"] + LABEL_CANDS_COMMON,
    "trustpilot": ["gender", "sex", "stars", "rating", "score"] + LABEL_CANDS_COMMON,
    "IMDB_reviews": ["sentiment", "label", "rating", "score", "stars"],
    "mini_yelp": ["sentiment", "label", "rating", "stars", "author", "author_id"],
    "name_redacted_bios": LABEL_CANDS_COMMON,  # may be missing; script will skip
}

# Possible prediction columns if you already have them
PRED_CANDS = ["pred", "prediction", "y_pred", "pred_label"]

def safe_read(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[skip] Cannot read {p}: {e}")
        return None

def _lower_map(cols):
    return {c.lower(): c for c in cols}

def find_label_column(task: str, cols: list[str]) -> str | None:
    lc = _lower_map(cols)
    cands = LABEL_CANDS.get(task, LABEL_CANDS["default"])
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    return None

def find_pred_column(cols: list[str]) -> str | None:
    lc = _lower_map(cols)
    for c in PRED_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    return None

def coerce_series(s: pd.Series) -> pd.Series:
    # Keep as string labels for consistency across tasks
    # (MCC supports multiclass; MRPC F1 macro needs string/consistent classes)
    return s.astype(str)

def majority_baseline_preds(y_true: pd.Series) -> pd.Series:
    # simple + realistic baseline when no predictions are present
    if len(y_true) == 0:
        return y_true
    maj = y_true.mode().iloc[0]
    return pd.Series([maj] * len(y_true), index=y_true.index, dtype=str)

def evaluate_task(df: pd.DataFrame, task: str, metric_name: str) -> float | None:
    lab_col = find_label_column(task, list(df.columns))
    if lab_col is None:
        print(f"[warn] {task}: no label column found in {list(df.columns)} — skipping")
        return None

    y_true = coerce_series(df[lab_col].dropna())
    if len(y_true) == 0:
        print(f"[warn] {task}: empty labels after dropna — skipping")
        return None

    pred_col = find_pred_column(list(df.columns))
    if pred_col is not None:
        y_pred_full = coerce_series(df[pred_col])
        y_pred = y_pred_full.loc[y_true.index]
    else:
        # No model predictions: use majority-class baseline
        y_pred = majority_baseline_preds(y_true)

    if metric_name == "mcc":
        # MCC supports multiclass; labels already coerced to str
        return float(matthews_corrcoef(y_true, y_pred))
    elif metric_name == "acc":
        return float(accuracy_score(y_true, y_pred))
    elif metric_name == "f1_acc":
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        acc = float(accuracy_score(y_true, y_pred))
        return (f1m + acc) / 2.0
    else:
        print(f"[warn] Unknown metric '{metric_name}' for {task}")
        return None

def main():
    results = []
    for task, metric in TASK_METRIC.items():
        task_dir = OUT_ROOT / task
        if not task_dir.exists():
            print(f"[skip] {task}: no folder found at {task_dir}")
            continue

        for eps in EPS:
            eps_dir = task_dir / f"eps_{eps}"
            for split in SPLITS:
                f = eps_dir / f"{split}.csv"
                if not f.exists():
                    print(f"[skip] {task} ε={eps} {split}: missing at {f}")
                    continue

                df = safe_read(f)
                if df is None:
                    continue

                score = evaluate_task(df, task, metric)
                if score is None:
                    continue

                results.append({
                    "task": task,
                    "epsilon": eps,
                    "split": split,
                    "metric": metric,
                    "score": round(score, 6),
                })
                print(f"[OK] {task} | ε={eps} | {split} | {metric}: {score:.6f}")

    if results:
        OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(OUT_SUMMARY, index=False)
        print(f"\n[Done] Metrics summary saved → {OUT_SUMMARY}")
    else:
        print("\n[No metrics computed — check folders/labels]")

if __name__ == "__main__":
    main()
