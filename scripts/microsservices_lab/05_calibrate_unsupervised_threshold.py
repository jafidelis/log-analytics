from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def summarize(y_true: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (predicted == 1)).sum())
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "artifacts/microsservices_lab/stage4_unsupervised_scores.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage5_threshold_calibration.json"))
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--holdout-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, dtype={"tc_trace_id": str, "manual_label": str})
    labeled = scores[scores["manual_label"].isin(["anomalia", "nao_anomalia"])].copy()
    labeled["target"] = (labeled["manual_label"] == "anomalia").astype(int)
    if labeled["target"].sum() < 2 or (labeled["target"] == 0).sum() < 2:
        raise ValueError("São necessários exemplos positivos e negativos suficientes.")

    calibration_ids, holdout_ids = train_test_split(
        labeled.index,
        test_size=args.holdout_size,
        random_state=args.seed,
        stratify=labeled["target"],
    )
    calibration = labeled.loc[calibration_ids]
    holdout = labeled.loc[holdout_ids]
    candidates = sorted(calibration["anomaly_score"].unique(), reverse=True)
    selected = None
    for threshold in candidates:
        pred = (calibration["anomaly_score"] >= threshold).astype(int)
        current = summarize(calibration["target"], pred)
        if current["recall"] < args.min_recall:
            continue
        key = (current["precision"], current["f1"], current["specificity"], -threshold)
        if selected is None or key > selected["key"]:
            selected = {"threshold": float(threshold), "metrics": current, "key": key}
    if selected is None:
        raise RuntimeError("Nenhum limiar atingiu o recall mínimo na calibração.")

    threshold = selected["threshold"]
    holdout_pred = (holdout["anomaly_score"] >= threshold).astype(int)
    all_pred = (labeled["anomaly_score"] >= threshold).astype(int)
    report = {
        "stage": 5,
        "stage_name": "calibrate_unsupervised_threshold",
        "source_scores": str(args.scores),
        "labeled_traces": int(len(labeled)),
        "incerto_excluded": True,
        "split": {"calibration": int(len(calibration)), "holdout": int(len(holdout)), "random_state": args.seed},
        "selection": {"minimum_recall": args.min_recall, "threshold": threshold,
                       "calibration_metrics": selected["metrics"]},
        "holdout_metrics": summarize(holdout["target"], holdout_pred),
        "all_labeled_metrics_reference_only": summarize(labeled["target"], all_pred),
        "pr_auc_reference": float(average_precision_score(labeled["target"], labeled["anomaly_score"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
