from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def metrics(y_true: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (predicted == 1)).sum())
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "artifacts/microsservices_lab/stage4_unsupervised_scores.csv"))
    parser.add_argument("--features", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage8_two_level_alerts.json"))
    parser.add_argument("--alerts-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage8_trace_alerts.csv"))
    parser.add_argument("--threshold", type=float, default=0.6713793902714876)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, dtype={"tc_trace_id": str, "manual_label": str})
    features = pd.read_csv(args.features, dtype={"tc_trace_id": str})
    frame = scores.merge(features, on="tc_trace_id", how="left", validate="one_to_one")
    frame["score_alert"] = (frame["anomaly_score"] >= args.threshold).astype(int)
    frame["alert_level"] = "none"
    frame.loc[frame["score_alert"].eq(1) & frame["has_stack_trace"].gt(0), "alert_level"] = "high_confidence"
    frame.loc[frame["score_alert"].eq(1) & frame["has_stack_trace"].eq(0), "alert_level"] = "review"
    args.alerts_output.parent.mkdir(parents=True, exist_ok=True)
    frame[["tc_trace_id", "anomaly_score", "has_stack_trace", "error_count", "warn_count",
           "event_count", "service_count", "alert_level", "manual_label"]].sort_values(
               "anomaly_score", ascending=False).to_csv(args.alerts_output, index=False)

    labeled = frame[frame["manual_label"].isin(["anomalia", "nao_anomalia"])].copy()
    labeled["target"] = (labeled["manual_label"] == "anomalia").astype(int)
    _, holdout_ids = train_test_split(labeled.index, test_size=0.30, random_state=args.seed,
                                      stratify=labeled["target"])
    holdout = labeled.loc[holdout_ids]
    high = (holdout["alert_level"] == "high_confidence").astype(int)
    any_alert = holdout["score_alert"]
    review = (holdout["alert_level"] == "review").astype(int)
    report = {"stage": 8, "stage_name": "two_level_alerts", "threshold": args.threshold,
              "labels_used_for_model_fit": False, "incerto_excluded": True,
              "alert_counts_all_traces": frame["alert_level"].value_counts().to_dict(),
              "holdout_metrics": {"high_confidence": metrics(holdout["target"], high),
                                   "all_score_alerts": metrics(holdout["target"], any_alert),
                                   "review_only": metrics(holdout["target"], review)},
              "alerts_output": str(args.alerts_output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
