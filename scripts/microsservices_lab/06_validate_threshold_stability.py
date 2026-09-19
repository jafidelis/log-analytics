from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def summarize(y_true: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (predicted == 1)).sum())
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def choose_threshold(frame: pd.DataFrame, minimum_recall: float) -> tuple[float, dict]:
    selected = None
    for threshold in sorted(frame["anomaly_score"].unique(), reverse=True):
        predicted = (frame["anomaly_score"] >= threshold).astype(int)
        current = summarize(frame["target"], predicted)
        if current["recall"] < minimum_recall:
            continue
        key = (current["precision"], current["f1"], -threshold)
        if selected is None or key > selected["key"]:
            selected = {"threshold": float(threshold), "metrics": current, "key": key}
    if selected is None:
        raise RuntimeError("Nenhum limiar atingiu o recall mínimo.")
    return selected["threshold"], selected["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "artifacts/microsservices_lab/stage4_unsupervised_scores.csv"))
    parser.add_argument("--features", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage6_threshold_stability.json"))
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--holdout-size", type=float, default=0.30)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260915, 20260916, 20260917, 20260918, 20260919])
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, dtype={"tc_trace_id": str, "manual_label": str})
    labeled = scores[scores["manual_label"].isin(["anomalia", "nao_anomalia"])].copy()
    labeled["target"] = (labeled["manual_label"] == "anomalia").astype(int)
    results = []
    for seed in args.seeds:
        calibration_ids, holdout_ids = train_test_split(
            labeled.index, test_size=args.holdout_size, random_state=seed,
            stratify=labeled["target"])
        calibration = labeled.loc[calibration_ids]
        holdout = labeled.loc[holdout_ids]
        threshold, calibration_metrics = choose_threshold(calibration, args.minimum_recall)
        holdout_metrics = summarize(holdout["target"],
                                    (holdout["anomaly_score"] >= threshold).astype(int))
        results.append({"seed": seed, "threshold": threshold,
                        "calibration_metrics": calibration_metrics,
                        "holdout_metrics": holdout_metrics})

    thresholds = pd.Series([item["threshold"] for item in results])
    median_threshold = float(thresholds.median())
    labeled["median_threshold_prediction"] = (labeled["anomaly_score"] >= median_threshold).astype(int)
    false_positives = labeled[(labeled["target"] == 0) &
                              (labeled["median_threshold_prediction"] == 1)].copy()
    features = pd.read_csv(args.features, dtype={"tc_trace_id": str})
    profile = false_positives.merge(features, on="tc_trace_id", how="left")
    profile_summary = {"count": int(len(profile))}
    for column in ["event_count", "service_count", "error_count", "warn_count",
                   "has_stack_trace", "text_signal_count"]:
        if column in profile:
            profile_summary[column] = {
                "mean": float(profile[column].mean()),
                "median": float(profile[column].median()),
                "max": float(profile[column].max()),
            }
    report = {
        "stage": 6, "stage_name": "validate_threshold_stability",
        "source_scores": str(args.scores), "labeled_traces": int(len(labeled)),
        "incerto_excluded": True, "minimum_recall": args.minimum_recall,
        "runs": results,
        "threshold_summary": {"median": median_threshold, "min": float(thresholds.min()),
                               "max": float(thresholds.max()), "mean": float(thresholds.mean()),
                               "std": float(thresholds.std(ddof=0))},
        "median_threshold_all_labeled_metrics": summarize(
            labeled["target"], labeled["median_threshold_prediction"]),
        "false_positive_profile_at_median_threshold": profile_summary,
        "labels_used_for_model_fit": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
