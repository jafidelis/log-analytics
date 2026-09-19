from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def choose_threshold(frame: pd.DataFrame, minimum_recall: float) -> float:
    best = None
    for threshold in sorted(frame["score"].unique(), reverse=True):
        predicted = (frame["score"] >= threshold).astype(int)
        current = metrics(frame["target"], predicted)
        if current["recall"] < minimum_recall:
            continue
        key = (current["precision"], current["f1"], -threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    if best is None:
        raise RuntimeError("Nenhum limiar atingiu o recall mínimo.")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "artifacts/microsservices_lab/stage4_unsupervised_scores.csv"))
    parser.add_argument("--features", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage7_fp_strategies.json"))
    parser.add_argument("--false-positives-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage7_false_positive_examples.csv"))
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, dtype={"tc_trace_id": str, "manual_label": str})
    features = pd.read_csv(args.features, dtype={"tc_trace_id": str})
    frame = scores.merge(features, on="tc_trace_id", how="left", validate="one_to_one")
    frame = frame[frame["manual_label"].isin(["anomalia", "nao_anomalia"])].copy()
    frame["target"] = (frame["manual_label"] == "anomalia").astype(int)
    calibration_ids, holdout_ids = train_test_split(
        frame.index, test_size=0.30, random_state=args.seed, stratify=frame["target"])
    calibration = frame.loc[calibration_ids]
    holdout = frame.loc[holdout_ids]
    base_threshold = 0.6713793902714876
    base_pred = (holdout["anomaly_score"] >= base_threshold).astype(int)
    stack_pred = ((holdout["anomaly_score"] >= base_threshold) &
                  (holdout["has_stack_trace"] > 0)).astype(int)

    revised_columns = ["log_event_count", "service_count", "error_rate", "warn_rate",
                       "other_rate", "has_stack_trace", "has_exception_term",
                       "has_failed_term", "has_timeout_term", "has_error_term"]
    revised = frame.copy()
    revised["log_event_count"] = __import__("numpy").log1p(revised["event_count"].astype(float))
    revised["other_rate"] = revised["other_count"] / revised["event_count"].clip(lower=1)
    x = StandardScaler().fit_transform(revised[revised_columns].astype(float))
    model = IsolationForest(n_estimators=300, max_samples="auto", contamination="auto",
                            random_state=20260915, n_jobs=-1)
    model.fit(x)
    revised["revised_score"] = -model.score_samples(x)
    revised_cal = revised.loc[calibration_ids].rename(columns={"revised_score": "score"})
    revised_hold = revised.loc[holdout_ids].rename(columns={"revised_score": "score"})
    revised_threshold = choose_threshold(revised_cal, args.minimum_recall)
    revised_pred = (revised_hold["score"] >= revised_threshold).astype(int)

    false_positives = frame[(frame["target"] == 0) &
                            (frame["anomaly_score"] >= base_threshold)].copy()
    example_columns = ["tc_trace_id", "anomaly_score", "event_count", "service_count",
                       "error_count", "warn_count", "has_stack_trace", "text_signal_count",
                       "manual_label"]
    args.false_positives_output.parent.mkdir(parents=True, exist_ok=True)
    false_positives.sort_values("anomaly_score", ascending=False)[example_columns].to_csv(
        args.false_positives_output, index=False)

    report = {
        "stage": 7, "stage_name": "compare_false_positive_strategies",
        "labeled_traces": int(len(frame)), "holdout": int(len(holdout)),
        "incerto_excluded": True, "labels_used_for_model_fit": False,
        "base_calibrated_threshold": {"threshold": base_threshold,
            "holdout_metrics": metrics(holdout["target"], base_pred)},
        "score_and_stack_trace": {"holdout_metrics": metrics(holdout["target"], stack_pred)},
        "revised_representation": {"features": revised_columns, "threshold": revised_threshold,
            "holdout_metrics": metrics(revised_hold["target"], revised_pred)},
        "false_positive_examples": str(args.false_positives_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
