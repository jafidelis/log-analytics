from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def calibrate(scores: pd.Series, y: pd.Series, minimum_recall: float) -> dict:
    candidates = sorted(scores.unique(), reverse=True)
    selected = None
    for threshold in candidates:
        current = metrics(y, (scores >= threshold).astype(int))
        if current["recall"] < minimum_recall:
            continue
        key = (current["precision"], current["f1"], current["specificity"], -threshold)
        if selected is None or key > selected["key"]:
            selected = {"threshold": float(threshold), "metrics": current, "key": key}
    if selected is None:
        raise RuntimeError("Nenhum limiar atingiu o recall mínimo.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/microsservices_lab/stage27_detector_comparison.json"))
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--holdout-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels, dtype=str)
    label_map = dict(zip(labels["tc_trace_id"].str.strip(), labels["label"].str.strip()))
    feature_columns = [c for c in features.columns if c != "tc_trace_id"]
    x = StandardScaler().fit_transform(features[feature_columns].astype(float))

    models = {
        "isolation_forest": IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=args.seed, n_jobs=-1),
        "one_class_svm": OneClassSVM(kernel="rbf", nu=0.01, gamma="scale"),
    }
    scores = {}
    for name, model in models.items():
        model.fit(x)
        scores[name] = pd.Series(-model.score_samples(x), index=features.index)

    result = features[["tc_trace_id"]].copy()
    result["manual_label"] = result["tc_trace_id"].map(label_map).fillna("")
    labeled = result[result["manual_label"].isin(["anomalia", "nao_anomalia"])].copy()
    labeled["target"] = (labeled["manual_label"] == "anomalia").astype(int)
    calibration_ids, holdout_ids = train_test_split(labeled.index, test_size=args.holdout_size, random_state=args.seed, stratify=labeled["target"])
    report = {
        "stage": 27,
        "stage_name": "compare_unsupervised_detectors",
        "features": str(args.features),
        "feature_columns": feature_columns,
        "labels_used_for_fit": False,
        "labeled_traces": int(len(labeled)),
        "incerto_excluded": True,
        "split": {"calibration": int(len(calibration_ids)), "holdout": int(len(holdout_ids)), "random_state": args.seed},
        "selection": {"minimum_recall": args.minimum_recall, "policy": "maximize precision, then F1 and specificity on calibration"},
        "models": {},
    }
    for name, score in scores.items():
        result[f"{name}_score"] = score
        calibration = labeled.loc[calibration_ids]
        holdout = labeled.loc[holdout_ids]
        selected = calibrate(score.loc[calibration_ids], calibration["target"], args.minimum_recall)
        threshold = selected["threshold"]
        holdout_metrics = metrics(holdout["target"], (score.loc[holdout_ids] >= threshold).astype(int))
        all_metrics = metrics(labeled["target"], (score.loc[labeled.index] >= threshold).astype(int))
        report["models"][name] = {
            "hyperparameters": {"nu": 0.01, "gamma": "scale"} if name == "one_class_svm" else {"n_estimators": 300, "random_state": args.seed},
            "calibration": {"threshold": threshold, "metrics": selected["metrics"]},
            "holdout": holdout_metrics,
            "all_labeled_reference": all_metrics,
            "pr_auc_reference": float(average_precision_score(labeled["target"], score.loc[labeled.index])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result.to_csv(args.output.with_suffix(".scores.csv"), index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
