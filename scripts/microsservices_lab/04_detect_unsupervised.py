from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    parser.add_argument("--labels", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage4_unsupervised_scores.csv"))
    args = parser.parse_args()
    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels, dtype=str)
    label_map = dict(zip(labels["tc_trace_id"].str.strip(), labels["label"].str.strip()))
    numeric = [c for c in features.columns if c != "tc_trace_id"]
    x = features[numeric].astype(float)
    x = StandardScaler().fit_transform(x)
    model = IsolationForest(n_estimators=300, max_samples="auto", contamination="auto",
                            random_state=20260915, n_jobs=-1)
    model.fit(x)
    result = features[["tc_trace_id"]].copy()
    result["anomaly_score"] = -model.score_samples(x)
    result["model_prediction"] = (model.predict(x) == -1).astype(int)
    result["manual_label"] = result["tc_trace_id"].map(label_map).fillna("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("anomaly_score", ascending=False).to_csv(args.output, index=False)
    labeled = result[result["manual_label"].isin(["anomalia", "nao_anomalia"])]
    report = {"stage": 4, "stage_name": "unsupervised_detection", "features_used": numeric,
              "labels_in_features": False, "labels_used_for_fit": False,
              "traces_scored": int(len(result)), "labeled_traces_for_evaluation": int(len(labeled)),
              "model": {"name": "IsolationForest", "n_estimators": 300, "random_state": 20260915}}
    if len(labeled):
        y = (labeled["manual_label"] == "anomalia").astype(int)
        pred = labeled["model_prediction"]
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        report["evaluation_auxiliary"] = {"precision": float(p), "recall": float(r), "f1": float(f),
                                           "pr_auc_score": float(average_precision_score(y, labeled["anomaly_score"]))}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
