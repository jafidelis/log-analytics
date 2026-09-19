from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = ["event_count", "info_count", "warn_count", "error_count", "other_count", "error_rate", "warn_rate", "has_stack_trace", "service_count", "unique_template_count", "unk_count", "top_template_count"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(y: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def select_threshold(scores: pd.Series, y: pd.Series, minimum_recall: float) -> dict:
    selected = None
    for threshold in sorted(scores.unique(), reverse=True):
        current = summarize(y, (scores >= threshold).astype(int))
        if current["recall"] < minimum_recall:
            continue
        key = (current["precision"], current["f1"], current["specificity"], -float(threshold))
        if selected is None or key > selected["key"]:
            selected = {"threshold": float(threshold), "metrics": current, "key": key}
    if selected is None:
        raise RuntimeError("Nenhum limiar atingiu o recall mínimo.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecção escalável em representações por trace.")
    parser.add_argument("--representations", type=Path, default=Path("data/processed/microservices_final/microservices_trace_representations.csv.gz"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--train-traces", type=int, default=200_000)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_detection_scores.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage07_scalable_detection.manifest.json"))
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Artefatos da detecção já existem; remova-os deliberadamente para reexecutar.")

    labels = pd.read_csv(args.labels, dtype=str)
    label_map = dict(zip(labels["tc_trace_id"].str.strip(), labels["label"].str.strip()))
    scaler = StandardScaler()
    train_parts = []
    total = 0
    for chunk in pd.read_csv(args.representations, compression="gzip", usecols=FEATURES, chunksize=args.chunksize):
        values = chunk[FEATURES].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()
        scaler.partial_fit(values)
        if sum(len(part) for part in train_parts) < args.train_traces:
            train_parts.append(values[: max(0, args.train_traces - sum(len(part) for part in train_parts))])
        total += len(chunk)
    x_train = scaler.transform(np.vstack(train_parts))
    models = {
        "isolation_forest": IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=args.seed, n_jobs=-1),
        "sgd_one_class_svm": SGDOneClassSVM(nu=0.01, random_state=args.seed, max_iter=1000, tol=1e-3, shuffle=True),
    }
    for model in models.values():
        model.fit(x_train)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out); writer.writerow(["analysis_trace_id", "manual_label", "isolation_forest_score", "sgd_one_class_svm_score"])
        offset = 0
        for chunk in pd.read_csv(args.representations, compression="gzip", usecols=["analysis_trace_id", *FEATURES], chunksize=args.chunksize):
            ids = chunk["analysis_trace_id"].astype(str).str.strip()
            values = chunk[FEATURES].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()
            transformed = scaler.transform(values)
            model_scores = [-model.score_samples(transformed) for model in models.values()]
            for trace_id, score_if, score_svm in zip(ids, model_scores[0], model_scores[1]):
                writer.writerow([trace_id, label_map.get(trace_id, ""), float(score_if), float(score_svm)])
            offset += len(chunk)
            if offset % 250_000 == 0: print(f"[detecção] scores {offset:,}/{total:,}", flush=True)
    os.replace(temporary, args.output)

    scores = pd.read_csv(args.output, compression="gzip", dtype={"analysis_trace_id": str, "manual_label": str})
    labeled = scores[scores["manual_label"].isin(["anomalia", "nao_anomalia"])].copy(); labeled["target"] = (labeled["manual_label"] == "anomalia").astype(int)
    calibration_ids, holdout_ids = train_test_split(labeled.index, test_size=0.30, random_state=args.seed, stratify=labeled["target"])
    report = {"stage": 7, "stage_name": "scalable_unsupervised_detection", "generated_at": datetime.now(timezone.utc).isoformat(), "representation": str(args.representations), "features_used": FEATURES, "labels_used_for_fit": False, "train_traces": len(x_train), "traces_scored": total, "labeled_traces": len(labeled), "split": {"calibration": len(calibration_ids), "holdout": len(holdout_ids), "random_state": args.seed}, "models": {}}
    for name in models:
        selected = select_threshold(scores.loc[calibration_ids, name + "_score"], labeled.loc[calibration_ids, "target"], 0.90)
        holdout = summarize(labeled.loc[holdout_ids, "target"], (scores.loc[holdout_ids, name + "_score"] >= selected["threshold"]).astype(int))
        report["models"][name] = {"threshold": selected["threshold"], "calibration": selected["metrics"], "holdout": holdout, "pr_auc_reference": float(average_precision_score(labeled["target"], scores.loc[labeled.index, name + "_score"]))}
    report["output"] = {"path": str(args.output), "sha256": sha256_file(args.output)}
    report["checks"] = {"traces_scored": total == len(scores), "scores_nonzero": len(scores) > 0, "output_hash_recorded": True}
    report["accepted"] = all(report["checks"].values())
    temporary_manifest = Path(str(args.manifest) + ".tmp"); temporary_manifest.parent.mkdir(parents=True, exist_ok=True); temporary_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary_manifest.replace(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
