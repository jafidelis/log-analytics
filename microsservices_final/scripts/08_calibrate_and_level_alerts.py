from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Tipo não serializável: {type(value).__name__}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0; specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra o Isolation Forest e cria níveis de alerta.")
    parser.add_argument("--scores", type=Path, default=Path("data/processed/microservices_final/microservices_detection_scores.csv.gz"))
    parser.add_argument("--representations", type=Path, default=Path("data/processed/microservices_final/microservices_trace_representations.csv.gz"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_alerts.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage08_alert_calibration.manifest.json"))
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Artefatos de alertas já existem; remova-os deliberadamente para reexecutar.")
    scores = pd.read_csv(args.scores, compression="gzip", dtype={"analysis_trace_id": str, "manual_label": str})
    stack = pd.read_csv(args.representations, compression="gzip", usecols=["analysis_trace_id", "has_stack_trace"], dtype={"analysis_trace_id": str})
    result = scores.merge(stack, on="analysis_trace_id", how="left", validate="one_to_one")
    if result["has_stack_trace"].isna().any():
        raise ValueError("Há scores sem representação de stack trace")
    labeled = result[result["manual_label"].isin(["anomalia", "nao_anomalia"])].copy(); labeled["target"] = (labeled["manual_label"] == "anomalia").astype(int)
    calibration_ids, holdout_ids = train_test_split(labeled.index, test_size=0.30, random_state=args.seed, stratify=labeled["target"])
    selected = None
    for threshold in sorted(labeled.loc[calibration_ids, "isolation_forest_score"].unique(), reverse=True):
        current = metrics(labeled.loc[calibration_ids, "target"], (labeled.loc[calibration_ids, "isolation_forest_score"] >= threshold).astype(int))
        if current["recall"] < args.minimum_recall:
            continue
        key = (current["precision"], current["f1"], current["specificity"], -float(threshold))
        if selected is None or key > selected["key"]:
            selected = {"threshold": float(threshold), "metrics": current, "key": key}
    if selected is None:
        raise RuntimeError("Nenhum limiar atingiu o recall mínimo")
    threshold = selected["threshold"]
    result["score_alert"] = (result["isolation_forest_score"] >= threshold).astype(int)
    result["alert_level"] = "none"
    result.loc[result["score_alert"].eq(1), "alert_level"] = "review"
    result.loc[result["score_alert"].eq(1) & result["has_stack_trace"].eq(1), "alert_level"] = "high_confidence"
    holdout_metrics = metrics(labeled.loc[holdout_ids, "target"], (labeled.loc[holdout_ids, "isolation_forest_score"] >= threshold).astype(int))
    high_labeled = labeled.loc[holdout_ids]
    high_pred = ((high_labeled["isolation_forest_score"] >= threshold) & (high_labeled["has_stack_trace"] == 1)).astype(int)
    report = {"stage": 8, "stage_name": "calibrate_and_level_alerts", "generated_at": datetime.now(timezone.utc).isoformat(), "threshold": threshold, "minimum_recall": args.minimum_recall, "labels_used_for_fit": False, "split": {"calibration": len(calibration_ids), "holdout": len(holdout_ids), "random_state": args.seed}, "calibration_metrics": selected["metrics"], "holdout_score_metrics": holdout_metrics, "holdout_high_confidence_metrics": metrics(high_labeled["target"], high_pred), "pr_auc_reference": float(average_precision_score(labeled["target"], labeled["isolation_forest_score"])), "alerts": result["alert_level"].value_counts().to_dict(), "output": {"path": str(args.output), "sha256": ""}, "checks": {"scores_representation_join_complete": len(result) == len(scores), "threshold_selected": True, "levels_present": result["alert_level"].isin(["none", "review", "high_confidence"]).all()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("isolation_forest_score", ascending=False).to_csv(args.output, index=False, compression="gzip")
    report["output"]["sha256"] = sha256_file(args.output); report["checks"]["output_hash_recorded"] = True; report["accepted"] = all(report["checks"].values())
    args.manifest.parent.mkdir(parents=True, exist_ok=True); temporary = Path(str(args.manifest) + ".tmp"); temporary.write_text(json.dumps(report, default=json_default, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(args.manifest)
    print(json.dumps(report, default=json_default, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
