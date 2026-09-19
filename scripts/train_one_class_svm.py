"""Etapa 5, incremento 4: One-Class SVM sobre contagens padronizadas.

Scaler ajustado em train_normal completo; SVM em subamostra de 20.000
(semente fixa). Calibração na validação; teste não é tocado.
"""

import csv
import gzip
import io
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.detection_metrics import sweep_best_threshold
from log_analytics.data.load_representation_hdfs_v1 import (
    load_session_counts,
    load_train_normal_ids,
    select_train_normal,
)
from log_analytics.data.session_hdfs_v1 import load_split_index
from log_analytics.data.split_hdfs_v1 import LABEL_ANOMALY

# ... mesmos caminhos de entrada do script do IF ...
COUNTS_PATH = Path(
    "artifacts/representation/hdfs_v1_session_counts.csv.gz"
)
COUNTS_MANIFEST_PATH = Path(
    "artifacts/representation/hdfs_v1_counts_manifest.json"
)
SPLIT_CSV_PATH = Path("artifacts/data_splits/hdfs_v1_session_split.csv")
SPLIT_MANIFEST_PATH = Path(
    "artifacts/data_splits/hdfs_v1_split_manifest.json"
)
TRAIN_NORMAL_LIST_PATH = Path(
    "artifacts/data_splits/hdfs_v1_train_normal_block_ids.txt"
)

OUTPUT_DIR = Path("artifacts/detection")
MODEL_PATH = OUTPUT_DIR / "hdfs_v1_ocsvm_model.joblib"
SCORES_PATH = OUTPUT_DIR / "hdfs_v1_ocsvm_scores.csv.gz"
REPORT_PATH = OUTPUT_DIR / "hdfs_v1_ocsvm_report.json"

RANDOM_STATE = 20260911
SUBSAMPLE_SIZE = 20_000
GRID = [
    {"nu": 0.01, "gamma": "scale"},
    {"nu": 0.01, "gamma": 0.05},
    {"nu": 0.05, "gamma": "scale"},
    {"nu": 0.05, "gamma": 0.05},
]


def main() -> None:
    existing = [
        str(p)
        for p in (MODEL_PATH, SCORES_PATH, REPORT_PATH)
        if p.exists()
    ]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    train = load_session_counts(
        COUNTS_PATH, COUNTS_MANIFEST_PATH, "train"
    )
    validation = load_session_counts(
        COUNTS_PATH, COUNTS_MANIFEST_PATH, "validation"
    )
    train_normal_ids = load_train_normal_ids(
        TRAIN_NORMAL_LIST_PATH, SPLIT_MANIFEST_PATH
    )
    train_normal = select_train_normal(train, train_normal_ids)
    split_index = load_split_index(SPLIT_CSV_PATH)

    x_fit_full = np.asarray(train_normal.features, dtype=np.float64)
    x_val = np.asarray(validation.features, dtype=np.float64)
    y_val = np.asarray(
        [
            split_index[block_id].label == LABEL_ANOMALY
            for block_id in validation.block_ids
        ],
        dtype=bool,
    )

    # Scaler no train_normal completo; SVM na subamostra com semente.
    scaler = StandardScaler().fit(x_fit_full)
    rng = np.random.default_rng(RANDOM_STATE)
    subsample_indices = np.sort(
        rng.choice(
            x_fit_full.shape[0], size=SUBSAMPLE_SIZE, replace=False
        )
    )
    x_fit = scaler.transform(x_fit_full[subsample_indices])
    x_val_scaled = scaler.transform(x_val)

    checks = {
        "fit_pool_expected": x_fit_full.shape[0] == len(train_normal_ids),
        "subsample_size": x_fit.shape[0] == SUBSAMPLE_SIZE,
        "validation_sessions_expected": x_val.shape[0] == 86_260,
        "validation_anomalies_expected": int(y_val.sum()) == 2_526,
        "feature_dimensions": x_fit.shape[1] == 20,
    }
    if not all(checks.values()):
        raise SystemExit(f"Pré-condições violadas: {checks}")

    combos = []
    best_combo = None

    for params in GRID:
        model = OneClassSVM(kernel="rbf", **params)
        model.fit(x_fit)

        val_scores = -model.decision_function(x_val_scaled)
        threshold_result = sweep_best_threshold(y_val, val_scores)
        pr_auc = round(
            float(average_precision_score(y_val, val_scores)), 4
        )

        combo = {
            "params": params,
            "pr_auc": pr_auc,
            "best_threshold": threshold_result,
        }
        combos.append(combo)
        print(
            f"nu={params['nu']} gamma={params['gamma']} -> "
            f"F1={threshold_result['f1']} PR-AUC={pr_auc}",
            file=sys.stderr,
        )

        if (
            best_combo is None
            or combo["best_threshold"]["f1"]
            > best_combo["combo"]["best_threshold"]["f1"]
        ):
            best_combo = {"combo": combo, "model": model}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = best_combo["combo"]
    model = best_combo["model"]

    model_tmp = Path(str(MODEL_PATH) + ".tmp")
    joblib.dump({"scaler": scaler, "model": model}, model_tmp)
    os.replace(model_tmp, MODEL_PATH)

    val_scores = -model.decision_function(x_val_scaled)
    fit_scores = -model.decision_function(
        scaler.transform(x_fit_full)
    )

    scores_tmp = Path(str(SCORES_PATH) + ".tmp")
    with scores_tmp.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", mtime=0
        ) as gz_stream:
            with io.TextIOWrapper(
                gz_stream, encoding="utf-8", newline=""
            ) as text_stream:
                writer = csv.writer(text_stream)
                writer.writerow(["BlockId", "Partition", "AnomalyScore"])
                for block_id, score in zip(
                    train_normal.block_ids, fit_scores
                ):
                    writer.writerow(
                        [block_id, "train_normal", repr(float(score))]
                    )
                for block_id, score in zip(
                    validation.block_ids, val_scores
                ):
                    writer.writerow(
                        [block_id, "validation", repr(float(score))]
                    )
    os.replace(scores_tmp, SCORES_PATH)

    report = {
        "stage": "etapa5_incremento4_one_class_svm",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": version("scikit-learn"),
            "numpy": version("numpy"),
        },
        "protocol": {
            "fit_partition": "train_normal",
            "scaler": "StandardScaler ajustado no train_normal completo",
            "subsample": {
                "size": SUBSAMPLE_SIZE,
                "seed": RANDOM_STATE,
                "method": "uniforme sem reposição",
            },
            "calibration_partition": "validation",
            "selection_rule": "max F1 (classe Anomaly)",
            "grid": GRID,
            "score": "-decision_function (maior = mais anômalo)",
            "test_touched": False,
        },
        "reference_baselines": {
            "B0c_f1": 0.5833,
            "isolation_forest_f1": 0.3021,
        },
        "combos": combos,
        "frozen_candidate": {
            "params": selected["params"],
            "threshold": selected["best_threshold"]["threshold"],
            "validation_metrics": selected["best_threshold"],
            "pr_auc": selected["pr_auc"],
        },
        "outputs": {
            "model_joblib": {
                "path": str(MODEL_PATH),
                "sha256": sha256_file(MODEL_PATH),
            },
            "scores_csv_gz": {
                "path": str(SCORES_PATH),
                "sha256": sha256_file(SCORES_PATH),
            },
        },
        "acceptance": {"checks": checks, "accepted": all(checks.values())},
    }

    report_tmp = Path(str(REPORT_PATH) + ".tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(report_tmp, REPORT_PATH)

    print(json.dumps(
        {
            "combos": [
                {
                    "params": c["params"],
                    "f1": c["best_threshold"]["f1"],
                    "precision": c["best_threshold"]["precision"],
                    "recall": c["best_threshold"]["recall"],
                    "pr_auc": c["pr_auc"],
                }
                for c in combos
            ],
            "frozen_candidate": report["frozen_candidate"],
        },
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()