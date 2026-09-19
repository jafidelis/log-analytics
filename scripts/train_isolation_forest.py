"""Etapa 5, incremento 3: Isolation Forest sobre vetores de contagem.

Ajuste somente em train_normal; calibração de limiar na validação.
O teste não é tocado — a configuração vencedora fica congelada
no relatório como candidata para o incremento 5.
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.detection_metrics import sweep_best_threshold
from log_analytics.data.load_representation_hdfs_v1 import (
    load_session_counts,
    load_train_normal_ids,
    select_train_normal,
)
from log_analytics.data.session_hdfs_v1 import load_split_index
from log_analytics.data.split_hdfs_v1 import LABEL_ANOMALY

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
MODEL_PATH = OUTPUT_DIR / "hdfs_v1_isolation_forest_model.joblib"
SCORES_PATH = OUTPUT_DIR / "hdfs_v1_isolation_forest_scores.csv.gz"
REPORT_PATH = OUTPUT_DIR / "hdfs_v1_isolation_forest_report.json"

RANDOM_STATE = 20260911
GRID = [
    {"n_estimators": 100, "max_samples": 256},
    {"n_estimators": 100, "max_samples": 2048},
    {"n_estimators": 300, "max_samples": 256},
    {"n_estimators": 300, "max_samples": 2048},
]

def main() -> None:
    existing = [
        str(p)
        for p in (MODEL_PATH, SCORES_PATH, REPORT_PATH)
        if p.exists()
    ]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    # 1. Cargas validadas (checksums e manifestos no carregador).
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

    x_fit = np.asarray(train_normal.features, dtype=np.float64)
    x_val = np.asarray(validation.features, dtype=np.float64)
    y_val = np.asarray(
        [
            split_index[block_id].label == LABEL_ANOMALY
            for block_id in validation.block_ids
        ],
        dtype=bool,
    )

    checks = {
        "fit_sessions_expected": x_fit.shape[0] == len(train_normal_ids),
        "validation_sessions_expected": x_val.shape[0] == 86_260,
        "validation_anomalies_expected": int(y_val.sum()) == 2_526,
        "feature_dimensions": x_fit.shape[1] == 20,
    }
    if not all(checks.values()):
        raise SystemExit(f"Pré-condições violadas: {checks}")

    # 2. Grade pré-registrada: ajustar, pontuar, calibrar.
    combos = []
    best_combo = None

    for params in GRID:
        model = IsolationForest(
            random_state=RANDOM_STATE,
            bootstrap=False,
            max_features=1.0,
            n_jobs=-1,
            **params,
        )
        model.fit(x_fit)

        val_scores = -model.score_samples(x_val)
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
            f"n_estimators={params['n_estimators']} "
            f"max_samples={params['max_samples']} -> "
            f"F1={threshold_result['f1']} PR-AUC={pr_auc}",
            file=sys.stderr,
        )

        if (
            best_combo is None
            or combo["best_threshold"]["f1"]
            > best_combo["combo"]["best_threshold"]["f1"]
        ):
            best_combo = {"combo": combo, "model": model}

    # 3. Persistir o modelo vencedor e os scores.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = best_combo["combo"]
    model = best_combo["model"]

    model_tmp = Path(str(MODEL_PATH) + ".tmp")
    joblib.dump(model, model_tmp)
    os.replace(model_tmp, MODEL_PATH)

    train_scores = -model.score_samples(x_fit)
    val_scores = -model.score_samples(x_val)

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
                    train_normal.block_ids, train_scores
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

    # 4. Relatório com a candidata congelada.
    report = {
        "stage": "etapa5_incremento3_isolation_forest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": version("scikit-learn"),
            "numpy": version("numpy"),
        },
        "protocol": {
            "fit_partition": "train_normal",
            "calibration_partition": "validation",
            "selection_rule": "max F1 (classe Anomaly)",
            "random_state": RANDOM_STATE,
            "grid": GRID,
            "score": "-score_samples (maior = mais anômalo)",
            "test_touched": False,
        },
        "reference_baseline": {"B0c_f1": 0.5833},
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