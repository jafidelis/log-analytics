"""Etapa 5, incremento 5: avaliação única no teste.

Configurações congeladas lidas dos relatórios dos incrementos 1 e 4.
Este script deve ser executado UMA vez; ele se recusa a repetir.
"""

import csv
import gzip
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.load_representation_hdfs_v1 import (
    load_session_counts,
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

TRIVIAL_REPORT_PATH = Path(
    "artifacts/detection/hdfs_v1_trivial_baselines.json"
)
OCSVM_REPORT_PATH = Path("artifacts/detection/hdfs_v1_ocsvm_report.json")
OCSVM_MODEL_PATH = Path(
    "artifacts/detection/hdfs_v1_ocsvm_model.joblib"
)

OUTPUT_DIR = Path("artifacts/detection")
TEST_SCORES_PATH = OUTPUT_DIR / "hdfs_v1_ocsvm_scores_test.csv.gz"
REPORT_PATH = OUTPUT_DIR / "hdfs_v1_test_evaluation.json"


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    if REPORT_PATH.exists() or TEST_SCORES_PATH.exists():
        raise SystemExit(
            "Avaliação no teste já executada. O protocolo prevê uma "
            "única execução; o resultado registrado é o definitivo."
        )

    # 1. Recuperar as configurações congeladas dos relatórios.
    trivial_report = json.loads(
        TRIVIAL_REPORT_PATH.read_text(encoding="utf-8")
    )
    ocsvm_report = json.loads(
        OCSVM_REPORT_PATH.read_text(encoding="utf-8")
    )

    length_limit = trivial_report["results"]["B0c_unk_or_length"][
        "selected_L"
    ]
    frozen = ocsvm_report["frozen_candidate"]
    threshold = frozen["threshold"]

    expected_model_sha256 = ocsvm_report["outputs"]["model_joblib"][
        "sha256"
    ]
    checks = {
        "trivial_report_accepted": (
            trivial_report["acceptance"]["accepted"] is True
        ),
        "ocsvm_report_accepted": (
            ocsvm_report["acceptance"]["accepted"] is True
        ),
        "model_sha256_matches_report": (
            sha256_file(OCSVM_MODEL_PATH) == expected_model_sha256
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Pré-condições violadas: {checks}")

    bundle = joblib.load(OCSVM_MODEL_PATH)
    scaler, model = bundle["scaler"], bundle["model"]

    # 2. Carregar o teste (primeira e única leitura com rótulos).
    test = load_session_counts(
        COUNTS_PATH, COUNTS_MANIFEST_PATH, "test"
    )
    split_index = load_split_index(SPLIT_CSV_PATH)

    x_test = np.asarray(test.features, dtype=np.float64)
    y_test = np.asarray(
        [
            split_index[block_id].label == LABEL_ANOMALY
            for block_id in test.block_ids
        ],
        dtype=bool,
    )
    unk_column = x_test[:, -1]
    lengths = np.asarray(test.lengths)

    checks["test_sessions_expected"] = x_test.shape[0] == 86_259
    checks["test_anomalies_expected"] = int(y_test.sum()) == 2_526
    if not all(checks.values()):
        raise SystemExit(f"Pré-condições violadas: {checks}")

    # 3. Aplicar as três configurações congeladas.
    scores = -model.decision_function(scaler.transform(x_test))

    predictions = {
        "B0c_frozen": (unk_column > 0) | (lengths <= length_limit),
        "OCSVM_frozen": scores >= threshold,
    }
    predictions["Hybrid_frozen"] = (
        predictions["B0c_frozen"] | predictions["OCSVM_frozen"]
    )

    results = {
        name: confusion(y_test, y_pred)
        for name, y_pred in predictions.items()
    }
    pr_auc_test = round(
        float(average_precision_score(y_test, scores)), 4
    )

    # 4. Gravar scores do teste e o relatório final.
    scores_tmp = Path(str(TEST_SCORES_PATH) + ".tmp")
    with scores_tmp.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", mtime=0
        ) as gz_stream:
            with io.TextIOWrapper(
                gz_stream, encoding="utf-8", newline=""
            ) as text_stream:
                writer = csv.writer(text_stream)
                writer.writerow(["BlockId", "Partition", "AnomalyScore"])
                for block_id, score in zip(test.block_ids, scores):
                    writer.writerow(
                        [block_id, "test", repr(float(score))]
                    )
    os.replace(scores_tmp, TEST_SCORES_PATH)

    accepted = all(checks.values())
    report = {
        "stage": "etapa5_incremento5_avaliacao_teste",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0]},
        "frozen_configurations": {
            "B0c": {"rule": f"unk > 0 ou Length <= {length_limit}"},
            "OCSVM": {
                "params": frozen["params"],
                "threshold": threshold,
                "model_sha256": expected_model_sha256,
            },
            "Hybrid": {"rule": "B0c OU OCSVM"},
        },
        "validation_reference": {
            "B0c_f1": 0.5833,
            "OCSVM_f1": frozen["validation_metrics"]["f1"],
            "OCSVM_pr_auc": frozen["pr_auc"],
        },
        "test": {
            "sessions": int(x_test.shape[0]),
            "anomalies": int(y_test.sum()),
            "results": results,
            "OCSVM_pr_auc": pr_auc_test,
        },
        "outputs": {
            "test_scores_csv_gz": {
                "path": str(TEST_SCORES_PATH),
                "sha256": sha256_file(TEST_SCORES_PATH),
            },
        },
        "acceptance": {"checks": checks, "accepted": accepted},
    }

    report_tmp = Path(str(REPORT_PATH) + ".tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(report_tmp, REPORT_PATH)

    print(json.dumps(
        {"test": report["test"], "acceptance": report["acceptance"]},
        ensure_ascii=False, indent=2,
    ))

    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()