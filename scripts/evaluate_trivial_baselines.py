"""Etapa 5, incremento 1: baselines triviais avaliados na validação.

B0a: unk > 0.  B0b: Length <= L.  B0c: unk > 0 ou Length <= L'.
Grades e seleção pré-registradas; teste não é tocado.
"""

import csv
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import sha256_file
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

OUTPUT_DIR = Path("artifacts/detection")
REPORT_PATH = OUTPUT_DIR / "hdfs_v1_trivial_baselines.json"

LENGTH_GRID = range(2, 51)  # pré-registrado; empate -> menor L


def load_accepted_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    acceptance = manifest["acceptance"]
    if acceptance.get("accepted") is not True or any(
        value is not True for value in acceptance["checks"].values()
    ):
        raise ValueError(f"Manifesto não aprovado: {path}")
    return manifest


def evaluate(rows: list[tuple[int, int, bool]], predict) -> dict:
    """Métricas da classe Anomaly para uma regra binária."""
    tp = fp = fn = tn = 0
    for length, unk, is_anomaly in rows:
        predicted = predict(length, unk)
        if predicted and is_anomaly:
            tp += 1
        elif predicted:
            fp += 1
        elif is_anomaly:
            fn += 1
        else:
            tn += 1

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


def select_length_threshold(
    rows: list[tuple[int, int, bool]],
    combine_with_unk: bool,
) -> tuple[int, dict, list[dict]]:
    """Varre a grade e devolve (L, métricas, grade completa)."""
    grid_results = []
    best = None

    for limit in LENGTH_GRID:
        if combine_with_unk:
            result = evaluate(
                rows, lambda length, unk: unk > 0 or length <= limit
            )
        else:
            result = evaluate(
                rows, lambda length, unk: length <= limit
            )
        grid_results.append({"L": limit, **result})

        if best is None or result["f1"] > best[1]["f1"]:
            best = (limit, result)

    return best[0], best[1], grid_results


def main() -> None:
    if REPORT_PATH.exists():
        raise SystemExit(
            "Relatório já existe; remova-o deliberadamente para reexecutar."
        )

    # 1. Validar entradas.
    counts_manifest = load_accepted_manifest(COUNTS_MANIFEST_PATH)
    load_accepted_manifest(SPLIT_MANIFEST_PATH)

    counts_sha256 = sha256_file(COUNTS_PATH)
    split_sha256 = sha256_file(SPLIT_CSV_PATH)
    checks = {
        "counts_sha256_matches_stage4": (
            counts_sha256
            == counts_manifest["outputs"]["session_counts_csv_gz"]["sha256"]
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    split_index = load_split_index(SPLIT_CSV_PATH)
    expected_sessions = sum(
        1 for a in split_index.values() if a.split == "validation"
    )
    expected_anomalies = sum(
        1
        for a in split_index.values()
        if a.split == "validation" and a.label == LABEL_ANOMALY
    )

    # 2. Carregar a validação: (Length, unk, is_anomaly).
    rows: list[tuple[int, int, bool]] = []
    with gzip.open(
        COUNTS_PATH, "rt", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        for record in reader:
            if record["Split"] != "validation":
                continue
            assignment = split_index[record["BlockId"]]
            rows.append((
                int(record["Length"]),
                int(record["unk"]),
                assignment.label == LABEL_ANOMALY,
            ))

    anomalies = sum(1 for _, _, is_anomaly in rows if is_anomaly)
    checks["validation_sessions_match_split"] = (
        len(rows) == expected_sessions
    )
    checks["validation_anomalies_match_split"] = (
        anomalies == expected_anomalies
    )

    # 3. Avaliar as três regras pré-registradas.
    b0a = evaluate(rows, lambda length, unk: unk > 0)
    b0b_limit, b0b, b0b_grid = select_length_threshold(
        rows, combine_with_unk=False
    )
    b0c_limit, b0c, b0c_grid = select_length_threshold(
        rows, combine_with_unk=True
    )

    # 4. Relatório.
    accepted = all(checks.values())
    report = {
        "stage": "etapa5_incremento1_baselines_triviais",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0]},
        "protocol": {
            "partition_evaluated": "validation",
            "selection_rule": "max F1 (classe Anomaly); empate -> menor L",
            "length_grid": [LENGTH_GRID.start, LENGTH_GRID.stop - 1],
            "test_touched": False,
        },
        "inputs": {
            "counts_sha256": counts_sha256,
            "split_csv_sha256": split_sha256,
        },
        "validation": {
            "sessions": len(rows),
            "anomalies": anomalies,
            "anomaly_rate": round(anomalies / len(rows), 4),
        },
        "results": {
            "B0a_unk_presence": b0a,
            "B0b_length": {"selected_L": b0b_limit, **b0b},
            "B0c_unk_or_length": {"selected_L": b0c_limit, **b0c},
        },
        "grids": {
            "B0b": b0b_grid,
            "B0c": b0c_grid,
        },
        "acceptance": {"checks": checks, "accepted": accepted},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_tmp = Path(str(REPORT_PATH) + ".tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(report_tmp, REPORT_PATH)

    summary = {
        key: value
        for key, value in report.items()
        if key in ("validation", "results", "acceptance")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()