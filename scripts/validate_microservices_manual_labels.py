from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VALID_LABELS = {
    "anomalia",
    "nao_anomalia",
    "incerto",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_trace_manual_labels.csv"
        ),
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=Path(
            "../../data/processed/microservices_sample/"
            "microservices_traces_anonymized.csv"
        ),
    )

    parser.add_argument(
        "--expected-labels",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    checks = {
        "labels_file_exists": args.labels.exists(),
        "events_file_exists": args.events.exists(),
    }

    if not all(checks.values()):
        raise FileNotFoundError(
            f"Arquivos ausentes: {checks}"
        )

    labels = pd.read_csv(
        args.labels,
        dtype=str,
        keep_default_na=False,
    )

    events = pd.read_csv(
        args.events,
        dtype=str,
        usecols=["tc_trace_id"],
        keep_default_na=False,
    )

    required_columns = {
        "tc_trace_id",
        "label",
    }

    checks["labels_schema_valid"] = required_columns.issubset(
        labels.columns
    )

    if not checks["labels_schema_valid"]:
        raise ValueError(
            f"Colunas obrigatórias ausentes: "
            f"{required_columns - set(labels.columns)}"
        )

    labels["tc_trace_id"] = labels["tc_trace_id"].str.strip()
    labels["label"] = labels["label"].str.strip()

    event_trace_ids = set(
        events.loc[
            events["tc_trace_id"].str.strip() != "",
            "tc_trace_id",
        ]
    )

    label_trace_ids = set(labels["tc_trace_id"])

    checks["expected_label_count"] = bool(
      len(labels) == args.expected_labels
    )

    checks["trace_ids_not_empty"] = bool(
        labels["tc_trace_id"].ne("").all()
    )

    checks["trace_ids_unique"] = bool(
        labels["tc_trace_id"].is_unique
    )

    checks["labels_valid"] = bool(
        set(labels["label"]).issubset(VALID_LABELS)
    )

    checks["all_label_traces_exist_in_events"] = bool(
        label_trace_ids.issubset(event_trace_ids)
    )

    missing_trace_ids = sorted(
        label_trace_ids - event_trace_ids
    )

    label_counts = {
        str(label): int(count)
        for label, count in (
            labels["label"].value_counts().to_dict().items()
        )
    }

    manifest = {
        "stage": "microservices_manual_labels_validation",
        "inputs": {
            "labels": str(args.labels),
            "events": str(args.events),
            "labels_sha256": sha256_file(args.labels),
            "events_sha256": sha256_file(args.events),
        },
        "results": {
            "label_rows": int(len(labels)),
            "distinct_label_trace_ids": int(
                labels["tc_trace_id"].nunique()
            ),
            "event_trace_ids": int(len(event_trace_ids)),
            "label_distribution": label_counts,
            "missing_trace_ids": missing_trace_ids,
        },
        "acceptance": {
            "checks": checks,
            "accepted": all(checks.values()),
        },
    }

    manifest_path = args.labels.with_suffix(
        ".manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if not manifest["acceptance"]["accepted"]:
        raise SystemExit(
            "Validação rejeitada. Consulte o manifesto."
        )


if __name__ == "__main__":
    main()