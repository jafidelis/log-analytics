from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "event_count",
    "info_count",
    "warn_count",
    "error_count",
    "has_stack_trace",
    "has_exception_term",
    "has_failed_term",
    "has_timeout_term",
    "has_error_term",
    "service_count",
]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_trace_signals.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_trace_features.csv"
        ),
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(args.input)

    if signals["tc_trace_id"].duplicated().any():
        raise ValueError("Existem tc_trace_id duplicados.")

    missing = [
        column
        for column in BASE_COLUMNS
        if column not in signals.columns
    ]

    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    for column in BASE_COLUMNS:
        signals[column] = pd.to_numeric(
            signals[column],
            errors="raise",
        )

    features = signals[["tc_trace_id", *BASE_COLUMNS]].copy()

    features["error_rate"] = (
        features["error_count"]
        / features["event_count"]
    )

    features["warn_rate"] = (
        features["warn_count"]
        / features["event_count"]
    )

    text_columns = [
        "has_exception_term",
        "has_failed_term",
        "has_timeout_term",
        "has_error_term",
    ]

    features["text_signal_count"] = features[
        text_columns
    ].sum(axis=1)

    features["has_any_signal"] = (
        (features["error_count"] > 0)
        | (features["warn_count"] > 0)
        | (features["has_stack_trace"] > 0)
        | (features["text_signal_count"] > 0)
    ).astype(int)

    features.to_csv(
        args.output,
        index=False,
    )

    manifest = {
        "stage": "microservices_trace_feature_matrix",
        "input": str(args.input),
        "output": str(args.output),
        "traces": int(len(features)),
        "events": int(features["event_count"].sum()),
        "feature_columns": [
            column
            for column in features.columns
            if column != "tc_trace_id"
        ],
        "labels_in_features": False,
        "trace_id_preserved": True,
    }

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()