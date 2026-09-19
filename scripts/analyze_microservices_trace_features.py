from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SIGNAL_COLUMNS = [
    "error_count",
    "warn_count",
    "has_stack_trace",
    "text_signal_count",
    "has_any_signal",
]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_trace_features.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/figures/microservices_exploratory"
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.input)

    if features["tc_trace_id"].duplicated().any():
        raise ValueError("Existem tc_trace_id duplicados.")

    # 1. Cobertura dos sinais
    coverage = pd.DataFrame(
        {
            "signal": SIGNAL_COLUMNS,
            "traces_with_signal": [
                int(features[column].gt(0).sum())
                for column in SIGNAL_COLUMNS
            ],
        }
    )

    coverage["percentage"] = (
        coverage["traces_with_signal"]
        / len(features)
        * 100
    )

    coverage.to_csv(
        args.output_dir / "signal_coverage.csv",
        index=False,
    )

    # 2. Estatísticas descritivas
    numeric = features.drop(
        columns=["tc_trace_id"],
    )

    descriptive = numeric.describe().T
    descriptive.to_csv(
        args.output_dir / "feature_descriptive_statistics.csv"
    )

    # 3. Combinações de sinais
    def signal_profile(row):
        profile = []

        if row["error_count"] > 0:
            profile.append("ERROR")

        if row["warn_count"] > 0:
            profile.append("WARN")

        if row["has_stack_trace"] > 0:
            profile.append("STACK_TRACE")

        if row["text_signal_count"] > 0:
            profile.append("TEXT")

        return "+".join(profile) if profile else "NONE"

    features["signal_profile"] = features.apply(
        signal_profile,
        axis=1,
    )

    combinations = (
        features["signal_profile"]
        .value_counts()
        .rename_axis("signal_profile")
        .reset_index(name="trace_count")
    )

    combinations["percentage"] = (
        combinations["trace_count"]
        / len(features)
        * 100
    )

    combinations.to_csv(
        args.output_dir / "signal_combinations.csv",
        index=False,
    )

    # 4. Correlação entre atributos
    correlation = numeric.corr(numeric_only=True)

    correlation.to_csv(
        args.output_dir / "feature_correlation.csv"
    )

    # 5. Gráfico de cobertura
    plt.figure(figsize=(9, 5))

    plt.bar(
        coverage["signal"],
        coverage["traces_with_signal"],
    )

    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Quantidade de traces")
    plt.title("Cobertura dos sinais por trace")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        args.output_dir / "signal_coverage.png",
        dpi=300,
    )
    plt.close()

    # 6. Gráfico de combinações
    top_combinations = combinations.head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_combinations["signal_profile"].iloc[::-1],
        top_combinations["trace_count"].iloc[::-1],
    )

    plt.xlabel("Quantidade de traces")
    plt.ylabel("Perfil de sinais")
    plt.title("Combinações mais frequentes de sinais")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        args.output_dir / "signal_combinations.png",
        dpi=300,
    )
    plt.close()

    # 7. Distribuição do tamanho dos traces
    plt.figure(figsize=(8, 5))

    plt.hist(
        features["event_count"],
        bins=30,
    )

    plt.xlabel("Eventos por trace")
    plt.ylabel("Quantidade de traces")
    plt.title("Distribuição do tamanho dos traces")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        args.output_dir / "trace_event_count.png",
        dpi=300,
    )
    plt.close()

    manifest = {
        "stage": "microservices_trace_feature_exploration",
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "traces_analyzed": int(len(features)),
        "feature_count": int(len(numeric.columns)),
        "labels_in_features": False,
        "outputs": [
            "signal_coverage.csv",
            "feature_descriptive_statistics.csv",
            "signal_combinations.csv",
            "feature_correlation.csv",
            "signal_coverage.png",
            "signal_combinations.png",
            "trace_event_count.png",
        ],
    }

    manifest_path = (
        args.output_dir
        / "microservices_feature_exploration_manifest.json"
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


if __name__ == "__main__":
    main()