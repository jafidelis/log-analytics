"""Etapa 4, incremento 2: distribuição de comprimentos de sessão."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.session_hdfs_v1 import load_split_index

PARSED_EVENTS_PATH = Path("artifacts/parsing/hdfs_v1_parsed_events.csv.gz")
PARSING_MANIFEST_PATH = Path("artifacts/parsing/hdfs_v1_parsing_manifest.json")
SPLIT_CSV_PATH = Path("artifacts/data_splits/hdfs_v1_session_split.csv")

PERCENTILES = (25, 50, 75, 90, 99)


def summarize(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    count = len(ordered)
    stats = {
        "sessions": count,
        "events": sum(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / count, 2),
    }
    for p in PERCENTILES:
        index = min(count - 1, max(0, round(p / 100 * count) - 1))
        stats[f"p{p}"] = ordered[index]
    return stats


def main() -> None:
    manifest = json.loads(
        PARSING_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest["acceptance"]["accepted"] is not True:
        raise ValueError("Manifesto da etapa 3 não aprovado.")
    if (
        sha256_file(PARSED_EVENTS_PATH)
        != manifest["outputs"]["parsed_events_csv_gz"]["sha256"]
    ):
        raise ValueError("parsed_events diverge do manifesto da etapa 3.")

    split_index = load_split_index(SPLIT_CSV_PATH)

    lengths: defaultdict[str, int] = defaultdict(int)
    with gzip.open(
        PARSED_EVENTS_PATH, "rt", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.reader(stream)
        next(reader)
        for _line_id, block_id, _split, _template_id in reader:
            lengths[block_id] += 1

    # Agrupar por partição; rótulo somente no treino.
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for block_id, length in lengths.items():
        assignment = split_index[block_id]
        by_group[assignment.split].append(length)
        if assignment.split == "train":
            by_group[f"train/{assignment.label}"].append(length)

    for group in (
        "train", "train/Normal", "train/Anomaly", "validation", "test"
    ):
        print(f"{group:>15}: {summarize(by_group[group])}")

    longest = sorted(
        lengths.items(), key=lambda item: -item[1]
    )[:5]
    print("\n5 sessões mais longas:")
    for block_id, length in longest:
        assignment = split_index[block_id]
        print(
            f"  {block_id} | {length} eventos"
            f" | {assignment.split}/{assignment.label}"
        )

    missing = len(split_index) - len(lengths)
    print(f"\nSessões do split sem eventos: {missing}")

if __name__ == "__main__":
    main()