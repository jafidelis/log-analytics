import gzip
import json
from pathlib import Path

import pytest

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.load_representation_hdfs_v1 import (
    load_session_counts,
    load_train_normal_ids,
    select_train_normal,
)

COLUMNS = ["BlockId", "Split", "Length", "t1", "t2", "unk"]
ROWS = [
    ["blk_1", "train", "3", "2", "1", "0"],
    ["blk_2", "validation", "2", "0", "1", "1"],
    ["blk_3", "train", "1", "1", "0", "0"],
]


def build_fixture(tmp_path: Path, rows=None) -> tuple[Path, Path]:
    counts_path = tmp_path / "counts.csv.gz"
    lines = [",".join(COLUMNS)] + [
        ",".join(row) for row in (rows if rows is not None else ROWS)
    ]
    with gzip.open(counts_path, "wt", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "representation": {"columns": COLUMNS},
        "outputs": {"session_counts_csv_gz": {
            "sha256": sha256_file(counts_path),
        }},
        "acceptance": {"checks": {"ok": True}, "accepted": True},
    }), encoding="utf-8")

    return counts_path, manifest_path


def test_loads_partition_in_order(tmp_path):
    counts_path, manifest_path = build_fixture(tmp_path)
    counts = load_session_counts(counts_path, manifest_path, "train")

    assert counts.block_ids == ["blk_1", "blk_3"]
    assert counts.features == [[2, 1, 0], [1, 0, 0]]
    assert counts.lengths == [3, 1]


def test_rejects_checksum_divergence(tmp_path):
    counts_path, manifest_path = build_fixture(tmp_path)
    with gzip.open(counts_path, "at", encoding="utf-8") as f:
        f.write("blk_9,train,1,1,0,0\n")

    with pytest.raises(ValueError, match="diverge"):
        load_session_counts(counts_path, manifest_path, "train")


def test_rejects_unaccepted_manifest(tmp_path):
    counts_path, manifest_path = build_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["acceptance"]["accepted"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="não aprovado"):
        load_session_counts(counts_path, manifest_path, "train")


def test_rejects_sum_mismatch(tmp_path):
    rows = [["blk_1", "train", "5", "2", "1", "0"]]
    counts_path, manifest_path = build_fixture(tmp_path, rows)

    with pytest.raises(ValueError, match="não somam"):
        load_session_counts(counts_path, manifest_path, "train")


def test_select_train_normal_filters_and_validates(tmp_path):
    counts_path, manifest_path = build_fixture(tmp_path)
    counts = load_session_counts(counts_path, manifest_path, "train")

    subset = select_train_normal(counts, frozenset({"blk_3"}))
    assert subset.block_ids == ["blk_3"]

    with pytest.raises(ValueError, match="ausentes"):
        select_train_normal(counts, frozenset({"blk_3", "blk_99"}))