import json
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import (
    DatasetExpectations,
    audit_dataset,
    extract_block_ids,
    parse_log_line,
    write_manifest,
)


def test_parse_log_line_and_timestamp() -> None:
    line = (
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
        "Receiving block blk_-1608999687919862906"
    )

    event = parse_log_line(line, line_id=1)

    assert event is not None
    assert event.line_id == 1
    assert event.pid == 143
    assert event.level == "INFO"
    assert event.component == "dfs.DataNode$DataXceiver"
    assert event.timestamp == "2008-11-09T20:35:18"


def test_extract_block_ids_removes_only_repetitions() -> None:
    content = "copy blk_10 to blk_-20; confirmation blk_10"

    assert extract_block_ids(content) == ("blk_10", "blk_-20")


def test_audit_dataset_with_synthetic_files(tmp_path: Path) -> None:
    labels_path = tmp_path / "anomaly_label.csv"
    labels_path.write_text(
        "BlockId,Label\nblk_10,Normal\nblk_-20,Anomaly\n",
        encoding="utf-8",
    )

    log_path = tmp_path / "HDFS.log"
    log_path.write_text(
        "081109 203518 143 INFO dfs.Component: copy blk_10 to blk_-20 blk_10\n"
        "081109 203519 144 INFO dfs.Component: message without session\n",
        encoding="utf-8",
    )

    expected = DatasetExpectations(
        total_log_lines=2,
        unique_labels=2,
        normal_labels=1,
        anomaly_labels=1,
    )

    manifest = audit_dataset(log_path, labels_path, expected)

    assert manifest["acceptance"]["accepted"] is True
    assert manifest["log"]["event_session_associations"] == 2
    assert manifest["log"]["lines_with_multiple_distinct_ids"] == 1
    assert manifest["log"]["lines_with_repeated_id_mentions"] == 1
    assert manifest["log"]["lines_without_block_id"] == 1


def test_unparseable_line_is_counted_and_reproves_audit(tmp_path: Path) -> None:
    labels_path = tmp_path / "anomaly_label.csv"
    labels_path.write_text("BlockId,Label\nblk_10,Normal\n", encoding="utf-8")

    log_path = tmp_path / "HDFS.log"
    log_path.write_text("invalid header with blk_10\n", encoding="utf-8")

    expected = DatasetExpectations(
        total_log_lines=1,
        unique_labels=1,
        normal_labels=1,
        anomaly_labels=0,
    )

    manifest = audit_dataset(log_path, labels_path, expected)

    assert manifest["acceptance"]["accepted"] is False
    assert manifest["log"]["unparseable_header_lines"] == 1
    assert manifest["log"]["labels_without_event"] == 1


def test_write_manifest_creates_valid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "audit.json"
    manifest = {"acceptance": {"accepted": True}}

    write_manifest(manifest, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest