from decimal import Decimal
from pathlib import Path

import pytest

from log_analytics.data.split_hdfs_v1 import (
    LABEL_ANOMALY,
    LABEL_NORMAL,
    SplitConfig,
    allocate_largest_remainder,
    create_stratified_split,
    stable_digest,
    write_split_csv,
    write_train_normal,
)


def test_largest_remainder_matches_hdfs_counts() -> None:
    config = SplitConfig()

    assert allocate_largest_remainder(558_223, config.ratios) == {
        "train": 390_756,
        "validation": 83_734,
        "test": 83_733,
    }
    assert allocate_largest_remainder(16_838, config.ratios) == {
        "train": 11_786,
        "validation": 2_526,
        "test": 2_526,
    }


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="soma"):
        allocate_largest_remainder(
            10,
            {
                "train": Decimal("0.80"),
                "validation": Decimal("0.15"),
                "test": Decimal("0.15"),
            },
        )


def test_digest_contract_is_stable() -> None:
    assert stable_digest(
        "blk_10",
        "hdfs-v1-master-split-v1",
    ).hex() == "0593608f90fc65666b7f94a0fb7987155065b8ccd2a073385ca254d1184dfa05"


def test_input_order_does_not_change_assignments() -> None:
    labels_a = {
        "blk_1": LABEL_NORMAL,
        "blk_2": LABEL_NORMAL,
        "blk_3": LABEL_ANOMALY,
        "blk_4": LABEL_ANOMALY,
        "blk_5": LABEL_NORMAL,
        "blk_6": LABEL_ANOMALY,
    }
    labels_b = dict(reversed(list(labels_a.items())))
    config = SplitConfig(
        train_ratio=Decimal("0.50"),
        validation_ratio=Decimal("0.25"),
        test_ratio=Decimal("0.25"),
    )

    assert create_stratified_split(labels_a, config) == create_stratified_split(
        labels_b,
        config,
    )


def test_each_session_receives_one_split() -> None:
    labels = {
        "blk_1": LABEL_NORMAL,
        "blk_2": LABEL_NORMAL,
        "blk_3": LABEL_ANOMALY,
        "blk_4": LABEL_ANOMALY,
    }
    config = SplitConfig(
        train_ratio=Decimal("0.50"),
        validation_ratio=Decimal("0.25"),
        test_ratio=Decimal("0.25"),
    )

    assignments = create_stratified_split(labels, config)

    assert len(assignments) == len(labels)
    assert {item.block_id for item in assignments} == set(labels)
    assert len({item.block_id for item in assignments}) == len(assignments)


def test_outputs_are_byte_identical(tmp_path: Path) -> None:
    labels = {
        "blk_1": LABEL_NORMAL,
        "blk_2": LABEL_NORMAL,
        "blk_3": LABEL_ANOMALY,
        "blk_4": LABEL_ANOMALY,
    }
    config = SplitConfig(
        train_ratio=Decimal("0.50"),
        validation_ratio=Decimal("0.25"),
        test_ratio=Decimal("0.25"),
    )
    assignments = create_stratified_split(labels, config)
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    first_normal = tmp_path / "first.txt"
    second_normal = tmp_path / "second.txt"

    assert write_split_csv(assignments, first_csv) == write_split_csv(
        assignments,
        second_csv,
    )
    assert write_train_normal(assignments, first_normal) == write_train_normal(
        assignments,
        second_normal,
    )
    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_normal.read_bytes() == second_normal.read_bytes()


def test_divergent_existing_artifact_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "split.csv"
    output.write_text("conteudo anterior\n", encoding="utf-8")
    labels = {"blk_1": LABEL_NORMAL, "blk_2": LABEL_ANOMALY}
    config = SplitConfig(
        train_ratio=Decimal("0.50"),
        validation_ratio=Decimal("0.25"),
        test_ratio=Decimal("0.25"),
    )

    with pytest.raises(FileExistsError):
        write_split_csv(create_stratified_split(labels, config), output)

    assert output.read_text(encoding="utf-8") == "conteudo anterior\n"