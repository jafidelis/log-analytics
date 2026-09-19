from log_analytics.data.fit_hdfs_v1 import (
    build_demo_parser,
    learn_if_train_normal,
)
from log_analytics.data.session_hdfs_v1 import associate_log_line
from log_analytics.data.split_hdfs_v1 import SplitAssignment

def catalog_snapshot(parser):
    return tuple(
        (cluster.cluster_id, cluster.get_template(), cluster.size)
        for cluster in sorted(
            parser.drain.clusters,
            key=lambda item: item.cluster_id,
        )
    )

def main() -> None:
    split_index = {
        "blk_1": SplitAssignment("blk_1", "Normal", "train"),
        "blk_2": SplitAssignment("blk_2", "Anomaly", "train"),
        "blk_3": SplitAssignment("blk_3", "Normal", "validation"),
        "blk_4": SplitAssignment("blk_4", "Normal", "test"),
        "blk_5": SplitAssignment("blk_5", "Anomaly", "validation"),
        "blk_6": SplitAssignment("blk_6", "Anomaly", "test"),
    }

    # Cada exemplo contém: severidade, mensagem, deve aprender?
    examples = [
        (
            "INFO",
            "Receiving block blk_1 "
            "src: /10.0.0.1:54106 dest: /10.0.0.2:50010",
            True,
        ),
        (
            "INFO",
            "Receiving block blk_2 "
            "src: /10.0.0.1:12345 dest: /10.0.0.2:50010",
            False,
        ),
        ("INFO", "Validation event blk_3", False),
        ("INFO", "Test event blk_4", False),
        ("ERROR", "Validation failure blk_5", False),
        ("ERROR", "Test failure blk_6", False),
        ("INFO", "Background event without session", False),
        (
            "ERROR",
            "Receiving block blk_1 "
            "src: /10.0.0.1:40524 dest: /10.0.0.2:50010",
            True,
        ),
    ]

    parser = build_demo_parser()
    learned = 0

    for line_id, (severity, content, expected) in enumerate(
        examples, start=1
    ):
        line = f"081109 203518 143 {severity} dfs.Component: {content}"
        item = associate_log_line(line, line_id, split_index)

        before = catalog_snapshot(parser)
        result = learn_if_train_normal(parser, item)

        assert (result is not None) == expected

        if result is None:
            assert catalog_snapshot(parser) == before
            status = "fora do ajuste; catálogo preservado"
        else:
            learned += 1
            status = result["change_type"]

        print(f"Evento {line_id}: {status}")

    assert learned == 2
    assert catalog_snapshot(parser) == (
        (
            1,
            "Receiving block <BLOCK_ID> src: <*> dest: /<IP>:50010",
            2,
        ),
    )

    print(f"Eventos usados no ajuste: {learned}")
    print(f"Catálogo final: {catalog_snapshot(parser)}")


if __name__ == "__main__":
    main()