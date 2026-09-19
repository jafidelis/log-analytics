from pathlib import Path
from tempfile import TemporaryDirectory

import jsonpickle
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.fit_hdfs_v1 import (
    build_demo_parser,
    learn_if_train_normal,
)
from log_analytics.data.match_hdfs_v1 import (
    UNK_TEMPLATE,
    match_hdfs_message,
)
from log_analytics.data.session_hdfs_v1 import associate_log_line
from log_analytics.data.split_hdfs_v1 import SplitAssignment


def catalog(parser: TemplateMiner) -> tuple:
    return tuple(
        (item.cluster_id, item.get_template(), item.size)
        for item in sorted(
            parser.drain.clusters,
            key=lambda item: item.cluster_id,
        )
    )


def run_demo(state_path: Path) -> None:
    training_parser = build_demo_parser(
        FilePersistence(str(state_path))
    )
    split_index = {
        "blk_1": SplitAssignment("blk_1", "Normal", "train"),
    }

    for line_id, source_port in enumerate(
        (54106, 40524, 40524), start=1
    ):
        line = (
            "081109 203518 143 INFO dfs.Component: "
            f"Receiving block blk_1 src: /10.0.0.1:{source_port} "
            "dest: /10.0.0.2:50010"
        )
        item = associate_log_line(line, line_id, split_index)
        result = learn_if_train_normal(training_parser, item)

        assert result is not None
        print(f"Ajuste {line_id}: {result['change_type']}")

    training_parser.save_state("fit_complete")

    assert state_path.is_file() and state_path.stat().st_size > 0
    saved_sha256 = sha256_file(state_path)
    print("Estado salvo em disco.")

    restored = build_demo_parser(
        FilePersistence(str(state_path))
    )

    expected_template = (
        "Receiving block <BLOCK_ID> src: <*> dest: /<IP>:50010"
    )
    assert catalog(restored) == catalog(training_parser) == (
        (1, expected_template, 3),
    )
    print("Restauração: mesmo catálogo, com 3 mensagens de ajuste.")

    before = jsonpickle.dumps(restored.drain, keys=True)

    examples = [
        (
            "Receiving block blk_99 "
            "src: /10.0.0.9:60001 dest: /10.0.0.8:50010",
            (1, expected_template),
        ),
        (
            "Receiving block blk_99 "
            "src: /10.0.0.9:60001 dest: /10.0.0.8:60000",
            (None, UNK_TEMPLATE),
        ),
    ]

    for number, (message, expected) in enumerate(examples, start=1):
        result = match_hdfs_message(restored, message)

        assert result == expected
        assert jsonpickle.dumps(restored.drain, keys=True) == before
        assert sha256_file(state_path) == saved_sha256

        print(f"Consulta {number}: {result}")

    print("Estado em memória preservado.")
    print("Checksum do arquivo preservado.")
    print(f"Catálogo final: {catalog(restored)}")


def main() -> None:
    with TemporaryDirectory(prefix="drain-increment5-") as directory:
        run_demo(Path(directory) / "parser.bin")

    print("Arquivo temporário removido ao encerrar o exercício.")


if __name__ == "__main__":
    main()