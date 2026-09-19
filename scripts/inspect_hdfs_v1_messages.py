from itertools import islice
from pathlib import Path

from log_analytics.data.prepare_hdfs_v1 import prepare_hdfs_message
from log_analytics.data.session_hdfs_v1 import (
    associate_log_line,
    load_split_index,
)


def main() -> None:
    split_index = load_split_index(
        Path("artifacts/data_splits/hdfs_v1_session_split.csv")
    )
    log_path = Path("data/raw/hdfs_v1/HDFS.log")

    with log_path.open(encoding="utf-8") as stream:
        for line_id, line in enumerate(islice(stream, 3), start=1):
            result = associate_log_line(line, line_id, split_index)

            if not result.is_train_normal:
                raise ValueError(
                    f"Linha {line_id}: fora da amostra de treino normal."
                )

            original = result.event.content
            prepared = prepare_hdfs_message(original)

            print(f"Linha {line_id}")
            print(f"Original:  {original}")
            print(f"Preparada: {prepared}")
            print()


if __name__ == "__main__":
    main()