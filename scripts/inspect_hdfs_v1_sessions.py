from itertools import islice
from pathlib import Path

from log_analytics.data.session_hdfs_v1 import (
    associate_log_line,
    load_split_index,
)

def main() -> None:
    split_index = load_split_index(
        Path("artifacts/data_splits/hdfs_v1_session_split.csv")
    )
    print(f"Sessões carregadas: {len(split_index)}")

    log_path = Path("data/raw/hdfs_v1/HDFS.log")

    with log_path.open(encoding="utf-8") as stream:
        for line_id, line in enumerate(islice(stream, 3), start=1):
            result = associate_log_line(line, line_id, split_index)
            assignment = result.assignment

            print({
                "line_id": result.event.line_id,
                "block_id": assignment.block_id if assignment else None,
                "split": assignment.split if assignment else None,
                "is_train_normal": result.is_train_normal,
                "severity": result.event.level,
                "component": result.event.component,
                "message": result.event.content,
            })


if __name__ == "__main__":
    main()