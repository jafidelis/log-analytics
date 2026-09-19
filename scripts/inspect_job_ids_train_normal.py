from collections import Counter
from pathlib import Path

from log_analytics.data.session_hdfs_v1 import associate_log_line, load_split_index
from log_analytics.data.prepare_hdfs_v1 import JOB_ID_PATTERN, prepare_hdfs_message_candidate_c

HDFS_LOG_PATH = Path("./data/raw/hdfs_v1/HDFS.log")            # mesmo caminho usado no piloto
SPLIT_CSV_PATH = Path("./artifacts/data_splits/hdfs_v1_session_split.csv")  # mesmo caminho usado no piloto
MAX_EXAMPLES = 5


def main() -> None:
    split_index = load_split_index(SPLIT_CSV_PATH)

    total_train_normal = 0
    lines_with_job_id = 0
    job_counter: Counter[str] = Counter()
    sessions_with_job_id: set[str] = set()
    examples: list[tuple[str, str]] = []

    with HDFS_LOG_PATH.open("r", encoding="ascii") as handle:
        for line_id, line in enumerate(handle, start=1):
            event = associate_log_line(line, line_id, split_index)
            if not event.is_train_normal:
                continue
            total_train_normal += 1
            message = event.event.content
            job_ids = JOB_ID_PATTERN.findall(message)
            if not job_ids:
                continue
            lines_with_job_id += 1
            job_counter.update(job_ids)
            sessions_with_job_id.add(event.event.line_id)  # ajuste ao seu acesso real ao BlockId
            if len(examples) < MAX_EXAMPLES:
                examples.append((message, prepare_hdfs_message_candidate_c(message)))

    print(f"Eventos train_normal percorridos: {total_train_normal}")
    print(f"Linhas com JobId: {lines_with_job_id}")
    print(f"JobIds distintos: {len(job_counter)}")
    print(f"Sessões com JobId: {len(sessions_with_job_id)}")
    print("Top 10 JobIds:")
    for job_id, count in job_counter.most_common(10):
        print(f"  {job_id}: {count}")
    print("Exemplos antes/depois da máscara C:")
    for original, prepared in examples:
        print(f"  - {original}")
        print(f"    {prepared}")


if __name__ == "__main__":
    main()