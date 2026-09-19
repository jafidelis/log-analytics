import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from log_analytics.data.audit_hdfs_v1 import (
    BLOCK_ID_PATTERN,
    LogEvent,
    extract_block_ids,
    parse_log_line,
)
from log_analytics.data.split_hdfs_v1 import (
    LABEL_ANOMALY,
    LABEL_NORMAL,
    SPLIT_ORDER,
    SPLIT_TRAIN,
    SplitAssignment,
)

@dataclass(frozen=True)
class SessionLogEvent:
    event: LogEvent
    assignment: SplitAssignment | None

    @property
    def is_train_normal(self) -> bool:
        return (
            self.assignment is not None
            and self.assignment.split == SPLIT_TRAIN
            and self.assignment.label == LABEL_NORMAL
        )

def load_split_index(path: Path) -> dict[str, SplitAssignment]:
    """Carrega o CSV mestre para consulta por BlockId."""
    index: dict[str, SplitAssignment] = {}
      
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)

        if next(reader, None) != ["BlockId", "Label", "Split"]:
            raise ValueError("Cabeçalho inválido no CSV mestre.")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(
                    f"Linha {row_number}: esperado 3 campos."
                )
            
            block_id, label, split = row

            if BLOCK_ID_PATTERN.fullmatch(block_id) is None:
                raise ValueError(
                    f"Linha {row_number}: BlockId inválido."
                )

            if label not in (LABEL_NORMAL, LABEL_ANOMALY):
                raise ValueError(
                    f"Linha {row_number}: rótulo inválido."
                )

            if split not in SPLIT_ORDER:
                raise ValueError(
                    f"Linha {row_number}: partição inválida."
                )

            if block_id in index:
                raise ValueError(
                    f"Linha {row_number}: BlockId duplicado."
                )

            index[block_id] = SplitAssignment(block_id, label, split)

    if not index:
        raise ValueError("CSV mestre sem sessões.")

    return index

def associate_log_line(
    line: str,
    line_id: int,
    split_index: Mapping[str, SplitAssignment],
) -> SessionLogEvent:
    """Associa um evento à divisão existente, preservando seu conteúdo."""
    event = parse_log_line(line.rstrip("\r\n"), line_id)

    if event is None:
        raise ValueError(f"Linha {line_id}: cabeçalho inválido.")

    block_ids = extract_block_ids(event.content)

    if not block_ids:
        return SessionLogEvent(event, assignment=None)

    if len(block_ids) > 1:
        raise ValueError(
            f"Linha {line_id}: mais de um BlockId distinto."
        )

    block_id = block_ids[0]

    if block_id not in split_index:
        raise ValueError(
            f"Linha {line_id}: {block_id} ausente no split."
        )

    return SessionLogEvent(event, split_index[block_id])
    