from drain3 import TemplateMiner
from typing import Callable

from log_analytics.data.prepare_hdfs_v1 import prepare_hdfs_message

UNK_TEMPLATE = "UNK_TEMPLATE"

def match_hdfs_message(
    parser: TemplateMiner,
    message: str,
    prepare: Callable[[str], str] = prepare_hdfs_message,
) -> tuple[int | None, str]:
    """Reconhece uma mensagem sem aprender novos templates."""
    prepared = prepare(message)

    cluster = parser.match(
        prepared,
        full_search_strategy="always",
    )

    if cluster is None:
        return None, UNK_TEMPLATE

    return cluster.cluster_id, cluster.get_template()