from typing import Any, Callable

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.persistence_handler import PersistenceHandler

from log_analytics.data.prepare_hdfs_v1 import prepare_hdfs_message
from log_analytics.data.session_hdfs_v1 import SessionLogEvent

def build_demo_parser(
    persistence_handler: PersistenceHandler | None = None,
) -> TemplateMiner:
    """Cria um parser didático, com persistência opcional."""
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.drain_sim_th = 0.8
    config.drain_extra_delimiters = []
    config.masking_instructions = []
    config.parametrize_numeric_tokens = False

    return TemplateMiner(
        persistence_handler=persistence_handler,
        config=config,
    )

def learn_if_train_normal(
      parser: TemplateMiner,
      item: SessionLogEvent,
      prepare: Callable[[str], str] = prepare_hdfs_message,
  ) -> dict[str, Any] | None:
    """Atualiza o parser somente com eventos de sessões normais do treino."""
    if not item.is_train_normal:
        return None

    return parser.add_log_message(prepare(item.event.content))