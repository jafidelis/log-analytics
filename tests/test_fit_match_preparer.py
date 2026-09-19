from log_analytics.data.fit_hdfs_v1 import (
    build_demo_parser,
    learn_if_train_normal,
)
from log_analytics.data.match_hdfs_v1 import (
    UNK_TEMPLATE,
    match_hdfs_message,
)
from log_analytics.data.prepare_hdfs_v1 import (
    prepare_hdfs_message_candidate_c,
)
from log_analytics.data.session_hdfs_v1 import associate_log_line
from log_analytics.data.split_hdfs_v1 import SplitAssignment

SPLIT_INDEX = {"blk_1": SplitAssignment("blk_1", "Normal", "train")}

ALLOCATE_FIRST_JOB = (
    "BLOCK* NameSystem.allocateBlock: "
    "/mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_1"
)
ALLOCATE_OTHER_JOB = (
    "BLOCK* NameSystem.allocateBlock: "
    "/mnt/hadoop/mapred/system/job_200811101024_0015/job.jar. blk_1"
)
EXPECTED_TEMPLATE = (
    "BLOCK* NameSystem.allocateBlock: "
    "/mnt/hadoop/mapred/system/<JOB_ID>/job.jar. <BLOCK_ID>"
)


def build_item(message: str):
    line = f"081109 203518 143 INFO dfs.Component: {message}"
    return associate_log_line(line, 1, SPLIT_INDEX)


def test_learn_uses_injected_preparer() -> None:
    parser = build_demo_parser()
    learn_if_train_normal(
        parser,
        build_item(ALLOCATE_FIRST_JOB),
        prepare_hdfs_message_candidate_c,
    )

    templates = [c.get_template() for c in parser.drain.clusters]
    assert templates == [EXPECTED_TEMPLATE]


def test_match_uses_injected_preparer() -> None:
    parser = build_demo_parser()
    learn_if_train_normal(
        parser,
        build_item(ALLOCATE_FIRST_JOB),
        prepare_hdfs_message_candidate_c,
    )

    cluster_id, template = match_hdfs_message(
        parser, ALLOCATE_OTHER_JOB, prepare_hdfs_message_candidate_c
    )
    assert cluster_id == 1
    assert template == EXPECTED_TEMPLATE


def test_match_with_default_preparer_misses_job_template() -> None:
    parser = build_demo_parser()
    learn_if_train_normal(
        parser,
        build_item(ALLOCATE_FIRST_JOB),
        prepare_hdfs_message_candidate_c,
    )

    result = match_hdfs_message(parser, ALLOCATE_OTHER_JOB)
    assert result == (None, UNK_TEMPLATE)