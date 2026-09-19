import pytest

from log_analytics.data.prepare_hdfs_v1 import (
    prepare_hdfs_message,
    prepare_hdfs_message_candidate,
    prepare_hdfs_message_candidate_c,
    prepare_hdfs_message_candidate_d,
)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (
            "Receiving blk_123 blk_-456",
            "Receiving <BLOCK_ID> <BLOCK_ID>",
        ),
        (
            "src: /10.0.0.1:54106 dest: /10.0.0.2:50010",
            "src: /<IP>:54106 dest: /<IP>:50010",
        ),
        (
            "version 1.2.3.4 error 500 retry 3",
            "version 1.2.3.4 error 500 retry 3",
        ),
        (
            "src: /999.0.0.1:80",
            "src: /999.0.0.1:80",
        ),
        (
            "src: /1.2.3.4.5:80",
            "src: /1.2.3.4.5:80",
        ),
        (
            "src: /1.2.3.4:80abc",
            "src: /1.2.3.4:80abc",
        ),
        (
            "/data/blk_123/file",
            "/data/<BLOCK_ID>/file",
        ),
        (
            "prefix_blk_123 blk_123_suffix",
            "prefix_blk_123 blk_123_suffix",
        ),
    ],
)
def test_preparation(original: str, expected: str) -> None:
    prepared = prepare_hdfs_message(original)

    assert prepared == expected
    assert prepare_hdfs_message(prepared) == prepared

@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (
            "10.250.14.224:50010:Transmitted block blk_1 "
            "to /10.251.215.16:50010",
            "<IP>:50010:Transmitted block <BLOCK_ID> "
            "to /<IP>:50010",
        ),
        (
            "Starting transfer to 10.1.2.3:50010",
            "Starting transfer to <IP>:50010",
        ),
        (
            "Received from /10.1.2.3",
            "Received from /10.1.2.3",
        ),
        (
            "version 1.2.3.4",
            "version 1.2.3.4",
        ),
        (
            "endpoint 999.1.2.3:50010",
            "endpoint 999.1.2.3:50010",
        ),
    ],
)
def test_candidate_preparation(
    original: str,
    expected: str,
) -> None:
    prepared = prepare_hdfs_message_candidate(original)

    assert prepared == expected
    assert prepare_hdfs_message_candidate(prepared) == prepared

@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (
            "job_200811092030_0001",
            "<JOB_ID>",
        ),
        (
            "/mnt/hadoop/mapred/system/"
            "job_200811092030_0001/job.jar",
            "/mnt/hadoop/mapred/system/<JOB_ID>/job.jar",
        ),
        (
            "job_200811092030_0001 "
            "job_200811092030_0002",
            "<JOB_ID> <JOB_ID>",
        ),
        (
            "prefix_job_200811092030_0001",
            "prefix_job_200811092030_0001",
        ),
        (
            "job_200811092030_0001_suffix",
            "job_200811092030_0001_suffix",
        ),
        (
            "job_20081109203_0001",
            "job_20081109203_0001",
        ),
        (
            "job_200811092030_00010",
            "job_200811092030_00010",
        ),
        (
            "job_200811092030_0001.jar",
            "job_200811092030_0001.jar",
        ),
        (
            "Receiving block blk_123 "
            "from 10.20.30.40:50010 "
            "job_200811092030_0001",
            "Receiving block <BLOCK_ID> "
            "from <IP>:50010 "
            "<JOB_ID>",
        ),
    ],
)
def test_candidate_c_preparation(
    original: str,
    expected: str,
) -> None:
    prepared = prepare_hdfs_message_candidate_c(original)

    assert prepared == expected
    assert prepare_hdfs_message_candidate_c(prepared) == prepared

def test_candidate_d_masks_bare_ipv4_with_slash() -> None:
    message = "Received block blk_1 of size 67108864 from /10.250.14.38"
    assert prepare_hdfs_message_candidate_d(message) == (
        "Received block <BLOCK_ID> of size 67108864 from /<IP>"
    )

def test_candidate_d_keeps_port_of_masked_endpoints() -> None:
    message = "Receiving block blk_1 src: /10.0.0.1:54106 dest: /10.0.0.2:50010"
    assert prepare_hdfs_message_candidate_d(message) == (
        "Receiving block <BLOCK_ID> src: /<IP>:54106 dest: /<IP>:50010"
    )

def test_candidate_d_masks_allocate_block_path() -> None:
    message = (
        "BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/"
        "_task_200811092030_0001_m_000079_0/part-00079. blk_1"
    )
    assert prepare_hdfs_message_candidate_d(message) == (
        "BLOCK* NameSystem.allocateBlock: <PATH> <BLOCK_ID>"
    )

def test_candidate_d_masks_path_containing_job_id() -> None:
    message = (
        "BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/"
        "job_200811092030_0001/job.jar. blk_1"
    )
    assert prepare_hdfs_message_candidate_d(message) == (
        "BLOCK* NameSystem.allocateBlock: <PATH> <BLOCK_ID>"
    )

def test_candidate_d_does_not_mask_invalid_ipv4() -> None:
    message = "Version 999.999.999.999 reported"
    assert prepare_hdfs_message_candidate_d(message) == message

def test_candidate_d_does_not_mask_single_segment_token() -> None:
    message = "Deleting file /tmp"
    assert prepare_hdfs_message_candidate_d(message) == message