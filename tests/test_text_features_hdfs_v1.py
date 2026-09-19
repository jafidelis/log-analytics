from log_analytics.data.text_features_hdfs_v1 import (
    analyze_log_document,
    tokenize_log_event,
)


def test_preserves_placeholders() -> None:
    assert tokenize_log_event(
        "<BLOCK_ID> <JOB_ID> <IP> <PATH> <*>"
    ) == [
        "<BLOCK_ID>",
        "<JOB_ID>",
        "<IP>",
        "<PATH>",
        "<*>",
    ]

def test_preserves_technical_tokens() -> None:
    assert tokenize_log_event(
        "NameSystem.allocateBlock BLOCK* "
        "java.net.SocketTimeoutException 67108864"
    ) == [
        "NameSystem.allocateBlock",
        "BLOCK*",
        "java.net.SocketTimeoutException",
        "67108864",
    ]

def test_generates_unigrams_and_bigrams() -> None:
    features = analyze_log_document(
        "Receiving block data"
    )

    assert features == [
        "Receiving",
        "block",
        "data",
        "Receiving block",
        "block data",
    ]


def test_does_not_cross_event_boundary() -> None:
    features = analyze_log_document(
        "Receiving block\nDeleting block"
    )

    assert "Receiving block" in features
    assert "Deleting block" in features
    assert "block Deleting" not in features


def test_preserves_repetitions() -> None:
    features = analyze_log_document(
        "Receiving block\nReceiving block"
    )

    assert features.count("Receiving") == 2
    assert features.count("Receiving block") == 2


def test_empty_document() -> None:
    assert analyze_log_document("") == []
    assert analyze_log_document("\n\n") == []