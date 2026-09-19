"""Tokenização textual dos logs HDFS para TF-IDF."""

import re

NGRAM_RANGE = (1, 2)

TOKEN_PATTERN = re.compile(
    r"<(?:BLOCK_ID|JOB_ID|IP|PATH|\*)>|"
    r"[A-Za-z0-9_]+(?:[.$:-][A-Za-z0-9_]+)*\*?"
)

def tokenize_log_event(event: str) -> list[str]:
    """Preserva placeholders, números e identificadores técnicos."""
    return TOKEN_PATTERN.findall(event)

def analyze_log_document(document: str) -> list[str]:
    """Gera unigramas e bigramas sem atravessar eventos."""
    features: list[str] = []

    for event in document.splitlines():
        tokens = tokenize_log_event(event)

        features.extend(tokens)
        features.extend(
            f"{left} {right}"
            for left, right in zip(tokens, tokens[1:])
        )

    return features