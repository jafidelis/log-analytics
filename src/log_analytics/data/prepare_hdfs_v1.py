import re
from ipaddress import IPv4Address

from log_analytics.data.audit_hdfs_v1 import BLOCK_ID_PATTERN

ENDPOINT_PATTERN = re.compile(
    r"(?P<prefix>\b(?:src|dest):[ \t]*/)"
    r"(?P<ip>[0-9]+(?:\.[0-9]+){3})"
    r"(?P<port>:[0-9]+)(?![\w.:])"
)

IPV4_PORT_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(?P<slash>/?)"
    r"(?P<ip>[0-9]+(?:\.[0-9]+){3})"
    r"(?P<port>:[0-9]+)"
    r"(?![\w.])"
)

JOB_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"job_[0-9]{12}_[0-9]{4}"
    r"(?![A-Za-z0-9_.-])"
)

IPV4_BARE_PATTERN = re.compile(
    r"(?<![\w.:])"
    r"(?P<slash>/?)"
    r"(?P<ip>[0-9]+(?:\.[0-9]+){3})"
    r"(?![\w.:])"
)

ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<!\S)"
    r"/(?:[A-Za-z0-9._<>-]+/)+[A-Za-z0-9._<>-]+"
    r"(?!\S)"
)

def mask_endpoint(match: re.Match[str]) -> str:
    """Mascara somente o IP de um endpoint reconhecido e válido."""
    try:
        IPv4Address(match.group("ip"))
    except ValueError:
        return match.group(0)

    return f"{match.group('prefix')}<IP>{match.group('port')}"

def mask_ipv4_port(match: re.Match[str]) -> str:
    """Mascara o IPv4 e preserva a barra e a porta."""
    try:
        IPv4Address(match.group("ip"))
    except ValueError:
        return match.group(0)

    return (
        f"{match.group('slash')}"
        f"<IP>"
        f"{match.group('port')}"
    )

def prepare_hdfs_message(message: str) -> str:
    """Gera texto derivado, generalizando BlockIds e IPs de src/dest."""
    prepared = BLOCK_ID_PATTERN.sub("<BLOCK_ID>", message)
    return ENDPOINT_PATTERN.sub(mask_endpoint, prepared)

def prepare_hdfs_message_candidate(message: str) -> str:
    """Preparação candidata para o experimento A/B."""
    prepared = BLOCK_ID_PATTERN.sub("<BLOCK_ID>", message)
    return IPV4_PORT_PATTERN.sub(mask_ipv4_port, prepared)

def prepare_hdfs_message_candidate_c(message: str) -> str:
    """Aplica a condição B e mascara o JobId no formato observado."""
    prepared = prepare_hdfs_message_candidate(message)
    return JOB_ID_PATTERN.sub("<JOB_ID>", prepared)

# Opção D: C + IPv4 sem porta + caminhos absolutos de arquivo.
def mask_bare_ipv4(match: re.Match[str]) -> str:
    """Mascara IPv4 sem porta, preservando a barra inicial."""
    try:
        IPv4Address(match.group("ip"))
    except ValueError:
        return match.group(0)

    return f"{match.group('slash')}<IP>"

def prepare_hdfs_message_candidate_d(message: str) -> str:
    """Condição D: C + IPv4 sem porta + caminhos absolutos de arquivo."""
    prepared = prepare_hdfs_message_candidate_c(message)
    prepared = IPV4_BARE_PATTERN.sub(mask_bare_ipv4, prepared)
    return ABSOLUTE_PATH_PATTERN.sub("<PATH>", prepared)

