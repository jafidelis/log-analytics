from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping, Sequence

from log_analytics.data.audit_hdfs_v1 import load_labels, sha256_file


LABEL_NORMAL = "Normal"
LABEL_ANOMALY = "Anomaly"
SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
EXPECTED_LABEL_COUNTS = {
    LABEL_NORMAL: 558_223,
    LABEL_ANOMALY: 16_838,
}
DEFAULT_SALT = "hdfs-v1-master-split-v1"
EXPECTED_LABELS_SHA256 = (
    "1c711ed6c8848fc3243fb4d092f172f31d128c8a6ec7f26ebba72ab931885ed8"
)
EXPECTED_LOG_SHA256 = (
    "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52"
)