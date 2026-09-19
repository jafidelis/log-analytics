from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


PATTERNS = {
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"),
    "timestamp": re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
    "url": re.compile(r"https?://[^\s,)] +".replace(" ", "")),
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "hex": re.compile(r"\b[A-Fa-f0-9]{16,}\b"),
    "number": re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])"),
}
WS = re.compile(r"\s+")


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def prepare(text: str, variant: str) -> str:
    text = WS.sub(" ", text.strip())
    text = PATTERNS["uuid"].sub("<UUID>", text)
    if variant in {"temporal", "aggressive"}:
        text = PATTERNS["timestamp"].sub("<TIMESTAMP>", text)
        text = PATTERNS["url"].sub("<URL>", text)
        text = PATTERNS["email"].sub("<EMAIL>", text)
        text = PATTERNS["ip"].sub("<IP>", text)
        text = PATTERNS["hex"].sub("<HEX>", text)
    text = PATTERNS["number"].sub("<NUM>", text)
    if variant == "aggressive":
        text = re.sub(r"(\b(?:id|Id|ID|token|correlationId|tenant|empresa|port|size|count)=)[^\s,;)]+", r"\1<VALUE>", text)
    return text


def miner(similarity_threshold: float) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.drain_sim_th = similarity_threshold
    config.drain_extra_delimiters = []
    config.masking_instructions = []
    config.parametrize_numeric_tokens = False
    return TemplateMiner(config=config)


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--events", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=Path("microsservices_final/manifests/stage04_preparer_tuning.json"))
    args = parser.parse_args()
    miners = {variant: miner(0.8) for variant in ("current", "temporal", "aggressive")}
    miners.update({f"temporal_sim_{threshold:.1f}": miner(threshold) for threshold in (0.6, 0.7, 0.8)})
    read = 0
    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            read += 1
            message = row.get("message") or ""
            for variant, model in miners.items():
                base_variant = "temporal" if variant.startswith("temporal_sim_") else variant
                prepared = prepare(message, base_variant)
                if prepared:
                    model.add_log_message(prepared)
            if read >= args.events:
                break
    results = {}
    for variant, model in miners.items():
        clusters = list(model.drain.clusters)
        counts = [cluster.size for cluster in clusters]
        results[variant] = {
            "templates": len(clusters),
            "singletons": sum(count == 1 for count in counts),
            "singleton_rate": sum(count == 1 for count in counts) / len(clusters) if clusters else 0.0,
            "max_template_count": max(counts) if counts else 0,
            "top_templates": [{"fit_count": cluster.size, "template": cluster.get_template()[:300]} for cluster in sorted(clusters, key=lambda c: (-c.size, c.cluster_id))[:10]],
        }
    report = {"stage": "4_tuning", "stage_name": "compare_drain3_preparers", "generated_at": datetime.now(timezone.utc).isoformat(), "input": str(args.input), "events_read": read, "variants": results, "decision": "select_by_lower_fragmentation_while_preserving_distinct_message_families"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
