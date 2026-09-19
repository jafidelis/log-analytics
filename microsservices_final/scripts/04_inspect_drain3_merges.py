from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from importlib.machinery import SourceFileLoader

TUNING = SourceFileLoader("tuning", str(Path(__file__).with_name("04_tune_drain3_preparer.py"))).load_module()


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def build_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.drain_sim_th = 0.6
    config.drain_extra_delimiters = []
    config.masking_instructions = []
    config.parametrize_numeric_tokens = False
    return TemplateMiner(config=config)


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--events", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=Path("microsservices_final/manifests/stage04_merge_inspection.json"))
    args = parser.parse_args()
    model = build_miner()
    examples = defaultdict(list)
    services = defaultdict(set)
    levels = defaultdict(set)
    read = 0
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            read += 1
            message = TUNING.prepare(row.get("message") or "", "temporal")
            if not message:
                continue
            result = model.add_log_message(message)
            cluster_id = result["cluster_id"]
            if len(examples[cluster_id]) < 3:
                examples[cluster_id].append(message[:500])
            services[cluster_id].add((row.get("tc_service") or "").strip())
            levels[cluster_id].add((row.get("level") or "").strip().upper())
            if read >= args.events:
                break
    clusters = {cluster.cluster_id: cluster for cluster in model.drain.clusters}
    inspected = []
    for cluster_id, cluster in clusters.items():
        template = cluster.get_template()
        inspected.append({
            "template_id": cluster_id,
            "fit_count": cluster.size,
            "wildcards": template.count("<*>") + template.count("<NUM>") + template.count("<UUID>"),
            "distinct_services": sorted(x for x in services[cluster_id] if x),
            "levels": sorted(x for x in levels[cluster_id] if x),
            "examples": examples[cluster_id],
            "template": template[:1000],
        })
    inspected.sort(key=lambda x: (-x["fit_count"], -x["wildcards"], x["template_id"]))
    report = {
        "stage": "4_tuning", "stage_name": "inspect_drain3_merge_coherence", "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_read": read, "similarity_threshold": 0.6, "preparer": "temporal",
        "results": {"templates": len(inspected), "templates_with_multiple_services": sum(len(x["distinct_services"]) > 1 for x in inspected), "templates_with_multiple_levels": sum(len(x["levels"]) > 1 for x in inspected)},
        "high_frequency_candidates": [x for x in inspected if x["fit_count"] >= 100][:50],
        "high_wildcard_candidates": sorted(inspected, key=lambda x: (-x["wildcards"], -x["fit_count"]))[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": report["stage_name"], "events_read": read, "results": report["results"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
