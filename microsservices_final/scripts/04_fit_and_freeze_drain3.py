from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from time import perf_counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
URL_RE = re.compile(r"https?://[^\s,)] +".replace(" ", ""))
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b[A-Fa-f0-9]{16,}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
WHITESPACE_RE = re.compile(r"\s+")


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_message(value: str) -> str:
    value = WHITESPACE_RE.sub(" ", value.strip())
    value = UUID_RE.sub("<UUID>", value)
    value = TIMESTAMP_RE.sub("<TIMESTAMP>", value)
    value = URL_RE.sub("<URL>", value)
    value = EMAIL_RE.sub("<EMAIL>", value)
    value = IP_RE.sub("<IP>", value)
    value = HEX_RE.sub("<HEX>", value)
    return NUMBER_RE.sub("<NUM>", value)


def build_parser(state_path: Path | None = None, similarity_threshold: float = 0.6) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.drain_sim_th = similarity_threshold
    config.drain_extra_delimiters = []
    config.masking_instructions = []
    config.parametrize_numeric_tokens = False
    persistence = FilePersistence(str(state_path)) if state_path is not None else None
    return TemplateMiner(persistence_handler=persistence, config=config)


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Ajusta e congela o Drain3 para microsserviços.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--fit-events", type=int, default=3_359_961)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--state", type=Path, default=Path("data/processed/microservices_final/microservices_drain3_state.bin"))
    parser.add_argument("--templates", type=Path, default=Path("data/processed/microservices_final/microservices_drain3_templates.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage04_drain3_fit.manifest.json"))
    args = parser.parse_args()
    if args.state.exists() or args.templates.exists() or args.manifest.exists():
        raise SystemExit("Artefatos Drain3 já existem; remova-os deliberadamente para reexecutar.")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.templates.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    # Durante o ajuste não persistir estado: o FilePersistence pode regravar
    # megabytes a cada novo cluster, o que degrada drasticamente o desempenho.
    # Ajuste temporal sem persistência; salvar somente após a janela completa.
    model = build_parser(similarity_threshold=args.similarity_threshold)
    events_seen = 0
    fit_events = 0
    started = perf_counter()
    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "message" not in (reader.fieldnames or []):
            raise ValueError("Campo message ausente")
        for row in reader:
            events_seen += 1
            if events_seen > args.fit_events:
                break
            message = prepare_message(row.get("message") or "")
            if not message:
                continue
            model.add_log_message(message)
            fit_events += 1
            if fit_events % 100_000 == 0:
                elapsed = perf_counter() - started
                rate = fit_events / elapsed if elapsed else 0.0
                remaining = (args.fit_events - fit_events) / rate if rate else 0.0
                print(
                    f"[drain3] ajuste {fit_events:,}/{args.fit_events:,} "
                    f"({fit_events / args.fit_events:.1%}) | "
                    f"templates={len(model.drain.clusters):,} | "
                    f"taxa={rate:,.0f} eventos/s | "
                    f"decorrido={elapsed / 60:.1f} min | "
                    f"ETA={remaining / 60:.1f} min",
                    flush=True,
                )

    # Persistência ocorre uma única vez, após o ajuste completo.
    model.persistence_handler = FilePersistence(str(args.state))
    model.save_state("fit_complete")
    clusters = sorted(model.drain.clusters, key=lambda cluster: cluster.cluster_id)
    temporary = Path(str(args.templates) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["template_id", "template", "fit_count"])
        writer.writeheader()
        for cluster in clusters:
            writer.writerow({"template_id": cluster.cluster_id, "template": cluster.get_template(), "fit_count": cluster.size})
    os.replace(temporary, args.templates)

    manifest = {
        "stage": 4, "stage_name": "fit_and_freeze_drain3", "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "fit_policy": {"events_requested": args.fit_events, "events_scanned": events_seen, "fit_events": fit_events, "selection": "first temporal window; no labels used"},
        "drain3": {"version": version("drain3"), "depth": 4, "similarity_threshold": args.similarity_threshold, "state_frozen": True, "preparer": "whitespace + UUID + timestamp + URL + email + IP + hex + numeric normalization"},
        "results": {"templates": len(clusters), "singleton_templates": sum(cluster.size == 1 for cluster in clusters)},
        "outputs": {"state": {"path": str(args.state), "sha256": sha256_file(args.state)}, "templates": {"path": str(args.templates), "sha256": sha256_file(args.templates)}},
        "checks": {"fit_events_nonzero": fit_events > 0, "templates_nonzero": bool(clusters), "state_hash_recorded": True, "templates_hash_recorded": True},
    }
    manifest["accepted"] = all(manifest["checks"].values())
    temporary_manifest = Path(str(args.manifest) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
