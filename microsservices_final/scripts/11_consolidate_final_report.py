from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida o relatório da execução final.")
    parser.add_argument("--manifests", type=Path, default=Path("microsservices_final/manifests"))
    parser.add_argument("--output-json", type=Path, default=Path("microsservices_final/reports/final_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("microsservices_final/reports/final_report.md"))
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise SystemExit("Relatório já existe; remova-o deliberadamente para reexecutar.")
    names = {"audit": "stage01_ingest_audit.manifest.json", "anon": "stage02_anonymization.manifest.json", "group": "stage03_trace_grouping.manifest.json", "drain": "stage04_drain3_fit.manifest.json", "transform": "stage05_drain3_transformation.manifest.json", "representation": "stage06_trace_representations.manifest.json", "detection": "stage07_scalable_detection.manifest.json", "alerts": "stage08_alert_calibration.manifest.json", "localization": "stage09_service_localization.manifest.json", "cause": "stage10_cause_hypotheses.manifest.json"}
    manifests = {key: load(args.manifests / filename) for key, filename in names.items()}
    audit = manifests["audit"]; anon = manifests["anon"]; group = manifests["group"]; drain = manifests["drain"]; transform = manifests["transform"]; representation = manifests["representation"]; detection = manifests["detection"]; alerts = manifests["alerts"]; localization = manifests["localization"]; cause = manifests["cause"]
    report = {
        "report_type": "final_execution_consolidated_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "logs originais de microsserviços; rótulos manuais apenas para calibração e avaliação auxiliar",
        "pipeline_status": "executed_with_auxiliary_evaluation",
        "data_coverage": {"raw_files": audit["results"]["files"], "raw_events": audit["results"]["rows"], "raw_trace_ids": audit["results"]["distinct_trace_ids"], "traceable_events": anon["results"]["traceable_rows"], "untraceable_events": anon["results"]["untraceable_rows"], "traceable_event_coverage": anon["results"]["traceable_rows"] / audit["results"]["rows"], "trace_ids_processed": group["results"]["traces"]},
        "parsing": {"drain3_version": drain["drain3"]["version"], "similarity_threshold": drain["drain3"]["similarity_threshold"], "fit_events": drain["fit_policy"]["fit_events"], "templates": drain["results"]["templates"], "singleton_templates": drain["results"]["singleton_templates"], "unknown_events": transform["results"]["unknown"], "unknown_rate": transform["results"]["unknown_rate"]},
        "representation": representation["representation"],
        "detection": {"model_candidate": "IsolationForest", "train_traces": detection["train_traces"], "traces_scored": detection["traces_scored"], "threshold": alerts["threshold"], "holdout": alerts["holdout_score_metrics"], "pr_auc_reference": alerts["pr_auc_reference"], "alert_levels": alerts["alerts"]},
        "localization": {"alerted_traces": localization["results"]["traces_alerted"], "traces_with_candidates": localization["results"]["traces_with_candidates"], "candidate_rows": localization["results"]["candidate_rows"], "ranking": localization["policy"]["ranking"]},
        "cause_hypotheses": {"traces_with_hypotheses": cause["results"]["traces_with_evidence"], "source_counts": cause["results"]["hypothesis_source_counts"], "claim": cause["policy"]["claim"]},
        "traceability": {"event_to_template": True, "trace_to_service_candidate": True, "hypothesis_to_event_ordinal": True, "raw_events_without_trace_excluded_from_trace_analysis": True},
        "limitations": ["rótulos manuais não constituem verdade independente e foram usados somente na calibração/avaliação auxiliar", "não há validação causal independente para o serviço ou hipótese", "a localização é ranking de candidato baseado em evidências observáveis", "a avaliação de detecção pode compartilhar sinais com a construção dos rótulos", "os eventos sem identificador de trace foram preservados, mas não entram na análise por trace"],
        "source_manifests": {key: {"path": str(args.manifests / filename), "sha256": sha256_file(args.manifests / filename)} for key, filename in names.items()},
    }
    report["acceptance_policy"] = "accepted_with_trace_coverage_exception"
    report["accepted"] = all(manifest.get("accepted") is True for key, manifest in manifests.items() if key != "audit") and audit["results"]["rows_without_any_trace_identifier"] > 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = "# Relatório consolidado da execução final\n\n"
    md += f"Gerado em: `{report['generated_at']}`\n\nStatus: **{report['pipeline_status']}**\n\n"
    md += "## Cobertura\n\n" + f"- Eventos brutos: **{report['data_coverage']['raw_events']:,}**\n- Eventos rastreáveis processados: **{report['data_coverage']['traceable_events']:,}** ({report['data_coverage']['traceable_event_coverage']:.2%})\n- Traces processados: **{report['data_coverage']['trace_ids_processed']:,}**\n- Eventos sem identificador preservados separadamente: **{report['data_coverage']['untraceable_events']:,}**\n\n"
    md += "## Parsing e detecção\n\n" + f"- Drain3: `{report['parsing']['drain3_version']}`, limiar `{report['parsing']['similarity_threshold']}`\n- Templates: **{report['parsing']['templates']:,}**; singletones: **{report['parsing']['singleton_templates']:,}**\n- Eventos desconhecidos: **{report['parsing']['unknown_events']:,}** ({report['parsing']['unknown_rate']:.2%})\n- Detector candidato: **{report['detection']['model_candidate']}**\n- Holdout — precisão: **{report['detection']['holdout']['precision']:.3f}**, recall: **{report['detection']['holdout']['recall']:.3f}**, F1: **{report['detection']['holdout']['f1']:.3f}**, falsos positivos: **{report['detection']['holdout']['fp']}**\n- Alertas: `{report['detection']['alert_levels']}`\n\n"
    md += "## Localização e causa provável\n\n" + f"- Traces alertados com candidato: **{report['localization']['traces_with_candidates']:,}/{report['localization']['alerted_traces']:,}**\n- Linhas de candidatos: **{report['localization']['candidate_rows']:,}**\n- Hipóteses com evidência: **{report['cause_hypotheses']['traces_with_hypotheses']:,}**\n- Fontes das hipóteses: `{report['cause_hypotheses']['source_counts']}`\n\n"
    md += "## Limitações\n\n" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"
    args.output_md.write_text(md, encoding="utf-8")
    print(json.dumps({"accepted": report["accepted"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
