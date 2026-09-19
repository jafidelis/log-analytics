from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seleciona métricas e exemplos para o TCC.")
    parser.add_argument("--report", type=Path, default=Path("microsservices_final/reports/final_report.json"))
    parser.add_argument("--alerts", type=Path, default=Path("data/processed/microservices_final/microservices_alerts.csv.gz"))
    parser.add_argument("--hypotheses", type=Path, default=Path("data/processed/microservices_final/microservices_cause_hypotheses.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("microsservices_final/reports/tcc_selected_material.md"))
    parser.add_argument("--output-csv", type=Path, default=Path("microsservices_final/reports/tcc_representative_examples.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage12_tcc_material_selection.manifest.json"))
    args = parser.parse_args()
    if args.output_md.exists() or args.output_csv.exists() or args.manifest.exists():
        raise SystemExit("Material do TCC já existe; remova-o deliberadamente para reexecutar.")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    alerts = pd.read_csv(args.alerts, compression="gzip", dtype=str)
    hypotheses = pd.read_csv(args.hypotheses, dtype=str)
    data = alerts.merge(hypotheses[["analysis_trace_id", "hypothesis_service", "hypothesis_source", "first_error_service", "entry_service", "evidence_event_ordinal", "evidence_level", "evidence_message"]], on="analysis_trace_id", how="inner", validate="one_to_one")
    data["score"] = data["isolation_forest_score"].astype(float)
    selections = []
    rules = [
        ("high_confidence_first_error", data[(data.alert_level == "high_confidence") & (data.hypothesis_source == "first_error_service")].sort_values("score", ascending=False)),
        ("review_candidate", data[(data.alert_level == "review") & (data.hypothesis_source == "candidate_service_top1")].sort_values("score", ascending=False)),
        ("highest_score", data.sort_values("score", ascending=False)),
        ("high_confidence_long_evidence", data[data.alert_level == "high_confidence"].assign(evidence_len=data.evidence_message.fillna("").str.len()).sort_values(["evidence_len", "score"], ascending=False)),
        ("review_with_error", data[(data.alert_level == "review") & data.first_error_service.fillna("").ne("")].sort_values("score", ascending=False)),
    ]
    used = set()
    for reason, frame in rules:
        if frame.empty:
            continue
        row = next((row for _, row in frame.iterrows() if row.analysis_trace_id not in used), None)
        if row is None:
            continue
        used.add(row.analysis_trace_id)
        selections.append({"selection_reason": reason, "analysis_trace_id": row.analysis_trace_id, "alert_level": row.alert_level, "score": row.score, "manual_label_auxiliary": row.manual_label, "hypothesis_service": row.hypothesis_service, "hypothesis_source": row.hypothesis_source, "first_error_service": row.first_error_service, "entry_service": row.entry_service, "evidence_event_ordinal": row.evidence_event_ordinal, "evidence_level": row.evidence_level, "evidence_message": row.evidence_message})
    fields = list(selections[0]) if selections else ["selection_reason"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(selections)
    detection = report["detection"]
    md = "# Material selecionado para o TCC\n\n"
    md += "## Métricas recomendadas\n\n"
    md += f"- Cobertura rastreável: **{report['data_coverage']['traceable_event_coverage']:.2%}** dos eventos brutos.\n- Eventos `UNK_TEMPLATE`: **{report['parsing']['unknown_rate']:.2%}**.\n- Templates Drain3: **{report['parsing']['templates']:,}**.\n- Detector: **{detection['model_candidate']}**.\n- Holdout auxiliar: precisão **{detection['holdout']['precision']:.3f}**, recall **{detection['holdout']['recall']:.3f}**, F1 **{detection['holdout']['f1']:.3f}**, PR-AUC **{detection['pr_auc_reference']:.3f}**.\n- Alertas: **{detection['alert_levels']}**.\n- Localização: apresentar cobertura de candidatos (**{report['localization']['traces_with_candidates']:,}/{report['localization']['alerted_traces']:,}**), não acurácia Top-1/Top-3, pois não há referência causal independente no conjunto completo.\n\n"
    md += "## Exemplos\n\n"
    md += "Os exemplos anonimizados estão em `tcc_representative_examples.csv`. Cada exemplo preserva trace, nível de alerta, score, serviço hipotético, origem da hipótese, ordinal e evidência textual.\n\n"
    md += "## Cuidados de interpretação\n\n- Os 1.000 rótulos manuais são auxiliares e não representam verdade independente.\n- A hipótese de causa raiz é uma inferência rastreável, não uma prova causal.\n- Eventos sem identificador de trace foram preservados e excluídos apenas da análise por trace.\n"
    args.output_md.write_text(md, encoding="utf-8")
    manifest = {"stage": 12, "stage_name": "select_tcc_material", "generated_at": datetime.now(timezone.utc).isoformat(), "inputs": {"report": {"path": str(args.report), "sha256": sha256_file(args.report)}, "alerts": {"path": str(args.alerts), "sha256": sha256_file(args.alerts)}, "hypotheses": {"path": str(args.hypotheses), "sha256": sha256_file(args.hypotheses)}}, "selection": {"metrics": ["traceable_event_coverage", "unknown_rate", "precision", "recall", "F1", "PR-AUC", "alert_levels", "candidate_coverage"], "examples": len(selections), "localization_accuracy_excluded": True}, "outputs": {"markdown": str(args.output_md), "examples_csv": str(args.output_csv), "examples_sha256": sha256_file(args.output_csv)}, "checks": {"examples_nonzero": bool(selections), "outputs_created": args.output_md.exists() and args.output_csv.exists()}}
    manifest["accepted"] = all(manifest["checks"].values())
    args.manifest.parent.mkdir(parents=True, exist_ok=True); temporary = Path(str(args.manifest) + ".tmp"); temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
