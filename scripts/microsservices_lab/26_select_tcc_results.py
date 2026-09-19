from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["tc_trace_id"]: row for row in csv.DictReader(stream)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path(
        "artifacts/microsservices_lab/stage20_trace_report.csv"))
    parser.add_argument("--manual", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--hypotheses", type=Path, default=Path(
        "artifacts/microsservices_lab/stage19_explainable_cause_evidence.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage26_tcc_metric_selection.md"))
    parser.add_argument("--examples", type=Path, default=Path(
        "artifacts/microsservices_lab/stage26_tcc_representative_examples.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage26_tcc_selection.manifest.json"))
    args = parser.parse_args()

    report_rows = list(read_map(args.report).values())
    manual = read_map(args.manual)
    hypotheses = read_map(args.hypotheses)
    stage5 = read_json(Path("artifacts/microsservices_lab/stage5_threshold_calibration.json"))
    stage13 = read_json(Path("artifacts/microsservices_lab/stage13_localization_evaluation.json"))
    stage23 = read_json(Path("artifacts/microsservices_lab/stage23_structured_review_quality.manifest.json"))
    stage24 = read_json(Path("artifacts/microsservices_lab/stage24_explainable_output_evaluation.manifest.json"))

    reviewed = []
    for trace_id, row in manual.items():
        if trace_id in hypotheses:
            merged = {"tc_trace_id": trace_id, **row, **hypotheses[trace_id]}
            reviewed.append(merged)
    examples = []
    categories = [
        ("high_confidence", lambda r: r.get("alert_level") == "high_confidence"),
        ("review", lambda r: r.get("alert_level") == "review"),
        ("multi_hypothesis", lambda r: r.get("hypothesis") == "MULTI_HYPOTHESIS"),
        ("scheduler", lambda r: r.get("root_cause_category_manual") == "scheduler_task_orfa"),
        ("normal_flow", lambda r: r.get("root_service_manual", "").strip().lower() == "nenhum"),
    ]
    for name, predicate in categories:
        matches = [r for r in reviewed if predicate(r)]
        if matches:
            matches.sort(key=lambda r: (-float(r.get("anomaly_score", 0)), r["tc_trace_id"]))
            row = matches[0]
            examples.append({"example_type": name, "tc_trace_id": row["tc_trace_id"],
                "anomaly_score": row.get("anomaly_score", ""), "alert_level": row.get("alert_level", ""),
                "root_service_manual": row.get("root_service_manual", ""),
                "root_cause_category_manual": row.get("root_cause_category_manual", ""),
                "hypothesis": row.get("hypothesis", ""),
                "hypothesis_source": row.get("evidence_type", ""),
                "evidence_manual": row.get("evidence_manual", "")})

    args.examples.parent.mkdir(parents=True, exist_ok=True)
    fields = list(examples[0])
    with args.examples.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(examples)
    holdout = stage5["holdout_metrics"]
    loc1 = stage13["top1"]
    loc3 = stage13["top3"]
    report = f"""# Seleção de resultados para o TCC — piloto de microsserviços

## Métricas selecionadas pelo protocolo

### Detecção

- unidade: trace (`tc_trace_id`);
- holdout: {holdout['n']} traces;
- precisão: {holdout['precision']:.3f}; recall: {holdout['recall']:.3f}; F1: {holdout['f1']:.3f};
- limiar calibrado: {stage5['selection']['threshold']:.10f};
- PR-AUC de referência: {stage5['pr_auc_reference']:.3f}.

### Localização

- referência manual: {loc1['total']} traces com serviço raiz preenchido;
- Top-1: {loc1['correct']}/{loc1['total']} = {loc1['rate']:.3f};
- Top-3: {loc3['correct']}/{loc3['total']} = {loc3['rate']:.3f}.

### Explicabilidade

- revisões estruturadas completas: {stage23['complete_reviews']}/{stage23['reviewed_traces']};
- completude média: {stage23['mean_score']:.2f}/{len(stage23['rubric'])};
- concordância da hipótese com o serviço manual: {stage24['overall']['hypothesis_match_rate']:.3f};
- rastreabilidade média: {stage24['overall']['mean_traceability_score']:.2f}/3.

## Exemplos selecionados

Os exemplos anonimizados estão em `stage26_tcc_representative_examples.csv`. Eles representam: alerta de alta confiança; alerta para revisão; hipótese múltipla; categoria `scheduler_task_orfa`; e fluxo normal sem falha.

## Limitações para redação

Os números são de um piloto com rótulos manuais, não de uma avaliação causal independente. Os 10 traces `incerto` foram excluídos das métricas binárias. A seleção de exemplos serve para demonstrar o funcionamento e a rastreabilidade do framework; não deve ser apresentada como amostra estatisticamente representativa da produção. O resultado final ainda exige execução do pipeline completo nos logs originais.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    manifest = {"stage": 26, "stage_name": "select_tcc_results", "source_trace_report": str(args.report),
                "selected_metrics": ["precision", "recall", "f1", "pr_auc", "top1", "top3", "completeness", "traceability", "hypothesis_match"],
                "representative_examples": len(examples), "output": str(args.output), "examples_output": str(args.examples)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
