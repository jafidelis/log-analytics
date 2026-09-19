# microsservices_lab — Etapas 1 a 4

Executar a partir da raiz do projeto `log-analytics`:

```bash
python scripts/microsservices_lab/01_validate_inputs.py
python scripts/microsservices_lab/02_build_trace_features.py
python scripts/microsservices_lab/03_score_signal_combinations.py
python scripts/microsservices_lab/04_detect_unsupervised.py
python scripts/microsservices_lab/05_calibrate_unsupervised_threshold.py
```

As Etapas 1–3 usam os rótulos manuais apenas para validação e pontuação. A Etapa 4 não inclui rótulos no ajuste do Isolation Forest; eles aparecem somente na avaliação auxiliar dos scores.
 A Etapa 5 escolhe o limiar apenas no subconjunto de calibração e mede o resultado em um holdout estratificado.

Para verificar a estabilidade do limiar e perfilar falsos positivos:

```bash
python scripts/microsservices_lab/06_validate_threshold_stability.py
python scripts/microsservices_lab/07_compare_fp_strategies.py
python scripts/microsservices_lab/08_two_level_alerts.py
python scripts/microsservices_lab/09_rank_service_candidates.py
python scripts/microsservices_lab/10_compare_service_rankings.py
python scripts/microsservices_lab/11_prepare_root_cause_annotation.py
python scripts/microsservices_lab/13_evaluate_service_localization.py
python scripts/microsservices_lab/15_rank_entity_candidates.py
python scripts/microsservices_lab/16_normalize_entity_ranking.py
python scripts/microsservices_lab/17_scheduler_specific_rule.py
python scripts/microsservices_lab/18_build_cause_hypotheses.py
python scripts/microsservices_lab/19_explainable_cause_evidence.py
python scripts/microsservices_lab/20_consolidate_trace_report.py
python scripts/microsservices_lab/21_score_causal_justifications.py
python scripts/microsservices_lab/22_prepare_structured_cause_review.py
python scripts/microsservices_lab/23_score_structured_review.py
python scripts/microsservices_lab/24_evaluate_explainable_outputs.py
python scripts/microsservices_lab/26_select_tcc_results.py
```
