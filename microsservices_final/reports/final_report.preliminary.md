# Relatório consolidado da execução final

Gerado em: `2026-09-16T01:22:53.497029+00:00`

Status: **executed_with_auxiliary_evaluation**

## Cobertura

- Eventos brutos: **14,805,016**
- Eventos rastreáveis processados: **11,199,871** (75.65%)
- Traces processados: **1,514,380**
- Eventos sem identificador preservados separadamente: **3,605,145**

## Parsing e detecção

- Drain3: `0.9.11`, limiar `0.6`
- Templates: **5,388**; singletones: **2,682**
- Eventos desconhecidos: **22,089** (0.20%)
- Detector candidato: **IsolationForest**
- Holdout — precisão: **0.541**, recall: **0.930**, F1: **0.684**, falsos positivos: **34**
- Alertas: `{'none': 1487403, 'review': 18283, 'high_confidence': 8694}`

## Localização e causa provável

- Traces alertados com candidato: **26,977/26,977**
- Linhas de candidatos: **183,410**
- Hipóteses com evidência: **26,977**
- Fontes das hipóteses: `{'first_error_service': 13271, 'candidate_service_top1': 13706}`

## Limitações

- rótulos manuais não constituem verdade independente e foram usados somente na calibração/avaliação auxiliar
- não há validação causal independente para o serviço ou hipótese
- a localização é ranking de candidato baseado em evidências observáveis
- a avaliação de detecção pode compartilhar sinais com a construção dos rótulos
- os eventos sem identificador de trace foram preservados, mas não entram na análise por trace
