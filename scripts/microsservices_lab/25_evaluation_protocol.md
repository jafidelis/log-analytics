# Protocolo de avaliação do framework de microsserviços

## Objetivo

Avaliar separadamente a capacidade de detectar traces suspeitos, localizar o serviço provavelmente responsável e apresentar uma hipótese de causa provável com evidências rastreáveis.

## Camadas e métricas

### 1. Detecção

- unidade: `tc_trace_id`;
- métricas: precisão, recall, F1, PR-AUC e quantidade de alertas;
- rótulos manuais: referência auxiliar, excluindo `incerto`;
- nenhum rótulo entra nas features ou no ajuste do detector.

### 2. Localização

- referência: `root_service_manual` preenchido;
- métricas: Top-1 e Top-3;
- `nenhum`/fluxo normal não entra no denominador;
- serviço emissor, primeiro serviço com erro e serviço-alvo devem permanecer diferenciados.

### 3. Explicabilidade

- completude: fato, interpretação, hipótese, evidência, confiança e alternativa;
- rastreabilidade: serviço/entidade, timestamp e evento observável;
- concordância: hipótese automática contra anotação manual;
- confiança manual não é probabilidade calibrada.

## Regras de validade

1. Reportar sempre o tamanho da amostra e a distribuição dos rótulos.
2. Manter `incerto` fora das métricas binárias e registrar sua quantidade.
3. Não usar os mesmos rótulos para ajustar regra e declarar validação independente.
4. Separar resultados exploratórios de decisões congeladas.
5. Não afirmar causa raiz comprovada sem referência causal independente.
6. Preservar `tc_trace_id`, serviço, timestamp, mensagem e evidência original.

## Resultado piloto atual

- detecção: Isolation Forest com limiar calibrado; holdout com precisão 0,423, recall 0,953 e F1 0,586;
- localização: Top-1 76,5% e Top-3 85,3% em 34 traces com serviço raiz manual;
- hipótese causal: 94,1% de concordância em 34 casos;
- explicabilidade: formulário estruturado completo em 50 traces, mas sem validação de correção causal.

## Limitações

Os resultados vêm de 1.000 traces rotulados manualmente, com apenas 145 anomalias, e a amostra foi construída com base em sinais observáveis. A avaliação é um piloto de engenharia e não representa ainda uma validação independente em produção.

## Próxima decisão

Aplicar o protocolo ao relatório consolidado por trace, revisar os casos discordantes e definir quais resultados serão apresentados como demonstração do framework e quais permanecerão como análise exploratória.
