# Material selecionado para o TCC

## Métricas recomendadas

- Cobertura rastreável: **75.65%** dos eventos brutos.
- Eventos `UNK_TEMPLATE`: **0.20%**.
- Templates Drain3: **5,388**.
- Detector: **IsolationForest**.
- Holdout auxiliar: precisão **0.541**, recall **0.930**, F1 **0.684**, PR-AUC **0.560**.
- Alertas: **{'none': 1487403, 'review': 18283, 'high_confidence': 8694}**.
- Localização: apresentar cobertura de candidatos (**26,977/26,977**), não acurácia Top-1/Top-3, pois não há referência causal independente no conjunto completo.

## Exemplos

Os exemplos anonimizados estão em `tcc_representative_examples.csv`. Cada exemplo preserva trace, nível de alerta, score, serviço hipotético, origem da hipótese, ordinal e evidência textual.

## Cuidados de interpretação

- Os 1.000 rótulos manuais são auxiliares e não representam verdade independente.
- A hipótese de causa raiz é uma inferência rastreável, não uma prova causal.
- Eventos sem identificador de trace foram preservados e excluídos apenas da análise por trace.
