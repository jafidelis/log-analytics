# Resultado final do estudo de microsserviços

Esta pasta contém exclusivamente os artefatos da execução final do framework sobre os logs originais.

## Estrutura

- `scripts/`: scripts usados na execução final;
- `manifests/`: manifestos, parâmetros, hashes e checks;
- `reports/`: relatórios consolidados por trace e resultados para o TCC;
- `tables/`: tabelas de métricas e avaliações;
- `figures/`: figuras selecionadas para análise e redação.

Os logs brutos não serão copiados para esta pasta. Cada execução deverá registrar caminho de entrada, SHA-256, configuração, versão, data e resultado dos checks.

## Critério de fechamento

A execução final somente será considerada concluída quando ingestão, anonimização, agrupamento por trace, parsing com Drain3, representação, detecção, localização, hipótese de causa raiz e avaliação estiverem registrados nos manifestos correspondentes.

## Configuração inicial

O manifesto de configuração está em `manifests/final_run_config.json`. O orquestrador começa em modo seguro:

```bash
python microsservices_final/scripts/run_final_pipeline.py
```

Esse comando apenas valida o diretório de entrada e exibe a ordem planejada. A execução completa não deve ser iniciada antes de revisar o manifesto, os caminhos de saída e os contratos de cada etapa.
