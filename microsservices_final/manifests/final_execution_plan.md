# Plano de fechamento da execução final

Data: 2026-09-15

## Objetivo

Aplicar aos logs originais de microsserviços o framework definido no laboratório,
mantendo a rastreabilidade desde o evento bruto até o alerta, o serviço candidato
e a hipótese de causa raiz. A causa raiz será apresentada como hipótese sustentada
por evidências, não como causalidade comprovada.

## Situação do mapeamento

| Etapa final | Contrato | Situação | Ação necessária |
|---|---|---|---|
| 01 ingestão e auditoria | arquivos originais, esquema, contagem, SHA-256 | concluída com ressalva | auditar a política para 3.605.145 eventos sem qualquer identificador de trace |
| 02 anonimização determinística | saída completa, regras e mapa/versionamento | concluída | anonimização integral aceita; 11.199.871 eventos rastreáveis e 3.605.145 preservados separadamente |
| 03 agrupamento por `tc_trace_id` | todos os eventos válidos agrupados por trace | concluída | 11.199.871 eventos conservados em 1.514.380 traces |
| 04 Drain3 | ajuste controlado, estado congelado, catálogo e desconhecidos | concluída como candidata | 5.388 templates, limiar efetivo 0,6 e estado congelado; transformação completa pendente |
| 05 representações | contagens/templates + sinais; representação semântica se aprovada | transformação aceita; representação pendente | construir vetores esparsos por trace e combinar com features estruturais |
| 06 detecção | score, limiar e níveis de alerta | candidato definido | Isolation Forest selecionado preliminarmente; SGD-OCSVM apresentou colapso para todos positivos |
| 07 localização | ranking de serviço candidato Top-1/Top-3 | pendente | aplicar ranking bruto aprovado nos traces alertados |
| 08 hipóteses causais | hipótese, evidência, origem e limitações | pendente | extrair componente lógico, primeiro erro e evidência dos traces alertados |
| 09 avaliação e relatório | manifestos, métricas, tabelas e exemplos | pendente | consolidar detecção, cobertura, localização e rastreabilidade com limitações |

## Componentes reutilizáveis

- A política de amostragem por trace e o tratamento de `tc_trace_id` dos scripts
  de microsserviços são referências de schema, não executores finais.
- O contrato de ajuste/congelamento, catálogo e validação de estado do Drain3 deve
  ser adaptado dos scripts `run_hdfs_v1_full_parsing.py` e módulos `fit/match` do HDFS.
- A avaliação de detecção deve reutilizar o limiar candidato `0.6713793902714876`
  somente como parâmetro inicial; a execução final deve registrar se ele foi mantido.
- O benchmark preliminar com as 15 features favoreceu o One-Class SVM (`nu=0.01`,
  `gamma=scale`): no holdout, F1 `0.7069`, precisão `0.5616`, recall `0.9535` e
  32 falsos positivos, contra F1 `0.5734` e 59 falsos positivos do Isolation
  Forest. Esse resultado é candidato, não decisão final.
- O Isolation Forest permanece como baseline obrigatório. O detector final somente
  será escolhido após a comparação com as representações Drain3 e, se executadas,
  TF-IDF/embeddings.
- A regra de alta confiança permanece `score >= threshold AND has_stack_trace`.
- A localização deve iniciar pelo ranking bruto aprovado; as variantes normalizada,
  temporal e por call stack permanecem diagnósticas.
- Os 1.000 rótulos manuais continuam sendo referência auxiliar para calibração e
  avaliação, sem entrar nas features nem no ajuste não supervisionado.

## Ordem operacional obrigatória

1. Auditoria dos arquivos originais e registro dos hashes.
2. Anonimização determinística completa, com escrita temporária e renomeação atômica.
3. Agrupamento/ordenação por trace e validação de conservação dos eventos.
4. Ajuste do Drain3 apenas na parcela de ajuste definida no manifesto; salvar e
   recarregar o estado antes da transformação completa.
5. Construção das representações e validação de invariantes por trace.
6. Detecção, calibração/limiar e níveis de alerta.
7. Ranking de serviço, hipóteses causais e evidências rastreáveis.
8. Avaliação final e geração dos artefatos da pasta `microsservices_final`.

## Critério de aceite desta etapa

Este documento fecha o desenho operacional, mas não autoriza declarar a execução
final concluída. O manifesto `final_run_config.json` deverá ser atualizado para
`executed` somente após todos os contratos acima possuírem script, saída, SHA-256,
checks e manifesto aceito.

## Achado da auditoria inicial

Os arquivos originais contêm 14.805.016 eventos e 1.506.116 `tc_trace_id`
distintos. Há 3.613.499 eventos sem `tc_trace_id`; somente 8.354 deles possuem
um valor alternativo em `traceId`, restando 3.605.145 eventos sem qualquer
identificador de trace. A execução final não poderá agrupá-los silenciosamente.
É necessário congelar uma política explícita para essa população antes da
anonimização e do agrupamento.

## Qualidade preliminar do catálogo Drain3

O primeiro ajuste produziu 76.675 templates, com 90,2% singletones. Um piloto
de 250.000 eventos comparou três preparadores: o atual (7.708 templates,
76,9% singletones), o temporal (7.284, 76,3%) e o agressivo (7.077, 77,9%).
A variante temporal reduziu a fragmentação sem o comportamento indesejado da
variante agressiva, mas ainda não atingiu um nível suficiente para ser adotada.
O próximo teste deve variar o limiar de similaridade do Drain3 usando a
preparação temporal.

O teste dos limiares mostrou que `0.6` produziu 1.581 templates e 38,2%
singletones em 250.000 eventos; `0.7` produziu 5.312 templates e 75,5%
singletones; `0.8` produziu 7.284 e 76,3%. O limiar `0.6` é o candidato
principal por reduzir a fragmentação, mas exige inspeção de possíveis fusões
semânticas antes do ajuste completo.
