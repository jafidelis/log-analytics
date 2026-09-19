# log-analytics

Implementação experimental do pipeline do TCC para detecção de anomalias em
logs de sistemas distribuídos, localização do serviço candidato e geração de
hipóteses de causa raiz com evidências rastreáveis.

O HDFS v1 é utilizado como laboratório público para comparação controlada de
parsing, representações e detectores. O framework final é aplicado aos logs de
um sistema privado de gestão empresarial para o varejo, construído em
arquitetura de microsserviços.

## Ambiente

- Python 3.12;
- dados brutos em `data/raw/`, fora do Git;
- artefatos experimentais em `artifacts/`;
- resultados finais de microsserviços em `microsservices_final/`.

Na raiz do projeto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Os scripts devem ser executados a partir da raiz do repositório.

## 1. Pipeline do HDFS v1

### 1.1 Obter os dados

O HDFS v1 é distribuído pelo repositório público [LogHub](https://github.com/logpai/loghub/tree/master/HDFS).
Baixe os dois arquivos para `data/raw/hdfs_v1/`:

```bash
mkdir -p data/raw/hdfs_v1
curl -L https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS.log \
  -o data/raw/hdfs_v1/HDFS.log
curl -L https://raw.githubusercontent.com/logpai/loghub/master/HDFS/anomaly_label.csv \
  -o data/raw/hdfs_v1/anomaly_label.csv
```

O arquivo `HDFS.log` contém os eventos brutos. O arquivo `anomaly_label.csv`
contém os rótulos por `BlockId`. Os scripts registram hashes e interrompem a
execução quando os artefatos esperados já existem.

### 1.2 Auditar e particionar

```bash
python scripts/audit_hdfs_v1.py \
  --log data/raw/hdfs_v1/HDFS.log \
  --labels data/raw/hdfs_v1/anomaly_label.csv \
  --output artifacts/data_audit/hdfs_v1_audit.json

python scripts/create_hdfs_v1_split.py \
  --labels data/raw/hdfs_v1/anomaly_label.csv \
  --audit-manifest artifacts/data_audit/hdfs_v1_audit.json \
  --output artifacts/data_splits/hdfs_v1_session_split.csv \
  --normal-train-output artifacts/data_splits/hdfs_v1_train_normal_block_ids.txt \
  --manifest artifacts/data_splits/hdfs_v1_split_manifest.json
```

O particionamento usa a sessão identificada por `BlockId` como unidade
indivisível e cria as partições de treinamento, validação e teste.

### 1.3 Fazer o parsing com Drain congelado

```bash
python scripts/run_hdfs_v1_full_parsing.py
```

Esse script ajusta o parser apenas em `train_normal`, congela o estado e
transforma as três partições. Os principais resultados são gravados em
`artifacts/parsing/`.

### 1.4 Construir as representações

```bash
python scripts/build_hdfs_v1_session_sequences.py
python scripts/build_hdfs_v1_session_counts.py
python scripts/build_hdfs_v1_tfidf.py
python scripts/build_hdfs_v1_session_embeddings.py
```

As representações incluem sequências e contagens de templates, TF-IDF,
embeddings semânticos e a combinação de embeddings com contagens. O modelo de
embeddings pode ser preparado previamente com:

```bash
python scripts/download_embedding_model.py
```

### 1.5 Treinar e avaliar os detectores

```bash
python scripts/train_isolation_forest.py
python scripts/train_one_class_svm.py
python scripts/compare_hdfs_v1_representations_validation.py
python scripts/evaluate_frozen_on_test.py
```

Os detectores são ajustados no conjunto previsto pelo protocolo, os limiares
são calibrados na validação e o teste é utilizado somente na avaliação final.
Os relatórios e modelos ficam em `artifacts/detection/`.

## 2. Pipeline final de microsserviços

### 2.1 Dados privados

Os logs de microsserviços pertencem a um sistema privado de gestão empresarial
para o varejo. Eles não são disponibilizados publicamente, não são baixados
por este projeto e não devem ser adicionados ao Git. Para executar o pipeline,
coloque localmente os arquivos brutos em:

```text
data/raw/microservices/
```

O caminho pode ser alterado no manifesto
`microsservices_final/manifests/final_run_config.json`. Os rótulos manuais,
quando usados para calibração e avaliação auxiliar, ficam em:

```text
data/processed/microservices_sample/microservices_trace_manual_labels.csv
```

Os manifestos registram o caminho de entrada, o SHA-256, as configurações e os
checks de cada etapa. O pipeline não publica o conteúdo dos logs privados.

### 2.2 Execução das etapas

Execute os comandos abaixo a partir da raiz do projeto:

```bash
python microsservices_final/scripts/01_ingest_and_audit.py
python microsservices_final/scripts/02_anonymize_deterministically.py
python microsservices_final/scripts/03_group_by_trace_id.py
python microsservices_final/scripts/04_fit_and_freeze_drain3.py
python microsservices_final/scripts/05_transform_with_frozen_drain3.py
python microsservices_final/scripts/06_build_sparse_trace_representations.py
python microsservices_final/scripts/07_detect_scalable_models.py
python microsservices_final/scripts/08_calibrate_and_level_alerts.py
python microsservices_final/scripts/09_localize_candidate_services.py
python microsservices_final/scripts/10_generate_cause_hypotheses.py
python microsservices_final/scripts/11_consolidate_final_report.py
python microsservices_final/scripts/12_select_tcc_material.py
```

As etapas realizam, em ordem:

1. auditoria e ingestão dos arquivos brutos;
2. anonimização determinística;
3. agrupamento por `tc_trace_id`;
4. ajuste e congelamento do Drain3;
5. transformação dos eventos com o parser congelado;
6. construção das representações esparsas por trace;
7. detecção não supervisionada em escala;
8. calibração do limiar e níveis de alerta;
9. localização dos serviços candidatos;
10. geração de hipóteses de causa raiz;
11. consolidação do relatório final;
12. seleção de tabelas e exemplos para o TCC.

O script `microsservices_final/scripts/run_final_pipeline.py` funciona como
validação inicial do manifesto e, por padrão, executa somente um *dry run*.
As etapas individuais acima são o procedimento reprodutível da execução final.

Os artefatos processados ficam em `data/processed/microservices_final/`, os
manifestos em `microsservices_final/manifests/` e os relatórios em
`microsservices_final/reports/`.

## Documentação complementar

- `scripts/microsservices_lab/README.md`: laboratório inicial de microsserviços
  e etapas de calibração e localização;
- `microsservices_final/README.md`: organização dos artefatos da execução final;
- `microsservices_lab/`: scripts auxiliares do laboratório;
- `microsservices_final/scripts/`: scripts da execução final em escala.
