# log-analytics

Implementação experimental do pipeline do TCC sobre análise automatizada de logs.

## Ambiente

- Python 3.12
- dados brutos em `data/raw/`, fora do Git
- resultados reproduzíveis em `artifacts/`

## Etapa atual

Auditoria de integridade do Loghub HDFS_v1 antes do parsing com Drain.

## Execução

```bash
python scripts/audit_hdfs_v1.py \
  --log data/raw/hdfs_v1/HDFS.log \
  --labels data/raw/hdfs_v1/anomaly_label.csv \
  --output artifacts/data_audit/hdfs_v1_audit.json
```
