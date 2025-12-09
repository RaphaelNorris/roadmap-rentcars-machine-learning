# Scripts - Automação

Scripts para automação de tarefas operacionais.

## Estrutura

```
scripts/
├── setup/                     # Setup de ambiente
│   ├── setup_ec2.sh           # Provisionar EC2
│   ├── install_dependencies.sh
│   └── configure_aws.sh
│
├── deployment/                # Deploy de modelos
│   ├── deploy_model.sh
│   ├── canary_deployment.sh
│   ├── blue_green_deployment.sh
│   └── rollback.sh
│
├── monitoring/                # Monitoramento
│   ├── check_drift.py
│   ├── check_performance.py
│   └── alert_degradation.py
│
├── backup/                    # Backup e DR
│   ├── backup_postgres.sh
│   ├── backup_mlflow_s3.sh
│   └── restore_from_backup.sh
│
└── utils/                     # Utilitários gerais
    ├── clean_old_models.py
    ├── export_metrics.py
    └── generate_reports.py
```

## Scripts Principais

### Setup

**setup_ec2.sh**
```bash
# Provisiona EC2 com todas as dependências
./scripts/setup/setup_ec2.sh --instance-type t3.large --project churn
```

**install_dependencies.sh**
```bash
# Instala Python, Docker, AWS CLI
./scripts/setup/install_dependencies.sh
```

### Deployment

**deploy_model.sh**
```bash
# Deploy de modelo para produção
./scripts/deployment/deploy_model.sh \
  --model-name churn_model \
  --version 5 \
  --strategy canary
```

**canary_deployment.sh**
```bash
# Canary deployment progressivo
./scripts/deployment/canary_deployment.sh \
  --model-version 5 \
  --stages "5,25,50,100" \
  --interval 300  # 5 minutos entre stages
```

**rollback.sh**
```bash
# Rollback para versão anterior
./scripts/deployment/rollback.sh \
  --model-name churn_model \
  --to-version 4
```

### Monitoring

**check_drift.py**
```bash
# Detecta drift em features
python scripts/monitoring/check_drift.py \
  --reference-data s3://bucket/baseline.parquet \
  --current-data s3://bucket/current.parquet \
  --threshold 0.05
```

**check_performance.py**
```bash
# Verifica degradação de performance
python scripts/monitoring/check_performance.py \
  --model-name churn_model \
  --window-days 7 \
  --threshold 0.05
```

### Backup

**backup_postgres.sh**
```bash
# Backup diário do PostgreSQL
./scripts/backup/backup_postgres.sh

# Output: backup-mlflow-20241123.sql → s3://bucket/backups/
```

**restore_from_backup.sh**
```bash
# Restore de backup
./scripts/backup/restore_from_backup.sh \
  --backup-date 20241123 \
  --from-s3 s3://bucket/backups/backup-mlflow-20241123.sql
```

### Utils

**clean_old_models.py**
```bash
# Remove modelos archived com > 90 dias
python scripts/utils/clean_old_models.py --days 90
```

**export_metrics.py**
```bash
# Exporta métricas para CSV/JSON
python scripts/utils/export_metrics.py \
  --model churn_model \
  --start-date 2024-01-01 \
  --format csv
```

## Agendamento (Cron)

```bash
# Backup diário às 3 AM
0 3 * * * /opt/mlops/scripts/backup/backup_postgres.sh

# Drift detection semanal (domingo 2 AM)
0 2 * * 0 python /opt/mlops/scripts/monitoring/check_drift.py

# Limpeza mensal (primeiro dia do mês)
0 4 1 * * python /opt/mlops/scripts/utils/clean_old_models.py --days 90
```

## Logs

Logs salvos em `/opt/mlops/logs/scripts/`:
- `setup_YYYYMMDD.log`
- `deployment_YYYYMMDD.log`
- `backup_YYYYMMDD.log`
