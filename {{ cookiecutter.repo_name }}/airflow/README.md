# Airflow - Orquestração de Pipelines

Diretório para orquestração de pipelines DataOps e MLOps usando Apache Airflow.

## Estrutura

```
airflow/
├── dags/
│   ├── dataops/              # DAGs de engenharia de dados
│   │   ├── dbt_*.py          # Transformações DBT
│   │   ├── athena_*.py       # Queries Athena
│   │   └── etl_*.py          # ETL pipelines
│   │
│   └── mlops/                # DAGs de Machine Learning
│       ├── ml_training_pipeline.py
│       ├── ml_batch_inference_pipeline.py
│       ├── ml_continuous_training_pipeline.py
│       └── ml_monitoring_pipeline.py
│
├── plugins/                  # Custom Airflow plugins
│   ├── operators/            # Custom operators
│   ├── sensors/              # Custom sensors
│   └── hooks/                # Custom hooks
│
├── config/                   # Configurações Airflow
│   ├── airflow.cfg
│   └── variables.json
│
├── logs/                     # Logs do Airflow
│
└── tests/                    # Testes de DAGs
    ├── test_dags_integrity.py
    └── test_dataops_mlops_integration.py
```

## Integração DataOps → MLOps

### Fluxo Recomendado

```
DAG DataOps (DBT):
  1. Extract → Lake Athena
  2. Transform → DBT (datamarts)
  3. Load → S3 Parquet
       ↓
  [Sensor] Aguarda datamart pronto
       ↓
DAG MLOps (Training):
  1. Feature engineering
  2. Model training
  3. Model validation
  4. Model deployment
```

### Exemplo de Conexão

**Opção 1: Sensor de arquivo S3**
```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_for_features = S3KeySensor(
    task_id='wait_for_datamart',
    bucket_name='{{ cookiecutter.s3_processed_bucket }}',
    bucket_key='datamarts/churn/latest.parquet',
    aws_conn_id='aws_default',
)
```

**Opção 2: TriggerDagRunOperator**
```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_ml_training = TriggerDagRunOperator(
    task_id='trigger_ml_training',
    trigger_dag_id='ml_training_pipeline',
    conf={'datamart_path': 's3://bucket/datamarts/latest.parquet'},
)
```

## Deployment

### Airflow Centralizado (DataOps Existente)

**Vantagens**:
- Orquestração unificada
- Dependências entre DataOps e MLOps
- Um único ponto de governança

**Como adicionar DAGs de ML**:
1. Copiar DAGs de `mlops/` para Airflow centralizado
2. Configurar conexões SSH/API para EC2s de ML
3. Executar comandos remotos via SSHOperator

**Exemplo**:
```python
from airflow.providers.ssh.operators.ssh import SSHOperator

train_model_ec2 = SSHOperator(
    task_id='train_model_remote',
    ssh_conn_id='ec2_ml_projeto1',
    command='cd /opt/mlops && python -m src.pipelines.DS.training_pipeline.train',
)
```

### Airflow Local (por EC2)

**Vantagens**:
- Execução local (baixa latência)
- Isolamento por projeto

**Como ativar**:
1. Adicionar serviço Airflow no `docker-compose.yaml`
2. DAGs executam localmente na EC2

## Configuração

### Variáveis Airflow

```json
{
  "s3_processed_bucket": "{{ cookiecutter.s3_processed_bucket }}",
  "s3_ml_artifacts_bucket": "{{ cookiecutter.s3_ml_artifacts_bucket }}",
  "mlflow_tracking_uri": "http://mlflow:5000",
  "model_registry_name": "{{ cookiecutter.project_name }}_model"
}
```

### Conexões

- `aws_default`: Credenciais AWS (S3, Athena)
- `ec2_ml_*`: SSH para EC2s de ML
- `mlflow_api`: API MLFlow
- `postgres_mlflow`: PostgreSQL MLFlow

## Monitoramento

- **Webserver UI**: http://localhost:8080
- **Logs**: `airflow/logs/`
- **Alertas**: Email on failure (configurado em default_args)
