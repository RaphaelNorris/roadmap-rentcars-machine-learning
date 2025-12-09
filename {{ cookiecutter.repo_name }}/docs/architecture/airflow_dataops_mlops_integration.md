# Integração Airflow: DataOps → MLOps

Guia completo para integrar pipeline de engenharia de dados (DBT + Athena) com pipeline de Machine Learning.

## Arquitetura de Integração

```
┌─────────────────────────────────────────────────────────────────┐
│                    Airflow Centralizado                         │
│                    (Infraestrutura DataOps Existente)           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────────┐
│   DAGs DataOps   │                    │    DAGs MLOps        │
│                  │                    │                      │
│  ├─ DBT          │                    │  ├─ Training         │
│  ├─ Athena       │                    │  ├─ Inference        │
│  └─ ETL          │                    │  ├─ Monitoring       │
│                  │                    │  └─ Cont. Training   │
└──────────────────┘                    └──────────────────────┘
        │                                           │
        │ (1) Transform Data                        │
        │     DBT → Datamarts                       │
        │                                           │
        ▼                                           │
┌──────────────────┐                                │
│   S3 Datamarts   │                                │
│   (Parquet)      │                                │
└──────────────────┘                                │
        │                                           │
        │ (2) S3 Sensor                             │
        │     Wait for datamart ready               │
        │                                           │
        └───────────────────┬───────────────────────┘
                            │
                            │ (3) Trigger ML Pipeline
                            │
                            ▼
               ┌────────────────────────────┐
               │    EC2 ML (Projeto)        │
               │  ├─ MLFlow                 │
               │  ├─ PostgreSQL             │
               │  └─ Training Pipeline      │
               └────────────────────────────┘
```

## Fluxo End-to-End

### 1. **Pipeline DataOps** (DBT + Athena)

```
Raw Data (Lake) → DBT Transform → Datamarts (S3 Parquet)
```

**Schedule**: Diariamente às 2 AM
**Output**: `s3://bucket/datamarts/project/latest.parquet`

### 2. **Sensor de Disponibilidade**

```python
wait_for_datamart = S3KeySensor(
    task_id='wait_for_datamart',
    bucket_name='processed-data-bucket',
    bucket_key='datamarts/churn/latest.parquet',
    timeout=3600,  # 1 hora
)
```

Aguarda datamart estar disponível no S3.

### 3. **Validação de Qualidade**

```python
def validate_datamart_quality(**context):
    # Checks:
    # - Schema correto
    # - Sem nulos em colunas críticas
    # - Tamanho mínimo de dados (> 1K rows)
    # - Sem duplicatas
```

### 4. **Feature Engineering**

```python
def create_ml_features(**context):
    # Transformações para ML:
    # - Feature engineering
    # - Encoding categóricas
    # - Normalização/scaling
    # - Salvar em S3
```

**Output**: `s3://bucket/ml/features/project/latest.parquet`

### 5. **Trigger ML Training**

**Três opções de execução**:

#### **Opção 1: TriggerDagRunOperator** (Airflow → Airflow)

Se ML training também for uma DAG no mesmo Airflow:

```python
trigger_ml_dag = TriggerDagRunOperator(
    task_id='trigger_ml_training',
    trigger_dag_id='ml_training_pipeline',
    conf={'features_path': 's3://bucket/ml/features/latest.parquet'},
    wait_for_completion=True,
)
```

**Vantagens**:
- ✅ Visibilidade unificada no Airflow UI
- ✅ Dependências explícitas entre DAGs
- ✅ Retry automático
- ✅ Logs centralizados

**Desvantagens**:
- ⚠️ Requer DAG de ML no mesmo Airflow
- ⚠️ Menos isolamento por projeto

#### **Opção 2: SSHOperator** (Airflow → EC2)

Se ML training roda em EC2 separada:

```python
train_via_ssh = SSHOperator(
    task_id='train_model_via_ssh',
    ssh_conn_id='ec2_ml_churn',
    command='''
        cd /opt/mlops && \
        source venv/bin/activate && \
        python -m src.pipelines.DS.training_pipeline.train \
            --features-path s3://bucket/ml/features/latest.parquet \
            --config config/prod/model_config.yaml
    ''',
    cmd_timeout=7200,  # 2 horas
)
```

**Vantagens**:
- ✅ Isolamento total por projeto
- ✅ Execução local na EC2 (baixa latência)
- ✅ Controle granular de recursos

**Desvantagens**:
- ⚠️ Requer configuração SSH
- ⚠️ Logs distribuídos (Airflow + EC2)

**Setup SSH Connection**:
```bash
# No Airflow: Admin → Connections → Add

Connection ID: ec2_ml_churn
Connection Type: SSH
Host: <ec2-public-ip>
Username: ec2-user
Private Key: <conteúdo da chave .pem>
Port: 22
```

#### **Opção 3: SimpleHttpOperator** (Airflow → API)

Se EC2 expõe API REST para treinamento:

```python
train_via_api = SimpleHttpOperator(
    task_id='trigger_training_via_api',
    http_conn_id='ec2_ml_api_churn',
    endpoint='/api/v1/training/trigger',
    method='POST',
    data=json.dumps({
        'features_path': 's3://bucket/ml/features/latest.parquet',
        'model_type': 'random_forest',
        'experiment_name': 'automated_training'
    }),
    headers={'Content-Type': 'application/json'},
    response_check=lambda r: r.status_code == 200,
)
```

**Vantagens**:
- ✅ Assíncrono (não bloqueia Airflow)
- ✅ API pode retornar imediatamente e treinar em background
- ✅ Melhor para treinos longos (> 1 hora)

**Desvantagens**:
- ⚠️ Requer API implementada na EC2
- ⚠️ Menos visibilidade do progresso

**API Endpoint Example**:
```python
# src/inference/api.py

from fastapi import FastAPI

app = FastAPI()

@app.post("/api/v1/training/trigger")
async def trigger_training(request: TrainingRequest):
    # Executar treinamento em background
    background_task = train_model_async(
        features_path=request.features_path,
        config=request.config
    )
    return {"status": "triggered", "job_id": background_task.id}
```

## Recomendação por Cenário

### Cenário 1: Airflow Centralizado + Múltiplas EC2s

**Arquitetura Recomendada**: SSHOperator

```
Airflow Centralizado (DataOps)
        ↓ SSH
EC2 ML Projeto 1 ← train via SSH
EC2 ML Projeto 2 ← train via SSH
EC2 ML Projeto 3 ← train via SSH
```

**Por quê?**
- Isolamento total por projeto
- Reuso de infraestrutura Airflow existente
- Execução local (melhor performance)

**Setup**:
1. Configurar SSH connections no Airflow (uma por EC2)
2. Criar DAG de integração por projeto
3. Schedule após pipeline DBT

### Cenário 2: Tudo no Mesmo Airflow

**Arquitetura Recomendada**: TriggerDagRunOperator

```
Airflow Centralizado
  ├─ DAGs DataOps
  └─ DAGs MLOps (todas)
```

**Por quê?**
- Visibilidade unificada
- Dependências explícitas
- Mais simples de gerenciar

**Setup**:
1. Adicionar DAGs de ML no mesmo Airflow
2. Usar TriggerDagRunOperator para conectar pipelines

### Cenário 3: Treinos Muito Longos (> 2 horas)

**Arquitetura Recomendada**: API + Background Jobs

```
Airflow → API (FastAPI) → Celery/Background Task
```

**Por quê?**
- Não bloqueia workers do Airflow
- Escalabilidade horizontal (múltiplos workers)
- Melhor para treinos longos

## Configuração de Dependências

### Opção 1: Schedule Sequencial

```python
# DAG DataOps
dag_dataops = DAG(
    'dbt_daily_transform',
    schedule_interval='0 2 * * *',  # 2 AM
)

# DAG MLOps (aguarda DataOps)
dag_mlops = DAG(
    'dataops_to_mlops_integration',
    schedule_interval='0 3 * * *',  # 3 AM (1 hora depois)
)
```

### Opção 2: TriggerDagRunOperator

```python
# Na DAG DataOps, adicionar última task:
trigger_mlops = TriggerDagRunOperator(
    task_id='trigger_mlops_pipeline',
    trigger_dag_id='dataops_to_mlops_integration',
    trigger_rule='all_success',
)
```

### Opção 3: ExternalTaskSensor

```python
# Na DAG MLOps, aguardar task específica da DAG DataOps:
from airflow.sensors.external_task import ExternalTaskSensor

wait_dbt_completion = ExternalTaskSensor(
    task_id='wait_for_dbt',
    external_dag_id='dbt_daily_transform',
    external_task_id='dbt_run',
    allowed_states=['success'],
    failed_states=['failed', 'skipped'],
)
```

## Monitoramento

### Métricas a Acompanhar

| Métrica | Threshold | Ação |
|---------|-----------|------|
| **Lag entre DataOps e MLOps** | < 2 horas | Alert se > 2h |
| **Data quality checks** | 100% pass | Alert se falhar |
| **Training duration** | < 1 hora | Otimizar se > 1h |
| **Model performance** | Accuracy > 0.80 | Rollback se < 0.80 |

### Dashboard Grafana

```yaml
panels:
  - title: "DataOps → MLOps Lag"
    query: airflow_dag_run_duration{dag_id="dataops_to_mlops_integration"}

  - title: "Data Quality Success Rate"
    query: sum(rate(data_quality_checks_passed[1h])) / sum(rate(data_quality_checks_total[1h]))

  - title: "Training Success Rate"
    query: sum(rate(ml_training_success[1d])) / sum(rate(ml_training_total[1d]))
```

### Alertas

```yaml
# Prometheus alerting rules
groups:
  - name: dataops_mlops_integration
    rules:
      - alert: DataOpsPipelineFailed
        expr: airflow_dag_run_failed{dag_id="dbt_daily_transform"} > 0
        for: 5m
        annotations:
          summary: "DataOps pipeline failed - MLOps blocked"

      - alert: MLOpsPipelineLag
        expr: (time() - airflow_dag_run_last_completion_time{dag_id="dataops_to_mlops_integration"}) > 7200
        for: 10m
        annotations:
          summary: "MLOps pipeline lag > 2 hours"
```

## Troubleshooting

### Problema 1: DAG MLOps não triggera

**Causas**:
- Datamart não salvo corretamente no S3
- S3KeySensor timeout
- Permissões AWS incorretas

**Solução**:
```bash
# Verificar se arquivo existe
aws s3 ls s3://bucket/datamarts/project/latest.parquet

# Verificar permissões Airflow
aws s3 cp s3://bucket/datamarts/project/latest.parquet /tmp/ --profile airflow

# Verificar logs do sensor
# Airflow UI → DAG → Task → Logs
```

### Problema 2: SSH timeout

**Causas**:
- EC2 não acessível
- Security group bloqueando porta 22
- Chave SSH incorreta

**Solução**:
```bash
# Testar SSH manual
ssh -i key.pem ec2-user@<ec2-ip>

# Verificar security group
# AWS Console → EC2 → Security Groups → Inbound Rules
# Adicionar: Type SSH, Port 22, Source <airflow-ip>/32

# Testar no Airflow
# Admin → Connections → Test (botão)
```

### Problema 3: Training falha

**Causas**:
- Features path incorreto
- Dependências Python faltando
- OOM (Out of Memory)

**Solução**:
```bash
# Verificar logs na EC2
ssh ec2-user@<ec2-ip>
tail -f /opt/mlops/logs/training/training_*.log

# Verificar features exist
aws s3 ls s3://bucket/ml/features/project/latest.parquet

# Verificar recursos EC2
htop  # RAM, CPU
df -h  # Disk
```

## Exemplo Completo

Ver arquivo: `airflow/dags/mlops/dataops_to_mlops_integration.py`

Esse DAG implementa:
- ✅ S3 Sensor para datamart
- ✅ Data quality validation
- ✅ Feature engineering
- ✅ Training trigger (3 opções comentadas)
- ✅ Error handling e retry
- ✅ XCom para passar dados entre tasks
