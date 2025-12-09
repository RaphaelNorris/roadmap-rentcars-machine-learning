# Documentacao de Pipelines MLOps

## Visao Geral

Este projeto implementa pipelines completos para MLOps seguindo a arquitetura **FTI (Feature, Training, Inference)**. Cada pipeline e projetado para ser modular, escalavel e facil de manter.

```
Feature Pipeline ’ Training Pipeline ’ Inference Pipeline
                         “
                   Model Registry
                         “
                    Monitoring
```

---

## Indice

1. [Feature Pipeline](#1-feature-pipeline)
2. [Training Pipeline](#2-training-pipeline)
3. [Inference Pipeline](#3-inference-pipeline)
4. [Monitoring Pipeline](#4-monitoring-pipeline)
5. [Data Engineering Pipelines](#5-data-engineering-pipelines)
6. [Airflow DAGs](#6-airflow-dags)

---

## 1. Feature Pipeline

**Objetivo**: Transformar dados raw em features prontas para ML

### 1.1 Arquitetura

```
Raw Data (Bronze/Silver)
    “
Feature Engineering
    “
Feature Validation
    “
Feature Store (Iceberg)
```

### 1.2 Implementacao

**Localizacao**: `src/pipelines/DS/feature_pipeline/`

**Componentes**:
- `extract.py`: Extrai dados das fontes
- `transform.py`: Aplica transformacoes
- `validate.py`: Valida qualidade das features
- `load.py`: Carrega no Feature Store

**Exemplo de Uso**:

```python
from src.pipelines.DS.feature_pipeline import FeaturePipeline

# Inicializar pipeline
pipeline = FeaturePipeline(
    input_path="s3://bucket/silver/customers/",
    output_path="ml/features/customer_features/",
    feature_date="2024-01-15"
)

# Executar pipeline
features = pipeline.run()

# Features criadas
print(features.columns)
# ['customer_id', 'total_transactions_30d', 'avg_value_30d', ...]
```

### 1.3 Features Implementadas

#### Features de Agregacao
```python
# Agregacoes temporais
- total_transactions_Nd (7, 30, 90 dias)
- avg_transaction_value_Nd
- max_transaction_value_Nd
- min_transaction_value_Nd
- std_transaction_value_Nd
```

#### Features Categoricas
```python
# Encoding de categorias
- customer_segment (premium, regular, bronze)
- preferred_category
- preferred_payment_method
```

#### Features Temporais
```python
# Padroes temporais
- days_since_last_transaction
- transactions_weekend_ratio
- peak_hour_transactions
- monthly_seasonality
```

#### Features Comportamentais
```python
# Comportamento do usuario
- churn_score
- engagement_score
- recency_score
- frequency_score
- monetary_score (RFM)
```

### 1.4 Feature Store

**Tecnologia**: Apache Iceberg via Athena

**Schema**:
```sql
CREATE TABLE feature_store.customer_features (
    customer_id STRING,
    feature_date DATE,

    -- Agregacoes
    total_transactions_30d INT,
    avg_transaction_value_30d DOUBLE,

    -- Categoricas
    customer_segment STRING,
    preferred_category STRING,

    -- Temporais
    days_since_last_transaction INT,
    transactions_weekend_ratio DOUBLE,

    -- Metadata
    _feature_version STRING,
    _created_at TIMESTAMP
)
USING iceberg
PARTITIONED BY (months(feature_date))
```

**Versionamento**:
```python
# Features sao versionadas automaticamente
{
    "_feature_version": "v1.2",
    "_created_at": "2024-01-15T12:00:00Z",
    "_feature_hash": "abc123...",
}
```

### 1.5 Validacao de Features

```python
import pandera as pa

# Schema de validacao
feature_schema = pa.DataFrameSchema({
    "customer_id": pa.Column(pa.String, nullable=False),
    "total_transactions_30d": pa.Column(pa.Int, ge=0),
    "avg_transaction_value_30d": pa.Column(pa.Float, ge=0),
    # ... mais validacoes
})

# Validar features
validated_features = feature_schema.validate(features)
```

---

## 2. Training Pipeline

**Objetivo**: Treinar, avaliar e registrar modelos de ML

### 2.1 Arquitetura

```
Feature Store
    “
Data Preparation
    “
Model Training + MLFlow Tracking
    “
Model Evaluation
    “
Model Registration (MLFlow Registry)
    “
Model Promotion (Staging/Production)
```

### 2.2 Implementacao

**Localizacao**: `src/pipelines/DS/training_pipeline/train.py`

**Fluxo Completo**:

```python
from src.pipelines.DS.training_pipeline import TrainingPipeline

# Inicializar pipeline
pipeline = TrainingPipeline(
    experiment_name="production_training",
    model_name="customer_churn_model"
)

# Executar training completo
results = pipeline.run(
    query="SELECT * FROM feature_store.customer_features WHERE feature_date >= '2024-01-01'",
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1
    },
    register_model=True
)

print(f"Model: {results['model_name']}")
print(f"Metrics: {results['metrics']}")
# Output:
# Model: customer_churn_model
# Metrics: {'accuracy': 0.85, 'f1': 0.82, 'roc_auc': 0.91}
```

### 2.3 Etapas do Pipeline

#### Etapa 1: Load Features
```python
def load_features(self, query=None, s3_path=None):
    """Carrega features do Feature Store"""
    if query:
        df = self.athena.query(query)
    else:
        df = self.iceberg.read_features("feature_store")
    return df
```

#### Etapa 2: Preprocess Data
```python
def preprocess_data(self, df, target_column="target"):
    """Preprocessamento e split"""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))

    # Feature scaling (opcional)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    return X, y
```

#### Etapa 3: Split Data
```python
def split_data(self, X, y, test_size=0.2, random_state=42):
    """Split train/test estratificado"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
```

#### Etapa 4: Train Model
```python
def train_model(self, X_train, y_train, model_type="xgboost"):
    """Treina modelo com autologging"""
    # Enable MLFlow autologging
    mlflow.sklearn.autolog()

    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    model.fit(X_train, y_train)
    return model
```

#### Etapa 5: Evaluate Model
```python
def evaluate_model(self, model, X_test, y_test):
    """Calcula metricas de avaliacao"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }

    # Log metrics to MLFlow
    mlflow.log_metrics(metrics)

    return metrics
```

### 2.4 Hyperparameter Tuning

**Com Optuna**:

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Definir hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

    # Treinar modelo
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Avaliar
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)

    return score

# Otimizar
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value}")
```

### 2.5 Model Registry

**Registro no MLFlow**:

```python
# Registrar modelo
registered_model = mlflow_manager.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="customer_churn_model",
    tags={
        "model_type": "xgboost",
        "data_version": "v1.2",
        "feature_version": "v1.2"
    },
    description="Customer churn prediction model"
)

# Transicionar para Production
mlflow_manager.transition_model_stage(
    name="customer_churn_model",
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True
)
```

---

## 3. Inference Pipeline

**Objetivo**: Gerar predicoes usando modelos em producao

### 3.1 Modos de Inferencia

#### 3.1.1 Online (Real-time)

**API FastAPI**: `src/inference/api.py`

```python
# Request
POST /predict
{
    "features": {
        "total_transactions_30d": 15,
        "avg_transaction_value_30d": 150.50,
        "days_since_last_transaction": 5,
        ...
    }
}

# Response
{
    "predictions": [0],
    "probability": [0.75, 0.25],
    "model_metadata": {
        "name": "customer_churn_model",
        "version": "3",
        "stage": "Production"
    },
    "inference_time_ms": 15.2
}
```

**Uso do Cliente**:

```python
import requests

# Fazer predicao
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {
            "total_transactions_30d": 15,
            "avg_transaction_value_30d": 150.50,
            ...
        }
    }
)

prediction = response.json()
print(f"Churn prediction: {prediction['predictions'][0]}")
print(f"Confidence: {max(prediction['probability'])}")
```

#### 3.1.2 Batch (Em lote)

**Pipeline Airflow**: `airflow/dags/ml_batch_inference_pipeline.py`

```python
from src.pipelines.DS.inference_pipeline import BatchInferencePipeline

# Inicializar pipeline
pipeline = BatchInferencePipeline(
    model_name="customer_churn_model",
    model_stage="Production"
)

# Executar batch inference
results = pipeline.run(
    input_data=customers_df,
    output_path="ml/predictions/batch_20240115.parquet"
)

print(f"Generated {results['n_predictions']} predictions")
```

### 3.2 Feature Consistency

**Importante**: Features de training devem ser identicas as de inference

```python
# Reutilizar feature engineering
from src.pipelines.DS.feature_pipeline import FeatureTransformer

# Training
transformer = FeatureTransformer()
train_features = transformer.fit_transform(train_data)

# Salvar transformer
joblib.dump(transformer, "artifacts/feature_transformer.pkl")

# Inference
transformer = joblib.load("artifacts/feature_transformer.pkl")
inference_features = transformer.transform(new_data)
```

### 3.3 Logging de Predicoes

Todas as predicoes sao logadas para monitoramento:

```python
{
    "prediction_id": "PRED_123456",
    "timestamp": "2024-01-15T15:00:00Z",
    "model_version": "v3",
    "features": {...},
    "prediction": 0,
    "probability": [0.75, 0.25],
    "confidence": 0.75,
    "inference_latency_ms": 15.2
}
```

---

## 4. Monitoring Pipeline

**Objetivo**: Monitorar performance e detectar drift

### 4.1 Componentes

#### 4.1.1 Data Drift Detection

```python
from src.inference.monitoring import ModelMonitor

# Inicializar monitor
monitor = ModelMonitor(
    reference_data_path="ml/training_sets/train_v1.parquet",
    drift_threshold=0.05
)

# Detectar drift
drift_results = monitor.detect_data_drift(
    current_data=recent_predictions_df
)

if drift_results["drift_detected"]:
    print(f"ALERT: Data drift detected!")
    print(f"Report: {drift_results['report_path']}")
```

#### 4.1.2 Model Performance Monitoring

```python
# Monitorar metricas
metrics = {
    "accuracy": current_accuracy,
    "f1_score": current_f1,
    "prediction_volume": n_predictions,
    "avg_latency_ms": avg_latency
}

# Enviar para Prometheus
for metric_name, value in metrics.items():
    prometheus_gauge.labels(model="churn_model").set(value)
```

#### 4.1.3 Alerting

```python
from src.inference.monitoring import AlertManager

alert_manager = AlertManager(
    alert_channels=["email", "slack"]
)

# Enviar alerta
alert_manager.send_alert(
    alert_type="drift",
    message="Data drift detected in customer features",
    severity="warning",
    metadata=drift_results
)
```

### 4.2 Dashboards Grafana

**Metricas Monitoradas**:
- Model accuracy over time
- Prediction volume
- Inference latency (p50, p95, p99)
- Data drift score
- Feature distributions

---

## 5. Data Engineering Pipelines

### 5.1 Data Ingestion Pipeline

**Localizacao**: `src/pipelines/DE/data_pipeline/`

**Objetivo**: Ingerir dados raw para Bronze layer

```python
from src.pipelines.DE.data_pipeline import DataIngestionPipeline

pipeline = DataIngestionPipeline(
    source="crm_database",
    destination="s3://bucket/bronze/crm/"
)

pipeline.run(
    extraction_query="SELECT * FROM customers WHERE updated_at >= CURRENT_DATE",
    partition_by="date"
)
```

### 5.2 Data Processing Pipeline

**Objetivo**: Processar Bronze ’ Silver

```python
from src.pipelines.DE.data_pipeline import DataProcessingPipeline

pipeline = DataProcessingPipeline(
    input_path="bronze/crm/customers/",
    output_path="silver/customers/"
)

pipeline.run(
    transformations=[
        "remove_duplicates",
        "standardize_columns",
        "validate_schema",
        "enrich_metadata"
    ]
)
```

---

## 6. Airflow DAGs

### 6.1 ML Training Pipeline DAG

**Arquivo**: `airflow/dags/ml_training_pipeline.py`

**Schedule**: Diario (2 AM)

**Tasks**:
1. `wait_for_data`: Aguarda novos dados no S3
2. `extract_features`: Extrai features do Feature Store
3. `prepare_training_data`: Prepara datasets train/test
4. `train_model`: Treina modelo com MLFlow
5. `evaluate_model`: Avalia performance
6. `promote_model`: Promove para Production (se aprovado)
7. `send_notification`: Notifica time

**Execucao Manual**:
```bash
# Via Airflow UI
http://localhost:8080 ’ DAGs ’ ml_training_pipeline ’ Trigger DAG

# Via CLI
airflow dags trigger ml_training_pipeline

# Via Make
airflow dags trigger ml_training_pipeline
```

### 6.2 Batch Inference Pipeline DAG

**Arquivo**: `airflow/dags/ml_batch_inference_pipeline.py`

**Schedule**: A cada 6 horas

**Tasks**:
1. `load_inference_data`: Carrega dados para inferencia
2. `load_production_model`: Carrega modelo do MLFlow
3. `generate_predictions`: Gera predicoes em lote
4. `save_to_iceberg`: Salva resultados no Iceberg
5. `monitor_predictions`: Monitora drift
6. `check_drift_status`: Branch baseado em drift
7a. `handle_drift`: Trata drift detectado
7b. `complete_pipeline`: Finaliza pipeline

---

## 7. Comandos Uteis

### 7.1 Executar Pipelines Localmente

```bash
# Feature pipeline
python -m src.pipelines.DS.feature_pipeline

# Training pipeline
make train

# Batch inference
python -m src.pipelines.DS.inference_pipeline.batch
```

### 7.2 Monitorar Pipelines

```bash
# Ver logs do Airflow
docker-compose logs airflow-scheduler airflow-webserver

# Ver metricas no Prometheus
open http://localhost:9090

# Ver dashboards no Grafana
open http://localhost:3000
```

### 7.3 Debug

```bash
# Teste unitario de pipeline
pytest tests/pipelines/test_training_pipeline.py -v

# Executar com debug
python -m pdb src/pipelines/DS/training_pipeline/train.py

# Ver traces do MLFlow
mlflow ui --port 5000
```

---

## 8. Melhores Praticas

### 8.1 Desenvolvimento de Pipelines

1. **Modularidade**: Cada etapa deve ser independente
2. **Testabilidade**: Escreva testes para cada componente
3. **Idempotencia**: Pipelines devem ser seguros para reexecucao
4. **Logging**: Log detalhado de cada etapa
5. **Error Handling**: Tratamento robusto de erros

### 8.2 Performance

1. **Particoes**: Use particoes para otimizar leitura
2. **Caching**: Cache features intermediarias
3. **Paralelizacao**: Processe em paralelo quando possivel
4. **Formato**: Use Parquet para eficiencia

### 8.3 Monitoramento

1. **Metricas**: Monitore tempo de execucao e recursos
2. **Alertas**: Configure alertas para falhas
3. **SLA**: Defina SLAs para cada pipeline
4. **Logs**: Centralize logs para analise

---

## 9. Troubleshooting

### Problema: Pipeline falha ao ler dados

**Solucao**:
```bash
# Verificar se dados existem
aws s3 ls s3://bucket/path/

# Verificar permissoes
aws s3 ls s3://bucket/ --profile your-profile

# Testar query Athena
aws athena start-query-execution \
  --query-string "SELECT * FROM table LIMIT 10"
```

### Problema: MLFlow nao registra modelo

**Solucao**:
```python
# Verificar conexao
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print(mlflow.get_tracking_uri())

# Verificar experimento
experiment = mlflow.get_experiment_by_name("production_training")
print(f"Experiment ID: {experiment.experiment_id}")

# Listar runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
print(runs)
```

### Problema: Airflow DAG nao aparece

**Solucao**:
```bash
# Verificar erros no DAG
airflow dags list-import-errors

# Recarregar DAGs
docker-compose restart airflow-scheduler

# Ver logs
docker-compose logs airflow-scheduler
```

---

## 10. Proximos Passos

Apos entender os pipelines:

1. Customize os pipelines para seu caso de uso
2. Adicione novas features ao Feature Pipeline
3. Experimente diferentes modelos no Training Pipeline
4. Configure alertas personalizados
5. Otimize performance dos pipelines

---

## Referencias

- [FTI Pipeline Pattern](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)
- [MLFlow Pipelines](https://mlflow.org/docs/latest/pipelines.html)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Feature Store Concepts](https://www.featurestore.org/)
