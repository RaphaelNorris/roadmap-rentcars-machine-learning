# Quick Start - Primeiros Passos com o Template MLOps

Este guia rapido vai te ajudar a ter o primeiro modelo em producao em menos de 30 minutos.

## Pre-requisitos Rapidos

```bash
# Verifique se tem tudo instalado
python --version    # 3.11+
docker --version    # 20.10+
aws --version       # 2.0+
```

---

## Passo 1: Criar Projeto (2 min)

```bash
# Instale cookiecutter
pip install cookiecutter

# Gere o projeto
cookiecutter https://github.com/RaphaelNorris/project-template-ds-rentcars.git

# Responda aos prompts
project_name: credit_risk_model
repo_name: credit-risk
author_name: Seu Nome
email: seu@email.com
description: Modelo de risco de credito

# Entre no diretorio
cd credit-risk
```

---

## Passo 2: Configurar Ambiente (5 min)

```bash
# Crie .env a partir do exemplo
cp .env.example .env

# Edite .env com suas configuracoes minimas
nano .env
```

**Configuracoes essenciais no .env**:
```env
# AWS - OBRIGATORIO
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=sua-chave
AWS_SECRET_ACCESS_KEY=seu-secret

# S3 Buckets - OBRIGATORIO
S3_RAW_BUCKET=seu-bucket-raw
S3_PROCESSED_BUCKET=seu-bucket-processed
S3_ML_ARTIFACTS_BUCKET=seu-bucket-ml

# Athena
ATHENA_DATABASE=ml_database
ATHENA_OUTPUT_LOCATION=s3://seu-bucket-athena-results/

# MLFlow - Deixe padrao para desenvolvimento local
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=quick_start_experiment
```

```bash
# Configure AWS CLI
aws configure

# Teste conexao
aws s3 ls
```

---

## Passo 3: Setup Automatizado (3 min)

```bash
# Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate  # Windows

# Instale dependencias
pip install -r requirements.txt

# Setup automatizado (cria buckets, ECR, etc)
make setup
```

**O que o setup faz**:
- Cria buckets S3
- Configura Athena database
- Cria repositories ECR
- Inicia MLFlow server
- Instala pre-commit hooks

---

## Passo 4: Iniciar Servicos (2 min)

```bash
# Inicie apenas MLFlow para comecar
make mlflow-up

# Aguarde ate ver:
# "Listening at: http://0.0.0.0:5000"

# Verifique
curl http://localhost:5000/health
```

**Acesse MLFlow UI**: http://localhost:5000

---

## Passo 5: Preparar Dados de Exemplo (5 min)

```bash
# Crie notebook para explorar dados
jupyter notebook notebooks/
```

**Notebook de exemplo** (`notebooks/1-data/load_sample_data.ipynb`):

```python
import pandas as pd
import numpy as np
from src.data.aws_integration import get_s3_client

# Gerar dados de exemplo
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
    'total_transactions_30d': np.random.randint(1, 50, n_samples),
    'avg_transaction_value_30d': np.random.uniform(10, 500, n_samples),
    'days_since_last_transaction': np.random.randint(0, 90, n_samples),
    'customer_segment': np.random.choice(['premium', 'regular', 'bronze'], n_samples),
    'target_churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

# Salvar em S3
s3 = get_s3_client()
s3.write_parquet(
    df=data,
    s3_key='ml/features/sample_features.parquet'
)

print(f"Dados salvos: {len(data)} registros")
print(f"Features: {list(data.columns)}")
```

---

## Passo 6: Treinar Primeiro Modelo (5 min)

```python
# notebooks/5-models/train_first_model.ipynb

from src.pipelines.DS.training_pipeline import TrainingPipeline
from src.model.mlflow_manager import MLFlowManager

# Inicializar pipeline
pipeline = TrainingPipeline(
    experiment_name="quick_start_experiment",
    model_name="churn_prediction_model"
)

# Treinar modelo
results = pipeline.run(
    s3_path="ml/features/sample_features.parquet",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    register_model=True
)

print(f"\nTreinamento concluido!")
print(f"Modelo: {results['model_name']}")
print(f"Metricas:")
for metric, value in results['metrics'].items():
    print(f"  {metric}: {value:.3f}")
```

**Output esperado**:
```
Treinamento concluido!
Modelo: churn_prediction_model
Metricas:
  accuracy: 0.850
  precision: 0.820
  recall: 0.880
  f1: 0.850
  roc_auc: 0.910
```

**Verifique no MLFlow**: http://localhost:5000

---

## Passo 7: Servir Modelo via API (3 min)

```bash
# Terminal 1: Inicie API
make serve

# Aguarde ate ver:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Teste a API**:

```bash
# Terminal 2: Teste predicao
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "total_transactions_30d": 25,
      "avg_transaction_value_30d": 150.5,
      "days_since_last_transaction": 5,
      "customer_segment": "premium"
    }
  }'
```

**Response esperada**:
```json
{
  "predictions": [0],
  "model_metadata": {
    "name": "churn_prediction_model",
    "version": "1",
    "stage": "None"
  },
  "inference_time_ms": 12.5
}
```

**Acesse documentacao da API**: http://localhost:8000/docs

---

## Passo 8: Promover para Production (2 min)

```python
from src.model.mlflow_manager import MLFlowManager

mlflow = MLFlowManager()

# Obter ultima versao
latest_version = mlflow.get_latest_model_version("churn_prediction_model")

# Promover para Production
mlflow.transition_model_stage(
    name="churn_prediction_model",
    version=latest_version.version,
    stage="Production"
)

print(f"Modelo v{latest_version.version} promovido para Production!")
```

**Recarregue modelo na API**:
```bash
curl -X POST http://localhost:8000/model/reload
```

---

## Passo 9: Monitorar Modelo (3 min)

```bash
# Inicie stack de monitoramento
make monitoring-up

# Aguarde servicos iniciarem
```

**Acesse dashboards**:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

**Configure dashboard basico no Grafana**:
1. Add Data Source → Prometheus → http://prometheus:9090
2. Import Dashboard → ID 12345 (exemplo)
3. Visualize metricas do modelo

---

## Passo 10: Fazer Predicoes em Batch (2 min)

```python
# notebooks/5-models/batch_predictions.ipynb

import pandas as pd
from src.inference.api import predict

# Carregar novos dados
new_customers = pd.read_parquet('s3://bucket/new_customers.parquet')

# Fazer predicoes em batch
predictions = []
for _, customer in new_customers.iterrows():
    pred = predict(features=customer.to_dict())
    predictions.append({
        'customer_id': customer['customer_id'],
        'churn_prediction': pred['predictions'][0],
        'churn_probability': max(pred['probability'])
    })

results_df = pd.DataFrame(predictions)

# Salvar resultados
from src.data.aws_integration import get_s3_client
s3 = get_s3_client()
s3.write_parquet(results_df, 'ml/predictions/batch_20240115.parquet')

print(f"Predicoes geradas: {len(results_df)}")
print(f"Churn previsto: {(results_df['churn_prediction'] == 1).sum()} clientes")
```

---

## Comandos Uteis para o Dia a Dia

```bash
# Ver status dos servicos
docker-compose ps

# Ver logs
docker-compose logs mlflow -f

# Treinar novo modelo
make train

# Rodar testes
make test

# Verificar qualidade do codigo
make quality

# Limpar ambiente
make clean
```

---

## Proximo Nivel

Agora que voce tem o basico funcionando:

### 1. Adicione Seus Dados Reais

```python
# Conecte ao seu banco de dados
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@host:5432/db')
df = pd.read_sql('SELECT * FROM customers', engine)

# Processe e salve
from src.pipelines.DS.feature_pipeline import FeaturePipeline
pipeline = FeaturePipeline()
features = pipeline.transform(df)
```

### 2. Configure Airflow para Automacao

```bash
# Inicie Airflow
docker-compose run --rm airflow-webserver airflow db init
docker-compose up -d airflow-webserver airflow-scheduler

# Acesse: http://localhost:8080
# Login: admin/admin

# Ative DAG
# DAGs → ml_training_pipeline → Toggle On
```

### 3. Configure CI/CD

```bash
# Configure secrets no GitHub
# Settings → Secrets → Actions → New secret

# Adicione:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - MLFLOW_TRACKING_URI
# - S3_ML_ARTIFACTS_BUCKET

# Push para main para trigger CI/CD
git push origin main
```

### 4. Melhore o Modelo

```python
# Hyperparameter tuning
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    # ... treinar e avaliar
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 5. Configure Alertas

```python
# Adicione webhook do Slack em .env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Configure alertas
from src.inference.monitoring import AlertManager

alerts = AlertManager(alert_channels=['slack'])
alerts.send_alert(
    alert_type='model_performance',
    message='Accuracy dropped below 80%',
    severity='warning'
)
```

---

## Troubleshooting Rapido

### MLFlow nao inicia
```bash
docker-compose restart mlflow postgres
docker-compose logs mlflow
```

### AWS Access Denied
```bash
aws sts get-caller-identity
aws configure
```

### API retorna erro 503
```bash
# Verifique se modelo esta carregado
curl http://localhost:8000/model/info

# Recarregue modelo
curl -X POST http://localhost:8000/model/reload
```

### Dados nao encontrados no S3
```bash
aws s3 ls s3://seu-bucket/ml/features/
# Se vazio, execute Passo 5 novamente
```

---

## Recursos de Aprendizado

### Documentacao
- [Arquitetura MLOps](mlops_architecture.md) - Visao completa
- [Estrutura de Dados](data_structure.md) - Organizacao dos dados
- [Pipelines](pipelines.md) - Detalhes dos pipelines

### Tutoriais
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Airflow Quick Start](https://airflow.apache.org/docs/apache-airflow/stable/start.html)

### Comunidade
- GitHub Issues: [Reporte bugs](https://github.com/RaphaelNorris/project-template-ds-rentcars/issues)
- Discussions: [Tire duvidas](https://github.com/RaphaelNorris/project-template-ds-rentcars/discussions)

---

## Parabens!

Voce agora tem um pipeline completo de MLOps funcionando:

- [x] Modelo treinado e versionado no MLFlow
- [x] API REST para inferencia em tempo real
- [x] Monitoramento com Prometheus e Grafana
- [x] Infraestrutura AWS configurada
- [x] Pipeline pronto para producao

**Proximos passos**: Customize para seu caso de uso e coloque em producao!
