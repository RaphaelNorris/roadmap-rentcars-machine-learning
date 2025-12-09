# Config - Configurações por Ambiente

Configurações separadas por ambiente (dev, staging, prod).

## Estrutura

```
config/
├── dev/
│   ├── model_config.yaml       # Hiperparâmetros para dev
│   ├── api_config.yaml         # Configurações de API
│   └── monitoring_config.yaml  # Thresholds de monitoramento
│
├── staging/
│   ├── model_config.yaml
│   ├── api_config.yaml
│   └── monitoring_config.yaml
│
├── prod/
│   ├── model_config.yaml
│   ├── api_config.yaml
│   └── monitoring_config.yaml
│
├── grafana/
│   ├── dashboards/             # JSON dashboards
│   └── datasources/            # Datasource configs
│
└── prometheus/
    └── prometheus.yml          # Prometheus config
```

## Uso

### Carregar Configurações

```python
from src.utils.config import load_config

# Carrega config baseado em ENV
config = load_config(env='prod')

# Acessar configurações
model_params = config['model']['hyperparameters']
api_timeout = config['api']['timeout']
```

### Exemplo: model_config.yaml (prod)

```yaml
model:
  type: random_forest
  hyperparameters:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 10
    random_state: 42

  training:
    test_size: 0.2
    cv_folds: 5
    random_state: 42

  performance_thresholds:
    min_accuracy: 0.80
    min_precision: 0.75
    min_recall: 0.70
    min_f1: 0.72

monitoring:
  drift_detection:
    enabled: true
    threshold: 0.05
    schedule: "0 2 * * *"  # Daily 2 AM

  performance_degradation:
    enabled: true
    threshold: 0.05
    window_days: 7

inference:
  batch_size: 1000
  timeout_seconds: 30
  cache_ttl: 3600
```

## Ambientes

### Dev
- **Propósito**: Desenvolvimento local
- **Dados**: Amostras pequenas (1K-10K rows)
- **Modelo**: Hiperparâmetros para treino rápido
- **Thresholds**: Relaxados

### Staging
- **Propósito**: Validação pré-produção
- **Dados**: Subset representativo (10%-20% prod)
- **Modelo**: Hiperparâmetros similares a prod
- **Thresholds**: Iguais a prod

### Prod
- **Propósito**: Produção
- **Dados**: Dados completos
- **Modelo**: Hiperparâmetros otimizados
- **Thresholds**: Rigorosos

## Grafana Dashboards

Dashboards pré-configurados:

1. **ML Model Performance**: Accuracy, precision, recall por versão
2. **Inference Metrics**: Latency, throughput, error rate
3. **Data Drift**: Drift score por feature
4. **Business Metrics**: ROI, false positives, false negatives

## Prometheus

Coleta métricas de:
- MLFlow Server
- Inference API
- PostgreSQL
- Airflow
