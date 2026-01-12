# Padrões de Nomenclatura e Desenvolvimento

## Visão Geral

Este documento define os padrões de nomenclatura e desenvolvimento para projetos de ML na RentCars.

---

## Nomenclatura de Projetos

### Nome do Projeto

| Regra | Exemplo |
|-------|---------|
| Lowercase | `recsys`, `churn` |
| Sem espaços (usar underscore) | `churn_prediction` |
| Descritivo e curto | `pricing_optimizer` |
| Sem prefixos redundantes | `recsys` (não `ml_recsys`) |

**Exemplos válidos:**
```
recsys
churn_prediction
pricing_optimizer
fraud_detection
nps_predictor
```

**Exemplos inválidos:**
```
RecSys              # Não usar CamelCase
ml-recsys           # Não usar hífen
ml_modelo_recsys    # Prefixo redundante
rec                 # Pouco descritivo
```

---

## Nomenclatura de Branches

### Padrão

```
<tipo>/<projeto>-<descricao>
```

### Tipos de Branch

| Tipo | Uso | Exemplo |
|------|-----|---------|
| `feature/` | Nova funcionalidade | `feature/recsys-collaborative-filter` |
| `fix/` | Correção de bug | `fix/churn-null-handling` |
| `refactor/` | Refatoração | `refactor/recsys-optimize-inference` |
| `docs/` | Documentação | `docs/recsys-add-readme` |
| `experiment/` | Experimento exploratório | `experiment/recsys-transformer` |

### Exemplos

```bash
# Features
feature/recsys-add-user-features
feature/churn-xgboost-model
feature/pricing-dynamic-rules

# Fixes
fix/recsys-memory-leak
fix/churn-missing-values

# Experiments
experiment/recsys-deep-learning
experiment/churn-automl
```

### Regras

- Sempre em lowercase
- Usar hífen para separar palavras
- Máximo 50 caracteres
- Sempre partir da branch `dev`

---

## Nomenclatura no MLFlow

### Experiments

```
<projeto>
```

| Projeto | Experiment Name |
|---------|-----------------|
| recsys | `recsys` |
| churn_prediction | `churn_prediction` |

### Registered Models

```
<projeto>
```

| Projeto | Model Name |
|---------|------------|
| recsys | `recsys` |
| churn_prediction | `churn_prediction` |

### Run Names

```
<descricao>-<data>
```

**Exemplos:**
```
baseline-20250106
xgboost-tuned-20250106
feature-v2-20250107
```

---

## Nomenclatura AWS

### ECR Images

```
<ecr-repo>:<projeto>-<versao>
```

**Exemplo:**
```
rentcars-data-platform-ecr-prd:recsys-1.0.0
```

### ECS Cluster

```
ml-cluster-<projeto>
```

**Exemplo:**
```
ml-cluster-recsys
```

### Step Functions

```
<projeto>-workflow
```

**Exemplo:**
```
recsys-workflow
```

### EventBridge Scheduler

```
sched-<projeto>
```

**Exemplo:**
```
sched-recsys
```

### CloudWatch Log Group

```
/ecs/<projeto>
```

**Exemplo:**
```
/ecs/recsys
```

---

## Estrutura de Projeto

### Estrutura Padrão

```
src/projects/<projeto>/
├── config.yaml              # Configuração do projeto
├── main.py                  # Entry point principal
├── steps/                   # Scripts de cada step (opcional)
│   ├── extract_features.py
│   ├── train_model.py
│   └── score_model.py
├── notebooks/               # Notebooks de desenvolvimento (opcional)
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── tests/                   # Testes (opcional)
│   └── test_model.py
├── requirements.txt         # Dependências
├── Dockerfile               # Imagem Docker
└── README.md                # Documentação do projeto
```

### Estrutura Mínima

```
src/projects/<projeto>/
├── config.yaml
├── main.py
├── requirements.txt
└── Dockerfile
```

---

## Nomenclatura de Arquivos

### Scripts Python

| Tipo | Padrão | Exemplo |
|------|--------|---------|
| Entry point | `main.py` | `main.py` |
| Step de features | `extract_features.py` | `extract_features.py` |
| Step de treino | `train_model.py` | `train_model.py` |
| Step de inferência | `score_model.py` | `score_model.py` |
| Step de pós-processamento | `post_process.py` | `post_process.py` |
| Utilitários | `utils.py` | `utils.py` |

### Notebooks

```
<numero>_<descricao>.ipynb
```

**Exemplos:**
```
01_eda.ipynb
02_feature_engineering.ipynb
03_modeling.ipynb
04_evaluation.ipynb
```

---

## Nomenclatura no Código

### Variáveis e Funções

| Tipo | Padrão | Exemplo |
|------|--------|---------|
| Variáveis | snake_case | `user_features` |
| Funções | snake_case | `extract_features()` |
| Constantes | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| Classes | PascalCase | `FeatureExtractor` |

### DataFrames

```python
# Padrão: df_<descricao>
df_raw = pd.read_parquet("...")
df_features = extract_features(df_raw)
df_predictions = model.predict(df_features)
```

### Modelos

```python
# Padrão: model ou model_<tipo>
model = xgb.XGBClassifier()
model_baseline = LogisticRegression()
model_tuned = xgb.XGBClassifier(**best_params)
```

---

## Nomenclatura de Métricas

### MLFlow Metrics

| Métrica | Nome | Exemplo |
|---------|------|---------|
| Classificação | `accuracy`, `precision`, `recall`, `f1`, `roc_auc` | `mlflow.log_metric("roc_auc", 0.85)` |
| Regressão | `rmse`, `mae`, `mape`, `r2` | `mlflow.log_metric("rmse", 0.12)` |
| Ranking | `ndcg`, `map`, `mrr` | `mlflow.log_metric("ndcg@10", 0.75)` |
| Custom | `<descricao>` | `mlflow.log_metric("conversion_rate", 0.15)` |

### MLFlow Parameters

```python
# Hiperparâmetros
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("max_depth", 6)
mlflow.log_param("n_estimators", 100)

# Configurações
mlflow.log_param("feature_version", "v2")
mlflow.log_param("train_size", 0.8)
```

---

## Versionamento

### Versão do Projeto (config.yaml)

```
<major>.<minor>.<patch>
```

| Tipo | Quando incrementar | Exemplo |
|------|-------------------|---------|
| Major | Mudança breaking | `1.0.0` → `2.0.0` |
| Minor | Nova feature | `1.0.0` → `1.1.0` |
| Patch | Bug fix | `1.0.0` → `1.0.1` |

**Exemplo no config.yaml:**
```yaml
project_name: recsys
version: "1.2.3"
```

### Versão do Modelo (MLFlow)

- Automático pelo MLFlow (v1, v2, v3...)
- Não controlar manualmente

---

## Padrões de Código

### Imports

```python
# Ordem dos imports
# 1. Standard library
import os
import json
from datetime import datetime

# 2. Third party
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

# 3. Local
from utils import load_data
```

### Docstrings

```python
def extract_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extrai features do DataFrame de entrada.

    Args:
        df: DataFrame com dados brutos
        config: Configurações de feature engineering

    Returns:
        DataFrame com features extraídas
    """
    pass
```

### Type Hints

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict
) -> xgb.XGBClassifier:
    """Treina modelo XGBoost."""
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Iniciando pipeline...")
    logger.info(f"Processando {len(df)} registros")
    logger.warning("Dados faltantes encontrados")
    logger.error("Falha ao carregar modelo")
```

---

## Padrões de Configuração

### config.yaml

```yaml
# Metadata
project_name: recsys                              # lowercase, sem espaços
description: "Sistema de recomendação"            # Descrição breve
version: "1.0.0"                                  # Semantic versioning

# Owners
owners:
  tech_owner: "email@rentcars.com"               # Email válido
  business_owner: "email@rentcars.com"

# Tags
tags:
  domain: "reservas"                              # Domínio de negócio
  criticality: "alta"                             # alta | media | baixa

# Schedule
schedule:
  enabled: true
  cron: "0 3 * * *"                               # Cron expression (AWS)
  timeout_seconds: 3600                           # Timeout em segundos
  retries: 2                                      # Número de retries

# Runtime
runtime:
  python_version: "3.11"                          # Versão do Python
  requirements_file: "requirements.txt"
  env_vars:
    ENV: "dev"                                    # Variáveis de ambiente

# Pipeline
batch:
  enabled: true
  orchestrator: "step_functions"
  pipeline:
    steps:
      - id: "extract_features"                   # ID único, snake_case
        name: "Extract Features"                  # Nome descritivo
        command: "python main.py"
        resources:
          cpu: 1.0
          memory_gb: 4

# Infra
infra:
  compute:
    provider: "ec2"
    ec2_type: "t3.xlarge"
    min_quantity: 0
    max_quantity: 2
    spot: true

# Notifications
notifications:
  on_success:
    - "team@rentcars.com"
  on_failure:
    - "team@rentcars.com"
    - "oncall@rentcars.com"
```

---

## Padrões de Commit

### Formato

```
<tipo>: <descrição curta>

<corpo opcional>
```

### Tipos

| Tipo | Uso |
|------|-----|
| `feat` | Nova feature |
| `fix` | Correção de bug |
| `docs` | Documentação |
| `refactor` | Refatoração |
| `test` | Testes |
| `chore` | Manutenção |

### Exemplos

```bash
feat: add collaborative filtering to recsys

fix: handle null values in churn features

docs: add README for pricing model

refactor: optimize feature extraction pipeline

test: add unit tests for scoring function
```

### Regras

- Primeira linha: máximo 72 caracteres
- Usar imperativo ("add" não "added")
- Não terminar com ponto
- Corpo opcional para detalhes

---

## Padrões de Pull Request

### Título

```
[<projeto>] <tipo>: <descrição>
```

**Exemplos:**
```
[recsys] feat: add user embedding features
[churn] fix: handle missing values
[pricing] refactor: optimize inference time
```

### Descrição

```markdown
## Descrição
Breve descrição do que foi feito.

## Mudanças
- Mudança 1
- Mudança 2

## Testes
- [ ] Testado localmente
- [ ] Métricas validadas no MLFlow

## Checklist
- [ ] Código segue os padrões
- [ ] Documentação atualizada
- [ ] Sem secrets no código
```

---

## Checklist de Qualidade

### Antes de abrir PR

- [ ] Código segue nomenclatura padrão
- [ ] Imports organizados
- [ ] Type hints nas funções principais
- [ ] Logging adequado
- [ ] Sem credentials hardcoded
- [ ] requirements.txt atualizado
- [ ] config.yaml válido
- [ ] Testado localmente
- [ ] Métricas logadas no MLFlow

### Antes de merge para master

- [ ] Code review aprovado
- [ ] Testado em ambiente dev
- [ ] Modelo registrado no MLFlow (Staging)
- [ ] Métricas validadas
- [ ] Documentação atualizada

---

## Referências

- [PEP 8 - Style Guide for Python](https://peps.python.org/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
