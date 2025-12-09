# ML Experiment Guidelines - Best Practices

Guia completo para experimentação em Machine Learning seguindo princípios de CD4ML e CRISP-DM.

---

## Table of Contents

1. [Experiment Planning](#1-experiment-planning)
2. [Experiment Setup](#2-experiment-setup)
3. [Experiment Execution](#3-experiment-execution)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Experiment Analysis](#5-experiment-analysis)
6. [Experiment Documentation](#6-experiment-documentation)
7. [Experiment Deployment](#7-experiment-deployment)
8. [Best Practices](#8-best-practices)

---

## 1. Experiment Planning

### 1.1 Define Experiment Objective

Antes de iniciar qualquer experimento, documente:

**Hipótese**:
- O que você está tentando provar/melhorar?
- Qual a métrica alvo?
- Qual o baseline atual?

**Exemplo**:
```
Hipótese: Adicionar features de comportamento temporal (RFM) vai melhorar
          recall de predição de churn de 0.70 para >0.75

Métrica alvo: Recall
Baseline atual: 0.70
Meta: > 0.75
```

### 1.2 Design Experimental

**Checklist de Design**:

- [ ] **Variável Independente**: O que você vai mudar?
  - Novo feature set
  - Novo algoritmo
  - Novos hiperparâmetros
  - Novo threshold de decisão

- [ ] **Variável Dependente**: O que você vai medir?
  - Métricas de ML (accuracy, precision, recall, F1, AUC)
  - Métricas de negócio (revenue, cost, NPS)
  - Métricas de sistema (latência, throughput)

- [ ] **Controles**: O que permanece constante?
  - Mesmo dataset de treino/teste
  - Mesma estratégia de validação
  - Mesmo random seed (para reprodutibilidade)

- [ ] **Critério de Sucesso**: Como saberá se foi bem-sucedido?
  - Métrica X > threshold Y
  - Melhoria estatisticamente significativa (p-value < 0.05)
  - Sem degradação em métricas de fairness

### 1.3 Experiment Naming Convention

Use nomes descritivos e consistentes:

```
YYMMDD_<author-initials>_<experiment-type>_<brief-description>

Exemplos:
- 241201_RN_features_add-rfm-temporal
- 241205_RN_model_xgboost-vs-rf
- 241210_RN_hyperparam_xgb-gridsearch
- 241215_RN_threshold_optimize-recall
```

---

## 2. Experiment Setup

### 2.1 Environment Setup

**Sempre use ambiente isolado**:

```bash
# Criar ambiente virtual específico para o experimento
python -m venv .venv_experiment_241201

# Ativar
source .venv_experiment_241201/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Congelar versões
pip freeze > experiments/241201_RN_features_add-rfm/requirements.txt
```

### 2.2 Data Versioning

**CRÍTICO**: Versione seus dados para reprodutibilidade.

```python
# Usar DVC para versionamento de dados
import dvc.api

# Fixar versão dos dados
data_version = "v1.2.0"  # Git tag ou commit hash

# Carregar dados versionados
with dvc.api.open(
    'data/processed/features.parquet',
    rev=data_version
) as f:
    df = pd.read_parquet(f)

# Documentar versão no experimento
experiment_metadata = {
    'data_version': data_version,
    'data_path': 'data/processed/features.parquet',
    'data_hash': hashlib.md5(df.to_string().encode()).hexdigest()[:8]
}
```

**Alternativa com S3 + MLFlow**:

```python
from src.data.aws_integration import get_s3_client

# Salvar snapshot dos dados para o experimento
s3_client = get_s3_client()

experiment_id = "241201_RN_features_add-rfm"
data_snapshot_path = f"ml/experiments/{experiment_id}/data_snapshot.parquet"

s3_client.write_parquet(df, s3_key=data_snapshot_path)

# Log no MLFlow
import mlflow
mlflow.log_param("data_snapshot", data_snapshot_path)
```

### 2.3 Code Versioning

**SEMPRE commite código antes de rodar experimento**:

```bash
# Criar branch para experimento (opcional mas recomendado)
git checkout -b experiment/241201-add-rfm-features

# Commit mudanças
git add -A
git commit -m "Experiment: Add RFM temporal features"

# Obter commit hash
COMMIT_HASH=$(git rev-parse HEAD)
echo "Experiment commit: $COMMIT_HASH"

# Log no MLFlow
```

```python
import mlflow
import subprocess

commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
mlflow.log_param("git_commit", commit_hash)
```

### 2.4 MLFlow Experiment Setup

```python
import mlflow
from src.model.mlflow_manager import MLFlowManager

# Inicializar MLFlow
mlflow_manager = MLFlowManager(
    experiment_name="churn_prediction_experiments"
)

# Criar run com nome descritivo
with mlflow.start_run(run_name="241201_RN_features_add-rfm"):

    # Log de configuração
    mlflow.log_params({
        "experiment_id": "241201_RN_features_add-rfm",
        "author": "{{ cookiecutter.author_name }}",
        "hypothesis": "Adding RFM features improves recall",
        "baseline_recall": 0.70,
        "target_recall": 0.75,
        "git_commit": commit_hash,
        "data_version": data_version,
    })

    # ... seu código de treinamento ...
```

---

## 3. Experiment Execution

### 3.1 Data Splitting Strategy

**CRÍTICO**: Use estratégia consistente de split.

#### Opção 1: Time-Series Split (Recomendado para dados temporais)

```python
from sklearn.model_selection import TimeSeriesSplit

# Ordenar por data
df = df.sort_values('date')

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]

    # Treinar e avaliar
    ...
```

#### Opção 2: Stratified K-Fold (Para dados não temporais)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Treinar e avaliar
    ...
```

#### Opção 3: Holdout com Test Set Fixo

```python
# Separar test set ANTES de qualquer experimento
from sklearn.model_selection import train_test_split

# Test set fixo (20%) - NUNCA tocar até validação final
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Salvar test set
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test
s3_client.write_parquet(test_df, "ml/test_sets/test_set_v1.parquet")

# Split train/validation para experimentos
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)
```

### 3.2 Feature Engineering

**Documente todas as transformações**:

```python
# Criar classe de transformação para reprodutibilidade
from sklearn.base import BaseEstimator, TransformerMixin

class RFMFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Adiciona features RFM (Recency, Frequency, Monetary).

    Params:
        reference_date: Data de referência para calcular recency
    """

    def __init__(self, reference_date=None):
        self.reference_date = reference_date or pd.Timestamp.now()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Recency: dias desde última transação
        X['recency'] = (self.reference_date - X['last_transaction_date']).dt.days

        # Frequency: número de transações
        X['frequency'] = X['transaction_count']

        # Monetary: valor total gasto
        X['monetary'] = X['total_spent']

        return X

# Usar no pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('rfm_features', RFMFeatureEngineering(reference_date=pd.Timestamp('2024-12-01'))),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Log transformações no MLFlow
mlflow.log_param("feature_engineering", "RFM + StandardScaler")
mlflow.log_param("rfm_reference_date", "2024-12-01")
```

### 3.3 Hyperparameter Tuning

**Use estratégias sistemáticas**:

#### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall',  # Métrica alvo
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Log resultados
mlflow.log_params(grid_search.best_params_)
mlflow.log_metric("best_cv_recall", grid_search.best_score_)
```

#### Optuna (Bayesian Optimization - Recomendado)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    }

    model = XGBClassifier(**params, random_state=42)

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')

    return scores.mean()

# Criar study
study = optuna.create_study(
    direction='maximize',
    study_name='241201_xgb_recall_optimization'
)

# Callback para log no MLFlow
def mlflow_callback(study, trial):
    mlflow.log_params(trial.params)
    mlflow.log_metric("trial_recall", trial.value)

# Otimizar
study.optimize(objective, n_trials=100, callbacks=[mlflow_callback])

# Log melhores params
mlflow.log_params(study.best_params)
mlflow.log_metric("best_recall", study.best_value)
```

### 3.4 Model Training

**Template de treinamento com boas práticas**:

```python
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

with mlflow.start_run(run_name="241201_RN_features_add-rfm"):

    # 1. Log experiment metadata
    mlflow.log_params({
        "experiment_id": "241201_RN_features_add-rfm",
        "model_type": "RandomForest",
        "feature_set": "baseline + RFM",
        "n_features": len(feature_cols),
    })

    # 2. Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob)
    }

    # Log métricas
    mlflow.log_metrics(metrics)

    # 4. Log artifacts

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')

    # 5. Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=None  # Registrar só se for promover
    )

    # 6. Comparar com baseline
    baseline_recall = 0.70
    improvement = metrics['recall'] - baseline_recall
    improvement_pct = (improvement / baseline_recall) * 100

    mlflow.log_metric("recall_improvement", improvement)
    mlflow.log_metric("recall_improvement_pct", improvement_pct)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS: {mlflow.active_run().info.run_id}")
    print(f"{'='*80}")
    print(f"Recall: {metrics['recall']:.4f} (baseline: {baseline_recall:.4f})")
    print(f"Improvement: {improvement_pct:+.2f}%")
    print(f"Target achieved: {'YES' if metrics['recall'] >= 0.75 else 'NO'}")
    print(f"{'='*80}\n")
```

---

## 4. Experiment Tracking

### 4.1 MLFlow Tracking Checklist

**SEMPRE log no mínimo**:

- [ ] **Parameters**:
  - Hiperparâmetros do modelo
  - Feature engineering steps
  - Data split strategy
  - Random seeds

- [ ] **Metrics**:
  - Todas as métricas de ML relevantes
  - Métricas de negócio
  - Comparação com baseline

- [ ] **Artifacts**:
  - Modelo treinado
  - Gráficos (confusion matrix, ROC curve, feature importance)
  - Feature list
  - Predictions (sample)

- [ ] **Tags**:
  - experiment_type (features, model, hyperparams, etc)
  - status (running, success, failed)
  - priority (high, medium, low)

```python
# Adicionar tags
mlflow.set_tags({
    "experiment_type": "feature_engineering",
    "status": "success",
    "priority": "high",
    "deployed": "false"
})
```

### 4.2 Comparing Experiments

```python
# Buscar experimentos
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Buscar runs do experimento
experiment = client.get_experiment_by_name("churn_prediction_experiments")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.recall DESC"],
    max_results=10
)

# Comparar top 5
comparison_df = pd.DataFrame([
    {
        'run_id': run.info.run_id[:8],
        'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
        'recall': run.data.metrics.get('recall', 0),
        'precision': run.data.metrics.get('precision', 0),
        'f1': run.data.metrics.get('f1', 0),
        'feature_set': run.data.params.get('feature_set', 'N/A'),
    }
    for run in runs[:5]
])

print("\nTop 5 Experiments by Recall:")
print(comparison_df.to_string(index=False))
```

---

## 5. Experiment Analysis

### 5.1 Statistical Significance Testing

**IMPORTANTE**: Não confie apenas em métricas. Teste significância estatística.

```python
from scipy import stats

# Comparar recall de dois modelos usando bootstrap
def bootstrap_metric(y_true, y_pred_1, y_pred_2, metric_fn, n_iterations=1000):
    """
    Compara duas predições usando bootstrap.

    Returns:
        p_value: p-value do teste
        ci_diff: Intervalo de confiança da diferença
    """
    n_samples = len(y_true)
    differences = []

    for _ in range(n_iterations):
        # Resample com reposição
        indices = np.random.choice(n_samples, n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_1_boot = y_pred_1[indices]
        y_pred_2_boot = y_pred_2[indices]

        # Calcular métricas
        metric_1 = metric_fn(y_true_boot, y_pred_1_boot)
        metric_2 = metric_fn(y_true_boot, y_pred_2_boot)

        differences.append(metric_2 - metric_1)

    # Calcular p-value (two-tailed)
    p_value = 2 * min(
        (np.array(differences) <= 0).mean(),
        (np.array(differences) >= 0).mean()
    )

    # Intervalo de confiança 95%
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)

    return p_value, (ci_lower, ci_upper)

# Usar
from sklearn.metrics import recall_score

p_value, ci_diff = bootstrap_metric(
    y_val,
    y_pred_baseline,
    y_pred_new,
    recall_score
)

print(f"P-value: {p_value:.4f}")
print(f"95% CI of difference: [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")

if p_value < 0.05:
    print("✓ Difference is statistically significant (p < 0.05)")
else:
    print("✗ Difference is NOT statistically significant")

# Log no MLFlow
mlflow.log_metric("statistical_significance_p_value", p_value)
mlflow.log_metric("recall_diff_ci_lower", ci_diff[0])
mlflow.log_metric("recall_diff_ci_upper", ci_diff[1])
```

### 5.2 Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5,
    scoring='recall',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), label='Train')
ax.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
ax.fill_between(
    train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.1
)
ax.fill_between(
    train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.1
)
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Recall')
ax.set_title('Learning Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('learning_curve.png')
mlflow.log_artifact('learning_curve.png')
```

---

## 6. Experiment Documentation

### 6.1 Experiment Log Template

Crie arquivo markdown para cada experimento:

**`experiments/241201_RN_features_add-rfm/README.md`**:

```markdown
# Experiment: Add RFM Temporal Features

## Metadata

- **Date**: 2024-12-01
- **Author**: {{ cookiecutter.author_name }}
- **MLFlow Run ID**: a1b2c3d4e5f6
- **Git Commit**: abc123def456
- **Status**: ✓ Success

## Hypothesis

Adding RFM (Recency, Frequency, Monetary) temporal features will improve
recall from 0.70 to >= 0.75 for churn prediction.

## Experiment Design

### What Changed
- Added 3 new features:
  - `recency`: Days since last transaction
  - `frequency`: Number of transactions in last 90 days
  - `monetary`: Total spent in last 90 days

### What Stayed Constant
- Same model: RandomForest(n_estimators=100, max_depth=10)
- Same train/val split: 80/20, stratified, seed=42
- Same evaluation metrics: accuracy, precision, recall, F1, AUC

## Results

| Metric | Baseline | New Model | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | 0.8200 | 0.8350 | +1.8% |
| Precision | 0.7500 | 0.7650 | +2.0% |
| **Recall** | **0.7000** | **0.7600** | **+8.6%** |
| F1 | 0.7241 | 0.7625 | +5.3% |
| ROC AUC | 0.8500 | 0.8720 | +2.6% |

**Statistical Significance**: p-value = 0.003 < 0.05 ✓

## Analysis

### What Worked
- RFM features significantly improved recall (8.6% increase)
- `recency` was the most important new feature (importance: 0.15)
- No degradation in other metrics

### What Didn't Work
- N/A - experiment was successful

### Feature Importance
Top 5 features:
1. total_transactions_lifetime (0.25)
2. avg_transaction_value (0.18)
3. **recency** (0.15) - NEW
4. customer_tenure_days (0.12)
5. **monetary** (0.10) - NEW

## Decision

✓ **APPROVED FOR PRODUCTION**

Reasons:
- Target recall >= 0.75 achieved
- Statistically significant improvement
- No fairness issues detected
- Passed robustness tests

## Next Steps

- [ ] Update feature pipeline with RFM features
- [ ] Retrain on full dataset
- [ ] Run fairness tests
- [ ] Deploy to staging
- [ ] A/B test for 2 weeks
- [ ] Promote to production

## Artifacts

- MLFlow Run: http://mlflow-server:5000/#/experiments/1/runs/a1b2c3d4
- Code: `experiments/241201_RN_features_add-rfm/train.py`
- Model: `s3://ml-artifacts/models/241201_rfm_model.pkl`
- Notebook: `notebooks/5-models/241201_rfm_experiment.ipynb`
```

---

## 7. Experiment Deployment

### 7.1 Criteria for Production Deployment

**Checklist antes de deploy**:

- [ ] **Performance**:
  - Métricas de ML >= threshold
  - Melhoria estatisticamente significativa
  - Performance consistente em cross-validation

- [ ] **Robustness**:
  - Passou testes de robustez (test_robustness.py)
  - Estável a perturbações adversariais
  - Não quebra em edge cases

- [ ] **Fairness**:
  - Passou testes de fairness (test_fairness.py)
  - Demographic parity < 0.1
  - Equal opportunity < 0.1

- [ ] **Business Impact**:
  - Aprovação do stakeholder de negócio
  - ROI estimado positivo
  - Alinhado com objetivos de negócio

- [ ] **Technical**:
  - Código revisado (code review)
  - Testes unitários passando
  - Documentação atualizada
  - Model Card criado

### 7.2 Promotion Workflow

```python
from src.model.mlflow_manager import MLFlowManager

mlflow_manager = MLFlowManager()

# 1. Registrar modelo no Model Registry
run_id = "a1b2c3d4e5f6"
model_uri = f"runs:/{run_id}/model"

model_version = mlflow_manager.register_model(
    model_uri=model_uri,
    registered_model_name=MODEL_REGISTRY_NAME,
    tags={
        "experiment_id": "241201_RN_features_add-rfm",
        "approved_by": "{{ cookiecutter.author_name }}",
        "deployment_date": datetime.now().isoformat()
    }
)

# 2. Adicionar descrição
model_version_details = mlflow_manager.client.update_model_version(
    name=MODEL_REGISTRY_NAME,
    version=model_version.version,
    description="RFM features model - 8.6% recall improvement over baseline"
)

# 3. Transição para Staging
mlflow_manager.transition_model_stage(
    name=MODEL_REGISTRY_NAME,
    version=model_version.version,
    stage="Staging"
)

print(f"Model v{model_version.version} promoted to Staging")

# 4. Após validação em staging, promover para Production
# (Fazer após A/B test, shadow mode, etc)
mlflow_manager.transition_model_stage(
    name=MODEL_REGISTRY_NAME,
    version=model_version.version,
    stage="Production"
)
```

---

## 8. Best Practices

### 8.1 DO's

#### Sempre:

1. **Version Everything**:
   - Code (Git)
   - Data (DVC ou S3 snapshots)
   - Environment (requirements.txt)
   - Models (MLFlow)

2. **Reproduzibilidade**:
   - Set random seeds
   - Documente TUDO
   - Use pipelines (sklearn.pipeline)
   - Salve artifacts

3. **Validação Rigorosa**:
   - Use cross-validation
   - Test set separado (nunca treinar nele)
   - Testes estatísticos
   - Testes de fairness e robustez

4. **Documentação**:
   - Hipótese clara
   - Resultados detalhados
   - Decisão e próximos passos
   - Atualizar Model Card

5. **Tracking**:
   - Log no MLFlow
   - Comparar com baseline
   - Manter experiment log

#### Nunca:

1. **Data Leakage**:
   - Nunca use dados de teste para seleção de features
   - Nunca ajuste hiperparâmetros no test set
   - Cuidado com feature engineering usando informação do futuro

2. **Cherry Picking**:
   - Não reporte só os melhores resultados
   - Documente experimentos que falharam também
   - Não ajuste threshold baseado no test set

3. **Overfitting**:
   - Não confie só em accuracy de treinamento
   - Use validação cruzada
   - Regularização quando apropriado

4. **P-Hacking**:
   - Não teste 100 hipóteses e reporte só a significativa
   - Ajuste para múltiplos testes (Bonferroni correction)
   - Pre-registre hipóteses quando possível

### 8.2 Experiment Lifecycle

```
1. PLAN
   ├── Define hypothesis
   ├── Design experiment
   └── Set success criteria

2. SETUP
   ├── Version code (git commit)
   ├── Version data (DVC/S3)
   ├── Create MLFlow run
   └── Document baseline

3. EXECUTE
   ├── Feature engineering
   ├── Model training
   ├── Hyperparameter tuning
   └── Evaluation

4. ANALYZE
   ├── Compare metrics
   ├── Statistical tests
   ├── Error analysis
   └── Fairness check

5. DOCUMENT
   ├── Update experiment log
   ├── Create Model Card
   ├── Present to team
   └── Get feedback

6. DECIDE
   ├── Approve/reject for production
   ├── Plan next experiments
   └── Archive artifacts

7. DEPLOY (if approved)
   ├── Register model
   ├── Staging validation
   ├── A/B test
   └── Production promotion
```

### 8.3 Common Pitfalls

#### Pitfall 1: Não documentar experimentos que falharam

**Problema**: Perde-se conhecimento valioso sobre o que NÃO funciona.

**Solução**: Documente TODOS os experimentos, incluindo falhas.

```markdown
## Experiment: Try Deep Learning Model

**Status**: ✗ Failed

**Why it failed**:
- Model overfit severely (train acc 0.95, val acc 0.62)
- Training time 10x longer than RF
- Não melhorou métricas vs baseline

**Learnings**:
- Dataset pequeno demais para DL (apenas 5k samples)
- RF é suficiente para este problema
- Não vale o custo computacional adicional

**Decision**: Stick with RandomForest
```

#### Pitfall 2: Comparar experimentos em datasets diferentes

**Problema**: Resultados não são comparáveis.

**Solução**: Use SEMPRE o mesmo test set fixo.

```python
# Carregar test set fixo
TEST_SET_PATH = "ml/test_sets/test_set_v1.parquet"
test_df = load_data_from_s3(TEST_SET_PATH)

# Todos os experimentos avaliam neste mesmo test set
```

#### Pitfall 3: Não testar em dados de produção

**Problema**: Modelo funciona bem em dados históricos mas falha em produção.

**Solução**: Validação com dados mais recentes + shadow mode.

```python
# Sempre reserve dados mais recentes para validação final
train_cutoff_date = "2024-10-31"
val_cutoff_date = "2024-11-30"

train_df = df[df['date'] <= train_cutoff_date]
val_df = df[(df['date'] > train_cutoff_date) & (df['date'] <= val_cutoff_date)]
test_df = df[df['date'] > val_cutoff_date]  # Dados mais recentes
```

---

## 9. Templates e Exemplos

### 9.1 Experiment Notebook Template

Ver: `notebooks/5-models/experiment_template.ipynb`

### 9.2 Experiment Script Template

Ver: `scripts/train_experiment.py`

### 9.3 MLFlow Comparison Dashboard

Acessar MLFlow UI:
```bash
mlflow ui --port 5000
```

Navegar para: http://localhost:5000

Comparar runs:
1. Selecionar múltiplos runs
2. Click "Compare"
3. Ver métricas lado a lado
4. Plotar learning curves

---

## 10. References

### ML Experimentation
- [Google - Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Continuous Delivery for Machine Learning (CD4ML)](https://martinfowler.com/articles/cd4ml.html)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)

### Statistical Testing
- [Bootstrap Methods](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Multiple Comparisons Problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)

### Responsible AI
- [Model Cards](https://arxiv.org/abs/1810.03993)
- [Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide)

---

**Última atualização**: YYYY-MM-DD
**Mantido por**: {{ cookiecutter.author_name }}
