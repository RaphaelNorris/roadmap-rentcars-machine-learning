# Template MLOps Completo para Projetos de Machine Learning

Template production-ready de MLOps para projetos de Machine Learning empresariais, com integracao completa AWS, MLFlow, Airflow e monitoramento continuo. Projetado para escalar de prototipo a producao com facilidade.

## Visao Geral

Este template implementa uma arquitetura completa de MLOps seguindo as melhores praticas da industria:
- **CRISP-DM** para metodologia de Data Science
- **CD4ML** (Continuous Delivery for Machine Learning) para entrega continua
- **MLOps Maturity Model** para operacionalizacao de modelos

## Principais Caracteristicas

### Infraestrutura e Deploy
- **Stack AWS completo**: S3, Athena, Iceberg, EC2, ECR, ECS
- **Containerizacao**: Docker e orquestracao docker-compose
- **CI/CD automatizado**: GitHub Actions com testes, build e deploy
- **Multi-ambiente**: Development, Staging, Production

### MLOps Core
- **MLFlow**: Tracking de experimentos e model registry versionado
- **Airflow**: Orquestracao de pipelines de ML e dados
- **FastAPI**: Model serving REST API com alta performance
- **Feature Store**: Apache Iceberg para gerenciamento de features

### Monitoramento e Qualidade
- **Drift Detection**: Evidently (opensource) para detectar data drift
- **Metricas**: Prometheus + Grafana para dashboards e alertas
- **Code Quality**: Ruff, Mypy, Bandit, Pytest com pre-commit hooks
- **Testing**: Unit, integration e smoke tests automatizados

### Integracao DataOps
- Compativel com stack Airflow + DBT existente
- Leitura de dados processados pela engenharia de dados
- Queries otimizadas no Athena com Iceberg
- Versionamento de dados e artefatos

---

## Inicio Rapido

### Opcao 1: Setup Completo (Recomendado)

```bash
# 1. Instale cookiecutter
pip install cookiecutter

# 2. Gere o projeto
cookiecutter https://github.com/RaphaelNorris/project-template-ds-rentcars.git

# Responda aos prompts:
# - project_name: nome_equipe_modelo (ex: credit_risk_model)
# - repo_name: nome do repositorio
# - author_name: Seu Nome
# - email: seu.email@empresa.com
# - description: Breve descricao do projeto

# 3. Entre no diretorio do projeto
cd seu-projeto

# 4. Configure variaveis de ambiente
cp .env.example .env
# Edite .env com suas credenciais AWS e configuracoes

# 5. Execute setup automatizado
chmod +x scripts/mlops.sh scripts/setup_mlops.sh
make setup

# 6. Inicie todos os servicos
make docker-up

# 7. Acesse as interfaces
# MLFlow: http://localhost:5000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# API Docs: http://localhost:8000/docs
```

### Opcao 2: Setup Manual (Desenvolvimento Local)

```bash
# 1. Clone/gere o projeto
cookiecutter https://github.com/RaphaelNorris/project-template-ds-rentcars.git
cd seu-projeto

# 2. Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate  # Windows

# 3. Instale dependencias
pip install -r requirements.txt

# 4. Configure AWS
aws configure
cp .env.example .env
# Edite .env

# 5. Inicie apenas MLFlow
make mlflow-up

# 6. Treine um modelo
make train

# 7. Sirva modelo via API
make serve
```

---

## Estrutura do Projeto Gerado

```
seu-projeto/
├── .github/                         # CI/CD workflows
│   └── workflows/
│       └── mlops-ci-cd.yml         # Pipeline completo
│
├── airflow/                         # DAGs do Airflow
│   └── dags/
│       ├── ml_training_pipeline.py
│       └── ml_batch_inference_pipeline.py
│
├── config/                          # Configuracoes
│   ├── aws_config.yaml             # Configuracao AWS
│   └── prometheus.yml              # Metricas
│
├── data/                            # Data Lake em camadas
│   ├── 01 - bronze/                # Raw data
│   ├── 02 - silver/                # Processed data
│   ├── 03 - ml/                    # ML artifacts
│   │   ├── features/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── evaluations/
│   └── 04 - gold/                  # Analytics ready
│
├── docker/                          # Dockerfiles
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.training
│   └── Dockerfile.inference
│
├── docs/                            # Documentacao
│   ├── mlops_architecture.md       # Arquitetura detalhada
│   ├── setup_guide.md              # Guia de setup
│   └── README.md                   # Indice
│
├── notebooks/                       # Jupyter notebooks
│   ├── 1-data/
│   ├── 2-exploration/
│   ├── 3-analysis/
│   ├── 4-feat_eng/
│   ├── 5-models/
│   └── notebook_template.ipynb
│
├── scripts/                         # Scripts utilitarios
│   └── setup_mlops.sh              # Setup automatizado
│
├── src/                             # Codigo fonte
│   ├── data/
│   │   └── aws_integration.py      # Cliente AWS (S3, Athena, Iceberg)
│   ├── model/
│   │   └── mlflow_manager.py       # Gerenciador MLFlow
│   ├── inference/
│   │   ├── api.py                  # FastAPI serving
│   │   └── monitoring.py           # Drift detection
│   └── pipelines/
│       ├── DE/                     # Data Engineering
│       └── DS/                     # Data Science
│           ├── training_pipeline/
│           ├── inference_pipeline/
│           └── feature_pipeline/
│
├── tests/                           # Testes
│   ├── data/
│   ├── model/
│   ├── inference/
│   └── pipelines/
│
├── .env.example                     # Template de variaveis
├── .gitignore
├── docker-compose.yaml              # Orquestracao de servicos
├── Makefile                         # Interface de comandos
├── pyproject.toml
├── requirements.txt                 # Dependencias Python
└── README.md
```

---

## Comandos Disponiveis

O projeto oferece **3 formas** de executar comandos:

### Opcao 1: Makefile (Recomendado)

Interface padrao da industria, simples e conhecida:

```bash
# Ver todos os comandos
make help

# Exemplos
make setup                # Setup completo
make train                # Treinar modelo
make serve                # API de inferencia
make docker-up            # Iniciar servicos
make quality              # Verificar codigo
make test                 # Rodar testes
```

### Opcao 2: Script Bash

Chamada direta ao script (mesma funcionalidade):

```bash
./scripts/mlops.sh help
./scripts/mlops.sh setup
./scripts/mlops.sh train
```

### Opcao 3: Comandos Diretos

Para quem prefere executar comandos nativos:

```bash
# Instalar dependencias
pip install -r requirements.txt

# Treinar modelo
python -m src.pipelines.DS.training_pipeline.train

# Iniciar API
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-compose up -d
docker-compose down
docker-compose logs -f

# Testes
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Qualidade
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

---

## Arquitetura MLOps

### Camadas da Arquitetura

#### 1. Data Layer
- **Bronze**: Dados raw do S3
- **Silver**: Dados processados e validados
- **Gold**: Features engenheiradas e prontas para ML
- **Feature Store**: Iceberg tables com versionamento

#### 2. Training Layer
- Pipeline automatizado de treinamento
- Tracking completo com MLFlow
- Hyperparameter tuning com Optuna
- Model Registry versionado
- Validacao e testes automaticos

#### 3. Inference Layer
- **Online**: API REST FastAPI para real-time
- **Batch**: Pipeline Airflow para inferencia em lote
- Carregamento dinamico de modelos do MLFlow
- Cache e otimizacao de performance

#### 4. Monitoring Layer
- Data drift detection (Evidently)
- Model performance metrics
- System metrics (Prometheus)
- Dashboards (Grafana)
- Alerting automatico

### Fluxo de Trabalho

```
1. Feature Engineering
   └─> Feature Store (Iceberg)

2. Model Training
   └─> MLFlow Tracking
   └─> Model Registry

3. Model Validation
   └─> Tests & Metrics
   └─> Approval Gate

4. Model Deployment
   └─> Staging Environment
   └─> Production Environment

5. Monitoring
   └─> Drift Detection
   └─> Performance Tracking
   └─> Alerts
```

---

## Integracao com Stack Existente

### Airflow + DBT Integration

Este template se integra perfeitamente com pipelines DataOps existentes:

**Cenario Tipico:**
```
DBT Pipeline (DataMart)
   └─> Silver Layer (S3/Iceberg)
       └─> ML Feature Pipeline (Este template)
           └─> Feature Store
               └─> ML Training Pipeline
                   └─> Model Registry
```

**Exemplo de integracao:**
```python
# No seu DAG DBT existente, adicione:
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_ml_pipeline = TriggerDagRunOperator(
    task_id='trigger_ml_training',
    trigger_dag_id='ml_training_pipeline',
    wait_for_completion=False,
)

dbt_task >> trigger_ml_pipeline
```

### AWS Resources Compartilhados

- **S3 Buckets**: Usa mesma estrutura de buckets
- **Athena Database**: Consulta tabelas criadas pela DE
- **Iceberg Tables**: Compativel com tables existentes
- **IAM Roles**: Reutiliza roles da infraestrutura

### Monitoramento Unificado

- Dashboards Grafana compartilhados
- Metricas consolidadas no Prometheus
- Alertas centralizados

---

## Melhores Praticas Implementadas

### Versionamento
- Codigo: Git + Semantic Versioning
- Dados: DVC + S3
- Modelos: MLFlow Model Registry
- Features: Timestamps + Iceberg snapshots

### Reproducibilidade
- Pin de versoes de dependencias
- Seeds aleatorios fixos
- Containerizacao completa
- Logs detalhados de experimentos

### Testes
- Unit tests para funcoes criticas
- Integration tests para pipelines
- Model tests (performance, fairness, robustez)
- API tests (load, stress, smoke)

### Seguranca
- Credentials via variaveis de ambiente
- IAM roles para servicos AWS
- Security scanning automatizado (Bandit)
- Secrets management

### Observabilidade
- Logging estruturado (Loguru)
- Metricas de negocio e tecnicas
- Tracing de requisicoes
- Audit logs

---

## Requisitos

### Software
- Sistema Operacional: Linux (Ubuntu 20.04+ ou similar)
- Python 3.10, 3.11 ou 3.12
- Docker 20.10+ e Docker Compose 2.0+
- Git 2.0+
- AWS CLI 2.0+
- Make (GNU Make 4.0+)
- Bash 4.0+

### Servicos AWS
- Conta AWS ativa
- Permissoes: S3, Athena, Glue, ECR, EC2/ECS, IAM
- Configuracao AWS CLI

### Conhecimentos Recomendados
- Python e ML basico (scikit-learn, pandas)
- Docker e containers
- Git e versionamento
- Conceitos basicos de AWS
- SQL para queries Athena

---

## Documentacao Completa

Para informacoes detalhadas, consulte:

- **[Arquitetura MLOps]({{ cookiecutter.repo_name }}/docs/mlops_architecture.md)**: Visao completa da arquitetura
- **[Guia de Setup]({{ cookiecutter.repo_name }}/docs/setup_guide.md)**: Passo a passo detalhado
- **[Estrutura de Dados]({{ cookiecutter.repo_name }}/docs/data_structure.md)**: Organizacao dos dados
- **[Pipelines]({{ cookiecutter.repo_name }}/docs/pipelines.md)**: Documentacao dos pipelines

---

## Troubleshooting Rapido

### Problema: Docker nao inicia
```bash
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

### Problema: MLFlow nao conecta
```bash
docker-compose logs mlflow
docker-compose restart mlflow
# Aguarde ate: "Listening at: http://0.0.0.0:5000"
```

### Problema: AWS Access Denied
```bash
aws sts get-caller-identity
aws configure
# Verifique IAM policies
```

### Problema: Import errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Ou adicione ao .env
```

Para mais detalhes, veja: [docs/setup_guide.md]({{ cookiecutter.repo_name }}/docs/setup_guide.md#troubleshooting)

---

## Governanca e Best Practices

Este template implementa um framework completo de governanca MLOps seguindo CRISP-DM e CD4ML:

### Documentacao de Governanca

**Templates de Documentacao**:
- **[Business Requirements]({{ cookiecutter.repo_name }}/templates/business_requirements_template.md)**: CRISP-DM Fase 1 - Business Understanding completo
- **[Data Card]({{ cookiecutter.repo_name }}/templates/data_card_template.md)**: Documentacao de datasets seguindo padroes de Data Cards
- **[Model Card]({{ cookiecutter.repo_name }}/templates/model_card_template.md)**: Documentacao de modelos seguindo Mitchell et al.

**Guias e Frameworks**:
- **[ML Governance]({{ cookiecutter.repo_name }}/docs/ml_governance.md)**: Framework completo de governanca MLOps
- **[Experiment Guidelines]({{ cookiecutter.repo_name }}/docs/experiment_guidelines.md)**: Best practices para experimentacao ML

### Responsible AI

**Testes Automatizados de Fairness e Bias**:
```bash
# Executar testes de fairness
pytest tests/model/test_fairness.py -v

# Metricas implementadas:
# - Demographic Parity
# - Equal Opportunity
# - Equalized Odds
# - Disparate Impact (4/5ths rule)
# - Predictive Parity
# - Calibration by group
```

**Testes de Robustez**:
```bash
# Executar testes de robustez
pytest tests/model/test_robustness.py -v

# Testes implementados:
# - Adversarial perturbations
# - Invariance tests
# - Edge case handling
# - Prediction stability
# - Feature range validation
```

### Continuous Training

**Pipeline de Retreinamento Automatico**:
- DAG de Continuous Training com triggers automaticos
- Deteccao de drift (data drift + concept drift)
- Validacao automatica de novos modelos
- Promocao ou rollback baseado em metricas

```bash
# DAG disponivel em:
# airflow/dags/ml_continuous_training_pipeline.py

# Triggers de retreinamento:
# - Data drift > threshold
# - Performance degradation
# - Novos dados disponiveis
# - Scheduled (semanal/mensal)
```

### Canary Deployment

**Progressive Rollout com Rollback Automatico**:
```bash
# Executar canary deployment
python scripts/canary_deployment.py \
  --canary-version 5 \
  --production-version 4 \
  --initial-traffic 5

# Fases automaticas:
# 5% -> 10% -> 25% -> 50% -> 100%
# Com monitoramento e rollback automatico se metricas degradarem
```

### Business Metrics Monitoring

**Monitoramento de Impacto no Negocio**:
- Revenue impact tracking
- Cost reduction measurement
- ROI calculation
- Customer satisfaction metrics
- Operational efficiency gains

```python
from src.inference.business_metrics import calculate_and_log_business_metrics

# Calcular metricas de negocio
metrics = calculate_and_log_business_metrics(
    predictions_df=predictions,
    actual_outcomes_df=actuals
)

# Metricas disponiveis:
# - revenue_impact
# - cost_reduction
# - roi
# - customer_satisfaction
# - churn_rate
```

### EDA Notebook Template

**Notebook CRISP-DM Completo**:
- Template estruturado seguindo CRISP-DM fases 1-2
- Profiling automatico com ydata-profiling
- Analise de qualidade de dados
- Testes estatisticos
- Hipothesis testing
- Feature engineering ideas

Disponivel em: `notebooks/1-data/01_eda_crisp_dm_template.ipynb`

---

## Roadmap

### Implementado Recentemente
- [x] ML Governance Framework completo
- [x] Testes de Fairness e Bias (Responsible AI)
- [x] Testes de Robustez (adversarial, invariance, edge cases)
- [x] Continuous Training Pipeline (automated retraining)
- [x] Canary Deployment strategy
- [x] Business Metrics Monitoring
- [x] Experiment Guidelines (CD4ML best practices)
- [x] EDA Notebook Template (CRISP-DM)
- [x] Model Card, Data Card, Business Requirements templates

### Proximamente
- [ ] Suporte a SageMaker para treinamento
- [ ] Integracao com Feast para Feature Store
- [ ] Templates de modelos pre-configurados (XGBoost, LightGBM, etc)
- [ ] Dashboard MLFlow customizado
- [ ] Suporte a GPU para deep learning
- [ ] A/B testing framework automatizado
- [ ] Model explainability integrado (SHAP, LIME)

### Contribuicoes

Contribuicoes sao bem-vindas! Por favor:
1. Fork o repositorio
2. Crie um branch para sua feature
3. Commit suas mudancas
4. Push para o branch
5. Abra um Pull Request

---

## Licenca

Este template e fornecido como-is, sem garantias. Use por sua conta e risco.

## Autor

Template desenvolvido e mantido por **[@RaphaelNorris](https://github.com/RaphaelNorris)**

Para questoes, sugestoes ou bugs, abra uma issue no repositorio.

---

## Referencias e Recursos

### MLOps
- [MLOps Principles](https://ml-ops.org/)
- [CD4ML - Martin Fowler](https://martinfowler.com/articles/cd4ml.html)
- [Google MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS MLOps Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/mlops/)

### Ferramentas
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Evidently AI Docs](https://docs.evidentlyai.com/)

### AWS
- [S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html)
- [Athena Performance Tuning](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html)
- [Apache Iceberg on AWS](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-iceberg.html)
