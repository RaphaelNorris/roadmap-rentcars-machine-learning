#!/bin/bash
# Script para criar estrutura completa de pastas MLOps
# Author: {{ cookiecutter.author_name }}

set -e

echo "Creating comprehensive MLOps directory structure..."

# ============================================================================
# AIRFLOW - Orquestração
# ============================================================================
mkdir -p airflow/plugins
mkdir -p airflow/config
mkdir -p airflow/logs
mkdir -p airflow/dags/dataops      # DAGs de engenharia de dados
mkdir -p airflow/dags/mlops        # DAGs de ML
mkdir -p airflow/tests

# ============================================================================
# CONFIG - Configurações por ambiente
# ============================================================================
mkdir -p config/dev
mkdir -p config/staging
mkdir -p config/prod
mkdir -p config/grafana/dashboards
mkdir -p config/grafana/datasources
mkdir -p config/prometheus

# ============================================================================
# DOCS - Documentação técnica
# ============================================================================
mkdir -p docs/architecture         # Diagramas de arquitetura
mkdir -p docs/governance           # Model cards, data cards, business reqs
mkdir -p docs/deployment           # Guias de deployment
mkdir -p docs/monitoring           # Monitoramento e alertas
mkdir -p docs/api                  # Documentação de APIs
mkdir -p docs/experiments          # Guidelines de experimentação

# ============================================================================
# NOTEBOOKS - Análise exploratória e desenvolvimento
# ============================================================================
mkdir -p notebooks/1-data-collection
mkdir -p notebooks/2-eda
mkdir -p notebooks/3-feature-engineering
mkdir -p notebooks/4-modeling
mkdir -p notebooks/5-evaluation
mkdir -p notebooks/6-deployment
mkdir -p notebooks/templates

# ============================================================================
# SCRIPTS - Automação
# ============================================================================
mkdir -p scripts/setup             # Setup EC2, instalação
mkdir -p scripts/deployment        # Deploy de modelos
mkdir -p scripts/monitoring        # Scripts de monitoramento
mkdir -p scripts/backup            # Backup PostgreSQL, S3
mkdir -p scripts/utils             # Utilitários gerais

# ============================================================================
# SRC - Código fonte
# ============================================================================
mkdir -p src/monitoring/drift      # Drift detection
mkdir -p src/monitoring/performance # Performance monitoring
mkdir -p src/monitoring/business   # Business metrics
mkdir -p src/deployment/canary     # Canary deployment
mkdir -p src/deployment/blue_green # Blue-green deployment
mkdir -p src/deployment/rollback   # Rollback strategies
mkdir -p src/utils/logging
mkdir -p src/utils/config
mkdir -p src/utils/aws

# ============================================================================
# TESTS - Testes
# ============================================================================
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/e2e
mkdir -p tests/responsible_ai/fairness
mkdir -p tests/responsible_ai/robustness
mkdir -p tests/responsible_ai/explainability
mkdir -p tests/performance

# ============================================================================
# MODELS - Cache de modelos
# ============================================================================
mkdir -p models/staging
mkdir -p models/production
mkdir -p models/archived

# ============================================================================
# LOGS - Logs da aplicação
# ============================================================================
mkdir -p logs/training
mkdir -p logs/inference
mkdir -p logs/monitoring
mkdir -p logs/deployment

# ============================================================================
# DATA - Dados por camada medallion
# ============================================================================
mkdir -p data/00-raw
mkdir -p data/01-bronze
mkdir -p data/02-silver
mkdir -p data/03-gold
mkdir -p data/04-ml/features
mkdir -p data/04-ml/predictions
mkdir -p data/04-ml/monitoring

# ============================================================================
# CI/CD - GitHub Actions
# ============================================================================
mkdir -p .github/workflows
mkdir -p .github/templates

echo "✓ MLOps directory structure created successfully!"
echo ""
echo "Directory structure:"
echo "├── airflow/           # Orchestration (DAGs, plugins, config)"
echo "├── config/            # Configurations (dev, staging, prod)"
echo "├── docs/              # Documentation (architecture, governance, API)"
echo "├── notebooks/         # Jupyter notebooks (EDA, modeling, evaluation)"
echo "├── scripts/           # Automation scripts (setup, deployment, monitoring)"
echo "├── src/               # Source code (pipelines, monitoring, deployment)"
echo "├── tests/             # Tests (unit, integration, e2e, responsible AI)"
echo "├── models/            # Model cache (staging, production, archived)"
echo "├── logs/              # Application logs"
echo "├── data/              # Data medallion architecture"
echo "└── .github/           # CI/CD workflows"
