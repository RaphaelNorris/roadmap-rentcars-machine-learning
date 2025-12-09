#!/bin/bash

# MLOps Project Management Script
# Comandos uteis para desenvolvimento e operacao

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funcoes auxiliares
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help
show_help() {
    cat << EOF
Uso: ./scripts/mlops.sh [comando]

SETUP E CONFIGURACAO:
  install              Instalar dependencias
  setup                Setup completo do ambiente
  init-aws             Criar recursos AWS (S3, ECR, Athena)

DESENVOLVIMENTO:
  notebook             Iniciar Jupyter notebook
  lab                  Iniciar Jupyter lab
  shell                Shell Python interativo

QUALIDADE DE CODIGO:
  lint                 Linting com Ruff
  format               Formatar codigo
  type-check           Type checking com mypy
  security             Security scan com bandit
  quality              Todos os checks acima
  test                 Rodar testes
  test-cov             Testes com cobertura

ML OPERATIONS:
  train                Treinar modelo
  serve                Iniciar API de inferencia
  serve-prod           API em modo producao

MONITORAMENTO:
  monitor              Monitorar modelos
  drift-check          Verificar data drift

INFRAESTRUTURA:
  docker-build         Build de imagens Docker
  docker-up            Iniciar todos servicos
  docker-down          Parar servicos
  docker-logs          Ver logs dos servicos
  mlflow-up            Apenas MLFlow
  monitoring-up        Prometheus + Grafana

LIMPEZA:
  clean                Limpar arquivos temporarios
  clean-docker         Limpar recursos Docker
  clean-all            Limpeza completa

OUTROS:
  help                 Mostrar esta mensagem
EOF
}

# Comandos

cmd_install() {
    info "Instalando dependencias..."
    pip install -r requirements.txt
}

cmd_setup() {
    info "Executando setup completo..."
    chmod +x scripts/setup_mlops.sh
    ./scripts/setup_mlops.sh
}

cmd_init_aws() {
    info "Inicializando recursos AWS..."

    # Carregar variaveis de ambiente
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi

    # Criar buckets S3
    aws s3 mb s3://${S3_RAW_BUCKET} --region ${AWS_REGION} 2>/dev/null || warn "Bucket ${S3_RAW_BUCKET} ja existe"
    aws s3 mb s3://${S3_PROCESSED_BUCKET} --region ${AWS_REGION} 2>/dev/null || warn "Bucket ${S3_PROCESSED_BUCKET} ja existe"
    aws s3 mb s3://${S3_ML_ARTIFACTS_BUCKET} --region ${AWS_REGION} 2>/dev/null || warn "Bucket ${S3_ML_ARTIFACTS_BUCKET} ja existe"

    # Criar repositories ECR
    aws ecr create-repository --repository-name ml-training --region ${AWS_REGION} 2>/dev/null || warn "Repository ml-training ja existe"
    aws ecr create-repository --repository-name ml-inference --region ${AWS_REGION} 2>/dev/null || warn "Repository ml-inference ja existe"

    info "Recursos AWS criados com sucesso"
}

cmd_notebook() {
    info "Iniciando Jupyter notebook..."
    jupyter notebook notebooks/
}

cmd_lab() {
    info "Iniciando Jupyter lab..."
    jupyter lab
}

cmd_shell() {
    info "Iniciando shell Python..."
    python -i -c "from src.data.aws_integration import *; from src.model.mlflow_manager import *"
}

cmd_lint() {
    info "Executando linting..."
    ruff check src/ tests/
}

cmd_format() {
    info "Formatando codigo..."
    ruff format src/ tests/
}

cmd_type_check() {
    info "Verificando tipos..."
    mypy src/ --config-file .code_quality/mypy.ini
}

cmd_security() {
    info "Executando security scan..."
    bandit -r src/ -c .code_quality/bandit.yaml
}

cmd_quality() {
    info "Executando verificacoes de qualidade..."
    cmd_lint
    cmd_format
    cmd_type_check
    cmd_security
}

cmd_test() {
    info "Executando testes..."
    pytest tests/ -v
}

cmd_test_cov() {
    info "Executando testes com cobertura..."
    pytest tests/ --cov=src --cov-report=html --cov-report=term
}

cmd_train() {
    info "Treinando modelo..."
    python -m src.pipelines.DS.training_pipeline.train
}

cmd_serve() {
    info "Iniciando API de inferencia..."
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload
}

cmd_serve_prod() {
    info "Iniciando API em modo producao..."
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --workers 4
}

cmd_monitor() {
    info "Executando monitoramento..."
    python scripts/monitor_model.py
}

cmd_drift_check() {
    info "Verificando drift..."
    python scripts/check_drift.py
}

cmd_docker_build() {
    info "Building imagens Docker..."
    docker-compose build
}

cmd_docker_up() {
    info "Iniciando servicos Docker..."
    docker-compose up -d
}

cmd_docker_down() {
    info "Parando servicos Docker..."
    docker-compose down
}

cmd_docker_logs() {
    info "Exibindo logs dos servicos..."
    docker-compose logs -f
}

cmd_mlflow_up() {
    info "Iniciando MLFlow..."
    docker-compose up -d mlflow postgres
}

cmd_monitoring_up() {
    info "Iniciando stack de monitoramento..."
    docker-compose up -d prometheus grafana
}

cmd_clean() {
    info "Limpando arquivos temporarios..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    rm -rf htmlcov/ .coverage dist/ build/ 2>/dev/null || true
    info "Limpeza concluida"
}

cmd_clean_docker() {
    info "Limpando recursos Docker..."
    docker-compose down -v
    docker system prune -f
}

cmd_clean_all() {
    cmd_clean
    cmd_clean_docker
}

# Main
case "${1:-help}" in
    install)        cmd_install ;;
    setup)          cmd_setup ;;
    init-aws)       cmd_init_aws ;;
    notebook)       cmd_notebook ;;
    lab)            cmd_lab ;;
    shell)          cmd_shell ;;
    lint)           cmd_lint ;;
    format)         cmd_format ;;
    type-check)     cmd_type_check ;;
    security)       cmd_security ;;
    quality)        cmd_quality ;;
    test)           cmd_test ;;
    test-cov)       cmd_test_cov ;;
    train)          cmd_train ;;
    serve)          cmd_serve ;;
    serve-prod)     cmd_serve_prod ;;
    monitor)        cmd_monitor ;;
    drift-check)    cmd_drift_check ;;
    docker-build)   cmd_docker_build ;;
    docker-up)      cmd_docker_up ;;
    docker-down)    cmd_docker_down ;;
    docker-logs)    cmd_docker_logs ;;
    mlflow-up)      cmd_mlflow_up ;;
    monitoring-up)  cmd_monitoring_up ;;
    clean)          cmd_clean ;;
    clean-docker)   cmd_clean_docker ;;
    clean-all)      cmd_clean_all ;;
    help|--help|-h) show_help ;;
    *)
        error "Comando desconhecido: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
