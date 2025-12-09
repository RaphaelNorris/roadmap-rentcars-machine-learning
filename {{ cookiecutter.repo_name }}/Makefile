.PHONY: help install setup init-aws notebook lab shell lint format type-check security quality
.PHONY: test test-cov train serve serve-prod monitor drift-check
.PHONY: docker-build docker-up docker-down docker-logs mlflow-up monitoring-up
.PHONY: clean clean-docker clean-all

# Default target
.DEFAULT_GOAL := help

# Script path
MLOPS_SCRIPT := ./scripts/mlops.sh

help: ## Mostrar esta mensagem de ajuda
	@echo "Comandos disponiveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Ou use diretamente: $(MLOPS_SCRIPT) [comando]"

# Setup e Configuracao
install: ## Instalar dependencias
	@$(MLOPS_SCRIPT) install

setup: ## Setup completo do ambiente
	@$(MLOPS_SCRIPT) setup

init-aws: ## Criar recursos AWS (S3, ECR, Athena)
	@$(MLOPS_SCRIPT) init-aws

# Desenvolvimento
notebook: ## Iniciar Jupyter notebook
	@$(MLOPS_SCRIPT) notebook

lab: ## Iniciar Jupyter lab
	@$(MLOPS_SCRIPT) lab

shell: ## Shell Python interativo
	@$(MLOPS_SCRIPT) shell

# Qualidade de Codigo
lint: ## Linting com Ruff
	@$(MLOPS_SCRIPT) lint

format: ## Formatar codigo
	@$(MLOPS_SCRIPT) format

type-check: ## Type checking com mypy
	@$(MLOPS_SCRIPT) type-check

security: ## Security scan com bandit
	@$(MLOPS_SCRIPT) security

quality: ## Todos os checks de qualidade
	@$(MLOPS_SCRIPT) quality

# Testes
test: ## Rodar testes
	@$(MLOPS_SCRIPT) test

test-cov: ## Testes com cobertura
	@$(MLOPS_SCRIPT) test-cov

# ML Operations
train: ## Treinar modelo
	@$(MLOPS_SCRIPT) train

serve: ## Iniciar API de inferencia
	@$(MLOPS_SCRIPT) serve

serve-prod: ## API em modo producao
	@$(MLOPS_SCRIPT) serve-prod

# Monitoramento
monitor: ## Monitorar modelos
	@$(MLOPS_SCRIPT) monitor

drift-check: ## Verificar data drift
	@$(MLOPS_SCRIPT) drift-check

# Infraestrutura
docker-build: ## Build de imagens Docker
	@$(MLOPS_SCRIPT) docker-build

docker-up: ## Iniciar todos servicos
	@$(MLOPS_SCRIPT) docker-up

docker-down: ## Parar servicos
	@$(MLOPS_SCRIPT) docker-down

docker-logs: ## Ver logs dos servicos
	@$(MLOPS_SCRIPT) docker-logs

mlflow-up: ## Apenas MLFlow
	@$(MLOPS_SCRIPT) mlflow-up

monitoring-up: ## Prometheus + Grafana
	@$(MLOPS_SCRIPT) monitoring-up

# Limpeza
clean: ## Limpar arquivos temporarios
	@$(MLOPS_SCRIPT) clean

clean-docker: ## Limpar recursos Docker
	@$(MLOPS_SCRIPT) clean-docker

clean-all: ## Limpeza completa
	@$(MLOPS_SCRIPT) clean-all
