# CI/CD - Esteira MLOps RentCars

## Visão Geral

Este documento descreve o fluxo de CI/CD da esteira de MLOps da RentCars, desde o desenvolvimento até a produção.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │ feature/ │────▶│   dev    │────▶│  master  │────▶│   AWS    │          │
│  │  branch  │ PR  │  branch  │ PR  │  branch  │ CI  │  Deploy  │          │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘          │
│                                                                             │
│  Desenvolvimento   Homologação      Produção         Step Functions        │
│                                                      ECS + ECR             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fluxo de Branches

### 1. Feature Branch (Desenvolvimento)

```bash
# Criar branch de feature
git checkout dev
git pull origin dev
git checkout -b feature/recsys-v1
```

**O que fazer:**
- Desenvolver o modelo
- Testar localmente
- Registrar experimentos no MLFlow
- Validar métricas

**Regras:**
- Nome da branch: `feature/<projeto>-<descrição>`
- Sempre partir da branch `dev`
- Não fazer push direto para `dev` ou `master`

---

### 2. Branch Dev (Homologação)

```bash
# Abrir PR de feature → dev
gh pr create --base dev --title "feat: adiciona modelo recsys v1"
```

**O que acontece no merge:**
1. GitHub Actions executa `build_apps.sh`
2. Ambiente: `ENV=dev`
3. Deploy em ambiente de homologação
4. Step Function criada/atualizada em dev

**Regras:**
- Requer aprovação de code review
- Testes devem passar
- Modelo deve estar registrado no MLFlow (Staging)

---

### 3. Branch Master (Produção)

```bash
# Abrir PR de dev → master
gh pr create --base master --title "release: recsys v1.0.0"
```

**O que acontece no merge:**
1. GitHub Actions executa `build_apps.sh`
2. Ambiente: `ENV=prd`
3. Deploy em produção
4. Step Function criada/atualizada em prod
5. Scheduler (cron) ativado

**Regras:**
- Requer aprovação obrigatória
- Só aceita PRs vindos de `dev`
- Modelo deve ser promovido para Production no MLFlow

---

## GitHub Actions

### Workflow: build.yml

**Trigger:**
- Push em `dev` ou `master`
- Workflow dispatch (manual)

**Jobs:**

```yaml
name: Build Apps

on:
  push:
    branches:
      - dev
      - master

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Set ENV (dev ou prd baseado na branch)
      - Configure AWS credentials
      - Run build_apps.sh
```

**Variáveis de ambiente:**
| Branch | ENV |
|--------|-----|
| dev | `dev` |
| master | `prd` |

---

## Build Script (build_apps.sh)

O script `build_apps.sh` é o coração do CI/CD. Ele:

### 1. Configuração Inicial
- Define variáveis de ambiente (região, VPC, subnets)
- Configura URIs do MLFlow
- Define roles IAM

### 2. Infraestrutura Base
- Cria Security Group para tasks de ML
- Cria roles para Step Functions
- Cria roles para EventBridge Scheduler

### 3. Para cada projeto em `src/projects/`:

```
Para cada config.yaml encontrado:
│
├── 1. Parse do config.yaml (yq)
│      - project_name, version
│      - schedule (cron, timeout, retries)
│      - runtime (python_version, env_vars)
│      - infra (ec2_type, min/max capacity)
│
├── 2. Docker Build & Push
│      - Build da imagem com Dockerfile do projeto
│      - Push para ECR: {account}.dkr.ecr.{region}.amazonaws.com/{ecr}:{project}-{version}
│
├── 3. ECS Infrastructure
│      - Cria cluster ECS
│      - Cria Launch Template
│      - Cria Auto Scaling Group
│      - Cria Capacity Provider
│      - Registra Task Definition
│
├── 4. Step Function
│      - Gera definição dinamicamente baseado nos steps do config.yaml
│      - Configura timeout e retry por step
│      - Injeta variáveis de ambiente (incluindo MLFlow)
│      - Cria ou atualiza State Machine
│
└── 5. EventBridge Scheduler
       - Configura cron baseado no config.yaml
       - Habilita ou desabilita baseado em schedule.enabled
```

---

## Estrutura de Projeto

```
src/projects/<nome>/
├── config.yaml          # Configurações do projeto
├── main.py              # Código principal (ou steps/)
├── requirements.txt     # Dependências Python
├── Dockerfile           # Imagem Docker
└── README.md            # Documentação do projeto
```

### config.yaml

```yaml
project_name: recsys
description: "Sistema de recomendação de veículos"
version: "1.0.0"

owners:
  tech_owner: "ml-team@rentcars.com"
  business_owner: "produto@rentcars.com"

schedule:
  enabled: true
  cron: "0 3 * * *"           # Todo dia às 3h
  timeout_seconds: 3600        # 1 hora
  retries: 2

runtime:
  python_version: "3.11"
  requirements_file: "requirements.txt"
  env_vars:
    ENV: "dev"

batch:
  enabled: true
  orchestrator: "step_functions"
  pipeline:
    steps:
      - id: "extract_features"
        name: "Extract Features"
        command: "python steps/extract_features.py"
        resources:
          cpu: 1.0
          memory_gb: 4

      - id: "score_model"
        name: "Score Model"
        command: "python steps/score_model.py"
        resources:
          cpu: 2.0
          memory_gb: 8

infra:
  compute:
    provider: "ec2"
    ec2_type: "t3.xlarge"
    min_quantity: 0
    max_quantity: 2
    spot: true

notifications:
  on_success:
    - "ml-team@rentcars.com"
  on_failure:
    - "ml-team@rentcars.com"
    - "oncall@rentcars.com"
```

---

## MLFlow

### Visão Geral

O MLFlow é o sistema central de gestão do ciclo de vida dos modelos. Ele é responsável por:

- **Tracking**: Registrar experimentos, métricas, parâmetros
- **Model Registry**: Versionar e gerenciar modelos
- **Artifacts**: Armazenar modelos e arquivos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLFlow Server                                        │
│                   http://mlflow.bi.rentcars.com                            │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │    Tracking     │  │ Model Registry  │  │   Artifacts     │            │
│  │  (experimentos) │  │   (versões)     │  │   (S3)          │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                      │
│           └────────────────────┼────────────────────┘                      │
│                                │                                           │
│                         ┌──────┴──────┐                                    │
│                         │  PostgreSQL │                                    │
│                         │  (metadata) │                                    │
│                         └─────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### URL de Acesso

| Ambiente | URL |
|----------|-----|
| MLFlow UI | http://mlflow.bi.rentcars.com |
| Tracking API | http://mlflow.bi.rentcars.com |

---

### Model Registry

O Model Registry gerencia as versões dos modelos:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Model Registry                                                             │
│                                                                             │
│  📦 recsys                                                                  │
│     ├── v1 (roc_auc: 0.78) .......... Archived                             │
│     ├── v2 (roc_auc: 0.82) .......... Archived                             │
│     ├── v3 (roc_auc: 0.85) .......... Production  ✅                        │
│     └── v4 (roc_auc: 0.87) .......... Staging                              │
│                                                                             │
│  📦 churn                                                                   │
│     ├── v1 (roc_auc: 0.80) .......... Production  ✅                        │
│     └── v2 (roc_auc: 0.79) .......... Archived                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Stages disponíveis:**

| Stage | Descrição |
|-------|-----------|
| `None` | Modelo recém registrado |
| `Staging` | Em validação/homologação |
| `Production` | Em produção |
| `Archived` | Versão antiga/descontinuada |

---

### Fluxo de Promoção de Modelos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. TREINO (branch feature)                                                 │
│                                                                             │
│  Cientista treina modelo                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  mlflow.log_model() ──────▶ Modelo registrado (Stage: None)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. VALIDAÇÃO (branch dev)                                                  │
│                                                                             │
│  Time valida métricas                                                      │
│       │                                                                     │
│       ▼                                                                     │
│  Promove para Staging ──────▶ Modelo em Staging                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. PRODUÇÃO (branch master)                                                │
│                                                                             │
│  Aprovação final                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  Promove para Production ──────▶ Modelo em Production                      │
│  (versão anterior → Archived)                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Variáveis de Ambiente

O `build_apps.sh` injeta automaticamente nas tasks:

| Variável | Valor |
|----------|-------|
| `MLFLOW_TRACKING_URI` | http://mlflow.bi.rentcars.com |
| `MLFLOW_S3_ENDPOINT_URL` | https://s3.us-east-1.amazonaws.com |
| `MLFLOW_EXPERIMENT_NAME` | {project_name} |

---

### Uso no Código

#### Registrar experimento e métricas

```python
import mlflow
import os

# Conecta ao MLFlow (variáveis já injetadas pelo Step Function)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

# Treina e registra
with mlflow.start_run(run_name="treino-v1"):
    # Log de parâmetros
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("max_depth", 6)

    # Treina modelo
    model = train_model(X_train, y_train)

    # Log de métricas
    mlflow.log_metric("roc_auc", 0.85)
    mlflow.log_metric("f1", 0.72)

    # Registra modelo
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="recsys"
    )
```

#### Carregar modelo para inferência

```python
import mlflow
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Carrega modelo em Production
model = mlflow.pyfunc.load_model("models:/recsys/Production")

# Faz predições
predictions = model.predict(features)
```

#### Promover modelo via código

```python
from mlflow import MlflowClient

client = MlflowClient("http://mlflow.bi.rentcars.com")

# Promover versão 4 para Production
client.transition_model_version_stage(
    name="recsys",
    version="4",
    stage="Production"
)
```

---

### Promover Modelo via UI

1. Acessar http://mlflow.bi.rentcars.com
2. Ir em **Models** → Selecionar modelo
3. Clicar na versão desejada
4. Clicar em **Stage** → **Transition to Production**

---

### Promover Modelo via CLI

```bash
# Promover para Staging
mlflow models transition-stage --name recsys --version 4 --stage Staging

# Promover para Production
mlflow models transition-stage --name recsys --version 4 --stage Production
```

---

## Fluxo de Deploy Completo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DESENVOLVIMENTO                                                         │
│                                                                             │
│  Cientista trabalha em feature/recsys-v1                                   │
│  ├── Desenvolve código                                                     │
│  ├── Testa localmente                                                      │
│  ├── Registra modelo no MLFlow (Staging)                                   │
│  └── Abre PR para dev                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. HOMOLOGAÇÃO (dev)                                                       │
│                                                                             │
│  PR aprovado e merged em dev                                               │
│  ├── GitHub Actions dispara                                                │
│  ├── build_apps.sh executa (ENV=dev)                                       │
│  ├── Imagem Docker → ECR                                                   │
│  ├── Step Function criada em dev                                           │
│  └── Time testa em ambiente de homologação                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. PRODUÇÃO (master)                                                       │
│                                                                             │
│  PR de dev → master aprovado e merged                                      │
│  ├── GitHub Actions dispara                                                │
│  ├── build_apps.sh executa (ENV=prd)                                       │
│  ├── Imagem Docker → ECR (prod)                                            │
│  ├── Step Function criada em prod                                          │
│  ├── Scheduler ativado (cron)                                              │
│  └── Modelo promovido para Production no MLFlow                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. EXECUÇÃO (diário)                                                       │
│                                                                             │
│  EventBridge Scheduler dispara no horário do cron                          │
│  ├── Step Function inicia                                                  │
│  ├── ECS provisiona instância (spot)                                       │
│  ├── Container executa steps em sequência                                  │
│  ├── Logs enviados para CloudWatch                                         │
│  └── Notificação enviada (sucesso/falha)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recursos AWS Criados

Para cada projeto, o CI/CD cria:

| Recurso | Nome | Descrição |
|---------|------|-----------|
| ECR Image | `{ecr}:{project}-{version}` | Imagem Docker do projeto |
| ECS Cluster | `ml-cluster-{project}` | Cluster para executar tasks |
| Launch Template | `lt-{project}` | Template de instâncias EC2 |
| Auto Scaling Group | `asg-{project}` | Gerencia capacidade |
| Capacity Provider | `cp-{project}` | Conecta ASG ao ECS |
| Task Definition | `{project}` | Definição da task ECS |
| Step Function | `{project}-workflow` | Orquestração dos steps |
| EventBridge Schedule | `sched-{project}` | Agendamento (cron) |
| CloudWatch Log Group | `/ecs/{project}` | Logs de execução |

---

## Comandos Úteis

### Executar Step Function manualmente

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account}:stateMachine:{project}-workflow
```

### Ver logs de execução

```bash
aws logs tail /ecs/{project} --follow
```

### Verificar status do scheduler

```bash
aws scheduler get-schedule --name sched-{project}
```

### Atualizar modelo no MLFlow

```python
from mlflow import MlflowClient

client = MlflowClient("http://mlflow.bi.rentcars.com")

# Promover modelo para Production
client.transition_model_version_stage(
    name="recsys",
    version="2",
    stage="Production"
)
```

---

## Troubleshooting

### Build falhou no GitHub Actions

1. Verificar logs do workflow no GitHub
2. Verificar se Dockerfile está correto
3. Verificar se requirements.txt está completo

### Step Function falhou

1. Verificar logs no CloudWatch: `/ecs/{project}`
2. Verificar se variáveis de ambiente estão corretas
3. Verificar se modelo existe no MLFlow

### Scheduler não está disparando

1. Verificar se `schedule.enabled: true` no config.yaml
2. Verificar expressão cron
3. Verificar status: `aws scheduler get-schedule --name sched-{project}`

---

## Padroes de Nomenclatura

### Projetos

| Regra | Exemplo |
|-------|---------|
| Lowercase | recsys, churn |
| Sem espacos (usar underscore) | churn_prediction |
| Descritivo e curto | pricing_optimizer |

### Branches

Padrao: `<tipo>/<projeto>-<descricao>`

| Tipo | Uso | Exemplo |
|------|-----|---------|
| feature/ | Nova funcionalidade | feature/recsys-add-embeddings |
| fix/ | Correcao de bug | fix/churn-null-handling |
| bugfix/ | Correcao de bug (alternativo) | bugfix/recsys-memory-leak |
| hotfix/ | Correcao urgente em producao | hotfix/pricing-critical-error |
| refactor/ | Refatoracao | refactor/recsys-optimize |
| docs/ | Documentacao | docs/recsys-readme |
| experiment/ | Experimento exploratorio | experiment/recsys-transformer |

### MLFlow

| Recurso | Padrao | Exemplo |
|---------|--------|---------|
| Experiment | {projeto} | recsys |
| Registered Model | {projeto} | recsys |
| Run Name | {descricao}-{data} | xgboost-tuned-20250106 |

### AWS

| Recurso | Padrao | Exemplo |
|---------|--------|---------|
| ECR Image | {ecr}:{projeto}-{versao} | ecr-prd:recsys-1.0.0 |
| ECS Cluster | ml-cluster-{projeto} | ml-cluster-recsys |
| Step Function | {projeto}-workflow | recsys-workflow |
| Scheduler | sched-{projeto} | sched-recsys |
| Log Group | /ecs/{projeto} | /ecs/recsys |

---

## Padroes de Commit

### Formato

```
<tipo>: <descricao curta>
```

### Tipos

| Tipo | Quando usar | Exemplo |
|------|-------------|---------|
| feat | Nova funcionalidade | feat: add user embeddings |
| fix | Correcao de bug | fix: handle null values |
| bug | Correcao de bug (alternativo) | bug: fix memory leak |
| hotfix | Correcao urgente | hotfix: fix critical error |
| refactor | Refatoracao sem mudar comportamento | refactor: optimize query |
| docs | Documentacao | docs: update README |
| test | Adicao ou correcao de testes | test: add unit tests |
| chore | Tarefas de manutencao | chore: update dependencies |
| style | Formatacao, sem mudanca de logica | style: fix indentation |
| perf | Melhoria de performance | perf: optimize feature extraction |
| ci | Mudancas no CI/CD | ci: add new workflow |
| build | Mudancas no build | build: update Dockerfile |

### Regras

- Primeira linha: maximo 72 caracteres
- Usar imperativo (add, fix, update)
- Nao terminar com ponto
- Em ingles ou portugues (manter consistencia no projeto)

### Exemplos

```bash
feat: add collaborative filtering model
fix: handle missing values in user_age
bug: fix null pointer in scoring
hotfix: fix critical pricing error
refactor: optimize feature extraction pipeline
docs: add API documentation
test: add unit tests for model
chore: update mlflow version
perf: improve inference latency
ci: add code quality checks
```

---

## Padroes de Codigo

### Estrutura de Arquivos

```
src/projects/{projeto}/
├── config.yaml           # Configuracao
├── main.py               # Entry point
├── steps/                # Logica de cada step
│   ├── extract_features.py
│   ├── train_model.py
│   └── score_model.py
├── utils/                # Funcoes auxiliares
├── tests/                # Testes
├── notebooks/            # Notebooks de desenvolvimento
├── requirements.txt      # Dependencias
├── Dockerfile            # Imagem Docker
└── README.md             # Documentacao
```

### Nomenclatura no Codigo

| Tipo | Padrao | Exemplo |
|------|--------|---------|
| Variaveis | snake_case | user_features |
| Funcoes | snake_case | extract_features() |
| Constantes | UPPER_SNAKE_CASE | MAX_ITERATIONS |
| Classes | PascalCase | FeatureExtractor |
| DataFrames | df_{descricao} | df_features |

### Imports

```python
# 1. Standard library
import os
import json
from datetime import datetime

# 2. Third party
import pandas as pd
import numpy as np
import mlflow

# 3. Local
from utils import load_data
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Iniciando processamento")
logger.warning("Dados faltantes encontrados")
logger.error("Falha ao carregar modelo")
```

---

## Boas Praticas

### Desenvolvimento

1. Sempre partir da branch dev atualizada
2. Testar localmente antes de abrir PR
3. Registrar experimentos no MLFlow
4. Usar variaveis de ambiente para configuracoes
5. Nao commitar credenciais ou dados sensiveis

### Dependencias

```
# BOM - versoes fixas
pandas==2.0.3
scikit-learn==1.3.0
mlflow==2.9.2

# RUIM - versoes abertas
pandas
scikit-learn
mlflow
```

### Dockerfile

```dockerfile
# Usar imagem base especifica
FROM python:3.11-slim

WORKDIR /app

# Copiar e instalar deps primeiro (cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo por ultimo
COPY . .

CMD ["python", "main.py"]
```

### Seguranca

```python
# ERRADO
password = "minha_senha_123"

# CERTO
password = os.environ["DB_PASSWORD"]
```

---

## Checklist

### Antes de abrir PR

- [ ] Codigo segue padroes de nomenclatura
- [ ] Sem credenciais ou dados sensiveis
- [ ] requirements.txt atualizado
- [ ] config.yaml valido
- [ ] Testado localmente
- [ ] Metricas logadas no MLFlow

### Antes de merge para master

- [ ] Code review aprovado
- [ ] Testado em ambiente dev
- [ ] Modelo registrado no MLFlow (Staging)
- [ ] Metricas validadas

---

## Referências

- [MLFlow Tracking Server](http://mlflow.bi.rentcars.com)
- [AWS Step Functions Console](https://console.aws.amazon.com/states)
- [AWS ECS Console](https://console.aws.amazon.com/ecs)
- [GitHub Actions](https://github.com/rentcars/rentcars-data-platform-science/actions)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8 - Style Guide](https://peps.python.org/pep-0008/)
