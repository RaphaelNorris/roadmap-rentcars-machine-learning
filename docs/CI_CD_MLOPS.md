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

## Integração com MLFlow

### Variáveis injetadas automaticamente

O `build_apps.sh` injeta automaticamente nas tasks:

| Variável | Valor |
|----------|-------|
| `MLFLOW_TRACKING_URI` | http://mlflow.bi.rentcars.com |
| `MLFLOW_S3_ENDPOINT_URL` | https://s3.us-east-1.amazonaws.com |
| `MLFLOW_EXPERIMENT_NAME` | {project_name} |

### Uso no código

```python
import mlflow
import os

# Variáveis já estão no ambiente
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

# Carregar modelo em Production
model = mlflow.pyfunc.load_model("models:/recsys/Production")
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

## Referências

- [MLFlow Tracking Server](http://mlflow.bi.rentcars.com)
- [AWS Step Functions Console](https://console.aws.amazon.com/states)
- [AWS ECS Console](https://console.aws.amazon.com/ecs)
- [GitHub Actions](https://github.com/rentcars/rentcars-data-platform-science/actions)
