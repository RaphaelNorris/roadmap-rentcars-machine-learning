# Integração com Repos Separados

Guia para integrar DataOps e MLOps quando os repositórios são separados.

## Cenário: Repos Separados

```
┌─────────────────────────────────────┐
│  Repo 1: dataops-dbt                │
│  ├── dbt/                           │
│  │   └── models/                    │
│  ├── airflow/                       │
│  │   └── dags/                      │
│  │       ├── dbt_daily.py           │
│  │       ├── athena_queries.py      │
│  │       └── mlops_integration.py   │ ← DAG de integração (ADICIONAR)
│  └── README.md                      │
└─────────────────────────────────────┘
                    ↓ SSHOperator
┌─────────────────────────────────────┐
│  Repo 2: mlops-projeto1 (EC2)       │
│  ├── src/                           │
│  │   ├── pipelines/                 │
│  │   │   └── DS/                    │
│  │   │       ├── feature_pipeline/  │ ← Executado via SSH
│  │   │       └── training_pipeline/ │ ← Executado via SSH
│  │   └── inference/                 │
│  ├── config/                        │
│  │   └── prod/                      │
│  ├── airflow/                       │ ← NÃO USADO (apenas referência)
│  └── README.md                      │
└─────────────────────────────────────┘

Airflow Centralizado
(lê apenas Repo 1: dataops-dbt)
```

---

## Arquitetura Recomendada

### Separação de Responsabilidades

**Repo dataops-dbt** (mantido pelo time DataOps):
- DAGs Airflow (orquestração)
- Transformações DBT
- Queries Athena
- **DAG de integração MLOps** (SSHOperator)

**Repo mlops-projeto1** (mantido pelo time ML):
- Código Python (pipelines, modelos)
- Configurações (dev/staging/prod)
- Testes (unit, integration, responsible AI)
- Documentação técnica

### Fluxo de Trabalho

```
1. Time DataOps:
   - Desenvolve transformações DBT
   - Cria/atualiza DAG de integração MLOps
   - Commit em dataops-dbt/
   - Airflow detecta mudanças automaticamente

2. Time ML:
   - Desenvolve pipelines Python
   - Atualiza configs de modelo
   - Commit em mlops-projeto1/
   - Deploy manual em EC2 (git pull)

3. Execução:
   - Airflow (repo dataops-dbt) executa DAG
   - DAG usa SSHOperator para executar código em EC2 (repo mlops-projeto1)
```

---

## Setup Passo-a-Passo

### 1. Preparar EC2 de MLOps

**1.1. Clonar repo MLOps na EC2**:

```bash
# SSH na EC2
ssh -i key.pem ec2-user@<ec2-ip>

# Clonar repositório
cd /opt
sudo git clone https://github.com/empresa/mlops-projeto1.git
sudo chown -R ec2-user:ec2-user mlops-projeto1
cd mlops-projeto1

# Setup ambiente
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Iniciar serviços (MLFlow, PostgreSQL, etc)
docker-compose up -d

# Testar pipeline
python -m src.pipelines.DS.training_pipeline.train --help
```

**1.2. Configurar atualização automática (opcional)**:

```bash
# Criar script de deploy
cat > /opt/mlops-projeto1/deploy.sh <<'EOF'
#!/bin/bash
cd /opt/mlops-projeto1
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
docker-compose restart
EOF

chmod +x /opt/mlops-projeto1/deploy.sh

# Criar webhook ou cron para atualização
# Exemplo: cron para atualizar a cada 5 minutos
# */5 * * * * /opt/mlops-projeto1/deploy.sh >> /var/log/mlops-deploy.log 2>&1
```

### 2. Configurar SSH no Airflow

**2.1. Gerar par de chaves SSH (se não existir)**:

```bash
# No servidor Airflow
ssh-keygen -t rsa -b 4096 -f ~/.ssh/ec2_mlops_projeto1 -N ""

# Copiar chave pública para EC2
ssh-copy-id -i ~/.ssh/ec2_mlops_projeto1.pub ec2-user@<ec2-ip>

# Testar conexão
ssh -i ~/.ssh/ec2_mlops_projeto1 ec2-user@<ec2-ip> "echo 'SSH OK'"
```

**2.2. Adicionar Connection no Airflow**:

```
Airflow UI → Admin → Connections → Add

Connection ID: ec2_mlops_projeto1
Connection Type: SSH
Host: <ec2-public-ip>
Username: ec2-user
Private Key: <conteúdo do arquivo ~/.ssh/ec2_mlops_projeto1>
Port: 22
```

**2.3. Testar conexão no Airflow**:

Airflow UI → Admin → Connections → ec2_mlops_projeto1 → Test

### 3. Criar DAG de Integração no Repo DataOps

**3.1. Copiar template**:

```bash
# No repo dataops-dbt
cd dataops-dbt/airflow/dags/

# Copiar template do repo mlops-projeto1
cp /caminho/para/mlops-projeto1/airflow/dags/mlops/TEMPLATE_for_separate_dataops_repo.py \
   mlops_churn_integration.py
```

**3.2. Ajustar configurações**:

```python
# mlops_churn_integration.py

# Ajustar:
PROJECT_NAME = "churn"  # Nome do projeto
S3_DATAMART_BUCKET = "empresa-processed-data"
S3_DATAMART_KEY = "datamarts/churn/latest.parquet"
S3_ML_BUCKET = "empresa-ml-artifacts"
EC2_SSH_CONN_ID = "ec2_mlops_projeto1"
EC2_MLOPS_PATH = "/opt/mlops-projeto1"
```

**3.3. Commit no repo DataOps**:

```bash
git add airflow/dags/mlops_churn_integration.py
git commit -m "Add MLOps integration DAG for churn project"
git push origin main
```

### 4. Verificar Security Groups AWS

**4.1. Security Group da EC2 de MLOps**:

```
Inbound Rules:
- Type: SSH (22)
  Source: <airflow-server-ip>/32
  Description: Airflow SSH access

- Type: Custom TCP (8000)
  Source: <your-ip>/32
  Description: FastAPI (opcional, para testes)
```

**4.2. IAM Role da EC2** (para acesso S3):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::empresa-processed-data/*",
        "arn:aws:s3:::empresa-ml-artifacts/*"
      ]
    }
  ]
}
```

### 5. Testar Integração

**5.1. Testar SSH manualmente**:

```bash
# No servidor Airflow
ssh -i ~/.ssh/ec2_mlops_projeto1 ec2-user@<ec2-ip> \
  "cd /opt/mlops-projeto1 && source venv/bin/activate && python -m src.pipelines.DS.training_pipeline.train --help"

# Deve retornar help do comando
```

**5.2. Testar DAG no Airflow**:

```
1. Airflow UI → DAGs → mlops_churn_integration
2. Verificar se DAG aparece (aguardar ~1 min)
3. Trigger DAG manualmente
4. Monitorar execução:
   - Graph View: Ver fluxo completo
   - Logs: Verificar saída de cada task
```

**5.3. Verificar resultados**:

```bash
# SSH na EC2
ssh -i key.pem ec2-user@<ec2-ip>

# Verificar logs
tail -f /opt/mlops-projeto1/logs/training/*.log

# Verificar MLFlow
# Browser: http://<ec2-ip>:5000
# Ver experiments e runs criados
```

---

## Estrutura de Arquivos

### Repo dataops-dbt (DataOps)

```
dataops-dbt/
├── dbt/
│   ├── models/
│   │   ├── staging/
│   │   └── marts/
│   └── dbt_project.yml
│
├── airflow/
│   ├── dags/
│   │   ├── dbt_daily.py                    # DAG DBT existente
│   │   ├── athena_queries.py               # DAG Athena existente
│   │   └── mlops_churn_integration.py      # DAG NOVA (integração)
│   └── plugins/
│
├── sql/
│   └── athena/
│
└── README.md
```

### Repo mlops-projeto1 (MLOps)

```
mlops-projeto1/
├── src/
│   ├── pipelines/
│   │   └── DS/
│   │       ├── feature_pipeline/
│   │       │   └── transform.py            # Executado via SSH
│   │       └── training_pipeline/
│   │           └── train.py                # Executado via SSH
│   ├── inference/
│   │   └── api.py
│   └── deployment/
│       └── promote_model.py                # Executado via SSH
│
├── config/
│   ├── dev/
│   ├── staging/
│   └── prod/
│       └── model_config.yaml               # Usado por training
│
├── tests/
│   └── responsible_ai/
│       ├── test_fairness.py                # Executado via SSH
│       └── test_robustness.py              # Executado via SSH
│
├── airflow/
│   └── dags/
│       └── mlops/
│           └── TEMPLATE_for_separate_dataops_repo.py  # Template para copiar
│
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## Manutenção e Updates

### Atualizar Código MLOps (Repo mlops-projeto1)

**Cenário**: Time ML fez mudanças em `training_pipeline/train.py`

```bash
# 1. Desenvolver localmente
git checkout -b feature/improve-training
# ... fazer mudanças ...
git commit -m "Improve training pipeline"
git push origin feature/improve-training

# 2. Merge para main
# GitHub: Create PR → Merge

# 3. Deploy na EC2
ssh ec2-user@<ec2-ip>
cd /opt/mlops-projeto1
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade  # Se mudaram dependências

# 4. Próxima execução do Airflow já usará código novo
```

### Atualizar DAG de Integração (Repo dataops-dbt)

**Cenário**: Mudança na interface do `train.py` (novos parâmetros)

```bash
# No repo dataops-dbt
git checkout -b update/mlops-dag

# Editar DAG
vim airflow/dags/mlops_churn_integration.py

# Exemplo: adicionar novo parâmetro --hyperparams
train_model = SSHOperator(
    task_id='train_model',
    command=f'''
        ...
        python -m src.pipelines.DS.training_pipeline.train \
            --features-path ... \
            --hyperparams '{{\"n_estimators\": 200}}'  # NOVO
    '''
)

git commit -m "Update MLOps DAG: add hyperparams parameter"
git push origin update/mlops-dag

# Merge PR
# Airflow detecta mudança automaticamente (~1 min)
```

### Adicionar Novo Projeto MLOps

**Cenário**: Criar projeto "credit_risk" além do "churn" existente

```bash
# 1. Criar nova EC2 para projeto credit_risk
# 2. Clonar repo mlops-projeto1 na nova EC2
# 3. Configurar SSH no Airflow (ec2_mlops_credit_risk)

# 4. Criar nova DAG no repo dataops-dbt
cd dataops-dbt/airflow/dags/
cp mlops_churn_integration.py mlops_credit_risk_integration.py

# 5. Ajustar configurações
vim mlops_credit_risk_integration.py
# PROJECT_NAME = "credit_risk"
# EC2_SSH_CONN_ID = "ec2_mlops_credit_risk"
# ...

git add mlops_credit_risk_integration.py
git commit -m "Add MLOps integration for credit_risk project"
git push
```

---

## Troubleshooting

### Problema 1: DAG não aparece no Airflow

**Causas**:
- Erro de sintaxe Python na DAG
- Importações faltando
- Airflow não atualizou DAGs

**Solução**:
```bash
# Verificar sintaxe
python airflow/dags/mlops_churn_integration.py

# Forçar refresh
# Airflow UI → DAGs → Refresh

# Ver logs do scheduler
tail -f /var/log/airflow/scheduler.log | grep mlops_churn
```

### Problema 2: SSHOperator timeout

**Causas**:
- Security group bloqueando porta 22
- Chave SSH incorreta
- EC2 não acessível

**Solução**:
```bash
# Testar SSH manual
ssh -i ~/.ssh/ec2_mlops_projeto1 ec2-user@<ec2-ip>

# Verificar security group
# AWS Console → EC2 → Security Groups → Inbound Rules

# Verificar logs do Airflow
# Airflow UI → DAG → Task → Logs
```

### Problema 3: Comando Python falha na EC2

**Causas**:
- Virtual env não ativado
- Dependências faltando
- Caminho incorreto

**Solução**:
```bash
# SSH na EC2 e testar comando manualmente
ssh ec2-user@<ec2-ip>
cd /opt/mlops-projeto1
source venv/bin/activate
python -m src.pipelines.DS.training_pipeline.train --help

# Verificar dependências
pip list | grep mlflow

# Verificar logs
tail -f logs/training/*.log
```

### Problema 4: S3 access denied

**Causas**:
- IAM role da EC2 sem permissões S3
- Bucket incorreto

**Solução**:
```bash
# Testar acesso S3 na EC2
ssh ec2-user@<ec2-ip>
aws s3 ls s3://empresa-processed-data/datamarts/churn/

# Verificar IAM role
# AWS Console → EC2 → Instance → IAM Role → Permissions

# Adicionar permissão S3 se necessário
```

---

## Monitoramento

### Métricas Importantes

| Métrica | Threshold | Ação |
|---------|-----------|------|
| Lag entre DAG DBT e MLOps | < 2 horas | Alert se > 2h |
| SSH connection failures | 0 | Alert imediatamente |
| Training duration | < 1 hora | Investigar se > 1h |
| Training failures | 0 | Alert + rollback |

### Alertas Prometheus

```yaml
groups:
  - name: mlops_integration_alerts
    rules:
      - alert: SSHConnectionFailed
        expr: airflow_task_failures{task_id=~".*ssh.*"} > 0
        for: 1m
        annotations:
          summary: "SSH connection to MLOps EC2 failed"
          description: "Task {{ $labels.task_id }} failed - check SSH connection"

      - alert: TrainingFailed
        expr: airflow_task_failures{task_id="train_model"} > 0
        for: 1m
        annotations:
          summary: "Model training failed on EC2"
          description: "Check logs in EC2: /opt/mlops-projeto1/logs/training/"
```

---

## Alternativas

Se SSHOperator não funcionar para seu caso, considere:

### 1. API REST (SimpleHttpOperator)

Ver: `docs/architecture/airflow_dataops_mlops_integration.md` - Opção 3

### 2. AWS Step Functions

Orquestrar via Step Functions em vez de Airflow.

### 3. Airflow em cada EC2

Cada EC2 tem Airflow próprio (descentralizado).

### 4. Git Hooks + CI/CD

Trigger training via GitHub Actions em vez de Airflow.

---

## Resumo

**✅ Repos separados NÃO é problema**

**Solução**: DAG no repo DataOps + SSHOperator

**Vantagens**:
- Separação clara de responsabilidades
- Times independentes (DataOps vs ML)
- Flexibilidade para escalar (múltiplas EC2s)

**Desvantagens**:
- Setup inicial mais complexo (SSH, security groups)
- Mudanças em interfaces requerem sync entre repos

**Recomendação**: Usar SSHOperator é a solução mais simples e robusta para seu cenário.
