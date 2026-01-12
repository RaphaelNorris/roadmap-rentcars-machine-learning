# Boas Praticas de Desenvolvimento e CI/CD - MLOps

## Introducao

Este documento define as boas praticas para desenvolvimento de modelos de Machine Learning e uso da esteira de CI/CD na RentCars.

---

## 1. Desenvolvimento Local

### 1.1 Ambiente de Desenvolvimento

**Configuracao inicial:**

```bash
# Clonar repositorio
git clone https://github.com/rentcars/rentcars-data-platform-science.git
cd rentcars-data-platform-science

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# Instalar dependencias do projeto
cd src/projects/<projeto>
pip install -r requirements.txt
```

**Variaveis de ambiente para desenvolvimento:**

```bash
export MLFLOW_TRACKING_URI="http://mlflow.bi.rentcars.com"
export MLFLOW_EXPERIMENT_NAME="<projeto>-dev"
export ENV="dev"
```

### 1.2 Estrutura de Codigo

**Separar responsabilidades:**

```
src/projects/<projeto>/
├── config.yaml           # Configuracao (NAO codigo)
├── main.py               # Entry point simples
├── steps/                # Logica de cada step
│   ├── extract_features.py
│   ├── train_model.py
│   └── score_model.py
├── utils/                # Funcoes auxiliares
│   └── data_utils.py
├── tests/                # Testes
│   └── test_model.py
├── requirements.txt
└── Dockerfile
```

**Entry point limpo:**

```python
# main.py - deve ser simples
import sys
from steps.extract_features import run as extract_features
from steps.train_model import run as train_model
from steps.score_model import run as score_model

def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step == "extract_features":
        extract_features()
    elif step == "train_model":
        train_model()
    elif step == "score_model":
        score_model()
    elif step == "all":
        extract_features()
        train_model()
        score_model()

if __name__ == "__main__":
    main()
```

### 1.3 Gerenciamento de Dependencias

**requirements.txt - ser especifico:**

```
# BOM - versoes fixas
pandas==2.0.3
scikit-learn==1.3.0
xgboost==1.7.6
mlflow==2.9.2

# RUIM - versoes abertas (pode quebrar)
pandas
scikit-learn
xgboost
mlflow
```

**Separar dependencias de dev:**

```
# requirements.txt - producao
pandas==2.0.3
scikit-learn==1.3.0

# requirements-dev.txt - desenvolvimento
-r requirements.txt
pytest==7.4.0
jupyter==1.0.0
black==23.7.0
```

---

## 2. Controle de Versao (Git)

### 2.1 Fluxo de Branches

```
master (producao)
   ^
   |  PR com aprovacao
   |
dev (homologacao)
   ^
   |  PR com code review
   |
feature/<projeto>-<descricao> (desenvolvimento)
```

**Regras:**

| Branch | Quem pode fazer push | Requer PR | Requer aprovacao |
|--------|---------------------|-----------|------------------|
| feature/* | Todos | Nao | Nao |
| dev | Ninguem direto | Sim | Sim (1 pessoa) |
| master | Ninguem direto | Sim | Sim (2 pessoas) |

### 2.2 Criacao de Branch

```bash
# Sempre partir de dev atualizado
git checkout dev
git pull origin dev

# Criar branch de feature
git checkout -b feature/recsys-add-embeddings

# Trabalhar...
git add .
git commit -m "feat: add user embeddings"
git push origin feature/recsys-add-embeddings
```

### 2.3 Commits

**Formato:**

```
<tipo>: <descricao curta>
```

**Tipos permitidos:**

| Tipo | Quando usar |
|------|-------------|
| feat | Nova funcionalidade |
| fix | Correcao de bug |
| refactor | Refatoracao sem mudar comportamento |
| docs | Documentacao |
| test | Adicao ou correcao de testes |
| chore | Tarefas de manutencao |

**Exemplos:**

```bash
# BOM
git commit -m "feat: add collaborative filtering model"
git commit -m "fix: handle null values in user_age"
git commit -m "refactor: optimize feature extraction query"

# RUIM
git commit -m "updates"
git commit -m "fix bug"
git commit -m "WIP"
```

### 2.4 O que NAO commitar

**Adicionar ao .gitignore:**

```gitignore
# Dados
*.csv
*.parquet
*.pkl
data/

# Credenciais
.env
*.pem
credentials.json

# Arquivos gerados
__pycache__/
*.pyc
.ipynb_checkpoints/
build_date.txt

# Modelos locais
models/
*.joblib

# IDE
.vscode/
.idea/
```

---

## 3. MLFlow

### 3.1 Experimentos

**Sempre usar experiment do projeto:**

```python
import mlflow
import os

# Usar variavel de ambiente (injetada pelo Step Function)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
```

**Nomear runs de forma descritiva:**

```python
# BOM
with mlflow.start_run(run_name="xgboost-tuned-v2"):
    ...

# RUIM
with mlflow.start_run():  # Nome automatico, dificil identificar
    ...
```

### 3.2 Logging de Parametros e Metricas

**Logar todos os hiperparametros:**

```python
# Hiperparametros do modelo
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("max_depth", 6)
mlflow.log_param("n_estimators", 100)
mlflow.log_param("subsample", 0.8)

# Configuracoes do experimento
mlflow.log_param("train_size", 0.8)
mlflow.log_param("feature_version", "v2")
mlflow.log_param("data_date", "2025-01-06")
```

**Logar metricas relevantes:**

```python
# Metricas de classificacao
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1", f1)
mlflow.log_metric("roc_auc", roc_auc)

# Metricas de negocio (quando aplicavel)
mlflow.log_metric("conversion_lift", lift)
mlflow.log_metric("revenue_impact", revenue)
```

### 3.3 Registro de Modelos

**Sempre registrar com nome do projeto:**

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="recsys"  # Nome do projeto
)
```

**Nao registrar modelos de teste:**

```python
# Em desenvolvimento/experimentacao - nao registrar
mlflow.sklearn.log_model(model, artifact_path="model")

# Modelo validado para producao - registrar
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="recsys"
)
```

### 3.4 Promocao de Modelos

**Fluxo de promocao:**

```
None -> Staging -> Production
         |
         v (se reprovado)
      Archived
```

**Quando promover:**

| De | Para | Quando |
|----|------|--------|
| None | Staging | Modelo validado em dev |
| Staging | Production | Aprovado para producao |
| Production | Archived | Substituido por versao melhor |
| Staging | Archived | Reprovado nos testes |

---

## 4. CI/CD

### 4.1 Pipeline de CI/CD

**Fluxo automatico:**

```
Push em dev/master
       |
       v
GitHub Actions dispara
       |
       v
build_apps.sh executa
       |
       +---> Docker build
       |
       +---> Push para ECR
       |
       +---> Cria/Atualiza ECS
       |
       +---> Cria/Atualiza Step Function
       |
       +---> Configura Scheduler
       |
       v
Deploy concluido
```

### 4.2 Ambientes

| Branch | Ambiente | ECR Tag | Step Function |
|--------|----------|---------|---------------|
| dev | Desenvolvimento | projeto-version (dev) | projeto-workflow (dev) |
| master | Producao | projeto-version (prd) | projeto-workflow (prd) |

### 4.3 config.yaml

**Campos obrigatorios:**

```yaml
project_name: recsys          # Obrigatorio
version: "1.0.0"              # Obrigatorio
schedule:
  enabled: true               # Obrigatorio
  cron: "0 3 * * *"          # Obrigatorio se enabled=true
batch:
  pipeline:
    steps:                    # Obrigatorio - pelo menos 1 step
      - id: "step_a"
        command: "python main.py"
```

**Validar antes de commitar:**

```bash
# Verificar YAML valido
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### 4.4 Dockerfile

**Boas praticas:**

```dockerfile
# Usar imagem base especifica
FROM python:3.11-slim

# Definir diretorio de trabalho
WORKDIR /app

# Copiar e instalar dependencias primeiro (cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo por ultimo
COPY . .

# Nao rodar como root
RUN useradd -m appuser
USER appuser

# Comando padrao
CMD ["python", "main.py"]
```

**NAO fazer:**

```dockerfile
# RUIM - imagem muito grande
FROM python:3.11

# RUIM - instalar coisas desnecessarias
RUN apt-get update && apt-get install -y vim curl wget

# RUIM - copiar tudo antes de instalar deps (quebra cache)
COPY . .
RUN pip install -r requirements.txt

# RUIM - rodar como root
# (sem USER)
```

---

## 5. Seguranca

### 5.1 Credenciais

**NUNCA commitar credenciais:**

```python
# ERRADO
aws_key = "AKIAIOSFODNN7EXAMPLE"
password = "minha_senha_123"

# CERTO - usar variaveis de ambiente
aws_key = os.environ["AWS_ACCESS_KEY_ID"]
password = os.environ["DB_PASSWORD"]
```

**Usar SSM Parameter Store:**

```python
import boto3

ssm = boto3.client('ssm')
response = ssm.get_parameter(
    Name='/rcdp/ml/recsys/db_password',
    WithDecryption=True
)
password = response['Parameter']['Value']
```

### 5.2 Dados Sensiveis

**Nao logar dados sensiveis:**

```python
# ERRADO
logger.info(f"Processando usuario {cpf}")
mlflow.log_param("sample_cpf", df['cpf'].iloc[0])

# CERTO
logger.info(f"Processando {len(df)} usuarios")
mlflow.log_param("sample_size", len(df))
```

---

## 6. Testes

### 6.1 Estrutura de Testes

```
tests/
├── test_features.py      # Testes de feature engineering
├── test_model.py         # Testes do modelo
├── test_inference.py     # Testes de inferencia
└── conftest.py           # Fixtures compartilhadas
```

### 6.2 Testes Minimos

**Testar carregamento de dados:**

```python
def test_load_data():
    df = load_data()
    assert df is not None
    assert len(df) > 0
    assert "user_id" in df.columns
```

**Testar feature engineering:**

```python
def test_extract_features():
    df_raw = load_sample_data()
    df_features = extract_features(df_raw)

    assert "feature_1" in df_features.columns
    assert df_features["feature_1"].isna().sum() == 0
```

**Testar modelo:**

```python
def test_model_predict():
    model = load_model()
    sample = get_sample_input()

    predictions = model.predict(sample)

    assert len(predictions) == len(sample)
    assert all(0 <= p <= 1 for p in predictions)
```

### 6.3 Rodar Testes

```bash
# Rodar todos os testes
pytest tests/

# Rodar com coverage
pytest tests/ --cov=src --cov-report=html

# Rodar teste especifico
pytest tests/test_model.py::test_model_predict
```

---

## 7. Monitoramento

### 7.1 Logging

**Usar logging, nao print:**

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BOM
logger.info("Iniciando processamento")
logger.warning("Dados faltantes: 5%")
logger.error("Falha ao carregar modelo")

# RUIM
print("Iniciando processamento")
```

**Niveis de log:**

| Nivel | Quando usar |
|-------|-------------|
| DEBUG | Informacoes detalhadas para debug |
| INFO | Confirmacao de que esta funcionando |
| WARNING | Algo inesperado, mas continua funcionando |
| ERROR | Erro que impede funcionalidade |
| CRITICAL | Erro grave, aplicacao pode parar |

### 7.2 Metricas de Execucao

**Logar metricas de execucao no MLFlow:**

```python
import time

start_time = time.time()

# ... executa pipeline ...

duration = time.time() - start_time

mlflow.log_metric("execution_time_seconds", duration)
mlflow.log_metric("records_processed", len(df))
mlflow.log_metric("predictions_generated", len(predictions))
```

---

## 8. Code Review

### 8.1 Checklist para Autor do PR

Antes de abrir PR, verificar:

- [ ] Codigo segue padroes de nomenclatura
- [ ] Sem credenciais ou dados sensiveis
- [ ] requirements.txt atualizado
- [ ] config.yaml valido
- [ ] Testes passando localmente
- [ ] Logging adequado
- [ ] Documentacao atualizada (se necessario)

### 8.2 Checklist para Revisor

Ao revisar PR, verificar:

- [ ] Codigo legivel e bem estruturado
- [ ] Logica correta
- [ ] Tratamento de erros adequado
- [ ] Sem codigo duplicado
- [ ] Sem dependencias desnecessarias
- [ ] Metricas sendo logadas no MLFlow
- [ ] Testes cobrindo casos importantes

### 8.3 Quando Reprovar PR

Reprovar se:

- Credenciais expostas no codigo
- Testes falhando
- config.yaml invalido
- Codigo sem tratamento de erros
- Mudancas nao relacionadas ao objetivo do PR

---

## 9. Troubleshooting

### 9.1 Build Falhou

**Verificar:**

1. Dockerfile esta correto?
2. requirements.txt tem todas as dependencias?
3. config.yaml esta valido?

```bash
# Testar build local
docker build -t teste .
docker run teste python main.py
```

### 9.2 Step Function Falhou

**Verificar:**

1. Logs no CloudWatch: `/ecs/<projeto>`
2. Modelo existe no MLFlow?
3. Variaveis de ambiente estao corretas?

```bash
# Ver logs
aws logs tail /ecs/recsys --follow
```

### 9.3 Modelo Nao Carrega

**Verificar:**

1. Modelo existe no MLFlow Registry?
2. Stage esta correto (Production)?
3. MLFLOW_TRACKING_URI esta configurado?

```python
# Debug
import mlflow
mlflow.set_tracking_uri("http://mlflow.bi.rentcars.com")

client = mlflow.MlflowClient()
versions = client.search_model_versions("name='recsys'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")
```

---

## 10. Resumo de Comandos

### Git

```bash
# Criar branch
git checkout dev && git pull && git checkout -b feature/projeto-descricao

# Commitar
git add . && git commit -m "feat: descricao"

# Push
git push origin feature/projeto-descricao
```

### Docker

```bash
# Build local
docker build -t projeto .

# Rodar local
docker run -e MLFLOW_TRACKING_URI=http://mlflow.bi.rentcars.com projeto
```

### MLFlow

```bash
# Promover modelo
mlflow models transition-stage --name projeto --version 1 --stage Production
```

### AWS

```bash
# Ver logs
aws logs tail /ecs/projeto --follow

# Executar Step Function manualmente
aws stepfunctions start-execution --state-machine-arn arn:aws:states:us-east-1:ACCOUNT:stateMachine:projeto-workflow
```

### Testes

```bash
# Rodar testes
pytest tests/ -v

# Com coverage
pytest tests/ --cov=src
```
