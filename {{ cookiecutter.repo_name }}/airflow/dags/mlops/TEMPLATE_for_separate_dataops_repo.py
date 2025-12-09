"""
DAG de Integração MLOps - Para Repo DataOps Separado

IMPORTANTE: Este arquivo deve ser copiado para o repositório dataops-dbt,
            NÃO para o repositório mlops-projeto1.

Localização correta: dataops-dbt/airflow/dags/mlops_<projeto>_integration.py

Esta DAG integra pipeline DBT (DataOps) com pipeline ML (MLOps) quando os
repositórios são separados.

Arquitetura:
  Repo DataOps (este arquivo)
    ↓ SSHOperator
  Repo MLOps (EC2)
    └── Código Python executado remotamente

Author: {{ cookiecutter.author_name }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÕES - AJUSTAR CONFORME SEU AMBIENTE
# =============================================================================

DEFAULT_ARGS = {
    'owner': '{{ cookiecutter.author_name }}',
    'depends_on_past': False,
    'email': ['{{ cookiecutter.email }}'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Projeto MLOps
PROJECT_NAME = "{{ cookiecutter.project_name }}"  # Ex: "churn", "credit_risk"

# S3 Paths (ajustar conforme seu ambiente)
S3_DATAMART_BUCKET = "processed-data-bucket"  # Bucket onde DBT salva datamarts
S3_DATAMART_KEY = f"datamarts/{PROJECT_NAME}/latest.parquet"
S3_ML_BUCKET = "ml-artifacts-bucket"  # Bucket para features/modelos ML

# EC2 MLOps
EC2_SSH_CONN_ID = f"ec2_mlops_{PROJECT_NAME}"  # Configurar no Airflow UI
EC2_MLOPS_PATH = "/opt/mlops-projeto"  # Caminho do repo MLOps na EC2

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_datamart_ready(**context):
    """
    Validação adicional de qualidade do datamart.
    Pode incluir checks adicionais além do S3 Sensor.
    """
    from src.data.aws_integration import get_s3_client
    import pandas as pd

    try:
        s3_client = get_s3_client()
        df = s3_client.read_parquet(
            s3_key=S3_DATAMART_KEY,
            bucket=S3_DATAMART_BUCKET
        )

        # Validações básicas
        assert len(df) > 1000, f"Insufficient data: {len(df)} rows"
        assert 'target' in df.columns, "Missing target column"

        logger.info(f"✓ Datamart validated: {len(df)} rows, {len(df.columns)} columns")

        # Push para XCom
        context['task_instance'].xcom_push(key='datamart_rows', value=len(df))

        return True

    except Exception as e:
        logger.error(f"Datamart validation failed: {e}")
        raise


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    f'mlops_{PROJECT_NAME}_integration',
    default_args=DEFAULT_ARGS,
    description=f'Integração DataOps → MLOps para projeto {PROJECT_NAME}',
    schedule_interval='0 3 * * *',  # Diariamente às 3 AM (1h após DBT)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'integration', PROJECT_NAME],
    doc_md=f"""
    # MLOps Integration DAG - {PROJECT_NAME}

    Esta DAG integra o pipeline DBT (DataOps) com o pipeline ML (MLOps).

    ## Fluxo
    1. Aguardar datamart DBT estar disponível no S3
    2. Validar qualidade do datamart
    3. Executar feature engineering na EC2 de MLOps
    4. Treinar modelo na EC2 de MLOps
    5. Validar e promover modelo (se passou nos testes)

    ## Repositórios
    - **Repo DataOps** (este arquivo): dataops-dbt/
    - **Repo MLOps** (código executado): mlops-{PROJECT_NAME}/ (EC2)

    ## Configuração
    - SSH Connection: `{EC2_SSH_CONN_ID}` (Airflow UI → Admin → Connections)
    - EC2 Path: `{EC2_MLOPS_PATH}`
    - Datamart S3: `s3://{S3_DATAMART_BUCKET}/{S3_DATAMART_KEY}`
    """,
) as dag:

    # =========================================================================
    # STAGE 1: Aguardar Datamart DBT
    # =========================================================================

    wait_for_datamart = S3KeySensor(
        task_id='wait_for_dbt_datamart',
        bucket_name=S3_DATAMART_BUCKET,
        bucket_key=S3_DATAMART_KEY,
        aws_conn_id='aws_default',
        timeout=3600,  # 1 hora
        poke_interval=300,  # Checar a cada 5 min
        mode='poke',
        doc_md=f"""
        Aguarda datamart gerado pelo pipeline DBT.

        O pipeline DBT deve salvar o resultado em:
        s3://{S3_DATAMART_BUCKET}/{S3_DATAMART_KEY}
        """
    )

    # =========================================================================
    # STAGE 2: Validação de Qualidade
    # =========================================================================

    validate_datamart = PythonOperator(
        task_id='validate_datamart_quality',
        python_callable=validate_datamart_ready,
        provide_context=True,
        doc_md="""
        Valida qualidade do datamart:
        - Mínimo 1000 rows
        - Coluna 'target' presente
        - Schema correto
        """
    )

    # =========================================================================
    # STAGE 3: Feature Engineering (via SSH)
    # =========================================================================

    create_features = SSHOperator(
        task_id='create_ml_features',
        ssh_conn_id=EC2_SSH_CONN_ID,
        command=f'''
            cd {EC2_MLOPS_PATH} && \
            source venv/bin/activate && \
            python -m src.pipelines.DS.feature_pipeline.transform \
                --input s3://{S3_DATAMART_BUCKET}/{S3_DATAMART_KEY} \
                --output s3://{S3_ML_BUCKET}/features/{PROJECT_NAME}/latest.parquet \
                --log-level INFO
        ''',
        cmd_timeout=1800,  # 30 minutos
        doc_md=f"""
        Executa feature engineering na EC2 de MLOps.

        Comando executado remotamente via SSH:
        ```bash
        cd {EC2_MLOPS_PATH}
        python -m src.pipelines.DS.feature_pipeline.transform
        ```

        Input: Datamart DBT (S3)
        Output: Features ML (S3)
        """
    )

    # =========================================================================
    # STAGE 4: Model Training (via SSH)
    # =========================================================================

    train_model = SSHOperator(
        task_id='train_model',
        ssh_conn_id=EC2_SSH_CONN_ID,
        command=f'''
            cd {EC2_MLOPS_PATH} && \
            source venv/bin/activate && \
            python -m src.pipelines.DS.training_pipeline.train \
                --features-path s3://{S3_ML_BUCKET}/features/{PROJECT_NAME}/latest.parquet \
                --config config/prod/model_config.yaml \
                --experiment-name {PROJECT_NAME}_automated_training \
                --register-model \
                --log-level INFO
        ''',
        cmd_timeout=7200,  # 2 horas
        doc_md=f"""
        Executa treinamento do modelo na EC2 de MLOps.

        Comando executado remotamente via SSH:
        ```bash
        cd {EC2_MLOPS_PATH}
        python -m src.pipelines.DS.training_pipeline.train
        ```

        - Treina modelo com features geradas
        - Registra no MLFlow Model Registry
        - Logs salvos em: {EC2_MLOPS_PATH}/logs/training/
        """
    )

    # =========================================================================
    # STAGE 5: Model Validation (via SSH)
    # =========================================================================

    validate_model = SSHOperator(
        task_id='validate_model',
        ssh_conn_id=EC2_SSH_CONN_ID,
        command=f'''
            cd {EC2_MLOPS_PATH} && \
            source venv/bin/activate && \
            python -m tests.responsible_ai.test_fairness && \
            python -m tests.responsible_ai.test_robustness
        ''',
        cmd_timeout=1800,  # 30 minutos
        doc_md="""
        Valida modelo novo:
        - Fairness tests (bias detection)
        - Robustness tests (adversarial)

        Se falhar, modelo NÃO é promovido para produção.
        """
    )

    # =========================================================================
    # STAGE 6: Model Promotion (via SSH)
    # =========================================================================

    promote_model = SSHOperator(
        task_id='promote_model_to_production',
        ssh_conn_id=EC2_SSH_CONN_ID,
        command=f'''
            cd {EC2_MLOPS_PATH} && \
            source venv/bin/activate && \
            python -m src.deployment.promote_model \
                --model-name {PROJECT_NAME}_model \
                --stage Production \
                --auto-archive-previous
        ''',
        cmd_timeout=300,  # 5 minutos
        doc_md="""
        Promove modelo validado para produção:
        - Transiciona para stage 'Production' no MLFlow
        - Arquiva modelo anterior
        - Notifica equipe (Slack/email)
        """
    )

    # =========================================================================
    # FLOW
    # =========================================================================

    wait_for_datamart >> validate_datamart >> create_features >> train_model >> validate_model >> promote_model


# =============================================================================
# INSTRUÇÕES DE SETUP
# =============================================================================

"""
## Setup no Repo DataOps

1. **Copiar este arquivo para repo dataops-dbt**:
   ```bash
   cp mlops_integration_template.py \\
      /path/to/dataops-dbt/airflow/dags/mlops_churn_integration.py
   ```

2. **Ajustar configurações** (seção CONFIGURAÇÕES acima):
   - PROJECT_NAME
   - S3_DATAMART_BUCKET
   - S3_ML_BUCKET
   - EC2_MLOPS_PATH

3. **Configurar SSH Connection no Airflow**:
   - Airflow UI → Admin → Connections → Add
   - Connection ID: `ec2_mlops_churn` (igual EC2_SSH_CONN_ID)
   - Connection Type: SSH
   - Host: <ec2-public-ip>
   - Username: ec2-user
   - Private Key: <conteúdo do arquivo .pem>
   - Port: 22

4. **Testar SSH**:
   ```bash
   # Testar manualmente
   ssh -i key.pem ec2-user@<ec2-ip>
   cd /opt/mlops-projeto
   source venv/bin/activate
   python -m src.pipelines.DS.training_pipeline.train --help
   ```

5. **Verificar Security Group da EC2**:
   - Permitir SSH (porta 22) do IP do Airflow
   - AWS Console → EC2 → Security Groups → Inbound Rules
   - Add: Type SSH, Port 22, Source <airflow-ip>/32

6. **Commit no repo DataOps**:
   ```bash
   cd dataops-dbt
   git add airflow/dags/mlops_churn_integration.py
   git commit -m "Add MLOps integration DAG for churn model"
   git push
   ```

7. **Airflow reload DAGs**:
   - Aguardar ~1 minuto (Airflow DAG refresh)
   - Verificar em: Airflow UI → DAGs → mlops_churn_integration

8. **Testar execução manual**:
   - Airflow UI → DAG mlops_churn_integration → Trigger DAG
   - Monitorar logs de cada task
   - Verificar em Graph View o fluxo completo
"""
