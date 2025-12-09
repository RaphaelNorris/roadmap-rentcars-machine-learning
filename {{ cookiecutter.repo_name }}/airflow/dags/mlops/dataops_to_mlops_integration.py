"""
DAG de Integração: DataOps → MLOps

Integra pipeline de engenharia de dados (DBT + Athena) com pipeline de ML.

Fluxo:
1. DBT transforma dados brutos → datamarts (silver/gold)
2. Sensor aguarda datamart pronto no S3
3. Feature engineering para ML
4. Trigger ML training pipeline
5. Model validation e deployment

Author: {{ cookiecutter.author_name }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.task_group import TaskGroup
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÕES
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

# Configuração de dados
S3_BUCKET = "{{ cookiecutter.s3_processed_bucket }}"
S3_ARTIFACTS_BUCKET = "{{ cookiecutter.s3_ml_artifacts_bucket }}"

# Caminhos S3
DATAMART_PATH = "datamarts/{{ cookiecutter.project_name }}/latest.parquet"
FEATURES_PATH = "ml/features/{{ cookiecutter.project_name }}/latest.parquet"

# Conexões
EC2_SSH_CONN_ID = "ec2_ml_{{ cookiecutter.project_name }}"  # Configurar no Airflow
EC2_API_CONN_ID = "ec2_ml_api_{{ cookiecutter.project_name }}"

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def validate_datamart_quality(**context):
    """
    Valida qualidade do datamart gerado pelo DBT.

    Checks:
    - Schema correto
    - Sem valores nulos em colunas críticas
    - Tamanho mínimo de dados
    - Sem duplicatas
    """
    logger.info("Validating datamart quality...")

    from src.data.aws_integration import get_s3_client
    import pandas as pd

    try:
        # Carregar datamart
        s3_client = get_s3_client()
        df = s3_client.read_parquet(s3_key=DATAMART_PATH)

        logger.info(f"Datamart loaded: {len(df)} rows, {len(df.columns)} columns")

        # Validação 1: Tamanho mínimo
        MIN_ROWS = 1000
        if len(df) < MIN_ROWS:
            raise ValueError(f"Insufficient data: {len(df)} rows < {MIN_ROWS}")

        # Validação 2: Colunas obrigatórias
        REQUIRED_COLS = ['customer_id', 'date', 'target']  # Ajustar conforme necessidade
        missing_cols = set(REQUIRED_COLS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validação 3: Nulos em colunas críticas
        null_counts = df[REQUIRED_COLS].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")

        # Validação 4: Duplicatas
        duplicates = df.duplicated(subset=['customer_id', 'date']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
            df = df.drop_duplicates(subset=['customer_id', 'date'], keep='last')

        # Push stats para XCom
        context['task_instance'].xcom_push(key='datamart_rows', value=len(df))
        context['task_instance'].xcom_push(key='datamart_cols', value=len(df.columns))

        logger.info("✓ Datamart validation passed")

        return {
            'rows': len(df),
            'columns': len(df.columns),
            'duplicates_removed': duplicates,
            'status': 'valid'
        }

    except Exception as e:
        logger.error(f"Datamart validation failed: {e}")
        raise


def create_ml_features(**context):
    """
    Cria features para ML a partir do datamart.

    Transformações:
    - Feature engineering específica para ML
    - Normalização/scaling
    - Encoding de categóricas
    - Salvar em formato otimizado
    """
    logger.info("Creating ML features...")

    from src.data.aws_integration import get_s3_client
    from src.pipelines.DS.feature_pipeline import FeaturePipeline

    try:
        # Carregar datamart
        s3_client = get_s3_client()
        df = s3_client.read_parquet(s3_key=DATAMART_PATH)

        logger.info(f"Processing {len(df)} rows for feature engineering")

        # Feature engineering
        feature_pipeline = FeaturePipeline()
        features_df = feature_pipeline.transform(df)

        logger.info(f"Features created: {len(features_df.columns)} columns")

        # Salvar features no S3
        s3_client.write_parquet(features_df, s3_key=FEATURES_PATH)

        logger.info(f"Features saved to s3://{S3_BUCKET}/{FEATURES_PATH}")

        # Push info para XCom
        context['task_instance'].xcom_push(key='features_path', value=f"s3://{S3_BUCKET}/{FEATURES_PATH}")
        context['task_instance'].xcom_push(key='n_features', value=len(features_df.columns))

        return {
            'features_count': len(features_df.columns),
            's3_path': f"s3://{S3_BUCKET}/{FEATURES_PATH}",
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


def prepare_training_config(**context):
    """
    Prepara configuração para treinamento do modelo.
    """
    ti = context['task_instance']

    # Obter informações das tasks anteriores
    features_path = ti.xcom_pull(task_ids='feature_engineering.create_features', key='features_path')
    n_features = ti.xcom_pull(task_ids='feature_engineering.create_features', key='n_features')
    datamart_rows = ti.xcom_pull(task_ids='datamart_validation.validate_quality', key='datamart_rows')

    training_config = {
        'features_path': features_path,
        'n_features': n_features,
        'n_samples': datamart_rows,
        'model_type': 'random_forest',
        'experiment_name': f"{{ cookiecutter.project_name }}_automated_training",
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'random_state': 42,
        },
        'register_model': True,
        'auto_promote': False,  # Requer validação manual
    }

    logger.info(f"Training config prepared: {training_config}")

    # Push config
    ti.xcom_push(key='training_config', value=training_config)

    return training_config


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    'dataops_to_mlops_integration',
    default_args=DEFAULT_ARGS,
    description='Integração DataOps (DBT) → MLOps (Training)',
    schedule_interval='0 3 * * *',  # Diariamente às 3 AM (após DBT)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['integration', 'dataops', 'mlops'],
) as dag:

    # =========================================================================
    # STAGE 1: Aguardar DataOps (DBT)
    # =========================================================================

    wait_for_datamart = S3KeySensor(
        task_id='wait_for_datamart',
        bucket_name=S3_BUCKET,
        bucket_key=DATAMART_PATH,
        aws_conn_id='aws_default',
        timeout=3600,  # 1 hora
        poke_interval=300,  # Checar a cada 5 min
        mode='poke',
        doc_md="""
        Aguarda datamart gerado pelo pipeline DBT estar disponível no S3.

        O pipeline DataOps deve:
        1. Executar transformações DBT
        2. Salvar resultado em S3 como Parquet
        3. Atualizar arquivo 'latest.parquet'
        """
    )

    # =========================================================================
    # STAGE 2: Validação de Qualidade
    # =========================================================================

    with TaskGroup('datamart_validation', tooltip='Validate datamart quality') as validation_group:

        validate_quality = PythonOperator(
            task_id='validate_quality',
            python_callable=validate_datamart_quality,
            provide_context=True,
        )

    # =========================================================================
    # STAGE 3: Feature Engineering
    # =========================================================================

    with TaskGroup('feature_engineering', tooltip='Create ML features') as feature_group:

        create_features = PythonOperator(
            task_id='create_features',
            python_callable=create_ml_features,
            provide_context=True,
        )

    # =========================================================================
    # STAGE 4: Preparar Treinamento
    # =========================================================================

    prep_training = PythonOperator(
        task_id='prepare_training_config',
        python_callable=prepare_training_config,
        provide_context=True,
    )

    # =========================================================================
    # STAGE 5: Trigger ML Training
    # =========================================================================

    # OPÇÃO 1: Trigger DAG do Airflow (se ML training também for DAG)
    trigger_ml_dag = TriggerDagRunOperator(
        task_id='trigger_ml_training_dag',
        trigger_dag_id='ml_training_pipeline',
        conf={
            'features_path': "{{ task_instance.xcom_pull(task_ids='prepare_training_config', key='training_config') }}",
            'triggered_by': 'dataops_integration',
        },
        wait_for_completion=True,
        poke_interval=60,
    )

    # OPÇÃO 2: Executar via SSH na EC2 de ML (se Airflow centralizado)
    # Descomentar se usar Airflow centralizado
    """
    train_via_ssh = SSHOperator(
        task_id='train_model_via_ssh',
        ssh_conn_id=EC2_SSH_CONN_ID,
        command='''
            cd /opt/mlops && \
            source venv/bin/activate && \
            python -m src.pipelines.DS.training_pipeline.train \
                --features-path {{ task_instance.xcom_pull(task_ids='prepare_training_config', key='training_config')['features_path'] }} \
                --config config/prod/model_config.yaml
        ''',
        cmd_timeout=7200,  # 2 horas
    )
    """

    # OPÇÃO 3: Trigger via API HTTP (se EC2 tiver API exposta)
    # Descomentar se usar API REST
    """
    train_via_api = SimpleHttpOperator(
        task_id='trigger_training_via_api',
        http_conn_id=EC2_API_CONN_ID,
        endpoint='/api/v1/training/trigger',
        method='POST',
        data=json.dumps({
            'features_path': "{{ task_instance.xcom_pull(task_ids='prepare_training_config', key='training_config')['features_path'] }}",
            'config': "{{ task_instance.xcom_pull(task_ids='prepare_training_config', key='training_config') }}"
        }),
        headers={'Content-Type': 'application/json'},
        response_check=lambda response: response.status_code == 200,
    )
    """

    # =========================================================================
    # FLOW
    # =========================================================================

    wait_for_datamart >> validation_group >> feature_group >> prep_training >> trigger_ml_dag

    # Se usar SSH ou API, trocar última linha por:
    # wait_for_datamart >> validation_group >> feature_group >> prep_training >> train_via_ssh
    # wait_for_datamart >> validation_group >> feature_group >> prep_training >> train_via_api
