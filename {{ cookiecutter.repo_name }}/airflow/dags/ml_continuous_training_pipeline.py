"""
Continuous Training Pipeline - CD4ML

DAG para retreinamento contínuo do modelo baseado em:
- Drift detection (data drift, concept drift)
- Performance degradation
- Scheduled retraining
- New data availability

Implementa princípios de Continuous Delivery for Machine Learning (CD4ML):
- Automated retraining triggers
- Automated validation
- Automated promotion
- Rollback capability

Author: {{ cookiecutter.author_name }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.task_group import TaskGroup
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Configuração de logging
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

# Thresholds para retraining triggers
DRIFT_THRESHOLD = 0.05  # 5% drift score
PERFORMANCE_DEGRADATION_THRESHOLD = 0.05  # 5% accuracy drop
MIN_IMPROVEMENT_THRESHOLD = 0.01  # Novo modelo deve ser pelo menos 1% melhor
MIN_NEW_DATA_PCT = 10  # Requer pelo menos 10% de dados novos

# Configurações de dados
S3_BUCKET = "{{ cookiecutter.s3_processed_bucket }}"
REFERENCE_DATA_PATH = "ml/reference_data/baseline.parquet"
CURRENT_DATA_PATH = "ml/features/current_features.parquet"
MODEL_REGISTRY_NAME = "{{ cookiecutter.project_name }}_model"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data_from_s3(s3_path: str) -> pd.DataFrame:
    """Carrega dados do S3."""
    from src.data.aws_integration import get_s3_client

    s3_client = get_s3_client()
    df = s3_client.read_parquet(s3_key=s3_path)

    logger.info(f"Loaded {len(df)} rows from {s3_path}")
    return df


def save_data_to_s3(df: pd.DataFrame, s3_path: str):
    """Salva dados no S3."""
    from src.data.aws_integration import get_s3_client

    s3_client = get_s3_client()
    s3_client.write_parquet(df, s3_key=s3_path)

    logger.info(f"Saved {len(df)} rows to {s3_path}")


def get_model_performance_metrics(model_version: str) -> Dict:
    """Obtém métricas do modelo do MLFlow."""
    from src.model.mlflow_manager import MLFlowManager

    mlflow_manager = MLFlowManager()

    # Buscar métricas no MLFlow
    client = mlflow_manager.client
    model_version_info = client.get_model_version(
        name=MODEL_REGISTRY_NAME,
        version=model_version
    )

    run_id = model_version_info.run_id
    run = client.get_run(run_id)

    metrics = {
        'accuracy': run.data.metrics.get('accuracy', 0),
        'precision': run.data.metrics.get('precision', 0),
        'recall': run.data.metrics.get('recall', 0),
        'f1': run.data.metrics.get('f1', 0),
        'roc_auc': run.data.metrics.get('roc_auc', 0),
    }

    return metrics


# =============================================================================
# TASK FUNCTIONS - DATA MONITORING
# =============================================================================

def check_new_data_availability(**context):
    """
    Verifica se há dados novos suficientes para retreinamento.

    Returns:
        bool: True se há dados novos suficientes
    """
    logger.info("Checking for new data availability...")

    try:
        # Carregar dados de referência
        reference_df = load_data_from_s3(REFERENCE_DATA_PATH)
        reference_count = len(reference_df)
        reference_date = reference_df['date'].max() if 'date' in reference_df.columns else None

        # Carregar dados atuais
        current_df = load_data_from_s3(CURRENT_DATA_PATH)
        current_count = len(current_df)
        current_date = current_df['date'].max() if 'date' in current_df.columns else None

        # Calcular % de dados novos
        new_data_pct = ((current_count - reference_count) / reference_count * 100) if reference_count > 0 else 100

        logger.info(f"Reference data: {reference_count} rows (up to {reference_date})")
        logger.info(f"Current data: {current_count} rows (up to {current_date})")
        logger.info(f"New data: {new_data_pct:.1f}%")

        # Push para XCom
        context['task_instance'].xcom_push(key='new_data_pct', value=new_data_pct)
        context['task_instance'].xcom_push(key='reference_count', value=reference_count)
        context['task_instance'].xcom_push(key='current_count', value=current_count)

        has_new_data = new_data_pct >= MIN_NEW_DATA_PCT

        logger.info(f"New data check: {'PASSED' if has_new_data else 'FAILED'} (threshold: {MIN_NEW_DATA_PCT}%)")

        return has_new_data

    except Exception as e:
        logger.error(f"Error checking new data: {e}")
        return False


def detect_data_drift(**context):
    """
    Detecta data drift usando Evidently.

    Returns:
        Dict com resultados de drift detection
    """
    logger.info("Detecting data drift...")

    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    try:
        # Carregar dados
        reference_df = load_data_from_s3(REFERENCE_DATA_PATH)
        current_df = load_data_from_s3(CURRENT_DATA_PATH)

        # Selecionar colunas numéricas para drift detection
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remover target se existir
        if 'target' in numeric_cols:
            numeric_cols.remove('target')

        # Criar relatório de drift
        drift_report = Report(metrics=[
            DataDriftPreset()
        ])

        drift_report.run(
            reference_data=reference_df[numeric_cols].sample(min(1000, len(reference_df))),
            current_data=current_df[numeric_cols].sample(min(1000, len(current_df)))
        )

        # Extrair métricas
        report_dict = drift_report.as_dict()

        # Calcular drift score geral
        drift_score = report_dict['metrics'][0]['result']['dataset_drift']
        n_drifted_features = report_dict['metrics'][0]['result']['number_of_drifted_columns']
        total_features = report_dict['metrics'][0]['result']['number_of_columns']

        drift_pct = (n_drifted_features / total_features * 100) if total_features > 0 else 0

        logger.info(f"Data drift score: {drift_score}")
        logger.info(f"Drifted features: {n_drifted_features}/{total_features} ({drift_pct:.1f}%)")

        # Push para XCom
        context['task_instance'].xcom_push(key='drift_score', value=float(drift_score))
        context['task_instance'].xcom_push(key='n_drifted_features', value=n_drifted_features)
        context['task_instance'].xcom_push(key='drift_pct', value=drift_pct)

        # Salvar relatório
        drift_report_path = f"ml/monitoring/drift_reports/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        drift_report.save_html(f"/tmp/drift_report.html")

        # Upload para S3
        s3_hook = S3Hook(aws_conn_id='aws_default')
        s3_hook.load_file(
            filename="/tmp/drift_report.html",
            key=drift_report_path,
            bucket_name=S3_BUCKET,
            replace=True
        )

        logger.info(f"Drift report saved to s3://{S3_BUCKET}/{drift_report_path}")

        has_drift = drift_score > DRIFT_THRESHOLD or drift_pct > 20

        return {
            'has_drift': has_drift,
            'drift_score': float(drift_score),
            'n_drifted_features': n_drifted_features,
            'drift_pct': drift_pct
        }

    except Exception as e:
        logger.error(f"Error detecting drift: {e}")
        return {'has_drift': False, 'error': str(e)}


def check_model_performance(**context):
    """
    Verifica se performance do modelo atual degradou.

    Compara métricas atuais com baseline.

    Returns:
        bool: True se performance degradou
    """
    logger.info("Checking model performance...")

    from src.model.mlflow_manager import MLFlowManager

    try:
        mlflow_manager = MLFlowManager()

        # Obter modelo em produção
        production_model = mlflow_manager.get_model_version(
            name=MODEL_REGISTRY_NAME,
            stage="Production"
        )

        if not production_model:
            logger.warning("No production model found")
            return True  # Trigger retraining se não há modelo

        # Obter métricas do modelo atual
        current_metrics = get_model_performance_metrics(production_model.version)

        # Obter baseline metrics (do run que criou o modelo)
        baseline_accuracy = current_metrics.get('accuracy', 0)

        # Avaliar modelo em dados recentes
        current_df = load_data_from_s3(CURRENT_DATA_PATH)

        # Carregar modelo
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/Production"
        model = mlflow_manager.load_model(model_uri)

        # Separar features e target
        feature_cols = [col for col in current_df.columns if col not in ['target', 'date', 'id']]
        X_current = current_df[feature_cols]
        y_current = current_df['target'] if 'target' in current_df.columns else None

        if y_current is None:
            logger.warning("No target column found in current data")
            return False

        # Fazer predições
        y_pred = model.predict(X_current)

        # Calcular accuracy atual
        from sklearn.metrics import accuracy_score
        current_accuracy = accuracy_score(y_current, y_pred)

        # Calcular degradação
        accuracy_drop = baseline_accuracy - current_accuracy
        degradation_pct = (accuracy_drop / baseline_accuracy * 100) if baseline_accuracy > 0 else 0

        logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        logger.info(f"Current accuracy: {current_accuracy:.4f}")
        logger.info(f"Degradation: {degradation_pct:.2f}%")

        # Push para XCom
        context['task_instance'].xcom_push(key='baseline_accuracy', value=baseline_accuracy)
        context['task_instance'].xcom_push(key='current_accuracy', value=current_accuracy)
        context['task_instance'].xcom_push(key='degradation_pct', value=degradation_pct)

        has_degraded = accuracy_drop > PERFORMANCE_DEGRADATION_THRESHOLD

        logger.info(f"Performance check: {'DEGRADED' if has_degraded else 'OK'}")

        return has_degraded

    except Exception as e:
        logger.error(f"Error checking model performance: {e}")
        return False


# =============================================================================
# TASK FUNCTIONS - RETRAINING DECISION
# =============================================================================

def decide_retraining(**context):
    """
    Decide se deve retreinar o modelo baseado em múltiplos fatores.

    Triggers de retraining:
    1. Data drift detectado
    2. Performance degradation
    3. Dados novos suficientes
    4. Scheduled retraining (semanal/mensal)

    Returns:
        str: 'retrain' ou 'skip_retrain'
    """
    ti = context['task_instance']

    # Obter resultados de tasks anteriores
    new_data_pct = ti.xcom_pull(task_ids='check_new_data', key='new_data_pct') or 0
    drift_score = ti.xcom_pull(task_ids='detect_drift', key='drift_score') or 0
    degradation_pct = ti.xcom_pull(task_ids='check_performance', key='degradation_pct') or 0

    logger.info("\n" + "="*80)
    logger.info("RETRAINING DECISION")
    logger.info("="*80)
    logger.info(f"New data: {new_data_pct:.1f}% (threshold: {MIN_NEW_DATA_PCT}%)")
    logger.info(f"Drift score: {drift_score:.3f} (threshold: {DRIFT_THRESHOLD})")
    logger.info(f"Performance degradation: {degradation_pct:.2f}% (threshold: {PERFORMANCE_DEGRADATION_THRESHOLD*100}%)")

    # Verificar triggers
    reasons = []

    if new_data_pct >= MIN_NEW_DATA_PCT:
        reasons.append(f"New data: {new_data_pct:.1f}%")

    if drift_score > DRIFT_THRESHOLD:
        reasons.append(f"Data drift: {drift_score:.3f}")

    if degradation_pct > PERFORMANCE_DEGRADATION_THRESHOLD * 100:
        reasons.append(f"Performance degradation: {degradation_pct:.2f}%")

    # Scheduled retraining (forçar se passou 30 dias)
    execution_date = context['execution_date']
    last_retrain_date = ti.xcom_pull(task_ids='decide_retraining', key='last_retrain_date')

    if last_retrain_date:
        days_since_retrain = (execution_date - pd.to_datetime(last_retrain_date)).days
        if days_since_retrain >= 30:
            reasons.append(f"Scheduled: {days_since_retrain} days since last retrain")

    # Decisão
    should_retrain = len(reasons) > 0

    if should_retrain:
        logger.info(f"\n✓ RETRAINING TRIGGERED")
        logger.info(f"Reasons: {', '.join(reasons)}")
        decision = 'retrain'

        # Salvar data de retreinamento
        ti.xcom_push(key='last_retrain_date', value=execution_date.isoformat())
    else:
        logger.info(f"\n✗ RETRAINING SKIPPED")
        logger.info("No triggers activated")
        decision = 'skip_retrain'

    logger.info("="*80 + "\n")

    return decision


# =============================================================================
# TASK FUNCTIONS - MODEL TRAINING
# =============================================================================

def retrain_model(**context):
    """
    Retreina o modelo com dados atualizados.

    Returns:
        Dict com informações do novo modelo
    """
    logger.info("Starting model retraining...")

    from src.pipelines.DS.training_pipeline import TrainingPipeline
    from src.model.mlflow_manager import MLFlowManager

    try:
        # Carregar dados
        current_df = load_data_from_s3(CURRENT_DATA_PATH)

        logger.info(f"Training with {len(current_df)} samples")

        # Inicializar pipeline
        pipeline = TrainingPipeline(
            experiment_name="continuous_training",
            model_name=MODEL_REGISTRY_NAME
        )

        # Treinar modelo
        results = pipeline.run(
            data=current_df,
            model_type="random_forest",  # ou obter de config
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            register_model=True
        )

        # Push resultados para XCom
        context['task_instance'].xcom_push(key='new_model_version', value=results['model_version'])
        context['task_instance'].xcom_push(key='new_model_metrics', value=results['metrics'])
        context['task_instance'].xcom_push(key='new_model_run_id', value=results['run_id'])

        logger.info(f"Model retrained successfully: version {results['model_version']}")
        logger.info(f"Metrics: {results['metrics']}")

        return results

    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise


def validate_new_model(**context):
    """
    Valida novo modelo antes de promover.

    Validações:
    1. Métricas melhores que modelo atual
    2. Testes de robustez
    3. Testes de fairness

    Returns:
        str: 'promote' ou 'rollback'
    """
    logger.info("Validating new model...")

    ti = context['task_instance']

    try:
        # Obter métricas do novo modelo
        new_metrics = ti.xcom_pull(task_ids='retrain_model', key='new_model_metrics')
        new_version = ti.xcom_pull(task_ids='retrain_model', key='new_model_version')

        # Obter métricas do modelo atual em produção
        from src.model.mlflow_manager import MLFlowManager
        mlflow_manager = MLFlowManager()

        production_model = mlflow_manager.get_model_version(
            name=MODEL_REGISTRY_NAME,
            stage="Production"
        )

        if production_model:
            current_metrics = get_model_performance_metrics(production_model.version)
            current_accuracy = current_metrics.get('accuracy', 0)
        else:
            current_accuracy = 0  # Não há modelo em produção

        new_accuracy = new_metrics.get('accuracy', 0)

        # Calcular melhoria
        improvement = new_accuracy - current_accuracy
        improvement_pct = (improvement / current_accuracy * 100) if current_accuracy > 0 else 100

        logger.info(f"\nModel Comparison:")
        logger.info(f"  Current accuracy: {current_accuracy:.4f}")
        logger.info(f"  New accuracy: {new_accuracy:.4f}")
        logger.info(f"  Improvement: {improvement_pct:+.2f}%")

        # Validação 1: Melhoria mínima
        validation_passed = True
        reasons = []

        if improvement < MIN_IMPROVEMENT_THRESHOLD:
            validation_passed = False
            reasons.append(f"Insufficient improvement: {improvement_pct:.2f}% < {MIN_IMPROVEMENT_THRESHOLD*100}%")

        # Validação 2: Testes de robustez (simplificado)
        # Em produção, executar test_robustness.py aqui

        # Validação 3: Testes de fairness (simplificado)
        # Em produção, executar test_fairness.py aqui

        # Decisão
        if validation_passed:
            logger.info("\n✓ VALIDATION PASSED - Model will be promoted")
            decision = 'promote'
        else:
            logger.info("\n✗ VALIDATION FAILED - Model will be rolled back")
            logger.info(f"Reasons: {', '.join(reasons)}")
            decision = 'rollback'

        # Push decisão
        ti.xcom_push(key='validation_decision', value=decision)
        ti.xcom_push(key='improvement_pct', value=improvement_pct)

        return decision

    except Exception as e:
        logger.error(f"Error validating model: {e}")
        return 'rollback'


def promote_model(**context):
    """
    Promove novo modelo para produção.
    """
    logger.info("Promoting new model to production...")

    ti = context['task_instance']

    from src.model.mlflow_manager import MLFlowManager

    try:
        mlflow_manager = MLFlowManager()

        # Obter versão do novo modelo
        new_version = ti.xcom_pull(task_ids='retrain_model', key='new_model_version')

        # Transicionar modelo atual para Archived
        production_model = mlflow_manager.get_model_version(
            name=MODEL_REGISTRY_NAME,
            stage="Production"
        )

        if production_model:
            mlflow_manager.transition_model_stage(
                name=MODEL_REGISTRY_NAME,
                version=production_model.version,
                stage="Archived"
            )
            logger.info(f"Archived previous production model: version {production_model.version}")

        # Promover novo modelo
        mlflow_manager.transition_model_stage(
            name=MODEL_REGISTRY_NAME,
            version=new_version,
            stage="Production"
        )

        logger.info(f"✓ Model version {new_version} promoted to Production!")

        # Atualizar dados de referência
        current_df = load_data_from_s3(CURRENT_DATA_PATH)
        save_data_to_s3(current_df, REFERENCE_DATA_PATH)
        logger.info("Reference data updated")

        # Enviar notificação (implementar conforme necessário)
        # send_slack_notification(f"New model v{new_version} deployed to production!")

        return {
            'status': 'success',
            'new_version': new_version,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise


def rollback_model(**context):
    """
    Rollback em caso de falha na validação.
    """
    logger.info("Rolling back model...")

    ti = context['task_instance']

    from src.model.mlflow_manager import MLFlowManager

    try:
        mlflow_manager = MLFlowManager()

        # Obter versão do modelo que falhou
        failed_version = ti.xcom_pull(task_ids='retrain_model', key='new_model_version')

        # Arquivar modelo que falhou
        mlflow_manager.transition_model_stage(
            name=MODEL_REGISTRY_NAME,
            version=failed_version,
            stage="Archived"
        )

        logger.info(f"✗ Model version {failed_version} rolled back (archived)")

        # Enviar notificação de falha
        # send_slack_notification(f"Model v{failed_version} failed validation and was rolled back")

        return {
            'status': 'rolled_back',
            'failed_version': failed_version,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        raise


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    'ml_continuous_training_pipeline',
    default_args=DEFAULT_ARGS,
    description='Continuous training pipeline with automated retraining triggers',
    schedule_interval='0 2 * * 0',  # Todo domingo às 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'continuous-training', 'cd4ml'],
) as dag:

    # Início
    start = DummyOperator(task_id='start')

    # Task Group: Monitoring
    with TaskGroup('monitoring', tooltip='Monitor data and model performance') as monitoring_group:

        check_new_data_task = PythonOperator(
            task_id='check_new_data',
            python_callable=check_new_data_availability,
            provide_context=True,
        )

        detect_drift_task = PythonOperator(
            task_id='detect_drift',
            python_callable=detect_data_drift,
            provide_context=True,
        )

        check_performance_task = PythonOperator(
            task_id='check_performance',
            python_callable=check_model_performance,
            provide_context=True,
        )

        [check_new_data_task, detect_drift_task, check_performance_task]

    # Decision: Retreinar ou não
    decide_task = BranchPythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining,
        provide_context=True,
    )

    # Task Group: Retraining
    with TaskGroup('retraining', tooltip='Retrain and validate model') as retraining_group:

        retrain_task = PythonOperator(
            task_id='retrain_model',
            python_callable=retrain_model,
            provide_context=True,
        )

        validate_task = BranchPythonOperator(
            task_id='validate_new_model',
            python_callable=validate_new_model,
            provide_context=True,
        )

        retrain_task >> validate_task

    # Promote ou Rollback
    promote_task = PythonOperator(
        task_id='promote',
        python_callable=promote_model,
        provide_context=True,
    )

    rollback_task = PythonOperator(
        task_id='rollback',
        python_callable=rollback_model,
        provide_context=True,
    )

    # Skip retraining
    skip_retrain_task = DummyOperator(task_id='skip_retrain')

    # Fim
    end = DummyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success'
    )

    # Flow
    start >> monitoring_group >> decide_task

    # Se decidir retreinar
    decide_task >> retraining_group  # 'retrain' branch
    retraining_group >> [promote_task, rollback_task]  # validate_task escolhe qual

    # Se decidir não retreinar
    decide_task >> skip_retrain_task  # 'skip_retrain' branch

    # Todos levam ao fim
    [promote_task, rollback_task, skip_retrain_task] >> end
