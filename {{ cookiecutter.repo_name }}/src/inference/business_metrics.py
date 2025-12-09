"""
Business Metrics Monitoring for ML Models

Monitora métricas de negócio (além de métricas técnicas de ML) para medir
impacto real do modelo na operação.

Métricas:
- Revenue impact
- Cost reduction
- Customer satisfaction
- Operational efficiency
- ROI

Author: {{ cookiecutter.author_name }}
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BusinessMetric:
    """Representa uma métrica de negócio."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


# =============================================================================
# BUSINESS METRICS CALCULATOR
# =============================================================================

class BusinessMetricsCalculator:
    """
    Calcula métricas de negócio relacionadas ao modelo ML.

    Exemplo de uso:
        calculator = BusinessMetricsCalculator()

        # Calcular revenue impact
        revenue_impact = calculator.calculate_revenue_impact(
            predictions_df=predictions,
            actual_outcomes_df=actuals,
            revenue_per_unit=1000
        )

        # Enviar para monitoring
        calculator.log_metric(revenue_impact)
    """

    def __init__(
        self,
        prometheus_enabled: bool = True,
        cloudwatch_enabled: bool = False
    ):
        """
        Inicializa calculadora de métricas.

        Args:
            prometheus_enabled: Se True, envia métricas para Prometheus
            cloudwatch_enabled: Se True, envia métricas para CloudWatch
        """
        self.prometheus_enabled = prometheus_enabled
        self.cloudwatch_enabled = cloudwatch_enabled

        self.metrics_history: List[BusinessMetric] = []

        # Inicializar clients
        if self.prometheus_enabled:
            self._init_prometheus()

        if self.cloudwatch_enabled:
            self._init_cloudwatch()

    def _init_prometheus(self):
        """Inicializa Prometheus metrics."""
        try:
            from prometheus_client import Gauge

            # Criar gauges para cada métrica de negócio
            self.prom_revenue_impact = Gauge(
                'ml_business_revenue_impact_total',
                'Total revenue impact from ML model predictions'
            )

            self.prom_cost_reduction = Gauge(
                'ml_business_cost_reduction_total',
                'Total cost reduction from ML model'
            )

            self.prom_roi = Gauge(
                'ml_business_roi_ratio',
                'Return on Investment ratio for ML model'
            )

            self.prom_customer_satisfaction = Gauge(
                'ml_business_customer_satisfaction_score',
                'Customer satisfaction score (NPS, CSAT, etc)'
            )

            logger.info("Prometheus metrics initialized")

        except ImportError:
            logger.warning("prometheus_client not installed. Prometheus metrics disabled.")
            self.prometheus_enabled = False

    def _init_cloudwatch(self):
        """Inicializa CloudWatch metrics."""
        try:
            import boto3
            self.cloudwatch_client = boto3.client('cloudwatch')
            logger.info("CloudWatch client initialized")

        except ImportError:
            logger.warning("boto3 not installed. CloudWatch metrics disabled.")
            self.cloudwatch_enabled = False

    # =========================================================================
    # REVENUE METRICS
    # =========================================================================

    def calculate_revenue_impact(
        self,
        predictions_df: pd.DataFrame,
        actual_outcomes_df: Optional[pd.DataFrame] = None,
        revenue_per_true_positive: float = 1000,
        cost_per_false_positive: float = 100,
        cost_per_false_negative: float = 2000
    ) -> BusinessMetric:
        """
        Calcula impacto em receita das predições do modelo.

        Lógica:
        - True Positive: Ganho de revenue (ex: previu churn, reteve cliente)
        - False Positive: Custo de ação desnecessária
        - False Negative: Custo de oportunidade perdida

        Args:
            predictions_df: DataFrame com predições (cols: customer_id, prediction)
            actual_outcomes_df: DataFrame com outcomes reais (cols: customer_id, actual)
                                Se None, usa apenas volume de predições
            revenue_per_true_positive: Receita ganha por TP
            cost_per_false_positive: Custo de FP
            cost_per_false_negative: Custo de FN

        Returns:
            BusinessMetric com revenue impact total
        """
        if actual_outcomes_df is None:
            # Sem ground truth - estimar baseado em volume de predições positivas
            n_positive_predictions = (predictions_df['prediction'] == 1).sum()
            estimated_revenue = n_positive_predictions * revenue_per_true_positive * 0.8  # Assume 80% precision

            metric = BusinessMetric(
                name='revenue_impact',
                value=float(estimated_revenue),
                unit='BRL',
                timestamp=datetime.now(),
                metadata={
                    'type': 'estimated',
                    'n_positive_predictions': int(n_positive_predictions),
                    'assumed_precision': 0.8
                }
            )

        else:
            # Com ground truth - calcular revenue exato
            merged = predictions_df.merge(actual_outcomes_df, on='customer_id', how='inner')

            # Confusion matrix
            tp = ((merged['prediction'] == 1) & (merged['actual'] == 1)).sum()
            fp = ((merged['prediction'] == 1) & (merged['actual'] == 0)).sum()
            fn = ((merged['prediction'] == 0) & (merged['actual'] == 1)).sum()
            tn = ((merged['prediction'] == 0) & (merged['actual'] == 0)).sum()

            # Calcular revenue impact
            revenue_from_tp = tp * revenue_per_true_positive
            cost_from_fp = fp * cost_per_false_positive
            cost_from_fn = fn * cost_per_false_negative

            total_revenue_impact = revenue_from_tp - cost_from_fp - cost_from_fn

            metric = BusinessMetric(
                name='revenue_impact',
                value=float(total_revenue_impact),
                unit='BRL',
                timestamp=datetime.now(),
                metadata={
                    'type': 'actual',
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn),
                    'revenue_from_tp': float(revenue_from_tp),
                    'cost_from_fp': float(cost_from_fp),
                    'cost_from_fn': float(cost_from_fn)
                }
            )

        self.log_metric(metric)
        return metric

    def calculate_cost_reduction(
        self,
        predictions_df: pd.DataFrame,
        baseline_cost_per_customer: float,
        model_cost_per_customer: float
    ) -> BusinessMetric:
        """
        Calcula redução de custo operacional devido ao modelo.

        Exemplo: Modelo reduz necessidade de intervenção manual.

        Args:
            predictions_df: DataFrame com predições
            baseline_cost_per_customer: Custo sem modelo (processo manual)
            model_cost_per_customer: Custo com modelo (processo automatizado)

        Returns:
            BusinessMetric com cost reduction
        """
        n_customers = len(predictions_df)

        baseline_total_cost = n_customers * baseline_cost_per_customer
        model_total_cost = n_customers * model_cost_per_customer

        cost_reduction = baseline_total_cost - model_total_cost
        cost_reduction_pct = (cost_reduction / baseline_total_cost * 100) if baseline_total_cost > 0 else 0

        metric = BusinessMetric(
            name='cost_reduction',
            value=float(cost_reduction),
            unit='BRL',
            timestamp=datetime.now(),
            metadata={
                'n_customers': int(n_customers),
                'baseline_total_cost': float(baseline_total_cost),
                'model_total_cost': float(model_total_cost),
                'cost_reduction_pct': float(cost_reduction_pct)
            }
        )

        self.log_metric(metric)
        return metric

    def calculate_roi(
        self,
        revenue_impact: float,
        development_cost: float,
        operational_cost_monthly: float,
        months_in_production: int = 1
    ) -> BusinessMetric:
        """
        Calcula ROI (Return on Investment) do modelo ML.

        ROI = (Revenue - Total Cost) / Total Cost * 100

        Args:
            revenue_impact: Receita gerada pelo modelo
            development_cost: Custo de desenvolvimento (one-time)
            operational_cost_monthly: Custo operacional mensal (infra, etc)
            months_in_production: Meses em produção

        Returns:
            BusinessMetric com ROI ratio
        """
        total_cost = development_cost + (operational_cost_monthly * months_in_production)

        if total_cost == 0:
            roi_ratio = 0
        else:
            roi_ratio = ((revenue_impact - total_cost) / total_cost) * 100

        metric = BusinessMetric(
            name='roi',
            value=float(roi_ratio),
            unit='percentage',
            timestamp=datetime.now(),
            metadata={
                'revenue_impact': float(revenue_impact),
                'total_cost': float(total_cost),
                'development_cost': float(development_cost),
                'operational_cost': float(operational_cost_monthly * months_in_production),
                'months_in_production': int(months_in_production)
            }
        )

        self.log_metric(metric)
        return metric

    # =========================================================================
    # CUSTOMER METRICS
    # =========================================================================

    def calculate_customer_satisfaction(
        self,
        customer_feedback_df: pd.DataFrame,
        score_column: str = 'nps_score'
    ) -> BusinessMetric:
        """
        Calcula customer satisfaction score.

        Pode ser NPS, CSAT, CES, etc.

        Args:
            customer_feedback_df: DataFrame com feedback de clientes
            score_column: Coluna com score (ex: nps_score, csat_score)

        Returns:
            BusinessMetric com satisfaction score
        """
        avg_score = customer_feedback_df[score_column].mean()
        std_score = customer_feedback_df[score_column].std()

        metric = BusinessMetric(
            name='customer_satisfaction',
            value=float(avg_score),
            unit='score',
            timestamp=datetime.now(),
            metadata={
                'score_type': score_column,
                'mean': float(avg_score),
                'std': float(std_score),
                'n_responses': int(len(customer_feedback_df)),
                'min': float(customer_feedback_df[score_column].min()),
                'max': float(customer_feedback_df[score_column].max())
            }
        )

        self.log_metric(metric)
        return metric

    def calculate_churn_rate(
        self,
        customers_df: pd.DataFrame,
        churned_column: str = 'churned'
    ) -> BusinessMetric:
        """
        Calcula churn rate.

        Args:
            customers_df: DataFrame com clientes
            churned_column: Coluna binária indicando churn (1=churned)

        Returns:
            BusinessMetric com churn rate
        """
        churn_rate = (customers_df[churned_column] == 1).mean() * 100

        metric = BusinessMetric(
            name='churn_rate',
            value=float(churn_rate),
            unit='percentage',
            timestamp=datetime.now(),
            metadata={
                'total_customers': int(len(customers_df)),
                'churned_customers': int((customers_df[churned_column] == 1).sum())
            }
        )

        self.log_metric(metric)
        return metric

    # =========================================================================
    # OPERATIONAL METRICS
    # =========================================================================

    def calculate_process_efficiency(
        self,
        before_automation_time_hours: float,
        after_automation_time_hours: float,
        n_processes: int
    ) -> BusinessMetric:
        """
        Calcula ganho de eficiência operacional.

        Args:
            before_automation_time_hours: Tempo antes da automação (horas)
            after_automation_time_hours: Tempo após automação (horas)
            n_processes: Número de processos executados

        Returns:
            BusinessMetric com time saved
        """
        time_saved_per_process = before_automation_time_hours - after_automation_time_hours
        total_time_saved = time_saved_per_process * n_processes
        efficiency_gain_pct = (time_saved_per_process / before_automation_time_hours * 100) if before_automation_time_hours > 0 else 0

        metric = BusinessMetric(
            name='process_efficiency_gain',
            value=float(total_time_saved),
            unit='hours',
            timestamp=datetime.now(),
            metadata={
                'time_saved_per_process': float(time_saved_per_process),
                'efficiency_gain_pct': float(efficiency_gain_pct),
                'n_processes': int(n_processes)
            }
        )

        self.log_metric(metric)
        return metric

    # =========================================================================
    # LOGGING & MONITORING
    # =========================================================================

    def log_metric(self, metric: BusinessMetric):
        """
        Registra métrica em sistemas de monitoring.

        Args:
            metric: BusinessMetric a ser registrada
        """
        # Adicionar ao histórico
        self.metrics_history.append(metric)

        # Log
        logger.info(
            f"Business Metric: {metric.name} = {metric.value} {metric.unit} "
            f"(timestamp: {metric.timestamp})"
        )

        # Enviar para Prometheus
        if self.prometheus_enabled:
            self._send_to_prometheus(metric)

        # Enviar para CloudWatch
        if self.cloudwatch_enabled:
            self._send_to_cloudwatch(metric)

    def _send_to_prometheus(self, metric: BusinessMetric):
        """Envia métrica para Prometheus."""
        metric_map = {
            'revenue_impact': self.prom_revenue_impact,
            'cost_reduction': self.prom_cost_reduction,
            'roi': self.prom_roi,
            'customer_satisfaction': self.prom_customer_satisfaction,
        }

        if metric.name in metric_map:
            metric_map[metric.name].set(metric.value)

    def _send_to_cloudwatch(self, metric: BusinessMetric):
        """Envia métrica para CloudWatch."""
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace='MLOps/BusinessMetrics',
                MetricData=[
                    {
                        'MetricName': metric.name,
                        'Value': metric.value,
                        'Unit': 'None',  # CloudWatch units são diferentes
                        'Timestamp': metric.timestamp
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error sending metric to CloudWatch: {e}")

    def get_metrics_summary(self, last_n_days: int = 7) -> pd.DataFrame:
        """
        Retorna summary de métricas dos últimos N dias.

        Args:
            last_n_days: Número de dias para incluir

        Returns:
            DataFrame com summary
        """
        cutoff_date = datetime.now() - timedelta(days=last_n_days)

        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_date
        ]

        if not recent_metrics:
            return pd.DataFrame()

        df = pd.DataFrame([m.to_dict() for m in recent_metrics])

        # Agrupar por métrica e calcular estatísticas
        summary = df.groupby('name')['value'].agg(['mean', 'std', 'min', 'max', 'count'])
        summary = summary.reset_index()

        return summary


# =============================================================================
# INTEGRATION WITH ML PIPELINE
# =============================================================================

def calculate_and_log_business_metrics(
    predictions_df: pd.DataFrame,
    actual_outcomes_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Helper function para calcular e logar métricas de negócio.

    Usar no pipeline de inferência para monitorar impacto contínuo.

    Args:
        predictions_df: Predições do modelo
        actual_outcomes_df: Outcomes reais (quando disponíveis)

    Returns:
        Dict com todas as métricas calculadas
    """
    calculator = BusinessMetricsCalculator()

    metrics = {}

    # Revenue impact
    revenue_metric = calculator.calculate_revenue_impact(
        predictions_df=predictions_df,
        actual_outcomes_df=actual_outcomes_df,
        revenue_per_true_positive=1000,  # Customizar conforme negócio
        cost_per_false_positive=100,
        cost_per_false_negative=2000
    )
    metrics['revenue_impact'] = revenue_metric.value

    # Cost reduction
    cost_metric = calculator.calculate_cost_reduction(
        predictions_df=predictions_df,
        baseline_cost_per_customer=50,  # Customizar
        model_cost_per_customer=10      # Customizar
    )
    metrics['cost_reduction'] = cost_metric.value

    # ROI
    roi_metric = calculator.calculate_roi(
        revenue_impact=revenue_metric.value,
        development_cost=100000,  # Customizar
        operational_cost_monthly=5000,  # Customizar
        months_in_production=1
    )
    metrics['roi'] = roi_metric.value

    return metrics


# =============================================================================
# GRAFANA DASHBOARD QUERIES
# =============================================================================

GRAFANA_QUERIES = """
# Grafana Dashboard Queries para Business Metrics

## Revenue Impact (Last 30 days)
```promql
sum(ml_business_revenue_impact_total)
```

## Cost Reduction Trend
```promql
rate(ml_business_cost_reduction_total[1h])
```

## ROI Over Time
```promql
ml_business_roi_ratio
```

## Customer Satisfaction Trend
```promql
avg_over_time(ml_business_customer_satisfaction_score[7d])
```

## Composite Business Health Score
```promql
(
  ml_business_revenue_impact_total +
  ml_business_cost_reduction_total
) / 100000
```
"""

if __name__ == '__main__':
    # Exemplo de uso
    print("Business Metrics Calculator - Example Usage")
    print("="*80)

    # Criar dados de exemplo
    predictions_df = pd.DataFrame({
        'customer_id': range(1000),
        'prediction': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })

    actuals_df = pd.DataFrame({
        'customer_id': range(1000),
        'actual': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
    })

    # Calcular métricas
    metrics = calculate_and_log_business_metrics(predictions_df, actuals_df)

    print("\nCalculated Business Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.2f}")
