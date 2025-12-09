"""
Fairness and Bias Tests for ML Models

Testa equidade e viés em modelos de machine learning seguindo princípios de
Responsible AI. Implementa métricas de fairness estabelecidas na literatura.

Referências:
- Fairness and Machine Learning (fairmlbook.org)
- Aequitas Bias Toolkit
- IBM AI Fairness 360

Author: {{ cookiecutter.author_name }}
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import warnings


class FairnessMetrics:
    """
    Calcula métricas de fairness para modelos de classificação.

    Métricas implementadas:
    - Demographic Parity (Statistical Parity)
    - Equal Opportunity
    - Equalized Odds
    - Disparate Impact
    - Predictive Parity
    - Calibration
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        favorable_label: int = 1
    ):
        """
        Inicializa calculadora de métricas de fairness.

        Args:
            y_true: Labels verdadeiros
            y_pred: Predições do modelo
            sensitive_features: DataFrame com features sensíveis (ex: gender, race, age_group)
            favorable_label: Label considerado favorável (default: 1)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sensitive_features = sensitive_features
        self.favorable_label = favorable_label

        # Validações
        assert len(self.y_true) == len(self.y_pred), "y_true e y_pred devem ter mesmo tamanho"
        assert len(self.y_true) == len(sensitive_features), "Tamanhos incompatíveis"

    def demographic_parity(self, sensitive_attr: str) -> Dict[str, float]:
        """
        Demographic Parity (Statistical Parity):
        P(Y_pred=1 | A=a) = P(Y_pred=1 | A=b) para todos os grupos a, b

        Violação: Quando a diferença entre grupos é > threshold (tipicamente 0.1)

        Args:
            sensitive_attr: Nome do atributo sensível a testar

        Returns:
            Dict com taxas de predição positiva por grupo e diferença
        """
        groups = self.sensitive_features[sensitive_attr].unique()

        positive_rates = {}
        for group in groups:
            mask = self.sensitive_features[sensitive_attr] == group
            positive_rate = (self.y_pred[mask] == self.favorable_label).mean()
            positive_rates[str(group)] = float(positive_rate)

        # Calcular diferença máxima
        max_diff = max(positive_rates.values()) - min(positive_rates.values())

        return {
            'positive_rates_by_group': positive_rates,
            'max_difference': float(max_diff),
            'passes_threshold_0.1': max_diff <= 0.1,
            'metric': 'demographic_parity'
        }

    def equal_opportunity(self, sensitive_attr: str) -> Dict[str, float]:
        """
        Equal Opportunity:
        TPR(A=a) = TPR(A=b) para todos os grupos

        Garante que True Positive Rate é igual entre grupos.
        Importante quando queremos garantir que oportunidades (ex: aprovação de crédito)
        sejam iguais para indivíduos qualificados de todos os grupos.

        Args:
            sensitive_attr: Nome do atributo sensível

        Returns:
            Dict com TPR por grupo e diferença
        """
        groups = self.sensitive_features[sensitive_attr].unique()

        tpr_by_group = {}
        for group in groups:
            mask = self.sensitive_features[sensitive_attr] == group

            # Filtrar apenas casos positivos verdadeiros
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            positive_mask = y_true_group == self.favorable_label
            if positive_mask.sum() > 0:
                tpr = (y_pred_group[positive_mask] == self.favorable_label).mean()
                tpr_by_group[str(group)] = float(tpr)
            else:
                tpr_by_group[str(group)] = np.nan

        # Remover NaN para cálculo
        valid_tprs = [v for v in tpr_by_group.values() if not np.isnan(v)]

        if len(valid_tprs) > 1:
            max_diff = max(valid_tprs) - min(valid_tprs)
        else:
            max_diff = np.nan

        return {
            'tpr_by_group': tpr_by_group,
            'max_difference': float(max_diff) if not np.isnan(max_diff) else None,
            'passes_threshold_0.1': max_diff <= 0.1 if not np.isnan(max_diff) else None,
            'metric': 'equal_opportunity'
        }

    def equalized_odds(self, sensitive_attr: str) -> Dict[str, float]:
        """
        Equalized Odds:
        TPR(A=a) = TPR(A=b) AND FPR(A=a) = FPR(A=b) para todos os grupos

        Mais rigoroso que Equal Opportunity: garante que tanto TPR quanto FPR
        são iguais entre grupos.

        Args:
            sensitive_attr: Nome do atributo sensível

        Returns:
            Dict com TPR, FPR por grupo e diferenças
        """
        groups = self.sensitive_features[sensitive_attr].unique()

        metrics_by_group = {}
        for group in groups:
            mask = self.sensitive_features[sensitive_attr] == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            # TPR (True Positive Rate)
            positive_mask = y_true_group == self.favorable_label
            tpr = (y_pred_group[positive_mask] == self.favorable_label).mean() if positive_mask.sum() > 0 else np.nan

            # FPR (False Positive Rate)
            negative_mask = y_true_group != self.favorable_label
            fpr = (y_pred_group[negative_mask] == self.favorable_label).mean() if negative_mask.sum() > 0 else np.nan

            metrics_by_group[str(group)] = {
                'tpr': float(tpr) if not np.isnan(tpr) else None,
                'fpr': float(fpr) if not np.isnan(fpr) else None
            }

        # Calcular diferenças
        tprs = [v['tpr'] for v in metrics_by_group.values() if v['tpr'] is not None]
        fprs = [v['fpr'] for v in metrics_by_group.values() if v['fpr'] is not None]

        tpr_diff = max(tprs) - min(tprs) if len(tprs) > 1 else np.nan
        fpr_diff = max(fprs) - min(fprs) if len(fprs) > 1 else np.nan

        # Equalized odds passa se AMBOS TPR e FPR passam
        passes = (tpr_diff <= 0.1 and fpr_diff <= 0.1) if (not np.isnan(tpr_diff) and not np.isnan(fpr_diff)) else None

        return {
            'metrics_by_group': metrics_by_group,
            'tpr_max_difference': float(tpr_diff) if not np.isnan(tpr_diff) else None,
            'fpr_max_difference': float(fpr_diff) if not np.isnan(fpr_diff) else None,
            'passes_threshold_0.1': passes,
            'metric': 'equalized_odds'
        }

    def disparate_impact(self, sensitive_attr: str) -> Dict[str, float]:
        """
        Disparate Impact (4/5ths Rule):
        P(Y_pred=1 | A=unprivileged) / P(Y_pred=1 | A=privileged) >= 0.8

        Usada em contextos legais (EEOC). Ratio < 0.8 indica possível discriminação.

        Args:
            sensitive_attr: Nome do atributo sensível

        Returns:
            Dict com disparate impact ratio
        """
        groups = self.sensitive_features[sensitive_attr].unique()

        if len(groups) != 2:
            warnings.warn(f"Disparate Impact funciona melhor com 2 grupos. Encontrados: {len(groups)}")

        positive_rates = {}
        for group in groups:
            mask = self.sensitive_features[sensitive_attr] == group
            positive_rate = (self.y_pred[mask] == self.favorable_label).mean()
            positive_rates[str(group)] = float(positive_rate)

        # Calcular ratio (min/max)
        max_rate = max(positive_rates.values())
        min_rate = min(positive_rates.values())

        ratio = min_rate / max_rate if max_rate > 0 else 0

        return {
            'positive_rates_by_group': positive_rates,
            'disparate_impact_ratio': float(ratio),
            'passes_4/5ths_rule': ratio >= 0.8,
            'metric': 'disparate_impact'
        }

    def predictive_parity(self, sensitive_attr: str) -> Dict[str, float]:
        """
        Predictive Parity (Precision Parity):
        PPV(A=a) = PPV(A=b) para todos os grupos

        PPV = Precision = TP / (TP + FP)

        Garante que predições positivas têm mesma precisão entre grupos.

        Args:
            sensitive_attr: Nome do atributo sensível

        Returns:
            Dict com precision por grupo e diferença
        """
        groups = self.sensitive_features[sensitive_attr].unique()

        ppv_by_group = {}
        for group in groups:
            mask = self.sensitive_features[sensitive_attr] == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            # Calcular precision
            predicted_positive = y_pred_group == self.favorable_label
            if predicted_positive.sum() > 0:
                precision = (y_true_group[predicted_positive] == self.favorable_label).mean()
                ppv_by_group[str(group)] = float(precision)
            else:
                ppv_by_group[str(group)] = np.nan

        valid_ppvs = [v for v in ppv_by_group.values() if not np.isnan(v)]

        if len(valid_ppvs) > 1:
            max_diff = max(valid_ppvs) - min(valid_ppvs)
        else:
            max_diff = np.nan

        return {
            'precision_by_group': ppv_by_group,
            'max_difference': float(max_diff) if not np.isnan(max_diff) else None,
            'passes_threshold_0.1': max_diff <= 0.1 if not np.isnan(max_diff) else None,
            'metric': 'predictive_parity'
        }

    def calibration_by_group(self, sensitive_attr: str, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
        """
        Calibration: Verifica se probabilidades preditas refletem taxas reais por grupo.

        P(Y=1 | Score=s, A=a) ≈ s para todos os grupos a

        Args:
            sensitive_attr: Nome do atributo sensível
            y_prob: Probabilidades preditas
            n_bins: Número de bins para calibração

        Returns:
            Dict com métricas de calibração por grupo
        """
        groups = self.sensitive_features[sensitive_attr].unique()

        calibration_by_group = {}
        for group in groups:
            mask = self.sensitive_features[sensitive_attr] == group
            y_true_group = self.y_true[mask]
            y_prob_group = y_prob[mask]

            # Dividir em bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_prob_group, bins) - 1

            bin_true_rates = []
            bin_pred_rates = []

            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    true_rate = y_true_group[bin_mask].mean()
                    pred_rate = y_prob_group[bin_mask].mean()
                    bin_true_rates.append(true_rate)
                    bin_pred_rates.append(pred_rate)

            # Calcular calibration error (ECE - Expected Calibration Error)
            if len(bin_true_rates) > 0:
                ece = np.mean(np.abs(np.array(bin_true_rates) - np.array(bin_pred_rates)))
            else:
                ece = np.nan

            calibration_by_group[str(group)] = {
                'expected_calibration_error': float(ece) if not np.isnan(ece) else None,
                'n_samples': int(mask.sum())
            }

        # Diferença de calibration entre grupos
        eces = [v['expected_calibration_error'] for v in calibration_by_group.values() if v['expected_calibration_error'] is not None]
        max_ece_diff = max(eces) - min(eces) if len(eces) > 1 else np.nan

        return {
            'calibration_by_group': calibration_by_group,
            'max_ece_difference': float(max_ece_diff) if not np.isnan(max_ece_diff) else None,
            'metric': 'calibration'
        }

    def compute_all_metrics(self, y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Calcula todas as métricas de fairness para todos os atributos sensíveis.

        Args:
            y_prob: Probabilidades (necessário para calibration)

        Returns:
            Dict com todas as métricas
        """
        all_metrics = {}

        for attr in self.sensitive_features.columns:
            all_metrics[attr] = {
                'demographic_parity': self.demographic_parity(attr),
                'equal_opportunity': self.equal_opportunity(attr),
                'equalized_odds': self.equalized_odds(attr),
                'disparate_impact': self.disparate_impact(attr),
                'predictive_parity': self.predictive_parity(attr),
            }

            if y_prob is not None:
                all_metrics[attr]['calibration'] = self.calibration_by_group(attr, y_prob)

        return all_metrics


# ============================================================================
# PYTEST TESTS
# ============================================================================

@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo para testes."""
    np.random.seed(42)
    n_samples = 1000

    # Criar dados sintéticos
    data = {
        'y_true': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'y_pred': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'y_prob': np.random.uniform(0, 1, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age_group': np.random.choice(['18-30', '31-50', '50+'], n_samples),
        'race': np.random.choice(['A', 'B', 'C'], n_samples),
    }

    return data


@pytest.fixture
def biased_data():
    """
    Fixture com dados intencionalmente enviesados para testar detecção de bias.

    Viés: Modelo tem TPR maior para grupo 'M' do que 'F'.
    """
    np.random.seed(42)
    n_samples = 1000

    # Ground truth balanceado
    y_true = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    gender = np.random.choice(['M', 'F'], n_samples)

    # Criar predições enviesadas
    y_pred = y_true.copy()

    # Para gênero F, reduzir TPR (mais falsos negativos)
    female_mask = gender == 'F'
    positive_females = female_mask & (y_true == 1)

    # Fazer 30% dos positivos de F serem preditos como negativos
    n_to_flip = int(positive_females.sum() * 0.3)
    flip_indices = np.random.choice(
        np.where(positive_females)[0],
        size=n_to_flip,
        replace=False
    )
    y_pred[flip_indices] = 0

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': np.random.uniform(0, 1, n_samples),
        'gender': gender,
    }


class TestFairnessMetrics:
    """Testes para as métricas de fairness."""

    def test_demographic_parity_calculation(self, sample_data):
        """Testa cálculo de demographic parity."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.demographic_parity('gender')

        assert 'positive_rates_by_group' in result
        assert 'max_difference' in result
        assert 'passes_threshold_0.1' in result
        assert len(result['positive_rates_by_group']) == 2  # M e F
        assert 0 <= result['max_difference'] <= 1

    def test_equal_opportunity_calculation(self, sample_data):
        """Testa cálculo de equal opportunity."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.equal_opportunity('gender')

        assert 'tpr_by_group' in result
        assert 'max_difference' in result
        assert len(result['tpr_by_group']) == 2

    def test_equalized_odds_calculation(self, sample_data):
        """Testa cálculo de equalized odds."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.equalized_odds('gender')

        assert 'metrics_by_group' in result
        assert 'tpr_max_difference' in result
        assert 'fpr_max_difference' in result

        # Verificar que cada grupo tem TPR e FPR
        for group_metrics in result['metrics_by_group'].values():
            assert 'tpr' in group_metrics
            assert 'fpr' in group_metrics

    def test_disparate_impact_calculation(self, sample_data):
        """Testa cálculo de disparate impact."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.disparate_impact('gender')

        assert 'disparate_impact_ratio' in result
        assert 'passes_4/5ths_rule' in result
        assert 0 <= result['disparate_impact_ratio'] <= 1
        assert isinstance(result['passes_4/5ths_rule'], bool)

    def test_predictive_parity_calculation(self, sample_data):
        """Testa cálculo de predictive parity."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.predictive_parity('gender')

        assert 'precision_by_group' in result
        assert 'max_difference' in result
        assert len(result['precision_by_group']) == 2

    def test_calibration_calculation(self, sample_data):
        """Testa cálculo de calibração."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.calibration_by_group('gender', sample_data['y_prob'])

        assert 'calibration_by_group' in result
        assert 'max_ece_difference' in result

        for group_calib in result['calibration_by_group'].values():
            assert 'expected_calibration_error' in group_calib
            assert 'n_samples' in group_calib

    def test_bias_detection(self, biased_data):
        """
        Testa se métricas detectam viés intencional nos dados.

        Dados enviesados: TPR menor para gênero F.
        Espera-se que equal_opportunity e equalized_odds falhem threshold.
        """
        sensitive_features = pd.DataFrame({
            'gender': biased_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=biased_data['y_true'],
            y_pred=biased_data['y_pred'],
            sensitive_features=sensitive_features
        )

        # Equal opportunity deve detectar diferença em TPR
        eo_result = metrics.equal_opportunity('gender')

        # Verificar que há diferença significativa
        assert eo_result['max_difference'] is not None
        assert eo_result['max_difference'] > 0.1  # Deve falhar threshold
        assert eo_result['passes_threshold_0.1'] is False

        # TPR para F deve ser menor que para M
        tpr_f = eo_result['tpr_by_group']['F']
        tpr_m = eo_result['tpr_by_group']['M']
        assert tpr_f < tpr_m

    def test_compute_all_metrics(self, sample_data):
        """Testa computação de todas as métricas."""
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender'],
            'age_group': sample_data['age_group']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        all_metrics = metrics.compute_all_metrics(y_prob=sample_data['y_prob'])

        # Verificar que todas as métricas foram calculadas para todos os atributos
        assert 'gender' in all_metrics
        assert 'age_group' in all_metrics

        for attr_metrics in all_metrics.values():
            assert 'demographic_parity' in attr_metrics
            assert 'equal_opportunity' in attr_metrics
            assert 'equalized_odds' in attr_metrics
            assert 'disparate_impact' in attr_metrics
            assert 'predictive_parity' in attr_metrics
            assert 'calibration' in attr_metrics


class TestFairnessThresholds:
    """
    Testes de threshold para fairness em produção.

    ATENÇÃO: Estes testes devem ser customizados para seu caso de uso específico.
    Thresholds podem variar dependendo do domínio, regulações e requisitos de negócio.
    """

    @pytest.mark.parametrize("sensitive_attr", ["gender", "age_group", "race"])
    def test_demographic_parity_threshold(self, sample_data, sensitive_attr):
        """
        Testa se demographic parity está dentro do threshold aceitável.

        Threshold: max_difference <= 0.1 (10%)
        """
        if sensitive_attr not in sample_data:
            pytest.skip(f"Atributo {sensitive_attr} não disponível nos dados de teste")

        sensitive_features = pd.DataFrame({
            sensitive_attr: sample_data[sensitive_attr]
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.demographic_parity(sensitive_attr)

        # Assertion com mensagem clara
        assert result['max_difference'] <= 0.1, (
            f"Demographic parity violation for {sensitive_attr}: "
            f"max difference = {result['max_difference']:.3f} > 0.1\n"
            f"Rates by group: {result['positive_rates_by_group']}"
        )

    @pytest.mark.parametrize("sensitive_attr", ["gender"])
    def test_disparate_impact_4_5ths_rule(self, sample_data, sensitive_attr):
        """
        Testa 4/5ths rule para disparate impact.

        Legal threshold (EEOC): ratio >= 0.8
        """
        sensitive_features = pd.DataFrame({
            sensitive_attr: sample_data[sensitive_attr]
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.disparate_impact(sensitive_attr)

        assert result['disparate_impact_ratio'] >= 0.8, (
            f"Disparate impact violation for {sensitive_attr}: "
            f"ratio = {result['disparate_impact_ratio']:.3f} < 0.8 (4/5ths rule)\n"
            f"Rates by group: {result['positive_rates_by_group']}"
        )

    def test_equal_opportunity_threshold(self, sample_data):
        """
        Testa se TPR é similar entre grupos.

        Threshold: max TPR difference <= 0.1
        """
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.equal_opportunity('gender')

        if result['max_difference'] is not None:
            assert result['max_difference'] <= 0.1, (
                f"Equal opportunity violation: "
                f"TPR difference = {result['max_difference']:.3f} > 0.1\n"
                f"TPR by group: {result['tpr_by_group']}"
            )

    def test_no_severe_bias_detected(self, sample_data):
        """
        Teste integrado: verifica se não há viés severo em nenhuma métrica.

        Viés severo: qualquer métrica com diferença > 0.2 (20%)
        """
        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        all_metrics = metrics.compute_all_metrics(y_prob=sample_data['y_prob'])

        violations = []

        for attr, attr_metrics in all_metrics.items():
            for metric_name, metric_result in attr_metrics.items():
                if metric_name == 'calibration':
                    continue  # Skip calibration for this test

                # Verificar max_difference
                if 'max_difference' in metric_result and metric_result['max_difference'] is not None:
                    if metric_result['max_difference'] > 0.2:
                        violations.append(
                            f"{metric_name} on {attr}: difference = {metric_result['max_difference']:.3f}"
                        )

        assert len(violations) == 0, (
            f"Severe bias detected in the following metrics:\n" +
            "\n".join(violations)
        )


class TestBiasAudit:
    """
    Testes de auditoria de viés para compliance e governança.

    Estes testes geram relatórios detalhados de fairness para documentação.
    """

    def test_generate_fairness_report(self, sample_data, tmp_path):
        """
        Gera relatório completo de fairness para auditoria.

        Este teste sempre passa, mas gera um arquivo de relatório que pode
        ser usado para documentação e compliance.
        """
        import json

        sensitive_features = pd.DataFrame({
            'gender': sample_data['gender'],
            'age_group': sample_data['age_group']
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        # Calcular todas as métricas
        all_metrics = metrics.compute_all_metrics(y_prob=sample_data['y_prob'])

        # Criar relatório
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_samples': len(sample_data['y_true']),
            'sensitive_attributes': list(sensitive_features.columns),
            'fairness_metrics': all_metrics
        }

        # Salvar relatório
        report_path = tmp_path / "fairness_audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📊 Fairness Audit Report generated: {report_path}")
        assert report_path.exists()

    def test_fairness_across_intersections(self, sample_data):
        """
        Testa fairness em interseções de atributos sensíveis.

        Exemplo: gender × age_group (mulheres jovens vs homens idosos)

        Intersectionality é importante para detectar viés composto.
        """
        # Criar atributo de interseção
        intersection = pd.Series(
            sample_data['gender'] + '_' + sample_data['age_group'],
            name='gender_age'
        )

        sensitive_features = pd.DataFrame({
            'intersection': intersection
        })

        metrics = FairnessMetrics(
            y_true=sample_data['y_true'],
            y_pred=sample_data['y_pred'],
            sensitive_features=sensitive_features
        )

        result = metrics.demographic_parity('intersection')

        # Verificar que há pelo menos 3 grupos de interseção
        assert len(result['positive_rates_by_group']) >= 3

        # Verificar se diferença é aceitável (mais leniente para interseções)
        # Threshold mais alto (0.15) pois interseções têm amostras menores
        assert result['max_difference'] <= 0.15, (
            f"Intersectional bias detected: {result['max_difference']:.3f} > 0.15"
        )


# ============================================================================
# HELPER FUNCTIONS PARA USO EM PRODUÇÃO
# ============================================================================

def assert_fairness_compliance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Helper function para verificar compliance de fairness em produção.

    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        sensitive_features: DataFrame com features sensíveis
        thresholds: Dict opcional com thresholds customizados

    Returns:
        Dict com resultados de compliance

    Raises:
        AssertionError: Se qualquer métrica falhar threshold
    """
    if thresholds is None:
        thresholds = {
            'demographic_parity': 0.1,
            'equal_opportunity': 0.1,
            'disparate_impact': 0.8,
        }

    metrics = FairnessMetrics(y_true, y_pred, sensitive_features)
    all_metrics = metrics.compute_all_metrics()

    violations = []

    for attr, attr_metrics in all_metrics.items():
        # Demographic Parity
        dp = attr_metrics['demographic_parity']
        if dp['max_difference'] > thresholds['demographic_parity']:
            violations.append(
                f"Demographic Parity violation for {attr}: "
                f"{dp['max_difference']:.3f} > {thresholds['demographic_parity']}"
            )

        # Equal Opportunity
        eo = attr_metrics['equal_opportunity']
        if eo['max_difference'] is not None and eo['max_difference'] > thresholds['equal_opportunity']:
            violations.append(
                f"Equal Opportunity violation for {attr}: "
                f"{eo['max_difference']:.3f} > {thresholds['equal_opportunity']}"
            )

        # Disparate Impact
        di = attr_metrics['disparate_impact']
        if di['disparate_impact_ratio'] < thresholds['disparate_impact']:
            violations.append(
                f"Disparate Impact violation for {attr}: "
                f"{di['disparate_impact_ratio']:.3f} < {thresholds['disparate_impact']}"
            )

    if violations:
        raise AssertionError(
            "Fairness compliance failed:\n" + "\n".join(violations)
        )

    return {
        'status': 'PASSED',
        'metrics': all_metrics,
        'violations': []
    }
