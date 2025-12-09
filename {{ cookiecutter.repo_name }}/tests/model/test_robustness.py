"""
Robustness Tests for ML Models

Testa robustez de modelos de machine learning contra:
- Perturbações adversariais
- Mudanças nos dados (invariance)
- Edge cases
- Data drift
- Model stability

Seguindo princípios de Responsible AI e CD4ML.

Author: {{ cookiecutter.author_name }}
"""

import pytest
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings


class RobustnessChecker:
    """
    Classe para testar robustez de modelos de ML.

    Implementa diversos testes de robustez:
    - Adversarial perturbations
    - Invariance tests
    - Edge case handling
    - Prediction stability
    - Input validation
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_types: Optional[Dict[str, str]] = None
    ):
        """
        Inicializa o testador de robustez.

        Args:
            model: Modelo treinado com método predict()
            feature_names: Lista de nomes das features
            feature_types: Dict mapeando feature -> tipo ('numeric', 'categorical')
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types or {}

        # Inferir tipos se não fornecidos
        if not self.feature_types:
            self._infer_feature_types()

    def _infer_feature_types(self):
        """Tenta inferir tipos de features (placeholder - customizar conforme modelo)."""
        # Por padrão, assume tudo como numérico
        for fname in self.feature_names:
            self.feature_types[fname] = 'numeric'

    def adversarial_perturbation_test(
        self,
        X: pd.DataFrame,
        epsilon: float = 0.1,
        n_samples: int = 100
    ) -> Dict:
        """
        Testa robustez contra perturbações adversariais.

        Adiciona ruído pequeno (epsilon) às features e verifica se predições mudam drasticamente.

        Args:
            X: Dataset de teste
            epsilon: Magnitude da perturbação (fração do std de cada feature)
            n_samples: Número de amostras a testar

        Returns:
            Dict com métricas de robustez
        """
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)

        # Predições originais
        y_pred_original = self.model.predict(X_sample)

        # Selecionar apenas features numéricas
        numeric_features = [
            f for f in self.feature_names
            if self.feature_types.get(f, 'numeric') == 'numeric' and f in X_sample.columns
        ]

        if len(numeric_features) == 0:
            return {
                'test': 'adversarial_perturbation',
                'status': 'SKIPPED',
                'reason': 'No numeric features found'
            }

        # Criar versão perturbada
        X_perturbed = X_sample.copy()

        for feat in numeric_features:
            # Adicionar ruído gaussiano
            std = X_sample[feat].std()
            noise = np.random.normal(0, epsilon * std, len(X_sample))
            X_perturbed[feat] = X_sample[feat] + noise

        # Predições perturbadas
        y_pred_perturbed = self.model.predict(X_perturbed)

        # Calcular taxa de mudança
        prediction_change_rate = (y_pred_original != y_pred_perturbed).mean()

        # Calcular magnitude média das mudanças (para regressão ou probabilidades)
        if hasattr(self.model, 'predict_proba'):
            prob_original = self.model.predict_proba(X_sample)
            prob_perturbed = self.model.predict_proba(X_perturbed)

            # Diferença média nas probabilidades
            prob_change = np.abs(prob_original - prob_perturbed).mean()
        else:
            prob_change = None

        return {
            'test': 'adversarial_perturbation',
            'epsilon': epsilon,
            'n_samples': n_samples,
            'prediction_change_rate': float(prediction_change_rate),
            'probability_change_mean': float(prob_change) if prob_change is not None else None,
            'is_robust': prediction_change_rate < 0.1,  # <10% mudança é considerado robusto
            'status': 'PASSED' if prediction_change_rate < 0.1 else 'FAILED'
        }

    def invariance_test(
        self,
        X: pd.DataFrame,
        invariant_features: List[str],
        n_samples: int = 100
    ) -> Dict:
        """
        Testa se predições são invariantes a mudanças em features específicas.

        Exemplo: Modelo de aprovação de crédito não deve mudar predição se gênero mudar,
        mantendo outras features constantes.

        Args:
            X: Dataset de teste
            invariant_features: Lista de features que NÃO devem afetar predição
            n_samples: Número de amostras a testar

        Returns:
            Dict com resultados de invariância
        """
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)

        # Predições originais
        y_pred_original = self.model.predict(X_sample)

        results_by_feature = {}

        for feat in invariant_features:
            if feat not in X_sample.columns:
                continue

            X_modified = X_sample.copy()

            # Modificar feature baseado no tipo
            if self.feature_types.get(feat) == 'categorical' or X_sample[feat].dtype == 'object':
                # Para categóricas, trocar por outra categoria
                unique_vals = X_sample[feat].unique()
                if len(unique_vals) > 1:
                    # Mapear cada valor para outro valor
                    mapping = {val: unique_vals[(i + 1) % len(unique_vals)] for i, val in enumerate(unique_vals)}
                    X_modified[feat] = X_sample[feat].map(mapping)
                else:
                    continue  # Skip se só tem 1 valor

            else:
                # Para numéricas, adicionar 1 std
                std = X_sample[feat].std()
                X_modified[feat] = X_sample[feat] + std

            # Predições modificadas
            y_pred_modified = self.model.predict(X_modified)

            # Taxa de mudança
            change_rate = (y_pred_original != y_pred_modified).mean()

            results_by_feature[feat] = {
                'prediction_change_rate': float(change_rate),
                'is_invariant': change_rate < 0.05,  # <5% mudança é considerado invariante
                'status': 'PASSED' if change_rate < 0.05 else 'FAILED'
            }

        # Status geral
        all_passed = all(r['status'] == 'PASSED' for r in results_by_feature.values())

        return {
            'test': 'invariance',
            'features_tested': invariant_features,
            'results_by_feature': results_by_feature,
            'all_invariant': all_passed,
            'status': 'PASSED' if all_passed else 'FAILED'
        }

    def edge_case_handling_test(
        self,
        X: pd.DataFrame,
        edge_cases: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Testa como modelo lida com edge cases.

        Edge cases testados:
        - Valores extremos (min, max)
        - Valores zero
        - Valores faltantes (se aplicável)
        - Combinações incomuns

        Args:
            X: Dataset de referência (para obter ranges)
            edge_cases: Dict opcional com edge cases customizados

        Returns:
            Dict com resultados
        """
        results = []

        # Selecionar apenas features numéricas
        numeric_features = [
            f for f in self.feature_names
            if self.feature_types.get(f, 'numeric') == 'numeric' and f in X.columns
        ]

        if len(numeric_features) == 0:
            return {
                'test': 'edge_case_handling',
                'status': 'SKIPPED',
                'reason': 'No numeric features found'
            }

        # Criar sample base (valores medianos)
        X_base = pd.DataFrame({
            feat: [X[feat].median()] for feat in X.columns
        })

        # Teste 1: Valores mínimos
        X_min = X_base.copy()
        for feat in numeric_features:
            X_min[feat] = X[feat].min()

        try:
            pred_min = self.model.predict(X_min)
            results.append({
                'case': 'all_min_values',
                'status': 'PASSED',
                'error': None
            })
        except Exception as e:
            results.append({
                'case': 'all_min_values',
                'status': 'FAILED',
                'error': str(e)
            })

        # Teste 2: Valores máximos
        X_max = X_base.copy()
        for feat in numeric_features:
            X_max[feat] = X[feat].max()

        try:
            pred_max = self.model.predict(X_max)
            results.append({
                'case': 'all_max_values',
                'status': 'PASSED',
                'error': None
            })
        except Exception as e:
            results.append({
                'case': 'all_max_values',
                'status': 'FAILED',
                'error': str(e)
            })

        # Teste 3: Todos valores zero (se zero está no range)
        X_zero = X_base.copy()
        for feat in numeric_features:
            if X[feat].min() <= 0 <= X[feat].max():
                X_zero[feat] = 0

        try:
            pred_zero = self.model.predict(X_zero)
            results.append({
                'case': 'zero_values',
                'status': 'PASSED',
                'error': None
            })
        except Exception as e:
            results.append({
                'case': 'zero_values',
                'status': 'FAILED',
                'error': str(e)
            })

        # Teste 4: Valores extremos (3 std além da média)
        X_extreme = X_base.copy()
        for feat in numeric_features:
            mean = X[feat].mean()
            std = X[feat].std()
            X_extreme[feat] = mean + 3 * std

        try:
            pred_extreme = self.model.predict(X_extreme)
            results.append({
                'case': 'extreme_values_3std',
                'status': 'PASSED',
                'error': None
            })
        except Exception as e:
            results.append({
                'case': 'extreme_values_3std',
                'status': 'FAILED',
                'error': str(e)
            })

        # Status geral
        all_passed = all(r['status'] == 'PASSED' for r in results)

        return {
            'test': 'edge_case_handling',
            'results': results,
            'total_cases': len(results),
            'passed': sum(1 for r in results if r['status'] == 'PASSED'),
            'failed': sum(1 for r in results if r['status'] == 'FAILED'),
            'status': 'PASSED' if all_passed else 'FAILED'
        }

    def prediction_stability_test(
        self,
        X: pd.DataFrame,
        n_trials: int = 10,
        sample_size: int = 100
    ) -> Dict:
        """
        Testa estabilidade das predições em amostras similares.

        Se modelo é estável, predições em amostras similares devem ser consistentes.

        Args:
            X: Dataset
            n_trials: Número de trials (amostras)
            sample_size: Tamanho de cada amostra

        Returns:
            Dict com métricas de estabilidade
        """
        if len(X) < sample_size:
            return {
                'test': 'prediction_stability',
                'status': 'SKIPPED',
                'reason': f'Dataset too small ({len(X)} < {sample_size})'
            }

        predictions_distribution = []

        for trial in range(n_trials):
            X_sample = X.sample(sample_size, random_state=trial)
            y_pred = self.model.predict(X_sample)

            # Taxa de predições positivas (para classificação binária)
            if len(np.unique(y_pred)) == 2:
                positive_rate = (y_pred == 1).mean()
                predictions_distribution.append(positive_rate)

        if len(predictions_distribution) == 0:
            return {
                'test': 'prediction_stability',
                'status': 'SKIPPED',
                'reason': 'Not a binary classification task'
            }

        # Calcular variabilidade
        std_predictions = np.std(predictions_distribution)
        mean_predictions = np.mean(predictions_distribution)
        cv = std_predictions / mean_predictions if mean_predictions > 0 else np.inf

        return {
            'test': 'prediction_stability',
            'n_trials': n_trials,
            'sample_size': sample_size,
            'mean_positive_rate': float(mean_predictions),
            'std_positive_rate': float(std_predictions),
            'coefficient_of_variation': float(cv),
            'is_stable': cv < 0.1,  # CV < 10% é considerado estável
            'status': 'PASSED' if cv < 0.1 else 'FAILED'
        }

    def feature_range_validation_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Dict:
        """
        Testa se dados de teste estão dentro do range visto no treinamento.

        Dados fora do range de treinamento podem causar predições não confiáveis.

        Args:
            X_train: Dataset de treinamento
            X_test: Dataset de teste

        Returns:
            Dict com resultados de validação
        """
        numeric_features = [
            f for f in self.feature_names
            if self.feature_types.get(f, 'numeric') == 'numeric' and f in X_test.columns
        ]

        results_by_feature = {}

        for feat in numeric_features:
            train_min = X_train[feat].min()
            train_max = X_train[feat].max()

            test_min = X_test[feat].min()
            test_max = X_test[feat].max()

            # Verificar se test está fora do range
            below_range = (X_test[feat] < train_min).sum()
            above_range = (X_test[feat] > train_max).sum()
            out_of_range_pct = (below_range + above_range) / len(X_test) * 100

            results_by_feature[feat] = {
                'train_range': [float(train_min), float(train_max)],
                'test_range': [float(test_min), float(test_max)],
                'samples_below_train_min': int(below_range),
                'samples_above_train_max': int(above_range),
                'out_of_range_pct': float(out_of_range_pct),
                'status': 'PASSED' if out_of_range_pct < 5 else 'WARNING' if out_of_range_pct < 10 else 'FAILED'
            }

        # Status geral
        failed_features = [
            f for f, r in results_by_feature.items()
            if r['status'] == 'FAILED'
        ]

        return {
            'test': 'feature_range_validation',
            'results_by_feature': results_by_feature,
            'features_with_issues': failed_features,
            'status': 'PASSED' if len(failed_features) == 0 else 'WARNING'
        }

    def run_all_tests(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        invariant_features: Optional[List[str]] = None
    ) -> Dict:
        """
        Executa todos os testes de robustez.

        Args:
            X_train: Dataset de treinamento
            X_test: Dataset de teste
            invariant_features: Features que devem ser invariantes

        Returns:
            Dict com todos os resultados
        """
        results = {}

        # Teste 1: Adversarial Perturbation
        print("Running adversarial perturbation test...")
        results['adversarial_perturbation'] = self.adversarial_perturbation_test(X_test)

        # Teste 2: Invariance (se fornecido)
        if invariant_features:
            print("Running invariance test...")
            results['invariance'] = self.invariance_test(X_test, invariant_features)

        # Teste 3: Edge Cases
        print("Running edge case handling test...")
        results['edge_case_handling'] = self.edge_case_handling_test(X_test)

        # Teste 4: Prediction Stability
        print("Running prediction stability test...")
        results['prediction_stability'] = self.prediction_stability_test(X_test)

        # Teste 5: Feature Range Validation
        print("Running feature range validation test...")
        results['feature_range_validation'] = self.feature_range_validation_test(X_train, X_test)

        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get('status') == 'PASSED')
        failed_tests = sum(1 for r in results.values() if r.get('status') == 'FAILED')

        results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'warnings': total_tests - passed_tests - failed_tests,
            'overall_status': 'PASSED' if failed_tests == 0 else 'FAILED'
        }

        return results


# ============================================================================
# PYTEST TESTS
# ============================================================================

@pytest.fixture
def sample_model_and_data():
    """Fixture com modelo e dados de exemplo."""
    np.random.seed(42)

    # Gerar dados sintéticos
    n_samples = 1000
    X = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, n_samples),
        'feature_2': np.random.uniform(0, 1, n_samples),
        'feature_3': np.random.exponential(2, n_samples),
        'category_A': np.random.choice(['A', 'B', 'C'], n_samples),
    })

    # Encode categorical
    X['category_A_encoded'] = pd.Categorical(X['category_A']).codes

    # Target
    y = (X['feature_1'] + X['feature_2'] * 50 + np.random.normal(0, 10, n_samples)) > 110
    y = y.astype(int)

    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Treinar modelo simples
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    features_for_training = ['feature_1', 'feature_2', 'feature_3', 'category_A_encoded']
    model.fit(X_train[features_for_training], y_train)

    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': features_for_training
    }


class TestRobustnessChecker:
    """Testes para RobustnessChecker."""

    def test_adversarial_perturbation(self, sample_model_and_data):
        """Testa robustez contra perturbações adversariais."""
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.adversarial_perturbation_test(
            X=data['X_test'],
            epsilon=0.1,
            n_samples=100
        )

        assert 'prediction_change_rate' in result
        assert 'is_robust' in result
        assert 0 <= result['prediction_change_rate'] <= 1

        # Modelo deve ser razoavelmente robusto a pequenas perturbações
        assert result['prediction_change_rate'] < 0.3, (
            f"Model is not robust to adversarial perturbations: "
            f"{result['prediction_change_rate']:.2%} predictions changed"
        )

    def test_invariance(self, sample_model_and_data):
        """Testa invariância a features específicas."""
        data = sample_model_and_data

        # Vamos testar se category_A_encoded é invariante (não deveria ser, é só exemplo)
        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.invariance_test(
            X=data['X_test'],
            invariant_features=['category_A_encoded'],
            n_samples=100
        )

        assert 'results_by_feature' in result
        assert 'category_A_encoded' in result['results_by_feature']
        assert 'prediction_change_rate' in result['results_by_feature']['category_A_encoded']

    def test_edge_case_handling(self, sample_model_and_data):
        """Testa handling de edge cases."""
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.edge_case_handling_test(X=data['X_test'])

        assert 'results' in result
        assert 'total_cases' in result
        assert result['total_cases'] > 0

        # Modelo deve conseguir fazer predições em edge cases
        assert result['passed'] >= result['total_cases'] * 0.75, (
            f"Model failed too many edge cases: {result['failed']}/{result['total_cases']}"
        )

    def test_prediction_stability(self, sample_model_and_data):
        """Testa estabilidade de predições."""
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.prediction_stability_test(
            X=data['X_test'],
            n_trials=10,
            sample_size=100
        )

        assert 'coefficient_of_variation' in result
        assert 'is_stable' in result

        # Predições devem ser razoavelmente estáveis
        assert result['coefficient_of_variation'] < 0.2, (
            f"Predictions are unstable: CV = {result['coefficient_of_variation']:.3f}"
        )

    def test_feature_range_validation(self, sample_model_and_data):
        """Testa se features de teste estão no range de treinamento."""
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.feature_range_validation_test(
            X_train=data['X_train'],
            X_test=data['X_test']
        )

        assert 'results_by_feature' in result
        assert 'features_with_issues' in result

        # Em dados sintéticos aleatórios, a maioria deve estar no range
        for feat, feat_result in result['results_by_feature'].items():
            assert feat_result['out_of_range_pct'] < 20, (
                f"Too many samples out of range for {feat}: "
                f"{feat_result['out_of_range_pct']:.1f}%"
            )

    def test_run_all_tests(self, sample_model_and_data):
        """Testa execução de todos os testes."""
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        results = checker.run_all_tests(
            X_train=data['X_train'],
            X_test=data['X_test']
        )

        assert 'summary' in results
        assert 'adversarial_perturbation' in results
        assert 'edge_case_handling' in results
        assert 'prediction_stability' in results
        assert 'feature_range_validation' in results

        # Pelo menos 80% dos testes devem passar
        summary = results['summary']
        pass_rate = summary['passed'] / summary['total_tests']
        assert pass_rate >= 0.8, (
            f"Too many robustness tests failed: {summary['failed']}/{summary['total_tests']}"
        )


class TestModelRobustnessInProduction:
    """
    Testes de robustez para modelos em produção.

    Estes testes devem ser executados antes de deploy.
    """

    def test_adversarial_robustness_threshold(self, sample_model_and_data):
        """
        Verifica se modelo é robusto o suficiente para produção.

        Threshold: <10% de predições devem mudar com perturbação de 10% do std.
        """
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.adversarial_perturbation_test(
            X=data['X_test'],
            epsilon=0.1,  # 10% do std
            n_samples=200
        )

        assert result['is_robust'], (
            f"Model is not robust enough for production: "
            f"{result['prediction_change_rate']:.2%} predictions changed "
            f"with epsilon={result['epsilon']}"
        )

    def test_edge_cases_must_not_crash(self, sample_model_and_data):
        """
        Garante que modelo não quebra em edge cases.

        CRÍTICO: Modelo deve sempre retornar predição, mesmo em edge cases.
        """
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.edge_case_handling_test(X=data['X_test'])

        # TODOS os edge cases devem passar (não crashear)
        assert result['failed'] == 0, (
            f"Model crashed on {result['failed']} edge cases. "
            f"All edge cases must be handled gracefully."
        )

    def test_prediction_consistency_requirement(self, sample_model_and_data):
        """
        Verifica consistência de predições em produção.

        Predições devem ser estáveis em amostras similares.
        """
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.prediction_stability_test(
            X=data['X_test'],
            n_trials=20,
            sample_size=150
        )

        # CV deve ser < 15% para produção
        assert result['coefficient_of_variation'] < 0.15, (
            f"Predictions are too unstable for production: "
            f"CV = {result['coefficient_of_variation']:.3f}"
        )

    def test_no_severe_extrapolation(self, sample_model_and_data):
        """
        Verifica que dados de teste não estão muito fora do range de treino.

        Extrapolação severa pode levar a predições não confiáveis.
        """
        data = sample_model_and_data

        checker = RobustnessChecker(
            model=data['model'],
            feature_names=data['feature_names']
        )

        result = checker.feature_range_validation_test(
            X_train=data['X_train'],
            X_test=data['X_test']
        )

        # Nenhuma feature deve ter >10% dos dados fora do range
        for feat, feat_result in result['results_by_feature'].items():
            assert feat_result['out_of_range_pct'] < 10, (
                f"Feature {feat} has {feat_result['out_of_range_pct']:.1f}% "
                f"of samples outside training range. This may indicate data drift."
            )


class TestModelStress:
    """Testes de stress para modelos."""

    def test_high_volume_prediction(self, sample_model_and_data):
        """
        Testa se modelo consegue processar alto volume de predições.

        Simula carga de produção.
        """
        data = sample_model_and_data

        # Criar dataset grande
        large_X = pd.concat([data['X_test']] * 100, ignore_index=True)

        # Tentar predizer
        try:
            predictions = data['model'].predict(large_X[data['feature_names']])
            assert len(predictions) == len(large_X)
        except Exception as e:
            pytest.fail(f"Model failed on high volume prediction: {e}")

    def test_concurrent_predictions(self, sample_model_and_data):
        """
        Testa se predições são thread-safe.

        Importante para APIs de produção com múltiplas requisições.
        """
        import threading

        data = sample_model_and_data
        results = []

        def make_prediction():
            try:
                pred = data['model'].predict(data['X_test'][data['feature_names']].head(10))
                results.append({'success': True, 'pred': pred})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})

        # Criar múltiplas threads
        threads = [threading.Thread(target=make_prediction) for _ in range(10)]

        # Executar
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verificar que todas tiveram sucesso
        assert all(r['success'] for r in results), (
            "Model is not thread-safe. Some predictions failed."
        )

        # Verificar que predições são consistentes
        first_pred = results[0]['pred']
        for r in results[1:]:
            np.testing.assert_array_equal(
                first_pred, r['pred'],
                err_msg="Predictions are not deterministic across threads"
            )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_robustness_report(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_path: str
) -> Dict:
    """
    Gera relatório completo de robustez para documentação.

    Args:
        model: Modelo treinado
        X_train: Dados de treinamento
        X_test: Dados de teste
        feature_names: Lista de features
        output_path: Caminho para salvar relatório

    Returns:
        Dict com resultados
    """
    import json

    checker = RobustnessChecker(model, feature_names)

    results = checker.run_all_tests(X_train, X_test)

    # Adicionar metadados
    results['metadata'] = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': len(feature_names),
        'features': feature_names
    }

    # Salvar
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results
