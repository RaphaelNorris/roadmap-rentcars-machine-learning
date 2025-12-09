# Model Card: {{ cookiecutter.project_name }}

Template baseado em [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993)

## Model Details

### Basic Information
- **Model Name**: [Nome do modelo]
- **Model Version**: [v1.0.0]
- **Model Date**: [Data de criação]
- **Model Type**: [Classification / Regression / Clustering / etc]
- **Model Architecture**: [Random Forest / XGBoost / Neural Network / etc]
- **Framework**: [scikit-learn / XGBoost / TensorFlow / PyTorch]
- **License**: [MIT / Apache 2.0 / Proprietary]

### Developers
- **Organization**: {{ cookiecutter.author_name }}
- **Contact**: {{ cookiecutter.email }}
- **Team Members**: [Lista de desenvolvedores]

### Model Versions
| Version | Date | Changes | Performance |
|---------|------|---------|-------------|
| v1.0.0  | YYYY-MM-DD | Initial release | Accuracy: 0.XX |
| v1.1.0  | YYYY-MM-DD | Added features X, Y | Accuracy: 0.XX |

---

## Intended Use

### Primary Intended Uses
**Descrição do caso de uso principal**:
- [Exemplo: Prever churn de clientes para ações de retenção]
- [Exemplo: Classificar transações fraudulentas em tempo real]

### Primary Intended Users
**Quem deve usar este modelo**:
- [Exemplo: Equipe de Customer Success]
- [Exemplo: Equipe de Prevenção a Fraudes]
- [Exemplo: Sistema automático de decisão]

### Out-of-Scope Use Cases
**Casos de uso NÃO recomendados**:
- [Exemplo: Não usar para decisões médicas]
- [Exemplo: Não usar fora do país X devido a regulações]
- [Exemplo: Não usar em dados anteriores a 2020]

---

## Factors

### Relevant Factors
**Fatores que podem afetar o desempenho do modelo**:
- **Demográficos**: Idade, gênero, localização geográfica
- **Temporais**: Sazonalidade, dia da semana, horário
- **Comportamentais**: Padrões de uso, frequência de transações
- **Externos**: Condições econômicas, eventos especiais

### Evaluation Factors
**Grupos usados na avaliação**:
- **Por gênero**: Masculino, Feminino, Não-binário
- **Por faixa etária**: 18-25, 26-35, 36-50, 50+
- **Por região**: Norte, Sul, Leste, Oeste
- **Por segmento**: Premium, Regular, Bronze

---

## Metrics

### Model Performance Metrics
**Métricas técnicas**:
| Metric | Overall | Grupo A | Grupo B | Grupo C |
|--------|---------|---------|---------|---------|
| Accuracy | 0.XX | 0.XX | 0.XX | 0.XX |
| Precision | 0.XX | 0.XX | 0.XX | 0.XX |
| Recall | 0.XX | 0.XX | 0.XX | 0.XX |
| F1-Score | 0.XX | 0.XX | 0.XX | 0.XX |
| AUC-ROC | 0.XX | 0.XX | 0.XX | 0.XX |

### Business Metrics
**Métricas de negócio**:
- **Revenue Impact**: [Aumento de X% na receita]
- **Cost Reduction**: [Redução de Y% em custos]
- **Customer Satisfaction**: [Aumento de Z pontos NPS]
- **Operational Efficiency**: [Redução de W% em tempo de processo]

### Decision Thresholds
**Thresholds de decisão**:
- **Default Threshold**: 0.5
- **High Precision Threshold**: 0.7 (para minimizar falsos positivos)
- **High Recall Threshold**: 0.3 (para maximizar detecção)

### Fairness Metrics
**Métricas de equidade entre grupos**:
| Group | Demographic Parity | Equal Opportunity | Predictive Parity |
|-------|-------------------|-------------------|-------------------|
| Grupo A | 0.XX | 0.XX | 0.XX |
| Grupo B | 0.XX | 0.XX | 0.XX |
| Grupo C | 0.XX | 0.XX | 0.XX |

**Análise de Fairness**:
- [Descrever se há disparidades significativas]
- [Explicar ações tomadas para mitigar]

---

## Training Data

### Datasets
**Fonte de dados**:
- **Nome**: [Nome do dataset]
- **Período**: [YYYY-MM-DD a YYYY-MM-DD]
- **Tamanho**: [X registros, Y features]
- **Fonte**: [S3, Database, API, etc]

### Data Processing
**Pré-processamento aplicado**:
1. Remoção de duplicatas
2. Tratamento de valores missing:
   - Features numéricas: [Média / Mediana / Forward fill]
   - Features categóricas: [Moda / Categoria "Unknown"]
3. Feature engineering: [Lista de features criadas]
4. Feature scaling: [StandardScaler / MinMaxScaler / None]
5. Encoding: [One-Hot / Label / Target]

### Data Splits
- **Training**: 70% (X registros)
- **Validation**: 15% (Y registros)
- **Test**: 15% (Z registros)
- **Split Strategy**: [Temporal / Random Stratified / K-Fold]

### Data Quality
- **Completeness**: X% de dados completos
- **Consistency**: Validações aplicadas
- **Timeliness**: Atualizado até [Data]
- **Known Issues**: [Lista de problemas conhecidos]

---

## Evaluation Data

### Test Dataset
- **Source**: [Mesmo dataset / Dataset separado]
- **Period**: [YYYY-MM-DD a YYYY-MM-DD]
- **Size**: [X registros]
- **Distribution**: [Mesma / Diferente da training]

### Evaluation Scenarios
**Cenários testados**:
1. **Cenário Normal**: Dados típicos do dia-a-dia
2. **Cenário Edge Case**: Casos raros mas importantes
3. **Cenário Adversarial**: Dados potencialmente problemáticos
4. **Cenário Drift**: Simulação de drift temporal

---

## Model Architecture

### Algorithm
**Algoritmo**: [Random Forest / XGBoost / Neural Network]

**Hyperparameters**:
```python
{
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1,
    "min_samples_split": 2,
    ...
}
```

### Features
**Features utilizadas** (Top 20 por importância):
1. feature_name_1 - Descrição - Importância: 0.XX
2. feature_name_2 - Descrição - Importância: 0.XX
3. ...

### Model Artifacts
- **Model File**: `models/model_v1.pkl` (Size: X MB)
- **Preprocessor**: `models/preprocessor_v1.pkl`
- **Feature List**: `models/features_v1.json`
- **MLFlow Run ID**: [run_id]
- **Model Registry URI**: `models:/model_name/Production`

---

## Ethical Considerations

### Potential Risks
**Riscos identificados**:
1. **Bias Risk**: [Descrever riscos de viés]
2. **Privacy Risk**: [Descrever riscos de privacidade]
3. **Security Risk**: [Descrever riscos de segurança]
4. **Misuse Risk**: [Descrever riscos de uso indevido]

### Mitigation Strategies
**Estratégias de mitigação**:
1. [Ação tomada para mitigar risco 1]
2. [Ação tomada para mitigar risco 2]
3. [Monitoramento contínuo de fairness]
4. [Processo de revisão humana para casos críticos]

### Data Privacy
- **PII Handling**: [Como dados pessoais são tratados]
- **LGPD/GDPR Compliance**: [Sim/Não - Detalhes]
- **Data Retention**: [Política de retenção]
- **Right to Explanation**: [Como explicações são fornecidas]

---

## Limitations

### Known Limitations
**Limitações conhecidas**:
1. **Performance**: [Descrever limitações de performance]
   - Exemplo: Accuracy menor que 0.8 para grupo X
2. **Coverage**: [Descrever limitações de cobertura]
   - Exemplo: Não funciona bem para região Y
3. **Temporal**: [Descrever limitações temporais]
   - Exemplo: Performance degrada após 6 meses
4. **Edge Cases**: [Casos especiais onde modelo falha]

### Recommendations
**Recomendações de uso**:
1. **Monitoring**: Monitorar drift a cada 7 dias
2. **Retraining**: Retreinar a cada 3 meses ou quando drift > threshold
3. **Human Review**: Revisar manualmente casos com probabilidade entre 0.4-0.6
4. **Fallback**: Ter processo manual para casos onde modelo não é confiável

---

## Monitoring and Maintenance

### Monitoring Strategy
**O que é monitorado**:
- **Model Performance**: Métricas diárias de accuracy, precision, recall
- **Data Drift**: Distribution shift nas features
- **Prediction Drift**: Mudanças na distribuição de predições
- **Business Metrics**: KPIs de negócio afetados pelo modelo

### Alerts
**Alertas configurados**:
| Alert | Threshold | Action |
|-------|-----------|--------|
| Accuracy drop | < 0.75 | Investigate + notify team |
| Data drift | > 0.05 | Review data quality |
| Prediction volume | > 2x normal | Check system |
| Latency | > 100ms p95 | Scale infrastructure |

### Retraining Schedule
- **Frequency**: [Mensal / Trimestral / Trigger-based]
- **Triggers**:
  - Accuracy drop > 5%
  - Drift score > threshold
  - New data available > X% of training set
- **Approval Process**: [Descrever processo de aprovação]

---

## Deployment

### Infrastructure
- **Environment**: [AWS / GCP / Azure / On-premise]
- **Compute**: [EC2 t3.medium / Lambda / ECS]
- **Storage**: [S3 / EBS]
- **Serving**: [FastAPI / SageMaker / Custom]

### Performance Requirements
- **Latency**: < 100ms p95
- **Throughput**: > 1000 requests/second
- **Availability**: 99.9%
- **Error Rate**: < 0.1%

### Rollout Strategy
- **Strategy**: [Blue/Green / Canary / Rolling]
- **Rollback Plan**: [Automatic rollback se error rate > 1%]
- **Monitoring Period**: [24 horas antes de full deployment]

---

## Responsible AI Checklist

### Fairness
- [ ] Model tested across demographic groups
- [ ] Fairness metrics calculated and documented
- [ ] Disparities identified and mitigated
- [ ] Ongoing fairness monitoring implemented

### Transparency
- [ ] Model Card completed and reviewed
- [ ] Feature importance documented
- [ ] Decision process explainable
- [ ] Stakeholders informed of limitations

### Privacy
- [ ] PII handling documented
- [ ] Data minimization applied
- [ ] Compliance verified (LGPD/GDPR)
- [ ] Data retention policy defined

### Security
- [ ] Model secured against adversarial attacks
- [ ] Access controls implemented
- [ ] Audit logging enabled
- [ ] Incident response plan defined

### Accountability
- [ ] Model owner identified
- [ ] Review process established
- [ ] Escalation path defined
- [ ] Documentation maintained

---

## Change Log

| Version | Date | Author | Changes | Review |
|---------|------|--------|---------|--------|
| v1.0.0 | YYYY-MM-DD | [Nome] | Initial model | [Revisor] |
| v1.1.0 | YYYY-MM-DD | [Nome] | Added feature X | [Revisor] |

---

## References

### Related Documents
- [Data Card](./data_card_{{ cookiecutter.project_name }}.md)
- [Business Requirements](./business_requirements.md)
- [Technical Documentation](../docs/mlops_architecture.md)
- [Experiment Log](../notebooks/experiments/experiment_log.md)

### External References
- [Model Cards Paper](https://arxiv.org/abs/1810.03993)
- [ML Fairness Guidelines](https://developers.google.com/machine-learning/fairness-overview)
- [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)

---

**Last Updated**: [YYYY-MM-DD]
**Next Review Date**: [YYYY-MM-DD]
**Status**: [Development / Staging / Production]
