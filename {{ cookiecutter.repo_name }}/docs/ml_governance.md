# ML Governance Framework
# {{ cookiecutter.project_name }}

Framework completo de governança para Machine Learning seguindo princípios de Responsible AI, CRISP-DM e CD4ML.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Governance Principles](#2-governance-principles)
3. [Roles and Responsibilities](#3-roles-and-responsibilities)
4. [Model Lifecycle Governance](#4-model-lifecycle-governance)
5. [Data Governance](#5-data-governance)
6. [Model Documentation](#6-model-documentation)
7. [Testing and Validation](#7-testing-and-validation)
8. [Deployment and Monitoring](#8-deployment-and-monitoring)
9. [Responsible AI](#9-responsible-ai)
10. [Compliance and Audit](#10-compliance-and-audit)

---

## 1. Overview

### 1.1 Purpose

Este documento define o framework de governança para todos os modelos de Machine Learning desenvolvidos e deployed no projeto {{ cookiecutter.project_name }}.

**Objetivos**:
- Garantir qualidade e confiabilidade dos modelos
- Assegurar compliance regulatório (LGPD, etc)
- Implementar princípios de Responsible AI
- Facilitar auditoria e rastreabilidade
- Mitigar riscos de negócio e técnicos

### 1.2 Scope

Este framework aplica-se a:
- Todos os modelos ML em desenvolvimento
- Modelos em staging e produção
- Dados usados para treinamento e inferência
- Pipelines e infraestrutura de MLOps

### 1.3 Governance Stack

```
┌─────────────────────────────────────────────────────────┐
│                 GOVERNANCE FRAMEWORK                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Data        │  │    Model     │  │  Deployment  │ │
│  │  Governance   │  │  Governance  │  │  Governance  │ │
│  └───────────────┘  └──────────────┘  └──────────────┘ │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │            Responsible AI Framework                │  │
│  │  - Fairness  - Transparency  - Privacy  - Safety  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Compliance & Audit Framework               │  │
│  │      - LGPD  - SOC2  - ISO27001  - GDPR           │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Governance Principles

### 2.1 Core Principles

#### Principle 1: Transparency

**Definição**: Todos os modelos devem ser explicáveis e documentados.

**Implementação**:
- Model Cards obrigatórios para todos os modelos
- Experiment logs detalhados
- Feature importance documentada
- Decision logic compreensível

#### Principle 2: Fairness

**Definição**: Modelos não devem discriminar grupos protegidos.

**Implementação**:
- Testes de fairness obrigatórios (test_fairness.py)
- Métricas de equidade monitoradas
- Análise de viés em dados e predições
- Mitigação ativa de discriminação

#### Principle 3: Accountability

**Definição**: Responsabilidade clara por decisões de modelos.

**Implementação**:
- Ownership definido no Model Card
- Approval workflow para deployment
- Audit trail completo
- Incident response plan

#### Principle 4: Privacy

**Definição**: Dados pessoais protegidos conforme regulações.

**Implementação**:
- PII handling documentado no Data Card
- Anonimização/pseudonimização quando aplicável
- Compliance LGPD/GDPR
- Data retention policies

#### Principle 5: Safety

**Definição**: Modelos são robustos e seguros.

**Implementação**:
- Testes de robustez obrigatórios (test_robustness.py)
- Adversarial testing
- Fallback mechanisms
- Automated rollback em caso de falha

---

## 3. Roles and Responsibilities

### 3.1 Governance Roles

| Role | Responsibilities | Authority |
|------|------------------|-----------|
| **Model Owner** | - Qualidade do modelo<br>- Documentação<br>- Maintenance | - Aprovar mudanças no modelo<br>- Decidir deprecação |
| **Data Steward** | - Qualidade de dados<br>- Data Cards<br>- Compliance de dados | - Aprovar uso de dados<br>- Definir retention policies |
| **ML Engineer** | - Deployment<br>- Monitoring<br>- Infrastructure | - Deploy para staging<br>- Configurar monitoring |
| **Compliance Officer** | - Regulatory compliance<br>- Audits<br>- Risk assessment | - Vetar deployments não-compliant<br>- Exigir correções |
| **Business Stakeholder** | - Definir requisitos<br>- Aprovar impacto de negócio | - Aprovar deployment para produção<br>- Definir SLAs |

### 3.2 RACI Matrix

**Deployment de Modelo para Produção**:

| Task | Data Scientist | ML Engineer | Model Owner | Business Stakeholder | Compliance Officer |
|------|----------------|-------------|-------------|----------------------|--------------------|
| Treinar modelo | R | C | C | I | - |
| Testes técnicos | R | C | A | - | I |
| Testes de fairness | R | C | A | - | C |
| Model Card | R | C | A | I | C |
| Deploy staging | C | R | A | I | - |
| Validação em staging | C | R | C | C | - |
| Aprovação para prod | C | C | R | A | C |
| Deploy produção | C | R | C | I | - |
| Monitoring | I | R | C | I | - |

**Legend**: R=Responsible, A=Accountable, C=Consulted, I=Informed

---

## 4. Model Lifecycle Governance

### 4.1 Development Phase

#### Requirements

- [ ] Business Requirements Document completed
- [ ] Data Card created for training data
- [ ] Experiment plan documented
- [ ] Success criteria defined (business + technical)

#### Approval Gate

**Who approves**: Model Owner + Business Stakeholder

**Criteria**:
- ROI estimado positivo
- Alinhamento com objetivos estratégicos
- Recursos disponíveis (time, budget, data)

### 4.2 Experimentation Phase

#### Requirements

- [ ] Experiments logged no MLFlow
- [ ] Baseline model documented
- [ ] Statistical significance tested
- [ ] Experiment guidelines seguidas

#### Approval Gate

**Who approves**: Model Owner

**Criteria**:
- Melhoria estatisticamente significativa vs baseline
- Métricas de negócio projetadas alcançadas
- Sem degradação em fairness

### 4.3 Validation Phase

#### Requirements

- [ ] Model Card completed
- [ ] Fairness tests passed (test_fairness.py)
- [ ] Robustness tests passed (test_robustness.py)
- [ ] Performance tests passed
- [ ] Security review completed

#### Approval Gate

**Who approves**: Model Owner + Compliance Officer

**Criteria**:
- Todos os testes automatizados passaram
- Demographic parity < 10%
- Equal opportunity < 10%
- Disparate impact ratio >= 0.8
- Robustez a perturbações confirmada
- Sem vulnerabilidades críticas de segurança

### 4.4 Staging Deployment

#### Requirements

- [ ] Modelo registrado no MLFlow Model Registry (stage: Staging)
- [ ] Canary deployment plan documented
- [ ] Rollback plan ready
- [ ] Monitoring dashboards configured

#### Approval Gate

**Who approves**: ML Engineer + Model Owner

**Criteria**:
- Testes de integração passaram
- API latency < SLA
- Infrastructure provisioned

### 4.5 Production Deployment

#### Requirements

- [ ] Staging validation successful (min 2 semanas)
- [ ] A/B test results positive (se aplicável)
- [ ] Business stakeholder sign-off
- [ ] Compliance review approved
- [ ] Incident response plan documented

#### Approval Gate

**Who approves**: Business Stakeholder + Model Owner + Compliance Officer

**Criteria**:
- Métricas de negócio melhoraram em staging
- Nenhum incidente crítico em staging
- Compliance requirements atendidos
- Budget de produção aprovado

### 4.6 Monitoring and Maintenance

#### Requirements

- [ ] Model performance monitored (daily)
- [ ] Data drift monitored (weekly)
- [ ] Business metrics tracked
- [ ] Fairness metrics monitored
- [ ] Incident response ready

#### Review Cadence

- **Daily**: Performance metrics
- **Weekly**: Data drift, business metrics
- **Monthly**: Comprehensive model review
- **Quarterly**: Governance compliance audit

### 4.7 Deprecation

#### Triggers

- Performance degradation não corrigível
- Modelo substituído por versão melhor
- Requisitos de negócio mudaram
- Compliance issues irremediáveis

#### Process

1. Notificação de deprecação (30 dias antecedência)
2. Migração de stakeholders para novo modelo
3. Arquivamento de modelo e documentação
4. Remoção de infraestrutura

---

## 5. Data Governance

### 5.1 Data Quality Standards

| Dimension | Threshold | Measurement | Action if Below |
|-----------|-----------|-------------|-----------------|
| Completeness | >= 95% | % non-null values | Investigate missing data |
| Accuracy | >= 98% | % correct values | Data quality audit |
| Consistency | >= 95% | Cross-field validation pass rate | Fix data pipeline |
| Timeliness | <= 24h latency | Time from event to availability | Optimize ETL |
| Uniqueness | >= 99% | % duplicate-free | Deduplication |

### 5.2 Data Card Requirements

**Obrigatório para todos os datasets**:

- [ ] Dataset Overview (purpose, size, timerange)
- [ ] Data Composition (features, target, distribution)
- [ ] Data Collection (method, sources)
- [ ] Data Quality (completeness, accuracy, issues)
- [ ] Privacy & Security (PII handling, LGPD compliance)
- [ ] Bias & Fairness (known biases, mitigation)
- [ ] Limitations (gaps, recommended/not-recommended uses)
- [ ] Schema (SQL/JSON/Parquet)
- [ ] Versioning (version history, update process)

### 5.3 PII Handling Policy

| PII Type | Handling | Retention | Access Control |
|----------|----------|-----------|----------------|
| Nome | Pseudonimização | 2 anos | Role: Data Steward |
| CPF | Hash SHA-256 | 2 anos | Role: Data Steward |
| Email | Encryption | 2 anos | Role: Data Steward |
| IP Address | Anonymização | 90 dias | Role: Security Team |
| Biometrics | **PROIBIDO** | N/A | N/A |

---

## 6. Model Documentation

### 6.1 Required Documentation

**Fase de Desenvolvimento**:
- Business Requirements Document
- Experiment Log (notebooks/experiments/)
- Data Card

**Fase de Deployment**:
- Model Card (completo)
- Deployment Guide
- Monitoring Guide
- Incident Response Plan

**Ongoing**:
- Performance Reports (monthly)
- Governance Audit Reports (quarterly)
- Incident Post-Mortems

### 6.2 Model Card Requirements

**Seções obrigatórias**:

1. Model Details (name, version, type, architecture)
2. Intended Use (primary uses, users, out-of-scope)
3. Factors (demographic, temporal, external)
4. Metrics (performance, business, fairness)
5. Training Data (source, processing, quality)
6. Evaluation Data (test sets, scenarios)
7. Ethical Considerations (risks, mitigation, privacy)
8. Limitations (known issues, recommendations)
9. Monitoring (strategy, alerts, retraining)
10. Deployment (infrastructure, performance req)
11. Responsible AI Checklist (fairness, transparency, privacy, security, accountability)

### 6.3 Documentation Review Cycle

- **Model Card**: Atualizado a cada nova versão de modelo
- **Data Card**: Revisado mensalmente ou quando dados mudam
- **Business Requirements**: Revisado trimestralmente
- **Governance Documents**: Revisado anualmente

---

## 7. Testing and Validation

### 7.1 Automated Testing Requirements

**Todos os modelos devem passar**:

| Test Suite | Tool | Threshold | Frequency |
|-------------|------|-----------|-----------|
| Unit Tests | pytest | 100% pass | Pre-commit |
| Integration Tests | pytest | 100% pass | CI/CD |
| Performance Tests | pytest | Latency < 100ms p95 | Pre-deploy |
| **Fairness Tests** | test_fairness.py | Demographic parity < 10% | Pre-deploy |
| **Robustness Tests** | test_robustness.py | Adversarial change < 10% | Pre-deploy |
| Data Quality Tests | pandera/great-expectations | 100% pass | Daily |
| Security Tests | bandit, safety | No critical issues | Pre-deploy |

### 7.2 Manual Review Requirements

**Antes de produção**:

- [ ] Code review (2 reviewers)
- [ ] Model Card review (Compliance Officer)
- [ ] Business impact review (Business Stakeholder)
- [ ] Security review (Security Team)
- [ ] Fairness review (Compliance Officer ou Ethics Committee)

---

## 8. Deployment and Monitoring

### 8.1 Deployment Strategy

**Progressiverollout obrigatório para todos os modelos**:

1. Deploy Staging (100% traffic)
   - Duração: Min 2 semanas
   - Validação: Performance, fairness, business metrics

2. Canary Deployment (5% → 10% → 25% → 50% → 100%)
   - Cada estágio: 30 minutos (ou configurável)
   - Automated rollback se métricas degradarem
   - Script: `scripts/canary_deployment.py`

3. A/B Testing (opcional mas recomendado)
   - Duração: 2-4 semanas
   - Análise de significância estatística

### 8.2 Monitoring Requirements

**Métricas Técnicas** (Prometheus + Grafana):
- API latency (p50, p95, p99)
- Error rate
- Throughput (requests/second)
- Model accuracy (online, se labels disponíveis)
- Data drift score
- Prediction drift

**Métricas de Negócio**:
- Revenue impact
- Cost reduction
- Customer satisfaction (NPS/CSAT)
- ROI
- Churn rate (ou métrica relevante ao negócio)

**Métricas de Fairness**:
- Demographic parity por grupo protegido
- Equal opportunity
- Disparate impact ratio

**SLAs**:
- Uptime: >= 99.9%
- Latency p95: <= 100ms
- Error rate: <= 0.1%

### 8.3 Alerting

**Severidade Critical** (PagerDuty + Email + Slack):
- Error rate > 5%
- Latency p95 > 200ms
- Model accuracy drop > 10%
- Fairness violation (demographic parity > 15%)

**Severidade Warning** (Slack + Email):
- Error rate > 1%
- Latency p95 > 150ms
- Accuracy drop > 5%
- Data drift score > threshold

**Severidade Info** (Slack):
- Deployment events
- Model version changes
- Retraining triggers

---

## 9. Responsible AI

### 9.1 Responsible AI Principles

Seguimos os [Google AI Principles](https://ai.google/principles/):

1. **Be socially beneficial**: Modelos devem criar valor para sociedade
2. **Avoid creating or reinforcing unfair bias**: Testar e mitigar viés
3. **Be built and tested for safety**: Testes de robustez obrigatórios
4. **Be accountable to people**: Human-in-the-loop quando apropriado
5. **Incorporate privacy design principles**: Privacy by design
6. **Uphold high standards of scientific excellence**: Rigor científico
7. **Be made available for uses that accord with these principles**: Uso ético

### 9.2 Fairness Framework

**Grupos Protegidos** (conforme LGPD e boas práticas):
- Gênero
- Raça/etnia
- Idade
- Deficiência
- Orientação sexual
- Religião
- Origem geográfica

**Métricas de Fairness Obrigatórias**:
- Demographic Parity: max diff <= 10%
- Equal Opportunity: TPR diff <= 10%
- Disparate Impact: ratio >= 0.8

**Processo de Mitigação**:
1. Identificar viés nos dados (Data Card)
2. Testar modelo (test_fairness.py)
3. Se falhar: aplicar técnicas de mitigação
   - Pre-processing: Reweighting, Sampling
   - In-processing: Adversarial debiasing, Fairness constraints
   - Post-processing: Threshold optimization per group
4. Re-testar
5. Documentar no Model Card

### 9.3 Explainability

**Requisitos de Explicabilidade**:

- **Global Explainability**: Feature importance documentada
- **Local Explainability**: SHAP/LIME para predições individuais
- **Model Interpretability**: Preferir modelos interpretáveis quando possível

**Quando explicação é obrigatória**:
- Decisões de alto impacto (crédito, contratação, saúde)
- Regulações exigem (LGPD Art. 20)
- Stakeholder request

**Ferramentas**:
- SHAP: `shap.TreeExplainer`, `shap.DeepExplainer`
- LIME: `lime.LimeTabularExplainer`
- Feature Importance: Nativo de tree-based models

### 9.4 Privacy

**Privacy by Design**:

1. **Data Minimization**: Coletar apenas dados necessários
2. **Purpose Limitation**: Usar dados apenas para propósito declarado
3. **Anonymization**: Anonimizar quando possível
4. **Encryption**: At rest e in transit
5. **Access Control**: RBAC estrito
6. **Audit Logging**: Log todos os acessos a dados sensíveis
7. **Right to Deletion**: Implementar LGPD Art. 18

**Técnicas**:
- Differential Privacy (quando aplicável)
- K-anonymity
- Pseudonymization
- Encryption (AES-256)

---

## 10. Compliance and Audit

### 10.1 Regulatory Compliance

**LGPD (Lei Geral de Proteção de Dados)**:

- [ ] Dados pessoais identificados e catalogados
- [ ] Base legal definida para processamento
- [ ] Consent management implementado (se aplicável)
- [ ] Right to access implementado
- [ ] Right to deletion implementado
- [ ] Data breach notification process definido
- [ ] DPO (Data Protection Officer) designado

**GDPR** (se aplicável):

- Similar a LGPD + requisitos adicionais

**SOC 2** (se aplicável):

- Security controls documentados
- Access controls implementados
- Change management process
- Monitoring e alerting

### 10.2 Audit Trail

**Rastreabilidade completa**:

```
Model Version X
├── Business Requirements: docs/requirements_v1.md
├── Data Card: data_card_v2.parquet
├── Experiments: mlflow.org/experiments/123
│   ├── Run 1: accuracy=0.80
│   ├── Run 2: accuracy=0.82
│   └── Run 3: accuracy=0.85 ← Selected
├── Model Card: model_card_v3.md
├── Tests:
│   ├── Fairness: PASSED (2024-01-15)
│   ├── Robustness: PASSED (2024-01-15)
│   └── Performance: PASSED (2024-01-15)
├── Approvals:
│   ├── Model Owner: John Doe (2024-01-16)
│   ├── Compliance: Jane Smith (2024-01-16)
│   └── Business: Bob Johnson (2024-01-17)
├── Deployment:
│   ├── Staging: 2024-01-18 (SUCCESS)
│   ├── Canary: 2024-01-25 (SUCCESS)
│   └── Production: 2024-01-25 (ACTIVE)
└── Monitoring:
    ├── Accuracy: 0.84 (7-day avg)
    ├── Fairness: COMPLIANT
    └── Business Impact: +R$ 500k revenue
```

### 10.3 Quarterly Governance Audit

**Checklist**:

- [ ] Todos os modelos em produção têm Model Cards atualizados
- [ ] Todos os datasets têm Data Cards
- [ ] Testes de fairness passando
- [ ] Testes de robustez passando
- [ ] Monitoring ativo e funcional
- [ ] Incident response testado
- [ ] Compliance requirements atendidos
- [ ] Nenhum alerta crítico não resolvido
- [ ] Documentation atualizada
- [ ] Training de equipe realizado

**Report para**:
- Executive Leadership
- Compliance Officer
- Board of Directors (se aplicável)

---

## 11. Incident Response

### 11.1 Incident Types

**Type 1: Performance Degradation**
- Accuracy drop > 10%
- Latency > 2x SLA
- Error rate > 5%

**Type 2: Fairness Violation**
- Demographic parity > 15%
- Disparate impact < 0.7

**Type 3: Security Breach**
- Unauthorized access
- Data leak
- Model manipulation

**Type 4: Compliance Violation**
- LGPD violation
- Regulatory non-compliance

### 11.2 Incident Response Process

```
1. DETECT
   ├── Automated monitoring alerts
   └── Manual reporting

2. TRIAGE
   ├── Assess severity (P0, P1, P2, P3)
   ├── Assign incident commander
   └── Notify stakeholders

3. INVESTIGATE
   ├── Collect logs
   ├── Analyze root cause
   └── Document findings

4. MITIGATE
   ├── Immediate: Rollback or traffic reduction
   ├── Short-term: Patch or hotfix
   └── Long-term: Permanent fix

5. RECOVER
   ├── Validate fix
   ├── Gradual rollout
   └── Resume normal operations

6. POST-MORTEM
   ├── Write incident report
   ├── Identify improvements
   ├── Update runbooks
   └── Share learnings
```

### 11.3 Incident Severity Levels

| Severity | Definition | Response Time | Escalation |
|----------|------------|---------------|------------|
| **P0 - Critical** | Produção down, data breach, compliance violation | Immediate | CEO, Legal |
| **P1 - High** | Performance degradation > 20%, fairness violation | < 1 hour | VP Engineering |
| **P2 - Medium** | Performance degradation 10-20%, non-critical bug | < 4 hours | Engineering Manager |
| **P3 - Low** | Minor issue, não afeta produção | < 24 hours | Team Lead |

---

## 12. Continuous Improvement

### 12.1 Governance Review Cycle

- **Monthly**: Métricas de governança revisadas
- **Quarterly**: Audit completo, report para leadership
- **Annually**: Framework de governança atualizado

### 12.2 Training and Education

**Obrigatório para toda a equipe**:
- ML Ethics training (anual)
- LGPD/GDPR training (anual)
- Security awareness (semestral)
- MLOps best practices (trimestral)

### 12.3 Framework Evolution

Este framework é vivo e deve evoluir:
- Novas regulações → Atualizar compliance
- Novas ferramentas → Atualizar stack
- Lições aprendidas → Atualizar processos
- Feedback da equipe → Melhorar usabilidade

---

## Appendices

### A. Templates

- [Business Requirements Template](../templates/business_requirements_template.md)
- [Data Card Template](../templates/data_card_template.md)
- [Model Card Template](../templates/model_card_template.md)

### B. Tools and Scripts

- Fairness Tests: `tests/model/test_fairness.py`
- Robustness Tests: `tests/model/test_robustness.py`
- Canary Deployment: `scripts/canary_deployment.py`
- Business Metrics: `src/inference/business_metrics.py`

### C. References

- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)
- [CD4ML by ThoughtWorks](https://martinfowler.com/articles/cd4ml.html)
- [Model Cards Paper](https://arxiv.org/abs/1810.03993)
- [Google AI Principles](https://ai.google/principles/)
- [LGPD](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)

---

**Última Atualização**: YYYY-MM-DD
**Versão**: 1.0
**Mantido por**: {{ cookiecutter.author_name }}
**Aprovado por**: [Compliance Officer, CTO]
