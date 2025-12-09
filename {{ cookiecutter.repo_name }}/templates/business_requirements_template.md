# Business Requirements Document
## {{ cookiecutter.project_name }}

**CRISP-DM Fase 1: Business Understanding**

## Document Control

| | |
|---|---|
| **Project Name** | {{ cookiecutter.project_name }} |
| **Document Owner** | {{ cookiecutter.author_name }} |
| **Contact** | {{ cookiecutter.email }} |
| **Version** | 1.0 |
| **Last Updated** | YYYY-MM-DD |
| **Status** | Draft / In Review / Approved |

---

## 1. Executive Summary

### 1.1 Business Problem
**Descrição do problema de negócio** (em linguagem não-técnica):

[Exemplo: A empresa está perdendo 20% dos clientes anualmente, resultando em perda de R$ XX milhões. Precisamos identificar quais clientes têm maior probabilidade de churnar para tomar ações preventivas.]

### 1.2 Proposed Solution
**Solução proposta**:

[Exemplo: Desenvolver um modelo de Machine Learning que preveja churn de clientes com 30 dias de antecedência, permitindo ações de retenção direcionadas.]

### 1.3 Expected Impact
**Impacto esperado**:

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Churn Rate | 20% | 15% | 6 meses |
| Revenue Loss | R$ XX M | R$ YY M | 6 meses |
| Retention Cost | R$ ZZ | R$ WW | 3 meses |

---

## 2. Business Context

### 2.1 Background
**Contexto do negócio**:

[Descrição do contexto empresarial, mercado, competição, etc.]

### 2.2 Current Situation
**Situação atual**:

- **As-Is Process**: [Descrição do processo atual]
- **Pain Points**: [Problemas atuais]
- **Workarounds**: [Soluções temporárias em uso]

### 2.3 Stakeholders
| Stakeholder | Role | Interest | Influence |
|-------------|------|----------|-----------|
| [Nome] | CEO | High | High |
| [Nome] | Marketing Director | High | Medium |
| [Nome] | Data Science Manager | Medium | High |
| [Nome] | IT Manager | Low | Medium |

---

## 3. Business Objectives

### 3.1 Primary Objectives
**Objetivos principais** (SMART - Specific, Measurable, Achievable, Relevant, Time-bound):

1. **Objetivo 1**: [Reduzir churn em 5 pontos percentuais em 6 meses]
   - Specific: Redução de churn de clientes premium
   - Measurable: De 20% para 15%
   - Achievable: Baseado em benchmarks da indústria
   - Relevant: Impacto direto em receita recorrente
   - Time-bound: 6 meses

2. **Objetivo 2**: [...]

### 3.2 Success Criteria
**Critérios de sucesso**:

| Criterion | Measurement | Target | Priority |
|-----------|-------------|--------|----------|
| Model Accuracy | Precision/Recall | Precision > 80% | P0 |
| Business Impact | Revenue saved | R$ XX M/year | P0 |
| User Adoption | % team usage | > 80% | P1 |
| ROI | Cost vs Benefit | > 300% | P0 |

### 3.3 Key Performance Indicators (KPIs)
**KPIs de negócio** (não confundir com métricas de ML):

1. **Revenue Impact**: [Receita salva/gerada]
2. **Cost Reduction**: [Redução de custos operacionais]
3. **Customer Satisfaction**: [NPS, CSAT]
4. **Operational Efficiency**: [Tempo economizado]
5. **Market Share**: [Aumento de market share]

---

## 4. Scope

### 4.1 In Scope
**Dentro do escopo**:

- [Item 1: Predição de churn para clientes premium]
- [Item 2: Dashboard de monitoramento]
- [Item 3: API de integração com CRM]

### 4.2 Out of Scope
**Fora do escopo**:

- [Item 1: Predição para clientes B2B]
- [Item 2: Ações automáticas de retenção]
- [Item 3: Integração com sistema legado X]

### 4.3 Assumptions
**Premissas**:

1. Dados históricos de pelo menos 2 anos estão disponíveis
2. Time de marketing pode agir sobre as predições
3. Budget de R$ XX está aprovado
4. Infraestrutura AWS está disponível

### 4.4 Constraints
**Restrições**:

1. **Técnicas**: [Modelo deve rodar em < 100ms]
2. **Regulatórias**: [Deve ser explicável para compliance]
3. **Orçamento**: [Máximo R$ XX mil]
4. **Prazo**: [6 meses para primeira versão]
5. **Recursos**: [2 data scientists, 1 eng]

---

## 5. Requirements

### 5.1 Functional Requirements

#### FR1: Prediction Capability
- **Description**: Sistema deve prever probabilidade de churn para cada cliente
- **Priority**: P0 (Must have)
- **Acceptance Criteria**:
  - Predição gerada para 100% dos clientes ativos
  - Atualização diária
  - Probabilidade entre 0 e 1

#### FR2: Explanation
- **Description**: Fornecer top 3 fatores que influenciam predição
- **Priority**: P0
- **Acceptance Criteria**:
  - Features com maior impacto identificadas
  - Explicação em linguagem de negócio
  - Disponível via API

#### FR3: Dashboard
- **Description**: Dashboard executivo com métricas principais
- **Priority**: P1 (Should have)
- **Acceptance Criteria**:
  - Atualização diária
  - Drill-down por segmento
  - Export para PDF

### 5.2 Non-Functional Requirements

#### NFR1: Performance
- **Latency**: < 100ms para predição individual
- **Throughput**: > 1000 predições/segundo
- **Batch Processing**: Completar em < 2 horas

#### NFR2: Availability
- **Uptime**: 99.9% (8.76 horas downtime/ano)
- **Recovery Time**: < 15 minutos

#### NFR3: Scalability
- **User Load**: Suportar 100 usuários concorrentes
- **Data Volume**: Processar até 10M clientes

#### NFR4: Security
- **Authentication**: SSO/LDAP
- **Authorization**: RBAC
- **Data Encryption**: At rest e in transit
- **Compliance**: LGPD

### 5.3 Data Requirements
**Dados necessários**:

| Data Source | Features | Update Frequency | Availability |
|-------------|----------|------------------|--------------|
| CRM | Customer demographics | Daily | Available |
| Transactions | Purchase history | Real-time | Available |
| Support | Tickets, interactions | Daily | Partial |
| Marketing | Campaign responses | Weekly | Available |

---

## 6. Use Cases

### UC1: Daily Churn Prediction
**Actor**: Marketing Analyst

**Preconditions**:
- User authenticated
- Fresh data available

**Flow**:
1. System runs daily batch prediction at 6 AM
2. Generates churn probability for all active customers
3. Sends high-risk list to CRM
4. Triggers email notification to marketing team

**Postconditions**:
- Predictions available in dashboard
- CRM updated with scores

**Business Value**: Enable proactive retention actions

### UC2: Individual Customer Check
**Actor**: Customer Success Manager

**Flow**:
1. CSM logs into dashboard
2. Searches for customer by ID/name
3. Views churn probability and factors
4. Decides on retention action

**Business Value**: Informed customer conversations

---

## 7. Business Rules

### BR1: Prediction Threshold
- Customers with probability > 0.7 = High Risk
- Customers with probability 0.4-0.7 = Medium Risk
- Customers with probability < 0.4 = Low Risk

### BR2: Action Rules
- High Risk: Direct contact by CSM + special offer
- Medium Risk: Email campaign + survey
- Low Risk: Monitor only

### BR3: Model Governance
- Model must be retrained if accuracy drops > 5%
- Manual review required for VIP customers (> R$ 100k/year)
- Predictions must be explainable to customer if requested

---

## 8. Data Science Translation

### 8.1 ML Problem Definition
**Tradução para problema de ML**:

- **Problem Type**: Binary Classification
- **Target Variable**: Churn (0/1) em próximos 30 dias
- **Features**: Customer demographics, transaction history, behavior
- **Prediction Horizon**: 30 dias
- **Model Update Frequency**: Mensal

### 8.2 Success Metrics (ML)
**Métricas técnicas que suportam objetivos de negócio**:

| Business Metric | ML Metric | Target | Rationale |
|-----------------|-----------|--------|-----------|
| Revenue saved | Precision | > 80% | Evitar custos com falsos positivos |
| Churn caught | Recall | > 70% | Capturar máximo de churns |
| ROI | F1-Score | > 0.75 | Balancear precision/recall |

### 8.3 Evaluation Strategy
- **Holdout Test Set**: 20% de dados mais recentes
- **Cross-Validation**: Time-series split, 5 folds
- **Baseline**: Current heuristic (30% precision)
- **Fairness**: Equal performance across customer segments

---

## 9. Timeline and Milestones

### Project Phases
| Phase | Duration | Deliverables | Dependencies |
|-------|----------|--------------|--------------|
| 1. Discovery | 2 weeks | Requirements doc | None |
| 2. Data Exploration | 3 weeks | EDA report | Data access |
| 3. Modeling | 6 weeks | Trained model | Features ready |
| 4. Evaluation | 2 weeks | Model report | Model trained |
| 5. Deployment | 3 weeks | Production API | Model approved |
| 6. Monitoring | Ongoing | Dashboards | Deployed |

### Key Milestones
- [ ] **M1**: Requirements approved (Week 2)
- [ ] **M2**: Data pipeline ready (Week 5)
- [ ] **M3**: Baseline model (Week 8)
- [ ] **M4**: Final model approved (Week 13)
- [ ] **M5**: Production deployment (Week 16)
- [ ] **M6**: Business impact validated (Week 24)

---

## 10. Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient data quality | Medium | High | Early data profiling, data quality gates |
| Model bias against segments | Low | Critical | Fairness testing, diverse training data |
| Low user adoption | High | High | Early stakeholder involvement, training |
| Concept drift | Medium | Medium | Continuous monitoring, auto-retraining |
| Regulatory changes | Low | High | Explainability built-in, legal review |

---

## 11. Budget and Resources

### 11.1 Team
| Role | Allocation | Duration | Cost |
|------|------------|----------|------|
| Data Scientist | 2 FTE | 4 months | R$ XX |
| ML Engineer | 1 FTE | 3 months | R$ XX |
| Data Engineer | 0.5 FTE | 2 months | R$ XX |
| Product Manager | 0.25 FTE | 6 months | R$ XX |

### 11.2 Infrastructure
| Resource | Type | Monthly Cost | Duration |
|----------|------|--------------|----------|
| AWS S3 | Storage | R$ XX | 12 months |
| AWS EC2 | Compute | R$ XX | 12 months |
| MLFlow | Tracking | R$ XX | 12 months |
| Monitoring | Tools | R$ XX | 12 months |

### 11.3 Total Budget
- **Personnel**: R$ XXX
- **Infrastructure**: R$ XXX
- **Tools/Software**: R$ XXX
- **Contingency (20%)**: R$ XXX
- **TOTAL**: R$ XXX

---

## 12. Success Measurement

### 12.1 Evaluation Plan
**Como medir sucesso**:

**Week 1-4**: Development metrics
- Model accuracy meeting targets
- Code quality passing gates

**Month 2-3**: Deployment metrics
- API latency < 100ms
- System uptime > 99.9%

**Month 4-6**: Business metrics
- Churn rate reduction
- ROI calculation

### 12.2 Go/No-Go Criteria
**Para aprovar deploy em produção**:

- [ ] Model precision > 80%
- [ ] Model recall > 70%
- [ ] No fairness issues detected
- [ ] Performance requirements met
- [ ] Security review approved
- [ ] Stakeholder sign-off

---

## 13. Change Management

### 13.1 Communication Plan
| Stakeholder | Frequency | Method | Content |
|-------------|-----------|--------|---------|
| Executives | Monthly | Email report | Business impact |
| Marketing | Weekly | Dashboard | Predictions, actions |
| Data Team | Daily | Slack | Technical updates |

### 13.2 Training Plan
- **Marketing Team**: 2-day workshop on using predictions
- **CS Team**: 1-day training on dashboard
- **Executives**: 2-hour overview presentation

### 13.3 Rollout Strategy
1. **Pilot** (Week 1-2): 10% of customers, selected segments
2. **Beta** (Week 3-4): 50% of customers, monitor closely
3. **Full Rollout** (Week 5+): 100% of customers

---

## 14. Governance

### 14.1 Decision Authority
| Decision Type | Decision Maker | Escalation Path |
|---------------|----------------|-----------------|
| Technical architecture | ML Lead | CTO |
| Feature prioritization | Product Manager | VP Product |
| Model deployment | Data Science Manager | VP Data |
| Budget changes | Project Sponsor | CFO |

### 14.2 Review Cadence
- **Weekly**: Team standup, progress review
- **Bi-weekly**: Stakeholder demo
- **Monthly**: Executive steering committee
- **Quarterly**: Business impact review

---

## 15. Appendices

### A. Glossary
| Term | Definition |
|------|------------|
| Churn | Customer canceling service in next 30 days |
| Precision | % of predicted churns that actually churned |
| Recall | % of actual churns that were predicted |

### B. References
- [Market Research Report](link)
- [Competitor Analysis](link)
- [Technical Feasibility Study](link)

### C. Approvals
| Name | Role | Signature | Date |
|------|------|-----------|------|
| [Nome] | Business Sponsor | | |
| [Nome] | Data Science Lead | | |
| [Nome] | Product Manager | | |

---

**Document Status**: [Draft / In Review / Approved]
**Next Review**: [YYYY-MM-DD]
**Contact**: {{ cookiecutter.email }}
