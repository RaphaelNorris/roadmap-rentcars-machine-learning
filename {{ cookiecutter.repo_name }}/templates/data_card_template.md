# Data Card: {{ cookiecutter.project_name }}

Template baseado em [Data Cards](https://arxiv.org/abs/2204.01075) e [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

## Dataset Overview

### Basic Information
- **Dataset Name**: [Nome do dataset]
- **Dataset Version**: [v1.0.0]
- **Creation Date**: [YYYY-MM-DD]
- **Last Updated**: [YYYY-MM-DD]
- **Dataset Owner**: {{ cookiecutter.author_name }}
- **Contact**: {{ cookiecutter.email }}

### Purpose
**Objetivo do dataset**:
[Descrição do propósito e caso de uso principal do dataset]

### Source
- **Origin**: [Database / API / Manual Collection / Web Scraping]
- **Location**: [S3 bucket / Database name / API endpoint]
- **Update Frequency**: [Real-time / Daily / Weekly / Monthly]

---

## Dataset Composition

### Size
- **Total Records**: [X registros]
- **Total Features**: [Y features]
- **File Size**: [Z GB]
- **Time Range**: [YYYY-MM-DD to YYYY-MM-DD]

### Features
| Feature Name | Type | Description | Missing % | Unique Values |
|--------------|------|-------------|-----------|---------------|
| customer_id | string | ID único do cliente | 0% | X |
| age | integer | Idade do cliente | 5% | - |
| gender | categorical | Gênero | 2% | 3 |
| ... | ... | ... | ... | ... |

### Target Variable
- **Name**: [target_column]
- **Type**: [Binary / Multi-class / Continuous]
- **Distribution**: [Balanceado / Desbalanceado - ratio X:Y]
- **Missing Values**: [X%]

### Data Distribution
**Distribuição por categoria/grupo**:
| Category | Count | Percentage |
|----------|-------|------------|
| Class A | X | XX% |
| Class B | Y | YY% |
| Class C | Z | ZZ% |

---

## Data Collection

### Collection Method
**Como os dados foram coletados**:
- [Descrição do processo de coleta]
- [Ferramentas utilizadas]
- [Período de coleta]

### Sampling Strategy
- **Strategy**: [Random / Stratified / Convenience / Census]
- **Sample Size**: [X% of population]
- **Sampling Bias**: [Descrever vieses conhecidos]

### Data Sources
1. **Source 1**: [Nome]
   - Type: [Database / API / File]
   - Frequency: [Daily]
   - Volume: [X records/day]

2. **Source 2**: [Nome]
   - Type: [...]
   - Frequency: [...]
   - Volume: [...]

---

## Data Quality

### Completeness
- **Overall Completeness**: [XX%]
- **Critical Features**: [Lista features críticas e completeness]
- **Missing Data Patterns**: [Random / Systematic]

### Accuracy
- **Validation Methods**: [Descrever métodos de validação]
- **Error Rate**: [X%]
- **Known Inaccuracies**: [Lista de problemas conhecidos]

### Consistency
- **Cross-field Validation**: [Regras aplicadas]
- **Temporal Consistency**: [Checks de consistência temporal]
- **Referential Integrity**: [Validações entre tabelas]

### Timeliness
- **Data Freshness**: [Atualizado há X horas/dias]
- **Latency**: [Tempo entre evento e disponibilidade]
- **Historical Coverage**: [Dados disponíveis desde YYYY-MM-DD]

### Quality Metrics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Completeness | XX% | >95% | Pass/Fail |
| Accuracy | XX% | >98% | Pass/Fail |
| Uniqueness | XX% | >99% | Pass/Fail |
| Consistency | XX% | >95% | Pass/Fail |

---

## Data Processing

### Cleaning Steps
1. **Remove Duplicates**: [Critério de duplicação]
2. **Handle Missing Values**:
   - Numeric: [Mean / Median / Forward fill / Drop]
   - Categorical: [Mode / "Unknown" / Drop]
3. **Outlier Treatment**: [IQR method / Z-score / Domain knowledge]
4. **Data Validation**: [Schema validation / Range checks]

### Transformations
1. **Feature Engineering**:
   - [Lista de features criadas]
   - [Lógica de criação]

2. **Encoding**:
   - Categorical: [One-Hot / Label / Target]
   - Text: [TF-IDF / Word2Vec / BERT]

3. **Scaling**:
   - [StandardScaler / MinMaxScaler / RobustScaler]
   - Applied to: [Lista de features]

### Derived Features
| Feature | Source Features | Formula/Logic | Purpose |
|---------|-----------------|---------------|---------|
| feature_1 | [source_cols] | [formula] | [purpose] |
| feature_2 | [source_cols] | [formula] | [purpose] |

---

## Privacy and Security

### Personal Data
- **Contains PII**: [Yes / No]
- **PII Fields**: [Lista de campos com PII]
- **PII Protection**: [Hashing / Encryption / Anonymization / Pseudonymization]

### Compliance
- **LGPD Compliant**: [Yes / No / In Progress]
- **GDPR Compliant**: [Yes / No / N/A]
- **Other Regulations**: [Lista outras regulações aplicáveis]

### Access Control
- **Who Can Access**: [Roles/teams com acesso]
- **Access Method**: [IAM / VPN / API Key]
- **Audit Logging**: [Enabled / Disabled]

### Data Retention
- **Retention Period**: [X meses/anos]
- **Deletion Policy**: [Automática / Manual]
- **Backup Strategy**: [Daily / Weekly]

---

## Bias and Fairness

### Known Biases
**Vieses identificados**:
1. **Selection Bias**: [Descrição]
2. **Sampling Bias**: [Descrição]
3. **Measurement Bias**: [Descrição]
4. **Historical Bias**: [Descrição]

### Demographic Representation
| Demographic | Dataset % | Population % | Representation |
|-------------|-----------|--------------|----------------|
| Gender Male | XX% | XX% | [Over/Under/Fair] |
| Gender Female | XX% | XX% | [Over/Under/Fair] |
| Age 18-25 | XX% | XX% | [Over/Under/Fair] |
| Region North | XX% | XX% | [Over/Under/Fair] |

### Mitigation Strategies
**Estratégias para mitigar vieses**:
1. [Estratégia 1]
2. [Estratégia 2]
3. [Estratégia 3]

---

## Limitations

### Known Issues
**Problemas conhecidos**:
1. **Issue 1**: [Descrição do problema]
   - Impact: [Alto / Médio / Baixo]
   - Mitigation: [O que foi feito]

2. **Issue 2**: [Descrição]
   - Impact: [...]
   - Mitigation: [...]

### Data Gaps
**Lacunas nos dados**:
- [Feature X não disponível para período Y]
- [Cobertura limitada para região Z]

### Temporal Limitations
- **Seasonality**: [Efeitos sazonais identificados]
- **Trend Changes**: [Mudanças de tendência]
- **Data Staleness**: [Dados podem ficar obsoletos após X meses]

### Recommended Uses
**Usos recomendados**:
- [Uso 1]
- [Uso 2]

### Not Recommended Uses
**Usos NÃO recomendados**:
- [Não usar para decisões médicas]
- [Não usar fora do contexto X]

---

## Versioning and Updates

### Version History
| Version | Date | Changes | Records Added | Records Removed |
|---------|------|---------|---------------|-----------------|
| v1.0.0 | YYYY-MM-DD | Initial version | X | 0 |
| v1.1.0 | YYYY-MM-DD | Added feature Y | Y | Z |

### Update Process
**Como o dataset é atualizado**:
1. [Passo 1: Extração de novos dados]
2. [Passo 2: Validação de qualidade]
3. [Passo 3: Merge com dados existentes]
4. [Passo 4: Versionamento]

### Deprecation Policy
- **Notification Period**: [30 dias antes da depreciação]
- **Support Period**: [90 dias após depreciação]
- **Migration Path**: [Como migrar para nova versão]

---

## Schema

### Database Schema
```sql
CREATE TABLE dataset_name (
    customer_id VARCHAR(50) PRIMARY KEY,
    feature_1 INTEGER NOT NULL,
    feature_2 FLOAT,
    feature_3 VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);
```

### JSON Schema
```json
{
  "type": "object",
  "properties": {
    "customer_id": {"type": "string"},
    "feature_1": {"type": "integer"},
    "feature_2": {"type": "number"},
    "feature_3": {"type": "string"}
  },
  "required": ["customer_id", "feature_1"]
}
```

### Parquet Schema
```
root
 |-- customer_id: string (nullable = false)
 |-- feature_1: long (nullable = true)
 |-- feature_2: double (nullable = true)
 |-- feature_3: string (nullable = true)
```

---

## Access and Usage

### How to Access
**Localização do dataset**:
- **S3 Path**: `s3://bucket-name/path/to/dataset/`
- **Athena Table**: `database_name.table_name`
- **API Endpoint**: `https://api.example.com/v1/dataset`

### Code Examples
```python
# Load from S3
import pandas as pd
df = pd.read_parquet('s3://bucket/dataset.parquet')

# Query from Athena
import awswrangler as wr
df = wr.athena.read_sql_query(
    "SELECT * FROM dataset_name WHERE date >= '2024-01-01'",
    database="ml_database"
)

# Load from feature store
from src.data.aws_integration import get_iceberg_manager
iceberg = get_iceberg_manager()
df = iceberg.read_features('feature_store')
```

### Dependencies
**Dependências necessárias**:
```
pandas>=2.0.0
pyarrow>=12.0.0
awswrangler>=3.5.0
```

---

## Statistics

### Descriptive Statistics
**Features Numéricas**:
| Feature | Mean | Median | Std | Min | Max | Q1 | Q3 |
|---------|------|--------|-----|-----|-----|----|----|
| age | XX | XX | XX | XX | XX | XX | XX |
| income | XX | XX | XX | XX | XX | XX | XX |

**Features Categóricas**:
| Feature | Mode | Unique Values | Top 5 Values |
|---------|------|---------------|--------------|
| category | X | Y | [A, B, C, D, E] |
| region | X | Y | [North, South, ...] |

### Correlations
**Top correlações** (|r| > 0.5):
- feature_1 ↔ feature_2: r = 0.XX
- feature_3 ↔ feature_4: r = 0.XX

---

## Monitoring

### Quality Monitoring
**Métricas monitoradas**:
- Completeness: Daily
- Data volume: Daily
- Schema changes: On change
- Distribution shifts: Weekly

### Alerts
| Alert | Threshold | Action |
|-------|-----------|--------|
| Completeness < 95% | Critical | Investigate source |
| Volume change > 50% | Warning | Review data pipeline |
| New nulls in critical fields | Critical | Block pipeline |

### Data Quality Dashboard
- **Location**: [Grafana dashboard URL]
- **Update Frequency**: Real-time
- **Owner**: [Team name]

---

## Related Resources

### Documentation
- [Model Card](./model_card_{{ cookiecutter.project_name }}.md)
- [Data Pipeline Documentation](../docs/pipelines.md)
- [Feature Engineering Guide](../notebooks/4-feat_eng/README.md)

### Tools
- **Profiling**: [pandas-profiling / ydata-profiling]
- **Validation**: [Great Expectations / Pandera]
- **Monitoring**: [Evidently / WhyLabs]

---

## Changelog

| Date | Author | Changes | Impact |
|------|--------|---------|--------|
| YYYY-MM-DD | [Nome] | Created data card | - |
| YYYY-MM-DD | [Nome] | Updated schema | Breaking change |

---

**Last Updated**: [YYYY-MM-DD]
**Next Review Date**: [YYYY-MM-DD]
**Status**: [Active / Deprecated]
**Maintainer**: {{ cookiecutter.author_name }}
