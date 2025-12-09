
# {{ cookiecutter.project_name }}

## 🧠 Arquitetura do Pipeline de Machine Learning

Este projeto implementa um pipeline completo de ciência de dados e machine learning, estruturado com base em duas metodologias amplamente reconhecidas:

- **CRISP-DM** (Cross Industry Standard Process for Data Mining): orienta a lógica analítica e investigativa do processo.
- **CD4ML** (Continuous Delivery for Machine Learning): garante robustez, reprodutibilidade e escalabilidade da entrega de modelos em produção.

### 🔁 Etapas baseadas em CRISP-DM

| Fase CRISP-DM              | Etapas correspondentes no pipeline                                 |
|---------------------------|---------------------------------------------------------------------|
| Entendimento do Negócio   | Requirements, Data Understand, Business Rules                      |
| Entendimento dos Dados    | EDA, validação inicial com apoio da engenharia de dados            |
| Preparação dos Dados      | Data Processing, Feature Engineering, Feature Store                |
| Modelagem                 | Train, Test, Experimentation, Model Tuning                         |
| Avaliação                 | Evaluation & Best Model Selection                                  |
| Deploy                    | Logging Artefacts, Register Model, Load Champion, Inference Model  |
| Monitoramento             | Model Monitoring (Data, Label, Concept Drift)                      |

### 🚀 Princípios de Continuous Delivery for ML (CD4ML)

A arquitetura implementa os principais pilares de CD4ML:

- **Pipelines desacoplados e versionáveis** (`src/pipelines/`)
- **Dados rastreáveis** com versionamento por camada (`data/bronze`, `silver`, `gold`, `gold_ml`)
- **Reuso de features** com armazenamento em **Feature Store**
- **Automação de entrega** com CI/CD (`.github/workflows`, `Makefile`, `pre-commit`)
- **Registro e rastreamento de modelos** (`ml/models`, `ml/evaluations`)
- **Monitoramento contínuo** pós-deploy para detectar mudanças ou degradação no desempenho

### 🧩 Componentes Chave

- 📦 **Feature Store**: versionamento e gerenciamento de features padronizadas
- 🧪 **Experimentações e Tuning**: separação clara entre treino/teste, múltiplos experimentos, hyperparameter tuning
- 🧠 **Model Registry**: controle de versão e metadata dos modelos aprovados
- 🔄 **Inference Pipeline**: serve o modelo campeão para geração de predições
- 📈 **Monitoramento**: verificação contínua de performance e detecção de *drift*

---

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

## Features and Tools

Features                                     | Package  | Why?
 ---                                         | ---      | ---
Dependencies and env                         | [pip] + [venv] | [article](https://realpython.com/python-virtual-environments-a-primer/)
Project configuration file                   | [Hydra]  |  [article](https://mathdatasimplified.com/2023/05/25/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/)
Lint - Format, sort imports  (Code Quality)  | [Ruff] | [article](https://www.sicara.fr/blog-technique/boost-code-quality-ruff-linter)
Static type checking                         | [Mypy] | [article](https://python.plainenglish.io/does-python-need-types-79753b88f521)
code security                                | [bandit] | [article](https://blog.bytehackr.in/secure-your-python-code-with-bandit)
Code quality & security each commit          | [pre-commit] | [article](https://dev.to/techishdeep/maximize-your-python-efficiency-with-pre-commit-a-complete-but-concise-guide-39a5)
Test code                                    | [Pytest] | [article](https://realpython.com/pytest-python-testing/)
Test coverage                                | [coverage.py] [codecov] | [article](https://martinxpn.medium.com/test-coverage-in-python-with-pytest-86-100-days-of-python-a3205c77296)
Project Template                             | [Cruft] or [Cookiecutter] | [article](https://medium.com/@bctello8/standardizing-dbt-projects-at-scale-with-cookiecutter-and-cruft-20acc4dc3f74)
Folder structure for data science projects   | [Data structure] | [article](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
Template for pull requests                   | [Pull Request template] | [article](https://www.awesomecodereviews.com/pull-request-template/)
Template for notebooks                       | [Notebook template] |

## Set up the environment

1. Initialize git in local:

    ```bash
    make init_git
    ```

2. Create and activate virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Optional: Install data science libraries:

    ```bash
    make install_data_libs
    ```

## Install new dependencies

To add a new package:

```bash
pip install <package-name>
````

To save it to `requirements.txt`:

```bash
pip freeze > requirements.txt
```

Alternatively, use [pip-tools](https://github.com/jazzband/pip-tools) or [Poetry](https://python-poetry.org/) for better dependency management.

## 🗃️ Project structure

```bash
.
├── README.md                           # Descrição geral do projeto
├── pyproject.toml                      # Gerenciamento de dependências e configuração de ferramentas Python
├── requirements.txt                    # (opcional) Lista congelada de dependências

# ▒▒▒ Configuração e qualidade de código ▒▒▒
├── .editorconfig                       # Padronização de estilo entre editores/IDEs
├── .gitignore                          # Arquivos e pastas ignoradas pelo Git
├── .pre-commit-config.yaml             # Hooks automáticos para lint, format, segurança, etc.
├── .code_quality/                      # Configurações de ferramentas de análise estática
│   ├── mypy.ini                        # Tipagem estática com Mypy
│   └── ruff.toml                       # Linter e formatter com Ruff
├── Makefile                            # Comandos úteis para setup, testes, lint, etc.
├── codecov.yml                         # Configuração de cobertura de testes com Codecov

# ▒▒▒ Dados ▒▒▒
├── data/                               # Camadas do data lake e artefatos de ML
│   ├── bronze/                         # Dados crus (raw) diretamente de fontes externas
│   ├── silver/                         # Dados limpos e tratados (prontos para consumo interno)
│   ├── gold/                           # Dados analíticos finais (features consolidadas)
│   ├── gold_ml/                        # Gold enriquecido com predições, scores, outputs de modelo
│   └── ml/                             # Artefatos do pipeline de machine learning
│       ├── features/                   # Features derivadas, transformadas e selecionadas
│       ├── training_sets/              # Dados finais de treino (features + target)
│       ├── models/                     # Modelos treinados (pkl, joblib, onnx, etc.)
│       ├── predictions/                # Saídas de inferência
│       └── evaluations/                # Métricas e validações dos modelos

# ▒▒▒ Notebooks ▒▒▒
├── notebooks/                          # Workflow exploratório modularizado
│   ├── 1-data/                         # Coleta, extração e estruturação de dados
│   ├── 2-exploration/                  # Análise exploratória (EDA)
│   ├── 3-analysis/                     # Testes estatísticos, hipóteses, correlações
│   ├── 4-feat_eng/                     # Engenharia de features
│   ├── 5-models/                       # Treinamento e avaliação de modelos
│   ├── 6-interpretation/               # Interpretação e explicabilidade (SHAP, LIME, etc.)
│   ├── 7-deploy/                       # Estratégias de empacotamento e deploy
│   ├── 8-reports/                      # Relatórios finais, storytelling, insights
│   ├── notebook_template.ipynb         # Template padrão para notebooks do time
│   └── README.md                       # Guia sobre uso e estrutura dos notebooks

# ▒▒▒ Fonte (src) ▒▒▒
├── src/                                # Código-fonte principal do projeto
│   ├── README.md                       # Documentação sobre a estrutura do `src`
│   ├── tmp_mock.py                     # Script exemplo ou temporário
│   ├── data/                           # Módulo de ingestão, limpeza e transformação de dados
│   ├── model/                          # Módulo de treinamento, tuning, validação, export de modelos
│   ├── inference/                      # Módulo de inferência, serving e integração com APIs
│   └── pipeline/                       # Pipelines de orquestração e execução
│       ├── DE/                         # Pipelines orientados à engenharia de dados
│       │   ├── data_pipeline/          # Pipeline de ingestão e transformação inicial
│       │   ├── feature_pipeline/       # Pipeline de criação e versionamento de features
│       │   └── serving_pipeline/       # Pipeline para dados em tempo real ou micro-batches
│       └── DS/                         # Pipelines orientados à ciência de dados
│           ├── feature_pipeline/       # Transformações e seleção de features para modelagem
│           ├── training_pipeline/      # Pipeline completo de treino e validação de modelo
│           └── inference_pipeline/     # Pipeline de inferência a partir de features e modelos

# ▒▒▒ Testes ▒▒▒
├── tests/                              # Código de testes automatizados
│   ├── data/                           # Testes para o módulo de dados
│   ├── model/                          # Testes para o módulo de modelos
│   ├── inference/                      # Testes para a inferência
│   └── pipelines/                      # Testes para os pipelines integrados

# ▒▒▒ Documentação e ferramentas de apoio ▒▒▒
├── docs/                               # Documentação funcional, técnica, manuais
├── flow/                               # Agentes internos automatizados com IA
│   ├── flow_token.py                   # Captura de token para acesso a sistemas externos
│   ├── generate_docs.py                # Geração automatizada de documentação com IA
│   ├── refactor_agent.py               # Agente para refatoração com boas práticas
│   └── review_agent.py                 # Agente de revisão técnica (docstrings, lint, segurança, etc.)

# ▒▒▒ Configurações de ambiente e CI/CD ▒▒▒
├── .github/                            # Configuração para GitHub Actions
│   ├── dependabot.md                   # Gerenciamento de dependências automáticas
│   ├── pull_request_template.md        # Template de PRs
│   └── workflows/                      # Workflows CI/CD
│       ├── ci.yml                      # Build e testes contínuos
│       ├── dependency_review.yml       # Análise de dependências
│       ├── docs.yml                    # Publicação de documentação (mkdocs)
│       └── pre-commit_autoupdate.yml   # Atualização de hooks do pre-commit
├── .vscode/                            # Configurações específicas para o VS Code
│   ├── extensions.json                 # Extensões recomendadas
│   ├── launch.json                     # Configuração de execução/debug
│   └── settings.json                   # Preferências do projeto no editor


```

---

## Créditos

Desenvolvido por **[@RaphaelNorris](https://github.com/RaphaelNorris)**.

---

<!-- Links -->

[Ruff]: https://docs.astral.sh/ruff/
[Mypy]: https://mypy-lang.org/
[Bandit]: https://github.com/PyCQA/bandit
[pre-commit]: https://pre-commit.com/
[coverage.py]: https://coverage.readthedocs.io/
[Data structure]: data/README.md

