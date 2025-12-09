# Pipelines de Feature / Treinamento / Inferência

Estrutura de arquivos baseada em:

<https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines>

---

## Estrutura de Pastas

- `src/`: código-fonte principal do projeto
    - `data/`: extração de dados, validação, processamento, transformação, salvamento e exportação.
    - `model/`: treinamento, avaliação, validação, salvamento e exportação dos modelos.
    - `inference/`: predição com os modelos, servimento (serving) e monitoramento.
    - `pipelines/`:
        - `feature_pipeline/`: recebe dados brutos e os transforma em features (e rótulos).
        - `training_pipeline/`: recebe features e rótulos e gera um modelo treinado.
        - `inference_pipeline/`: utiliza dados de entrada e o modelo treinado para gerar predições.

---

## Multiplicidade de Pipelines

Você pode (e geralmente deve) ter **vários pipelines distintos**, por exemplo:

- 3 pipelines de features que obtêm dados de diferentes fontes e os salvam em um **feature store**.
- 2 pipelines de treinamento que utilizam esses dados para treinar modelos distintos.
- 3 pipelines de inferência online (serviço REST) e 1 pipeline de inferência em lote (batch).

---

## Orquestração

É recomendado ter um **script de orquestração** para executar os pipelines na ordem correta.  

