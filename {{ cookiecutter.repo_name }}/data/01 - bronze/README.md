# Camada Bronze

A camada **Bronze** é responsável por armazenar os dados **brutos** extraídos diretamente das fontes originais (APIs, bancos de dados, arquivos, etc).

Nenhum tipo de transformação é aplicada neste estágio. O objetivo é garantir **rastreabilidade e reprodutibilidade**, mantendo o dado original exatamente como foi recebido.

## Exemplos de conteúdo
- Dumps completos de tabelas
- Arquivos CSV, JSON ou Parquet originais
- Respostas brutas de chamadas a APIs
- Logs ou arquivos de sensores

## Boas práticas
- Organizar por data de extração (`YYYY/MM/DD`)
- Evitar sobrescrever arquivos
- Registrar o processo de ingestão
