FROM python:3.11-slim

# Instala drivers necessários para o MLflow falar com S3 e Postgres
RUN apt-get update && apt-get install -y libpq-dev gcc && \
    pip install mlflow boto3 psycopg2-binary

WORKDIR /app
COPY . .

# O comando padrão será sobrescrito pela Step Function,
# mas deixamos um teste local
CMD ["python", "main.py"]