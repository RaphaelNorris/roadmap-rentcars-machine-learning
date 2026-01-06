import mlflow
import os
import time

def run_test():
    # As variáveis MLFLOW_TRACKING_URI e MLFLOW_EXPERIMENT_NAME
    # vêm automaticamente da Step Function
    print(f"Conectando ao MLflow em: {os.getenv('MLFLOW_TRACKING_URI')}")

    # Inicia uma rodada de experimento
    with mlflow.start_run(run_name="Test_Run_Rentcars"):
        mlflow.log_param("env", os.getenv("ENV", "dev"))
        mlflow.log_param("version", "0.0.1")

        print("Logando métricas de teste...")
        for i in range(10):
            mlflow.log_metric("accuracy_test", 0.8 + (i/100))
            time.sleep(1)

        print("✅ Teste finalizado com sucesso no MLflow!")

if __name__ == "__main__":
    run_test()