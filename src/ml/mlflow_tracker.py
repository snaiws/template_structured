import mlflow

# mlflow_tracking 데코레이터를 tracking URI와 experiment 이름을 인자로 받을 수 있도록 수정
def mlflow_tracking(tracking_uri, experiment_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # mlflow 설정 적용
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                model, metrics = func(*args, **kwargs)
                # 에포크별 메트릭 로깅
                for epoch_metric in metrics.get("epochs", []):
                    mlflow.log_metric("train_loss", epoch_metric["train_loss"], step=epoch_metric["epoch"])
                    mlflow.log_metric("val_f1", epoch_metric["val_f1"], step=epoch_metric["epoch"])
                mlflow.log_metric("test_f1", metrics.get("test_f1"))
                # 최종 모델 mlflow에 저장
                mlflow.pytorch.log_model(model, "pytorch_mlp_model")
                print("Model saved to MLflow")
                return model, metrics
        return wrapper
    return decorator