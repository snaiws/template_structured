import mlflow
from sklearn.metrics import f1_score

class MlflowLoggingCallbackLGBM:
    """
    LightGBM 학습 시 매 epoch마다 MLflow에 메트릭을 기록하는 콜백 클래스.
    """
    def __init__(self, X_test, y_test, metric):
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric

    def __call__(self, env):
        iteration = env.iteration + 1  # 0-indexed
        y_pred = env.model.predict(self.X_test, num_iteration=iteration)
        
        # f1_score를 사용할 경우 이진 분류의 확률값을 이진값으로 변환
        if self.metric == f1_score:
            y_pred_binary = (y_pred > 0.5).astype(int)
            score = self.metric(self.y_test, y_pred_binary)
        else:
            score = self.metric(self.y_test, y_pred)
        
        metrics = {"test": score}
        for name, metric_name, value, _ in env.evaluation_result_list:
            metrics[name] = value
        
        mlflow.log_metrics(metrics, step=iteration, synchronous=False)
