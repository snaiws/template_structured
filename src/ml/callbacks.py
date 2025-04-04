
from xgboost.callback import TrainingCallback

class MlflowLoggingCallbackXGB(TrainingCallback):
    def __init__(self, dtest, y_test, metric=f1_score):
        self.dtest = dtest
        self.y_test = y_test
        self.metric = metric

    def after_iteration(self, model, epoch, evals_log):
        y_pred = model.predict(self.dtest, iteration_range=(0, epoch + 1))
        if self.metric == f1_score:
            y_pred_binary = (y_pred > 0.5).astype(int)
            score = self.metric(self.y_test, y_pred_binary)
        else:
            score = self.metric(self.y_test, y_pred)
        
        # 로그 기록
        mlflow.log_metric("test_f1", score, step=epoch + 1)

        # 기존 eval metric도 로깅 (선택)
        for data_name in evals_log:
            for metric_name in evals_log[data_name]:
                val = evals_log[data_name][metric_name][-1]
                mlflow.log_metric(f"{data_name}_{metric_name}", val, step=epoch + 1)

        return False  # 학습 계속

class MlflowLoggingCallbackLGBM:
    def __init__(self, X_test, y_test, metric):
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric

    def __call__(self, env):
        iteration = env.iteration + 1  # 0-indexed
        y_pred = env.model.predict(self.X_test, num_iteration=iteration)

        # 이진 분류인 경우 확률값 → 이진값 변환
        if self.metric == f1_score:
            y_pred_binary = (y_pred > 0.5).astype(int)
            score = self.metric(self.y_test, y_pred_binary)
        else:
            score = self.metric(self.y_test, y_pred)

        metrics = {"test": score}

        # 기존 validation loss 들도 같이 기록
        for name, metric_name, value, _ in env.evaluation_result_list:
            metrics[name] = value

        mlflow.log_metrics(metrics, step=iteration, synchronous=False)