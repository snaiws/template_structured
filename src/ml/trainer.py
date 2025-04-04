import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from data.data_loader import load_data
from utils.logging_callback import MlflowLoggingCallbackLGBM
from configs.config import config

def train_model(train_csv: str, test_csv: str):
    # 데이터 로딩
    train, test = load_data(train_csv, test_csv)
    
    # train, validation, test 데이터셋 분리 (비율: 60:20:20)
    train, val = train_test_split(train, test_size=0.4, random_state=config["random_state"])
    val, test = train_test_split(val, test_size=0.5, random_state=config["random_state"])
    
    target = "채무 불이행 여부"
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_val = val.drop(columns=[target])
    y_val = val[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # MLflow 설정
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    
    with mlflow.start_run():
        params = config["lgb_params"]
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
            MlflowLoggingCallbackLGBM(X_test, y_test, f1_score)
        ]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=1000,
            callbacks=callbacks,
        )
        
        # 테스트 데이터에 대해 예측 및 평가
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_binary = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_test, y_pred_binary)
        print("f1:", f1)
        
        # MLflow에 파라미터 및 메트릭 기록
        mlflow.log_params(params)
        mlflow.log_metric("f1", f1)
        mlflow.lightgbm.log_model(model, "lightgbm_model")
        print("Model saved to MLflow")
