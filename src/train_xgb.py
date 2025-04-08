import os

from sklearn.metrics import f1_score
import xgboost as xgb

from configs import ConfigDefineTool
from data.data_loader import load_data
from machine import ModelXGB


def train(exp_name):
    # configs
    config = ConfigDefineTool(exp_name = exp_name)
    env = config.get_env()
    exp = config.get_exp()

    path_train = os.path.join(env.PATH_DATA_DIR, exp.train)

    model_name = exp.model_name
    model_params = exp.model_params


    # 데이터 로딩
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(path_train)

    # XGBoost용 DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # 모델 학습
    model = ModelXGB(model_name = model_name, model_params = model_params)
    model.train(
        data_train = dtrain, 
        data_valid = dval
    )
    
    # 테스트 데이터에 대해 예측 및 평가
    y_pred = model.predict(dtest)


    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_binary)
    print("f1:", f1)
    


if __name__ == "__main__":
    train("ExperimentXgbBase")