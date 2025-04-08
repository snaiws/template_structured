import os
from sklearn.metrics import f1_score

import lightgbm as lgb

from configs import ConfigDefineTool
from data.data_loader import load_data
from machine import ModelLGB

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
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 모델 학습
    model = ModelLGB(model_name = model_name, model_params = model_params)
    model.train(
        data_train = train_data, 
        data_valid = valid_data
    )
    y_pred = model.predict(X_test)

    # 후처리
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 평가
    f1 = f1_score(y_test, y_pred_binary)
    print("f1:", f1)



if __name__ == "__main__":
    train("ExperimentLgbBase")