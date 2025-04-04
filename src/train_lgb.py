import os
from sklearn.metrics import f1_score

import lightgbm as lgb

from configs import ConfigDefineTool
from data.data_loader import load_data

def train(exp_name):
    # configs
    config = ConfigDefineTool(exp_name = exp_name)
    env = config.get_env()
    exp = config.get_exp(exp_name)

    path_train = os.path.join(env.PATH_DATA_DIR, exp.train)


    model_params = exp.model_params
    training_params = exp.training_params

    stopping_rounds = training_params['stopping_rounds']
    period = training_params['period']
    num_boost_round = training_params['num_boost_round']


    # 데이터 로딩
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(path_train)

    # 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


    callbacks = [
        lgb.early_stopping(stopping_rounds=stopping_rounds),
        lgb.log_evaluation(period=period),
    ]

    model = lgb.train(
        model_params,
        train_data,
        valid_sets=[train_data,valid_data],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
        # feval = root_mean_squared_log_error_lgbm,
    )


    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # rmsle = root_mean_squared_log_error(y_test, y_pred)
    f1 = f1_score(y_test, y_pred_binary)
    print("f1:", f1)

    # MLflow에 파라미터 및 메트릭 기록

    # 모델 저장
    print("Model saved to MLflow")