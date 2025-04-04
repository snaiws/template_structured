import os

from sklearn.metrics import f1_score
import xgboost as xgb

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

    num_boost_round = training_params['num_boost_round']
    earlystopping_round = training_params['earlystopping_round']


    # 데이터 로딩
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(path_train)

    # XGBoost용 DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # XGBoost 파라미터 설정


    evals = [(dtrain, "train"), (dval, "valid")]
    # xgb.callback.EarlyStopping은 지정한 round 동안 성능 향상이 없으면 학습 조기 종료
    callbacks = [
        xgb.callback.EarlyStopping(rounds=earlystopping_round),
        MlflowLoggingCallbackXGB(dtest, y_test, f1_score)
    ]
    
    model = xgb.train(
        model_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        callbacks=callbacks,
    )
    
    # 테스트 데이터에 대해 예측 및 평가
    if hasattr(model, "best_ntree_limit"):
        y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    else:
        y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_binary)
    print("f1:", f1)
    
    # 파라미터 및 최종 메트릭 기록
    
    # MLflow에 모델 저장
    print("Model saved to MLflow")
