import os
import sys
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import xgboost as xgb
from xgboost.callback import TrainingCallback

import mlflow
import mlflow.xgboost

from data.data_loader import load_data

# 데이터 파일 경로 설정
train_path = "/workspace/Storage/template_structured/Data/raw/train.csv"
test_path = "/workspace/Storage/template_structured/Data/raw/test.csv"

# 데이터 로딩
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 지정된 컬럼을 범주형으로 변환
categorical_cols = ['UID', '주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']
for col in categorical_cols:
    train[col] = train[col].astype('category')

# train 데이터셋 분할 (train: 60%, validation: 20%, test: 20%)
train, val = train_test_split(train, test_size=0.4, random_state=421)
val, test = train_test_split(val, test_size=0.5, random_state=421)

target = "채무 불이행 여부"
X_train, y_train = train.drop(columns=[target]), train[target]
X_val, y_val = val.drop(columns=[target]), val[target]
X_test, y_test = test.drop(columns=[target]), test[target]

# XGBoost용 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
dtest  = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# XGBoost 파라미터 설정
params = {
    "objective": "binary:logistic",  # 이진 분류
    "eval_metric": "logloss",
    "eta": 0.05,                     # 학습률
    "max_depth": 6,
    "subsample": 0.9
}

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

# MLflow 설정
mlflow.set_tracking_uri("http://175.214.62.133:50001/")
mlflow.set_experiment("xgboost_experiment")

with mlflow.start_run():
    evals = [(dtrain, "train"), (dval, "valid")]
    # xgb.callback.EarlyStopping은 지정한 round 동안 성능 향상이 없으면 학습 조기 종료
    callbacks = [
        xgb.callback.EarlyStopping(rounds=50),
        MlflowLoggingCallbackXGB(dtest, y_test, f1_score)
    ]
    
    num_boost_round = 1000
    model = xgb.train(
        params,
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
    mlflow.log_params(params)
    mlflow.log_metric("f1", f1)
    
    # MLflow에 모델 저장
    mlflow.xgboost.log_model(model, "xgboost_model")
    print("Model saved to MLflow")
