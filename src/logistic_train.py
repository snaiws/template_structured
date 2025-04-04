import os
import sys
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

# 데이터 파일 경로 설정
train_path = "/workspace/Storage/template_structured/Data/raw/train.csv"

# 데이터 로딩
df = pd.read_csv(train_path)

# 지정된 범주형 컬럼을 범주형으로 변환 후, 원-핫 인코딩 처리
categorical_cols = ['UID', '주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']
for col in categorical_cols:
    df[col] = df[col].astype('category')
df = pd.get_dummies(df, columns=categorical_cols)

# 타겟 및 피처 설정
target = "채무 불이행 여부"
X = df.drop(columns=[target])
y = df[target]

# train, validation, test 데이터셋 분할 (train: 60%, validation: 20%, test: 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=421)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=421)

# SVM과 마찬가지로 Logistic Regression도 스케일링이 중요하므로 StandardScaler 적용
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Logistic Regression 모델 정의
lr_model = LogisticRegression(penalty='l2',
                              C=1.0,
                              solver='lbfgs',
                              max_iter=1000,
                              random_state=421)

# MLflow 설정
mlflow.set_tracking_uri("http://175.214.62.133:50001/")
mlflow.set_experiment("logistic_regression_experiment")

with mlflow.start_run():
    # 모델 학습
    lr_model.fit(X_train, y_train)
    
    # 검증 데이터에 대한 예측 및 평가
    val_preds = lr_model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds)
    print(f"Validation F1: {val_f1:.4f}")
    
    # 테스트 데이터에 대한 예측 및 평가
    test_preds = lr_model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds)
    print(f"Test F1: {test_f1:.4f}")
    
    # MLflow에 파라미터 및 메트릭 기록
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("test_f1", test_f1)
    
    # MLflow에 모델 저장 (scikit-learn 모델 저장)
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    print("Model saved to MLflow")
