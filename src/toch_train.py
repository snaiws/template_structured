import os
import sys
import datetime
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

import mlflow
import mlflow.pytorch

# 데이터 파일 경로 설정
train_path = "/workspace/Storage/template_structured/Data/raw/train.csv"
test_path = "/workspace/Storage/template_structured/Data/raw/test.csv"  # 필요시 사용

# 데이터 로딩
df = pd.read_csv(train_path)

# 지정된 컬럼을 범주형으로 변환 후, 원-핫 인코딩 처리 (모델 입력은 모두 수치형이어야 함)
categorical_cols = ['UID', '주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']
for col in categorical_cols:
    df[col] = df[col].astype('category')
df = pd.get_dummies(df, columns=categorical_cols)
df = df.astype(float)

scaler = MinMaxScaler()

target = "채무 불이행 여부"
# target을 제외한 수치 데이터만 스케일링
feature_cols = df.columns.drop(target)
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# train 데이터셋 분할 (train: 60%, validation: 20%, test: 20%)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=421)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=421)

X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_val = val_df.drop(columns=[target])
y_val = val_df[target]
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# PyTorch Dataset 정의
class TabularDataset(Dataset):
    def __init__(self, X, y):
        # 데이터프레임을 텐서로 변환 (실수형)
        self.X = torch.tensor(X.values, dtype=torch.float32)
        # 이진 분류이므로 target은 (N, 1) shape로 변환
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = TabularDataset(X_train, y_train)
val_dataset = TabularDataset(X_val, y_val)
test_dataset = TabularDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 커스텀 MLP 모델 정의 (입력 차원에 맞게 구성)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 출력: 1차원 (로짓)
        )
    
    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = MLP(input_dim, hidden_dim=64)

# 손실함수 및 옵티마이저 정의
criterion = nn.BCEWithLogitsLoss()  # 출력이 로짓이므로 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# MLflow 설정
mlflow.set_tracking_uri("http://175.214.62.133:50001/")
mlflow.set_experiment("pytorch_mlp_experiment")

# GPU 사용 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
best_val_f1 = 0.0

with mlflow.start_run():
    for epoch in range(1, num_epochs + 1):
        # 학습 모드
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 검증 모드
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                # 로짓을 시그모이드로 변환한 후 0.5 기준 이진 분류
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()
                all_preds.append(preds_binary.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # MLflow에 메트릭 로깅
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        
        # 검증 성능이 개선되면 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
    
    # 테스트 데이터에 대해 평가 (검증 성능이 가장 좋았던 모델 사용)
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            all_preds.append(preds_binary.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    test_f1 = f1_score(all_labels, all_preds)
    print("Test F1:", test_f1)
    mlflow.log_metric("test_f1", test_f1)
    
    # 최종 모델 MLflow에 저장
    mlflow.pytorch.log_model(model, "pytorch_mlp_model")
    print("Model saved to MLflow")
