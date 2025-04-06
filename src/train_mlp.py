import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from configs import ConfigDefineTool
from data.data_loader import load_data
from model.models.mlp_bc import MLP



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



def train(exp_name):
    # configs
    config = ConfigDefineTool(exp_name = exp_name)
    env = config.get_env()
    exp = config.get_exp()

    path_train = os.path.join(env.PATH_DATA_DIR, exp.train)


    model_params = exp.model_params
    training_params = exp.training_params

    hidden_dims = model_params['hidden_dims']

    batch_size = training_params['batch_size']
    lr_optimizer = training_params['lr_optimizer']
    num_epochs = training_params['num_epochs']


    # 데이터 로딩
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(path_train)



    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    input_dim = X_train.shape[1]
    model = MLP(input_dim, hidden_dims=hidden_dims)

    # 손실함수 및 옵티마이저 정의
    criterion = nn.BCEWithLogitsLoss()  # 출력이 로짓이므로 사용
    optimizer = optim.Adam(model.parameters(), lr=lr_optimizer)


    # GPU 사용 설정 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_f1 = 0.0
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
        
        # 최종 모델 MLflow에 저장
        print("Model saved to MLflow")  


if __name__ == "__main__":
    train("ExperimentMlpBase")