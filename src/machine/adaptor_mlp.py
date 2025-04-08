import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from .base import ModelAdaptor
from .models import MLP

class ModelMLP(ModelAdaptor):
    def __init__(self, model_name, model_params):
        """모델 선언"""
        self.model_name = model_name
        self.model = None
        
        self.input_dim = model_params.get("input_dim", None)
        self.hidden_dims = model_params.get("hidden_dims", None)
        self.batch_size = model_params.get("batch_size", None)
        self.num_epochs = model_params.get("num_epochs", None)
        self.lr_optimizer = model_params.get("lr_optimizer", None)
        self.metric = f1_score

        self.model = MLP(self.input_dim, hidden_dims=self.hidden_dims)

        self.criterion = nn.BCEWithLogitsLoss()  # 출력이 로짓이므로 사용
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_optimizer)

        # GPU 사용 설정 (가능한 경우)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        

    def train(self, data_train, data_valid):
        """모델 학습"""
        best_val_f1 = 0.0
        for epoch in range(1, self.num_epochs + 1):
            # 학습 모드
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in data_train:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(data_train.dataset)
            
            # 검증 모드
            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in data_valid:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    # 로짓을 시그모이드로 변환한 후 0.5 기준 이진 분류
                    preds = torch.sigmoid(outputs)
                    preds_binary = (preds > 0.5).float()
                    all_preds.append(preds_binary.cpu().numpy())
                    all_labels.append(y_batch.cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            val_f1 = self.metric(all_labels, all_preds)
            
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # 검증 성능이 개선되면 모델 저장
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict()
        self.model.load_state_dict(best_model_state)

    def predict(self, data):
        """예측 실행"""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch in data:
                # X_batch가 튜플인 경우 (TensorDataset에서 반환) 첫 번째 원소 추출
                if isinstance(X_batch, tuple):
                    X_batch = X_batch[0]
                    
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()
                all_preds.append(preds_binary.cpu().numpy())
        
        return np.concatenate(all_preds)

    def load(self, path):
        """모델 불러오기"""
        self.model.load_state_dict(torch.load(path), map_location=self.device)
        self.model.eval()

    def save(self, path):
        """모델 저장하기"""
        torch.save(self.model.state_dict(), path)

