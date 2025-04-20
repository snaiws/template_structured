import torch
from torch.utils.data import Dataset

# PyTorch Dataset 정의
class DatasetMLPTrain(Dataset):
    def __init__(self, X, y):
        # 데이터프레임을 텐서로 변환 (실수형)
        self.X = torch.tensor(X.values, dtype=torch.float32)
        # 이진 분류이므로 target은 (N, 1) shape로 변환
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DatasetMLPInfer(Dataset):
    def __init__(self, X):
        # 데이터프레임을 텐서로 변환 (실수형)
        self.X = torch.tensor(X.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]
