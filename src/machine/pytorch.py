import torch

from .base import BaseAdaptor

class PytorchAdapter(BaseAdaptor):
    def __init__(self, model, criterion, optimizer, device='cpu'):
        """
        model: nn.Module 기반 PyTorch 모델
        criterion: 손실 함수 (예: nn.CrossEntropyLoss())
        optimizer: 최적화 기법 (예: torch.optim.Adam(model.parameters()))
        device: 'cpu' 또는 'cuda'
        """
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, epochs=10, **kwargs):
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            outputs = self.model(X)
        return outputs.cpu()