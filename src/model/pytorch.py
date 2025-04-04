import torch

class PytorchAdapter(BaseModel):
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

    def fit(self, train_loader, epochs=10, **kwargs):
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

    def evaluate(self, test_loader, **kwargs):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
        return total_loss
