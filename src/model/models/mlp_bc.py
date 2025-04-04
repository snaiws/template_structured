import torch.nn as nn

# 커스텀 MLP 모델 정의 (입력 차원에 맞게 구성)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        # 출력 레이어: 마지막 hidden dimension에서 1로 변환 (로짓)
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)