from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, X):
        """예측 실행"""
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        """모델 평가"""
        pass
