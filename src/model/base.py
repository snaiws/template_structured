from abc import ABC, abstractmethod

class BaseAdaptor(ABC):
    @abstractmethod
    def __init__(self, model, model_params):
        """모델 선언"""
        pass
        
    @abstractmethod
    def train(self, data, **training_params):
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, data):
        """예측 실행"""
        pass