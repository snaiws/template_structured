from abc import ABC, abstractmethod

class ModelAdaptor(ABC):
    @abstractmethod
    def __init__(self, model_name, model_params):
        """모델 선언"""
        pass
        
    @abstractmethod
    def train(self, data_train, data_valid):
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, data):
        """예측 실행"""
        pass

    @abstractmethod
    def load(self, path):
        """모델 불러오기"""
        pass

    @abstractmethod
    def save(self, path):
        """모델 저장하기"""
        pass