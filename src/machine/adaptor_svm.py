import pickle

from .base import ModelAdaptor

from sklearn.svm import SVC



class ModelSVM(ModelAdaptor):
    def __init__(self, model_name, model_params):
        """모델 선언"""
        self.model_name = model_name
        self.model = None
        self.kernel = model_params.get("kernel", None)
        self.C = model_params.get("C", None)
        self.gamma = model_params.get("gamma", None)
        self.probability = model_params.get("probability", None)
        self.random_state = model_params.get("random_state", None)

        self.model = SVC(
            kernel = self.kernel,
            C = self.C,
            gamma = self.gamma,
            probability = self.probability,
            random_state = self.random_state
        )

        
    def train(self, data_train, data_valid=None):
        """모델 학습"""
        X_train = data_train.X
        y_train = data_train.y
        self.model.fit(X_train, y_train)


    def predict(self, data):
        """예측 실행"""
        return self.model.predict(data)


    def load(self, path):
        """모델 불러오기"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


    def save(self, path):
        """모델 저장하기"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)