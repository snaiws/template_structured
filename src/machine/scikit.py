from .base import BaseAdaptor



class SklearnAdapter(BaseAdaptor):
    def __init__(self, model):
        """
        model: scikit-learn 호환 모델 (예: LinearRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier 등)
        """
        self.model = model

    def train(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)
