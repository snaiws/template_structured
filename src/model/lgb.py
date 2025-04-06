import lightgbm as lgb

from .base import BaseAdaptor

class LightGBMAdapter(BaseAdaptor):
    def __init__(self, model, model_params):
        """
        model: lightgbm.LGBMClassifier 또는 lightgbm.LGBMRegressor
        """
        self.model = model

    def train(self, data):
        self.model.fit(X, y, eval_set=eval_set, **kwargs)
        return self

    def predict(self, data):
        return self.model.predict(data)

