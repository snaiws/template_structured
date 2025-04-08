from .base import ModelAdaptor



class XGBoostAdapter(ModelAdaptor):
    def __init__(self, model, model_params):
        """
        model: xgboost.XGBClassifier 또는 xgboost.XGBRegressor
        """
        self.model = model
        self.model_params = model_params

    def train(self, data, training_params):
        self.model = self.model.train(
            self.model_params,
            data,
            **training_params
            )
        return self

    def predict(self, data):
        if hasattr(self.model, "best_ntree_limit"):
            return self.model.predict(data, ntree_limit=self.model.best_ntree_limit)
        else:
            return self.model.predict(data)
