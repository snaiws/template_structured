import os

import lightgbm as lgb

from .base import ModelAdaptor

class ModelLGB(ModelAdaptor):
    def __init__(self, model_name:str, model_params:dict):
        """
        model: lightgbm.LGBMClassifier 또는 lightgbm.LGBMRegressor
        """
        self.model_name = model_name
        self.model = None
        self.objective = model_params.get("objective", None)
        self.metric = model_params.get("metric", None)
        self.boosting_type = model_params.get("boosting_type", None)
        self.num_leaves = model_params.get("num_leaves", None)
        self.learning_rate = model_params.get("learning_rate", None)
        self.feature_fraction = model_params.get("feature_fraction", None)
        self.stopping_rounds = model_params.get("stopping_rounds", None)
        self.period = model_params.get("period", None)
        self.params = {
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
        }

        self.callbacks = []
        if self.stopping_rounds is not None:
            self.stopping_rounds = model_params["stopping_rounds"]
            self.callbacks.append(lgb.early_stopping(stopping_rounds=self.stopping_rounds))
        if self.period is not None:
            self.period = model_params["period"]
            self.callbacks.append(lgb.log_evaluation(period=self.period))
        
            
    def train(self, data_train, data_valid, **kwargs):
        self.model = lgb.train(
            params = self.params,
            train_set = data_train,
            valid_sets = [data_train, data_valid],
            **kwargs
            )
        return self

    def predict(self, data):
        y_pred = self.model.predict(data, num_iteration = self.model.best_iteration)
        return y_pred

    def load(self, path):
        self.model = lgb.Booster(model_file=path)

    def save(self, path):
        self.model.save(path)