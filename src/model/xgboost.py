# XGBoost 어댑터 예시
import xgboost as xgb

class XGBoostAdapter(BaseModel):
    def __init__(self, model):
        """
        model: xgboost.XGBClassifier 또는 xgboost.XGBRegressor
        """
        self.model = model

    def fit(self, X, y, eval_set=None, **kwargs):
        self.model.fit(X, y, eval_set=eval_set, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, **kwargs):
        return self.model.score(X, y, **kwargs)
