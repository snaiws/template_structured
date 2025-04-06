import xgboost as xgb


class model_xgb:
    def __init__(self, model_params):
        self.model = xgb
        self.model_params = model_params