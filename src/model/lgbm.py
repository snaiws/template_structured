import lightgbm as lgb

class LightGBMAdapter(BaseModel):
    def __init__(self, model):
        """
        model: lightgbm.LGBMClassifier 또는 lightgbm.LGBMRegressor
        """
        self.model = model

    def fit(self, X, y, eval_set=None, **kwargs):
        self.model.fit(X, y, eval_set=eval_set, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, **kwargs):
        return self.model.score(X, y, **kwargs)




def get_lightgbm_model(params: dict, train_data, valid_data, callbacks):
    """
    LightGBM 모델을 학습하는 함수.
    """
    import lightgbm as lgb
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=1000,
        callbacks=callbacks,
    )
    return model
