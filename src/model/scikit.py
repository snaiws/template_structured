class SklearnAdapter(BaseModel):
    def __init__(self, model):
        """
        model: scikit-learn 호환 모델 (예: LinearRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier 등)
        """
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, **kwargs):
        # scikit-learn의 score 메서드 활용
        return self.model.score(X, y, **kwargs)
