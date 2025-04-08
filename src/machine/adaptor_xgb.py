import xgboost as xgb

from .base import ModelAdaptor



class ModelXGB(ModelAdaptor):
    def __init__(self, model_name, model_params):
        """모델 선언"""
        self.model_name = model_name
        self.model = None
        
        self.objective = model_params.get("objective", None)
        self.eval_metric = model_params.get("eval_metric", None)
        self.eta = model_params.get("eta", None)
        self.max_depth = model_params.get("max_depth", None)
        self.subsample = model_params.get("subsample", None)
        self.num_boost_round = model_params.get("num_boost_round", None)
        self.earlystopping_round = model_params.get("earlystopping_round", None)
        
        self.callbacks = []
        if self.earlystopping_round is not None:
            self.earlystopping_round = model_params["earlystopping_round"]
            self.callbacks.append(xgb.callback.EarlyStopping(rounds=self.earlystopping_round))
        
        self.params = {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "eta": self.eta,                  # 학습률
            "max_depth": self.max_depth,
            "subsample": self.subsample,
        }
        
    def train(self, data_train, data_valid):
        """모델 학습"""
        evals = [(data_train, "train"), (data_valid, "valid")]
        self.model = xgb.train(
            self.params,
            data_train,
            num_boost_round = self.num_boost_round,
            evals = evals,
            callbacks = self.callbacks
        )
        
    def predict(self, data):
        """예측 실행"""
        if hasattr(self.model, "best_ntree_limit"):
            return self.model.predict(data, ntree_limit=self.model.best_ntree_limit)
        else:
            return self.model.predict(data)
        
    def load(self, path):
        """모델 불러오기"""
        self.model = xgb.Booster()
        self.model.load_model(path)
    
    def save(self, path):
        """모델 저장하기"""
        self.model.save_model(path)