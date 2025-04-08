import lightgbm as lgb


class DatasetLGB:
    def __init__(self, features, label, reference):
        self.dataset = lgb.Dataset(features, label=label, reference = reference)