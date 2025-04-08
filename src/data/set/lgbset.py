import lightgbm as lgb



class DatasetLGB(lgb.Dataset):
    def __init__(self, features, label, reference = None):
        super().__init__(features, label, reference)