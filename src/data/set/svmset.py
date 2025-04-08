
class DatasetSVM:
    def __init__(self, X, y):
        self.X = X.to_numpy()
        self.y = y.to_numpy()