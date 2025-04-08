import os

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from configs import ConfigDefineTool
from data.data_loader import load_data
from data.set import DatasetLGR
from machine import ModelLGR


def train(exp_name):
    # configs
    config = ConfigDefineTool(exp_name = exp_name)
    env = config.get_env()
    exp = config.get_exp()

    path_train = os.path.join(env.PATH_DATA_DIR, exp.train)


    model_name = exp.model_name
    model_params = exp.model_params


    # 데이터 로딩
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(path_train)

    data_train = DatasetLGR(X_train, y_train)

    # 모델 학습
    model = ModelLGR(model_name = model_name, model_params = model_params)
    model.train(
        data_train = data_train
    )
    
    # 검증 데이터에 대한 예측 및 평가
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds)
    print(f"Validation F1: {val_f1:.4f}")
    
    # 테스트 데이터에 대한 예측 및 평가
    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds)
    print(f"Test F1: {test_f1:.4f}")
    
    


if __name__ == "__main__":
    train("ExperimentLgrBase")