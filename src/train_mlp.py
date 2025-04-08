import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from configs import ConfigDefineTool
from data.data_loader import load_data
from data.set import DatasetMLPTrain, DatasetMLPInfer
from machine.models.mlp_bc import MLP
from machine import ModelMLP


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



    train_dataset = DatasetMLPTrain(X_train, y_train)
    val_dataset = DatasetMLPTrain(X_val, y_val)
    test_dataset = DatasetMLPInfer(X_test)

    batch_size = model_params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    # 모델 학습
    model = ModelMLP(model_name = model_name, model_params = model_params)
    model.train(
        data_train = train_loader, 
        data_valid = val_loader
    )

        
    # 테스트 데이터에 대해 평가 (검증 성능이 가장 좋았던 모델 사용)
    y_preds = model.predict(test_loader)
    test_f1 = f1_score(y_test, y_preds)
    print("Test F1:", test_f1)
    


if __name__ == "__main__":
    train("ExperimentMlpBase")