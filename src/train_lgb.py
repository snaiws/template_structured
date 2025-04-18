import os
import asyncio

from sklearn.metrics import f1_score

from configs import ConfigDefineTool
from dataset.dataset_loaninfo import DatasetLoaninfoRaw
from machine import ModelLGB

async def train(exp_name):
    # configs
    config = ConfigDefineTool(exp_name = exp_name)
    env = config.get_env()
    exp = config.get_exp()

    path_train = os.path.join(env.PATH_DATA_DIR, exp.train)

    model_name = exp.model_name
    model_params = exp.model_params

    data_params = {
        "csv_path": path_train,
        "ratio": (0.6,0.2,0.2),
        "random_state":42
    }

    # 데이터 로딩
    X_train, y_train, X_val, y_val, X_test, y_test = await DatasetLoaninfoRaw(data_params).data

    # 모델 학습
    model = ModelLGB(model_name = model_name, model_params = model_params)
    model.train(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)

    # 후처리
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 평가
    f1 = f1_score(y_test, y_pred_binary)
    print("f1:", f1)



if __name__ == "__main__":
    asyncio.run(train("ExperimentLgbBase"))