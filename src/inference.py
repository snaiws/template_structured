import mlflow.lightgbm
import pandas as pd

def inference():
    # 예시: MLflow에 저장된 모델 로드 후 새로운 데이터에 대해 예측 수행
    model_uri = "runs:/{run_id}/lightgbm_model"  # 실제 run_id로 대체
    model = mlflow.lightgbm.load_model(model_uri)
    
    # 새 데이터 로드 (예시)
    data = pd.read_csv("path/to/new/data.csv")
    # 전처리 수행 (train 시 사용했던 전처리와 동일하게)
    # ...
    
    predictions = model.predict(data)
    print("Predictions:", predictions)

if __name__ == '__main__':
    inference()
