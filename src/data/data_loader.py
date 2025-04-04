import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(data_path: str):
    """CSV 파일로부터 train, test 데이터를 로딩하고, 카테고리형 컬럼을 처리합니다."""
    df = pd.read_csv(data_path)
    
    categorical_cols = ['UID', '주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']
    df = pd.get_dummies(df, columns=categorical_cols)
    df = df.astype(float)
    
    target = "채무 불이행 여부"

    # train 데이터셋 분할 (train: 60%, validation: 20%, test: 20%) 이건 원래 따로 샘플링 버전 저장해야 함
    train, val = train_test_split(train, test_size=0.4, random_state=421)
    val, test = train_test_split(val, test_size=0.5, random_state=421)

    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
