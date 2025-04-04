from ml.trainer import train_model
import os

def main():
    # 데이터 파일 경로 설정 (필요에 따라 절대경로 또는 환경변수로 처리)
    train_csv = "/workspace/Storage/template_structured/Data/raw/train.csv"
    test_csv = "/workspace/Storage/template_structured/Data/raw/test.csv"
    
    train_model(train_csv, test_csv)

if __name__ == '__main__':
    main()
