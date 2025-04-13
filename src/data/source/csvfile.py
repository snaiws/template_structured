import pandas as pd

from .filesystem import DataSourceFile



class DataSourceCSV(DataSourceFile):
    """
    CSV 데이터를 로드하는 클래스
    """
    def __init__(
        self, 
        csv_path: str,
        **read_csv_kwargs
    ):
        """
        csv_path: 데이터 파일 경로
        read_csv_kwargs: pandas read_csv 옵션들
        """
        super().__init__(file_path = csv_path)
        self.csv_path = csv_path
        self.read_csv_kwargs = read_csv_kwargs
        

    def _load_from_file(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path, **self.read_csv_kwargs)



if __name__ == "__main__":
    path = "/workspace/Storage/template_structured/Data/raw/train.csv"
    data = DataSourceCSV(path)
    print(data.get_data())