import pandas as pd




class DataSourceCSV:
    """
    CSV 데이터를 로드하는 클래스로 데이터노드의 덕타이핑
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
        self.csv_path = csv_path
        self.read_csv_kwargs = read_csv_kwargs
        

    async def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path, **self.read_csv_kwargs)
