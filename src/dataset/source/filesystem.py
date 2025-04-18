import os

from .base import DataSource



class DataSourceFile(DataSource):
    """파일에서 데이터를 불러오는 기본 노드"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    

    def _load_data(self):
        """파일 존재 확인 후 상속 클래스에서 구체적인 로딩 구현"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
        return self._load_from_file()
    

    def _load_from_file(self):
        """구체적인 파일 로딩 로직 (상속 클래스에서 구현)"""
        raise NotImplementedError("상속 클래스에서 구현해야 합니다")


