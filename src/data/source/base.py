from abc import ABC, abstractmethod



class DataSource(ABC):
    """
    데이터를 불러오는 책임을 가진 추상 기본 클래스
    기존에 데이터 노드에 대한 정의와 불러오기책임으로 진행했는데, 추상화가 심해져서 일단 이것으로 진행
    이대로 source로 진행하면 추후 쓰기에 대한 기능을 붙여야 할 것으로 보인다.
    또한 파일시스템에서 lock 기능이 필요할 지도 모른다.
    이전에 정리했던 db와 s3에 대한 객체를 불러오자.
    """
    def __init__(self):
        """
        이전 노드로부터 하달받는 방식
        Args:
            parent_node: 이전 처리 단계의 DataSource 인스턴스 (선택적)
        """
        self._data = None
        

    @abstractmethod
    def _load_data(self):
        """구체적인 데이터 로딩 로직 (상속 클래스에서 구현)"""
        pass


    @property
    def data(self):
        """데이터에 대한 지연 로딩 구현"""
        if self._data is None:
            self._data = self._load_data()
        return self._data
    
    
    def get_data(self):
        """데이터 접근 메서드"""
        return self.data
    
    
