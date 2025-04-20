from abc import ABC, abstractmethod



class DataSource(ABC):
    """
    데이터를 불러오는 책임을 가진 추상 기본 클래스
    기존에 데이터 노드에 대한 정의와 불러오기책임으로 진행했는데, 추상화가 심해져서 일단 이것으로 진행
    이대로 source로 진행하면 추후 쓰기에 대한 기능을 붙여야 할 것으로 보인다.
    또한 파일시스템에서 lock 기능이 필요할 지도 모른다.
    이전에 정리했던 db와 s3에 대한 객체를 불러오자.
    """
    @abstractmethod
    async def get_data(self):
        """구체적인 데이터 로딩 로직 (상속 클래스에서 구현)"""
        pass


