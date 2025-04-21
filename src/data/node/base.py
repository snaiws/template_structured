from abc import ABC, abstractmethod
import asyncio
import weakref



# 데이터노드 인터페이스
class BaseDataNode(ABC):
    """
    프록시 패턴을 통해 데이터를 제어, 로깅, 지연호출
    """
    def __init__(self, *prior_nodes, element, process=None):
        self.element = element # id or path
        self.prior_nodes = prior_nodes # 지연호출
        self.process = process
        self._cache:dict = None
        self._cache_lock = asyncio.Lock()  # 락 추가
        self.pre_hooks = []   # 데이터 처리 전 실행할 훅(로깅, api훅 등)
        self.post_hooks = []  # 데이터 처리 후 실행할 훅(로깅, api훅 등)


    async def get_data(self):
        if self._cache is not None:
            return self._cache
        async with self._cache_lock:
            # 락 획득 후 다시 확인 (다른 태스크가 이미 계산했을 수 있음)
            if self._cache is not None:
                return self._cache
            # 프리훅
            for hook in self.pre_hooks:
                asyncio.create_task(hook(self))
            # 데이터 처리
            result = await self._get_data() # 지연호출
            self._cache = result
            # 포스트훅
            for hook in self.post_hooks:
                asyncio.create_task(hook(self))
            return result
    

    @abstractmethod
    async def _get_data(self):
        '''
        실제 데이터 호출
        '''
        pass
    
    
    def add_pre_hook(self, hook):
        """
        데이터 처리 전에 실행할 훅 추가
        hook : 비동기 함수
        """
        self.pre_hooks.append(hook)
        return self  # 메서드 체이닝을 위한 반환
    

    def add_post_hook(self, hook):
        """
        데이터 처리 전에 실행할 훅 추가
        hook : 비동기 함수
        """
        self.post_hooks.append(hook)
        return self  # 메서드 체이닝을 위한 반환

