import asyncio

from .base import BaseDataNode



class PubNode(BaseDataNode):
    async def get_data(self, key):
        if self._cache is not None:
            return self._cache.get(key)
        async with self._cache_lock:
            # 락 획득 후 다시 확인 (다른 태스크가 이미 계산했을 수 있음)
            if self._cache is not None:
                return self._cache.get(key)
            # 프리훅
            for hook in self.pre_hooks:
                asyncio.create_task(hook(self))
            # 데이터 처리
            result = await self._get_data() # 지연호출
            self._cache = result
            # 포스트훅
            for hook in self.post_hooks:
                asyncio.create_task(hook(self))
            return result.get(key)
    
    async def _get_data(self):
        data = await asyncio.gather(*[node.get_data() for node in self.prior_nodes])
        if self.process is not None:
            data = self.process(*data, **self.process_kwargs)
        return data
    
    
class SubNode(BaseDataNode):
    def __init__(self, prior_nodes, key, node_name, process=None):
        self.node_name = node_name
        self.prior_nodes = prior_nodes # 지연호출
        self.process = process
        self.key = key
        self._cache:dict = None
        self._cache_lock = asyncio.Lock()  # 락 추가
        self.pre_hooks = []   # 데이터 처리 전 실행할 훅(로깅, api훅 등)
        self.post_hooks = []  # 데이터 처리 후 실행할 훅(로깅, api훅 등)


    async def _get_data(self):
        data = await self.prior_nodes.get_data(self.key)
        if self.process is not None:
            data = self.process(*data, **self.process_kwargs)
        return data