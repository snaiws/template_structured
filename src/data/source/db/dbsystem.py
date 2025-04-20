from typing import Dict, ClassVar, List, Tuple
import asyncio

import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncEngine


class DBManager:
    _instances: ClassVar[Dict[str, 'DBManager']] = {}
    _lock = asyncio.Lock()
    
    def __new__(cls, host, *args, **kwargs):
        # host를 키로 사용하여 인스턴스 관리
        if host not in cls._instances:
            cls._instances[host] = super(DBManager, cls).__new__(cls)
        return cls._instances[host]
    
    def __init__(self, db_type: str, user: str, password: str, host: str, port: int, database: str, driver: str = None):
        # 이미 초기화된 인스턴스인 경우 중복 초기화 방지
        if hasattr(self, '_initialized') and self._initialized and self.host == host:
            return
        
        self.engine: AsyncEngine = None
        self.db_type = db_type
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.driver = driver
        # 비동기 초기화는 직접 호출해야 함
        self._initialized = True

    async def connect(self):
        """데이터베이스 연결 비동기 설정"""
        try:
            # 연결 문자열 생성
            connection_string = self._get_connection_string()
            self.engine = create_async_engine(connection_string)
            print(f"Connected to {self.db_type} database successfully.")
            return self
        except Exception as e:
            print(f"Error connecting to {self.db_type} database: {e}")
            self.engine = None
            raise e

    @classmethod
    async def create(cls, db_type: str, user: str, password: str, host: str, port: int, database: str, driver: str = None):
        """비동기 팩토리 메서드로 인스턴스 생성 및 초기화"""
        async with cls._lock:
            instance = cls(db_type, user, password, host, port, database, driver)
            await instance.connect()
            return instance

    def _get_connection_string(self) -> str:
        """데이터베이스에 맞는 비동기 연결 문자열 생성"""
        if self.db_type == "mysql":
            driver = self.driver or "aiomysql"  # 비동기 드라이버
            return f"mysql+{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == "postgresql":
            driver = self.driver or "asyncpg"  # 비동기 드라이버
            return f"postgresql+{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == "sqlite":
            # SQLite는 호스트나 포트가 필요 없음
            return f"sqlite+aiosqlite:///{self.database}"
        else:
            raise ValueError(f"Unsupported database type for async: {self.db_type}")
        
    async def getter(self, query: str, params: dict = None) -> List[Tuple]:
        """SELECT 쿼리 비동기 실행 및 튜플 형태로 반환"""
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(text(query), params or {})
                return [tuple(row) for row in await result.fetchall()]
        except Exception as e:
            print(f"Error in async getter: {e}")
            raise e

    async def setter(self, query: str, params: dict = None) -> None:
        """INSERT, UPDATE, DELETE 쿼리 비동기 실행"""
        try:
            async with self.engine.begin() as connection:  # 트랜잭션 자동 처리
                await connection.execute(text(query), params or {})
        except Exception as e:
            print(f"Error in async setter: {e}")
            raise e

    async def close(self):
        """연결 비동기 종료"""
        try:
            if self.engine:
                await self.engine.dispose()
                print("Database connection closed.")
        except Exception as e:
            print(f"Error closing database connection: {e}")




if __name__ == "__main__":
    from dotenv import load_dotenv
    
    async def main():
        # .env 파일 로드
        load_dotenv()
        
        # 비밀 변수 가져오기
        DB_TYPE = os.getenv("DB_TYPE")  # mysql, postgresql, sqlite
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = int(os.getenv("DB_PORT"))
        DB_NAME = os.getenv("DB_NAME")
        DB_DRIVER = os.getenv("DB_DRIVER", None)  # 선택적
        
        # Database 인스턴스 비동기 생성
        db = await Database.create(
            db_type=DB_TYPE,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            driver=DB_DRIVER
        )
        
        # getter 예제
        select_query = "SELECT * FROM some_table LIMIT 10"
        try:
            data = await db.getter(select_query)
            print(data)
        except Exception as e:
            print(f"Query failed: {e}")
        
        # 연결 종료
        await db.close()
    
    # 비동기 메인 함수 실행
    asyncio.run(main())