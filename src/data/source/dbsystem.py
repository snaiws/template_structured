from .base import DataSource


#추후 수정!
class DataSourceDB(DataSource):
    """데이터베이스에서 데이터를 불러오는 기본 노드"""
    
    def __init__(self, query, x_cols, y_col=None, **kwargs):
        super().__init__(**kwargs)
        self.query = query
        self.x_cols = x_cols
        self.y_col = y_col
    
    def _load_data(self):
        """데이터베이스 연결, 쿼리 실행 및 데이터 로딩"""
        # 데이터베이스 연결 객체 획득
        conn = self._get_connection()
        
        try:
            # 쿼리 실행 및 데이터프레임으로 변환
            df = self._execute_query(conn, self.query)
            
            # 필요한 열 추출
            X = df[self.x_cols].values if isinstance(self.x_cols, list) else df[self.x_cols].values
            y = None
            if self.y_col and self.y_col in df.columns:
                y = df[self.y_col].values
                
            return X, y
        finally:
            # 연결 종료
            self._close_connection(conn)