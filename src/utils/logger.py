import os
import time
import psutil
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

# GPU 사용량 로깅을 위한 pynvml 초기화 (GPU가 없는 경우 예외 처리)
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
except Exception as e:
    logging.warning("pynvml 모듈이 없거나 GPU 사용량 로깅은 비활성화됩니다.")
    gpu_available = False

# 로그 파일 설정 함수 (INFO, DEBUG, ERROR 각각 별도 파일, 하루 단위 회전 및 보관)
def setup_logger(LOG_DIR, now):
    """
    LOG_DIR 디렉토리에 현재 시간(now)를 포함한 파일명으로 INFO, DEBUG, ERROR 로그 파일을 생성합니다.
    INFO, DEBUG 로그는 7일, ERROR 로그는 30일간 보관하며, 하루 단위로 로그 파일을 회전합니다.
    로그 포매터는 파이프(|) 구분자를 사용하여 CSV 형태로 기록합니다.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 파이프(|) 구분자를 사용하여 각 필드(타임스탬프, 레벨, 메시지)를 기록
    formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # INFO 로그: INFO 이상의 로그 기록 (INFO, WARNING, ERROR, CRITICAL)
    info_log_file = os.path.join(LOG_DIR, f"{now}_info.log")
    info_handler = logging.handlers.TimedRotatingFileHandler(info_log_file, when="D", interval=1, backupCount=7, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    
    # DEBUG 로그: DEBUG 이상의 로그 기록 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    debug_log_file = os.path.join(LOG_DIR, f"{now}_debug.log")
    debug_handler = logging.handlers.TimedRotatingFileHandler(debug_log_file, when="D", interval=1, backupCount=7, encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)
    
    # ERROR 로그: ERROR 이상의 로그 기록 (ERROR, CRITICAL)
    error_log_file = os.path.join(LOG_DIR, f"{now}_error.log")
    error_handler = logging.handlers.TimedRotatingFileHandler(error_log_file, when="D", interval=1, backupCount=30, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

# 리소스 상태 수집 함수
def inspect_server_resources(unit="MB", round_val=2):
    """
    psutil을 사용하여 CPU, 메모리, 디스크, 네트워크 사용량을 지정한 단위(unit)와 반올림(round_val)으로 반환합니다.
    """
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()

    unit_converter = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    converter = lambda x: round(x / unit_converter[unit], round_val)
    
    metrics = {
        "cpu_usage": psutil.cpu_percent(interval=0),
        "memory_used": converter(memory.used),
        "memory_total": converter(memory.total),
        "disk_used": converter(disk.used),
        "disk_total": converter(disk.total),
        "network_sent": converter(network.bytes_sent),
        "network_received": converter(network.bytes_recv),
    }
    return metrics

# 현재 리소스 상태를 로그에 기록
def logging_resource():
    metrics = inspect_server_resources()
    message = " | ".join([
        f"cpu_usage: {metrics['cpu_usage']}%",
        f"memory_used: {metrics['memory_used']}",
        f"memory_total: {metrics['memory_total']}",
        f"disk_used: {metrics['disk_used']}",
        f"disk_total: {metrics['disk_total']}",
        f"network_sent: {metrics['network_sent']}",
        f"network_received: {metrics['network_received']}"
    ])
    logging.info(message)

# 데코레이터: 함수 실행 전후의 시작/종료 시각과 소요시간, 리소스 상태를 파이프(|) 구분자로 로그 기록
def log_resources(func):
    def wrapper(*args, **kwargs):
        # 실행 시작 시간 측정
        start_time = time.time()
        start_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        
        # (옵션) 실행 전 리소스 상태 로그 기록
        mem_before = psutil.virtual_memory().used
        msg_start = f"FUNCTION_START|{func.__name__}|{start_str}|RAM={mem_before}"
        if gpu_available:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            msg_start += f"|GPU={gpu_mem_before}"
        logging.info(msg_start)
        
        result = func(*args, **kwargs)
        
        # 실행 종료 시간 측정
        end_time = time.time()
        end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        mem_after = psutil.virtual_memory().used
        msg_end = f"FUNCTION_END|{func.__name__}|{end_str}|RAM={mem_after}"
        if gpu_available:
            gpu_mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            msg_end += f"|GPU={gpu_mem_after}"
        logging.info(msg_end)
        
        # 실행 소요시간 로그 (분석에 사용)
        duration = end_time - start_time
        # 메시지 형식: FUNCTION_EXECUTION|함수명|시작시간|종료시간|소요시간(초)
        logging.info(f"FUNCTION_EXECUTION|{func.__name__}|{start_str}|{end_str}|{duration:.3f}")
        return result
    return wrapper

# 데코레이터: 함수의 입출력 값을 파이프(|) 구분자로 로그 기록
def log_io(func):
    def wrapper(*args, **kwargs):
        logging.info(f"FUNCTION_IO|{func.__name__}|INPUT|{args}|{kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"FUNCTION_IO|{func.__name__}|OUTPUT|{result}")
        return result
    return wrapper

# 두 데코레이터를 적용한 예시 함수
@log_resources
@log_io
def sample_function(x, y):
    time.sleep(1)  # 실행 시간 측정을 위한 지연
    return x + y

# 로그 분석 함수 (pandas 사용)
def analyze_execution_times_pandas(log_file: str):
    """
    지정된 로그 파일에서 FUNCTION_EXECUTION 메시지를 분석하여 함수별 평균 실행 시간을 계산하여 출력합니다.
    
    로그 메시지 예시:
    2025-03-13 12:34:56|INFO|FUNCTION_EXECUTION|sample_function|2025-03-13 12:34:56|2025-03-13 12:34:57|1.000
    """
    import pandas as pd

    # 로그 파일 전체를 파이프(|) 구분자로 읽음
    # 각 로그 라인의 필드 수가 다를 수 있으므로, 최소 7개 컬럼까지 읽음
    df = pd.read_csv(log_file, delimiter="|", header=None, names=["timestamp", "level", "field1", "field2", "field3", "field4", "field5"], engine="python")
    
    # FUNCTION_EXECUTION 메시지의 경우, field1는 "FUNCTION_EXECUTION", field2: 함수명, field3: 시작시간, field4: 종료시간, field5: 소요시간
    df_exec = df[df["field1"] == "FUNCTION_EXECUTION"].copy()
    if df_exec.empty:
        print("FUNCTION_EXECUTION 로그 메시지를 찾을 수 없습니다.")
        return

    # 소요시간을 숫자로 변환 (초)
    df_exec["duration"] = pd.to_numeric(df_exec["field5"], errors="coerce")
    
    # 함수명별 평균 소요시간 계산
    result = df_exec.groupby("field2")["duration"].mean().reset_index()
    result.columns = ["Function", "Average Execution Time (sec)"]
    print(result)

# 테스트용 코드
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(verbose=False)
    
    # 현재 시각 문자열을 생성하는 함수 (없을 경우 기본 제공)
    try:
        from now import get_now
    except ImportError:
        def get_now():
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    LOG_DIR = os.getenv("PATH_LOG_VIRTUAL", ".")
    now = get_now()
    setup_logger(LOG_DIR, now)
    
    # 기본 로그 메시지 테스트
    logging.info("This is an INFO log.")
    logging.debug("This is a DEBUG log.")
    logging.error("This is an ERROR log.")
    
    # 리소스 정보 로그 기록
    logging.info(f"Resource metrics: {inspect_server_resources()}")
    logging_resource()
    
    # 샘플 함수 실행 테스트
    sample_function(5, 7)
    
    # 로그 분석 예시 (여기서는 INFO 로그 파일을 대상으로 함)
    # 실제 분석 시 로그 파일 경로를 지정하세요.
    log_file_path = os.path.join(LOG_DIR, f"{now}_info.log")
    analyze_execution_times_pandas(log_file_path)
