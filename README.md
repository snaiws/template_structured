# 1. 설명

1. 디렉토리 구조(주요파일, 폴더만 표시)
    
    ```
    Project
    ├src
    │├configs # 파라미터관리모듈
    │├data # 데이터모듈
    ││├node # 데이터노드
    ││├source # 데이터소스노드(덕타이핑, 미완)
    ││├process # 전처리 파이프라인 및 함수
    ││├dataset_loaninfo.py # 이것을 예시로 보시면 됩니다
    │├machine # 머신러닝모델모듈
    │├train_lgb.py # 이것을 예시로 보시면 됩니다
    ```
    
2. `configs` 파라미터 관리 모듈 설명
    
    이 프로젝트에선 실험 파라미터와 환경변수 파라미터로 나누었습니다.
    
    실험파라미터의 경우 `experiments` 폴더에서 상속과 데코레이터를 사용하여 정의하면 클래스명입력만으로 해당 실험파라미터를 불러올 수 있습니다.
    
    또한 `to_dict` 메소드를 통해 dictionary로 바꾼 후 kwargs로 넣거나 json/yaml로 내보낼 수 있습니다.
    
3. `data` 모듈 설명
    
    ![image.png](attachment:b35b1fec-b6ba-4ba8-80ca-9ab200d184a5:image.png)
    
    머신러닝 서비스 배포를 하던 저는 경험에 의해 데이터는 이러한 성질을 가진다는 것을 알게되었고, 입출력부가 특별한 DAG 방식으로 데이터를 관리하면 좋을 것 같다고 생각했습니다.
    
    1. `data node`
        
        base는 프록시패턴, 옵저버패턴을 사용하여 구현
        
        프록시 패턴으로 간편하게 추가기능 부착 등 가능
        
        옵저버패턴으로 pull방식 lazy loading 구현
        
        pub노드의 경우 sub노드 생성에 팩토리패턴을 사용
        
        미구현 : __gt__ 메소드 오버라이딩을 사용하여 node1 > node2만으로 연결하는 기능 → 시도했는데 pubsub노드를 고쳐야해서 스킵 / 멀티프로세스
        
    2. `source`
        
        덕타이핑을 사용하여 `data node`와 동일 메소드 공유하여 `data node`처럼 취급
        
        미구현 : get data를 제외한 부가적인 인터페이스
        
    3. `process`
        - 전처리 함수들
            
            데이터소스별로 폴더에 `preprocess.py`, `postprocess.py` 로 구분하여 보관(EDA 혹은 baseline 모델에서 사용한 전처리를 여기에 담아 재사용성 증가)
            
        - `pipeline`
            
            __call__ 메소드에 위의 전처리 함수들을 불러와 조합, `data node`의 
            
    4. 미구현
        - path 관리 클래스… Pathlib을 쓸지 컴포짓패턴 클래스로 관리할 지 아직 미정
4. `machine(core, model)` 모듈 설명
    
    어댑터패턴 사용하여 여러 프레임워크를 동일메소드로 사용
    
    미구현한 ml 모듈(추후 learning 혹은 프로세스 관련 이름으로 변경 예정)에서 사용함으로써 프레임워크가 달라도 인자에 모델을 넣어 동일 기능으로 작동
    
    ```
    def some_kind_of_ML_process(machine,...):
    	...
    	machine.train(...)
    	...
    	machine.predict(...)
    ```
    
5. 기타 미구현
    
    ml 모듈 - airflow의 DAG는 함수형프로그래밍이라서 일단 모듈화 안함
    
    optuna 이식
    
    자원관리, 멀티프로세스를 위한 브로커(airflow, celery 사용 예상)
