from dataclasses import dataclass, field

from .base import Experiment  # 별도 파일에 선언된 dataclass



@dataclass    
class ExperimentSvmBase(Experiment):
    '''
    실험 파라미터
    '''
    # data 파라미터
    train: str = 'raw/train.csv'
    val: str = 'raw/val.csv'
    test: str =  'raw/test.csv'
    
    # model 파라미터
    model_name : str = "lgr_base"
    model_params : dict = field(default_factory=lambda: 
        {

        }
    )
    
    
    training_params : dict = field(default_factory=lambda: 
        {
            "kernel":'rbf', 
            "C":1.0, 
            "gamma":'scale', 
            "probability":False, 
            "random_state":42
        }
    )

