from dataclasses import dataclass, field

from .base import Experiment  # 별도 파일에 선언된 dataclass



@dataclass    
class ExperimentLgrBase(Experiment):
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
            "penalty" : 'l2',
            "C" : 1.0,
            "solver" : "lbfgs",
            "max_iter" : 1000,
            "random_state" : 42
        }
    )
