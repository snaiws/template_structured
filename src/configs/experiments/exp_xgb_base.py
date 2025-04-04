from dataclasses import dataclass, field

from .base import Experiment  # 별도 파일에 선언된 dataclass



@dataclass
class ExperimentXgbBase(Experiment):
    '''
    실험 파라미터
    '''
    # data 파라미터
    train: str = 'raw/train.csv'
    
    # model 파라미터
    model_name : str = "xbg_base"
    model_params : dict = field(default_factory=lambda: 
        {
            "objective": "binary:logistic",  # 이진 분류
            "eval_metric": "logloss",
            "eta": 0.05,                     # 학습률
            "max_depth": 6,
            "subsample": 0.9
        }
    )

    # train 파라미터터
    training_params : dict = field(default_factory=lambda: 
        {
            "num_boost_round":1000,
            "earlystopping_round":50
        }
    )