from dataclasses import dataclass, field

from .base import Experiment  # 별도 파일에 선언된 dataclass



@dataclass    
class ExperimentLgbBase(Experiment):
    '''
    실험 파라미터
    '''
    # data 파라미터
    train: str = 'raw/train.csv'
    
    # model 파라미터
    model_name : str = "lgb_base"
    model_params : dict = field(default_factory=lambda: 
        {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
        }
    )

    training_params : dict = field(default_factory=lambda: 
        {
            "earlystopping_round" : 50,
            "log_eval_period" : 100,
            "num_boost_round" : 1000
        }
    )


