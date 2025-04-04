from dataclasses import dataclass, field

from .base import Experiment  # 별도 파일에 선언된 dataclass



@dataclass    
class ExperimentMlpBase(Experiment):
    '''
    실험 파라미터
    '''
    # data 파라미터
    train: str = 'raw/train.csv'
    
    # model 파라미터
    model_name : str = "mlp_base"
    model_params : dict = field(default_factory=lambda: 
        {
            "hidden_dim" : [64,64]
        }
    )

    training_params : dict = field(default_factory=lambda: 
        {
            "batch_size" : 32,
            "num_epochs" : 20,
            "lr_optimizer" : 0.001
        }
    )


