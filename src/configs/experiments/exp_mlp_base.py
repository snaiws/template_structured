from dataclasses import dataclass, field

from .base import Experiment, register  # 별도 파일에 선언된 dataclass



@register
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
            "input_dim" : 43,
            "hidden_dims" : [64,64],
            "batch_size" : 32,
            "num_epochs" : 20,
            "lr_optimizer" : 0.001
        }
    )


