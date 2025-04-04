from dataclasses import dataclass, field

from .exp_5 import Experiment_5  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_7(Experiment_5):
    exp_name : str = "exp_7"
    version_prompt_precendent : str = "exp_4" # 주요변경
    version_prompt_question : str = "exp_3" # 주요변경