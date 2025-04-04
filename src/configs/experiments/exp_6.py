from dataclasses import dataclass, field

from .exp_5 import Experiment_5  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_6(Experiment_5):
    exp_name : str = "exp_6"
    chain_strategy : str = "baseline"
    
    prompt_template_format : str = "exp_3"

    embedding_model_name_guideline :str = None
    retriever_guideline_name : str = None
    retriever_guideline_params : tuple = None
    splitter_guideline_name : str = None
    splitter_guideline_kwargs : dict = None