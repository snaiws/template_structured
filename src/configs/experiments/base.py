from dataclasses import dataclass, asdict, field



@dataclass
class Experiment:
    exp_name: str = field(init=False)

    def __post_init__(self):
        self.exp_name = self.__class__.__name__
        
    def to_dict(self):
        return asdict(self)
