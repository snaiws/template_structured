from .environments import EnvDefineUnit
from .experiments import versions

    

class Configs:
    def __init__(self, exp_name="Experiment_0"):
        self.exp_name = exp_name

    @property
    def env(self):
        return EnvDefineUnit()

    @property
    def exp(self):
        exp_class = versions[self.exp_name]
        return exp_class()
    


if __name__ == "__main__":
    print(versions)
    tool = Configs("ExperimentLgbBase")
    environment = tool.env  # 한번만 호출하고 반환값 사용
    experiment = tool.exp  # 한번만 호출하고 반환값 사용
    print(environment)
    print(experiment)