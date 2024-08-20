from class_argparse import ClassArgParser
from .experiments import experiments, EXPERIMENT_KEYS, TASK_KEYS


class Main(ClassArgParser):
    def __init__(self) -> None:
        super().__init__(name="DIVR SSL")

    def experiment(self, experiment_key: EXPERIMENT_KEYS, task_key: TASK_KEYS) -> None:
        experiments[experiment_key](task_key)


if __name__ == "__main__":
    Main()()
