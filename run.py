"""
To run the experiment:
$ python run.py config.yaml

To see more options:
$ python run.py config.yaml -h
"""

from rlberry.experiment import experiment_generator
from rlberry.manager import MultipleManagers

if __name__ == "__main__":
    managers = MultipleManagers()

    for agent_manager in experiment_generator():
        managers.append(agent_manager)

    managers.run(save=True)
