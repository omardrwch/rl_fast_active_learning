# Fast active learning for pure exploration in reinforcement learning

Paper available [here](https://arxiv.org/abs/2007.13442).

The algorithm is implemented in the folder `algorithms/`. The folder `config/` contains the parameters defining the experiments.

* Requirements:
    * Python 3.7
    * [`rlberry`](https://github.com/rlberry-py/rlberry) version 0.1
    * pyyaml

* Create and activate conda virtual environment (optional)

```bash
$ conda create -n ucbmq_env python=3.7
$ conda activate ucbmq_env
```

* Install requirements

```bash
$ pip install 'rlberry[full]==0.1'
$ pip install pyyaml
```

* Run and plot

```bash
$ python run.py config/experiment.yaml --n_fit=4
$ python plot.py
```
