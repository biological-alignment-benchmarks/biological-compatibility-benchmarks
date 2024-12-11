# aintelope

We operationalize conjectures from Steven Byrnes’
[“Reverse-engineer human social instincts”](https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8/p/tj8AC3vhTnBywdZoA)
research program and extend existing research into brain-like AI. We simulate
agents with reinforcement learning over selected cue-response patterns in
environments that could give rise to humanl-iike complex behaviors. To do so we
select cue-responses from a pre-existing list of more than 60 candidates for
human affects and instincts from affective neuroscience and other sources
including original patterns. Cue-responses are conjectured to form hierarchies
and the project will start the simulation of lower-level patterns first. We
intend to verify the general principles and the ability of our software to model
and simulate the agents and the environment to a sufficient degree.

Working document:
https://docs.google.com/document/d/1qc6a3MY2_guCZH8XJjutpaASNE7Zy6O5z1gblrfPemk/edit#

## Project setup


### Installation under Linux

The project installation is managed via `make` and `pip`. Please see the respective commands in the `Makefile`. To setup the environment follow these steps:

1. Install CPython. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager. 

Under Linux, run the following commands:

`sudo add-apt-repository ppa:deadsnakes/ppa`
<br>`sudo apt update`
<br>`sudo apt install python3.10 python3.10-dev python3.10-venv`
<br>`sudo apt install curl`
<br>`sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10`

2. Get the code from repo:

`sudo apt install git-all`
<br>`git clone https://github.com/GunnarZarncke/aintelope.git`

3. Create a virtual python environment:

`make venv-310`
<br>`source venv_aintelope/bin/activate`

4. Install dependencies:

`sudo apt update`
<br>`sudo apt install build-essential`
<br>`make install`

5. If you use VSCode, then set up your launch configurations file:

`cp .vscode/launch.json.template .vscode/launch.json`

Edit the launch.json so that the PYTHONPATH variable points to the folder where you downloaded the repo and installed virtual environment:

replace all
<br>//"PYTHONPATH": "your_path_here"
<br>with
<br>"PYTHONPATH": "your_local_repo_path"

6. For development and testing:

* Install development dependencies: `make install-dev`
* Run tests: `make tests-local`

7. Location of an example agent you can use as a template for building your custom agent: 
[`aintelope/agents/example_agent.py`](aintelope/agents/example_agent.py)


### Installation under Windows

1. Install CPython from python.org. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager.

You can download the latest installer from https://www.python.org/downloads/release/python-31010/ or if you want to download a newer 3.10.x version then from https://github.com/adang1345/PythonWindows

2. Get the code from repo:
* Install Git from https://gitforwindows.org/
* Open command prompt and navigate top the folder you want to use for repo
* Run `git clone https://github.com/GunnarZarncke/aintelope.git`
* Run `cd biological-compatibility-benchmarks`

3. Create a virtual python environment by running: 
<br>`python -m venv venv_aintelope`
<br>`venv_aintelope\scripts\activate`

4. Install dependencies by running:
<br>`pip uninstall -y ai_safety_gridworlds >nul 2>&1`
<br>`pip install -r requirements/api.txt`

5. If you use VSCode, then set up your launch configurations file:

`copy .vscode\launch.json.template .vscode\launch.json`

Edit the launch.json so that the PYTHONPATH variable points to the folder where you downloaded the repo and installed virtual environment:

replace all
<br>//"PYTHONPATH": "your_path_here"
<br>with
<br>"PYTHONPATH": "your_local_repo_path"

6. For development and testing:

* Install development dependencies: `pip install -r requirements/dev.txt`
* Run tests: `python -m pytest --tb=native --cov="aintelope tests"`

7. Location of an example agent you can use as a template for building your custom agent: 
[`aintelope\agents\example_agent.py`](aintelope/agents/example_agent.py)


### Code formatting and style

To automatically sort the imports you can run
[`isort aintelope tests`](https://github.com/PyCQA/isort) from the root level of the project.
To autoformat python files you can use [`black .`](https://github.com/psf/black) from the root level of the project.
Configurations of the formatters can be found in `pyproject.toml`.
For linting/code style use [`flake8`](https://flake8.pycqa.org/en/latest/).

These tools can be invoked via `make`:

```bash
make isort
make format
make flake8
```

## Executing `aintelope`

Try `make run-training`. Then look in `aintelope/outputs/memory_records`. (WIP)
There should be two new files named `Record_{current timestamp}.csv` and
`Record_{current timestamp}_plot.png`. The plot will be an image of the path the
agent took during the test episode, using the best agent that the training
produced. Green dots are food in the environment, blue dots are water.

## Experiment Analysis

To see the results, do the following:
1. Run the following n-times (you can choose n, say 3, this is just for statistical significance):
  `make run-training-baseline`
  `make run-training-instinct`
2. Run `jupyter lab`, and run the blocks by targeting them and Shift+Enter/play button.
  Initialize: run the first three blocks to start
  Then run the blocks under a title to show those results
There are currently three distinct plots available, training plots, E(R) -plots and simulations of the trained models. 

Some metrics and visualizations are logged with
[`tensorboard`](https://www.tensorflow.org/tensorboard). This information can be
accessed by starting a `tensorboard` server locally. To do that switch to the
directory where pytorch-lightning stores the experiments (e.g.
`outputs/lightning_logs`). Your `aintelope` environment needs to be _active_
(`tensorboard` is installed automatically from the requirements). Within you
find one folder for each experiment containing `events.out.tfevents.*` files.
Start the server via

```
cd outputs/lightning_logs
tensorboard --logdir=. --bind_all
```

You can access the dashboard using your favorit browser at `127.0.0.1:6006` (the
port is also shown in the command line).

## Logging

The logging level can be controlled via hydra. By adding `hydra.verbose=True`
all loggers will be executed with level `DEBUG`. Alternatively a string or list
of loggers can be provided. See the
[documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/)
for more details.

## Windows

Aintelope code base is compatible with Windows. No extra steps needed. GPU computation works fine as well. WSL is not needed.

# Differences to regular RL

For alignment and cognitive research, the internal reward of the agent and 
the actual score from the desired behaviour are measured separately. 
The reward comes from the agent.py itself, while the desired score comes from 
the environment (and thus the test). Both of these values are then recorded and 
compared during analysis.

# License

This project is licensed under the Mozilla Public License 2.0. You are free to use, modify, and distribute this code under the terms of this license.

**Attribution Requirement**: If you use this benchmark suite, please cite the source as follows:

Roland Pihlakas and Joel Pyykkö. From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks. Arxiv, a working paper, September 2024 (https://arxiv.org/abs/2410.00081).

**Use of Entire Suite**: We encourage the inclusion of the entire benchmark suite in derivative works to maintain the integrity and comprehensiveness of AI safety assessments.

For more details, see the [LICENSE.txt](LICENSE.txt) file.
