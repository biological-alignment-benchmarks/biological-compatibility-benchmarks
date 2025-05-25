# aintelope

We operationalize conjectures from Steven Byrnes’
[“Reverse-engineer human social instincts”](https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8/p/tj8AC3vhTnBywdZoA)
research program and extend existing research into brain-like AI. We simulate
agents with reinforcement learning over selected cue-response patterns in
environments that could give rise to human-like complex behaviors. To do so we
select cue-responses from a pre-existing list of more than 60 candidates for
human affects and instincts from affective neuroscience and other sources
including original patterns. Cue-responses are conjectured to form hierarchies
and the project will start the simulation of lower-level patterns first. We
intend to verify the general principles and the ability of our software to model
and simulate the agents and the environment to a sufficient degree.

Working document:
https://docs.google.com/document/d/1qc6a3MY2_guCZH8XJjutpaASNE7Zy6O5z1gblrfPemk/edit#

## Project setup

This readme contains instructions for both Linux and Windows installation. Windows installation instructions are located after Linux installation instructions.

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
<br>Run `git clone https://github.com/GunnarZarncke/aintelope.git`
<br>Run `cd aintelope`

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
* Open command prompt and navigate to the folder you want to use for repo
* Run `git clone https://github.com/GunnarZarncke/aintelope.git`
* Run `cd aintelope`

3. Create a virtual python environment by running: 
<br>3.1. To activate VirtualEnv with Python 3.10:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`virtualenv -p python3.10 venv_aintelope` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(or if you want to use your default Python version: 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python -m venv venv_aintelope`)
<br>3.2. `venv_aintelope\scripts\activate`

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


### Setting up the LLM API access

Set environment variable:
`OPENAI_API_KEY`.

Ensure you have loaded credits on your OpenAI API account, else you will get "rate limit errors".


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

In the folder `.vscode` there is a file named `launch.json.template`. Copy that file to `launch.json`. This is a VSCode launch configurations file, containing many launch configurations. (The original file named `launch.json.template` is necessary so that your local changes to launch configurations do not end up in the Git repository.)

Alternatively, try executing `make run-training-baseline`. You do not need VSCode for running this command. Then look in `aintelope/outputs`. This command will execute only one of many available launch configurations present in `launch.json`.

### Executing LLM agent

For LLM agent, there are the following launch configurations in `launch.json`:
- Run single environment with LLM agent and default params
- Run pipeline with LLM agent and default params
- Run BioBlue pipeline with LLM agent and default params
- Run multiple trials pipeline with LLM agent and default params
- Run multiple trials BioBlue pipeline with LLM agent and default params


## Experiment Analysis

To see the results, do the following:
Run the following:
  `make run-training-baseline`
  `make run-training-instinct`


To see the Tensorboard, run
`tensorboard --log_dir outputs/TIMESTAMP/EXPERIMENT/tensorboard/`
then go to http://localhost:6006/ in the browser.


## Actions map

The actions the agents can take have the following mapping:
```
  NOOP = 0
  LEFT = 1
  RIGHT = 2
  UP = 3
  DOWN = 4
```

Eating and drinking are not separate actions. Eating and drinking occurs always when an action ends with the agent being on top of a food or water tile, correspondingly. If the agent continues to stay on that tile then eating and drinking continues until the agent leaves. Likewise with collecting gold and silver. The agent is harmed by danger tile or predator, when the agent action ends up on a danger tile or predator tile.

Additionally, when `observation_direction_mode` = 2 or `action_direction_mode` = 2 then the following actions become available:
```
  TURN_LEFT_90 = 5
  TURN_RIGHT_90 = 6
  TURN_LEFT_180 = 7
  TURN_RIGHT_180 = 8
```
By default, the separate turning actions are turned off.


## Human-playable demos

In the folder `aintelope\environments\demos\gridworlds\` are located the human-playable demo environments, which have same configuration as the benchmarks in our pipeline. Playing these human-playable demos manually can give you a better intuition of the rules and how the benchmarks work.

You can launch these Python files without additional arguments. 

You can move the agents around using arrow keys (left, right, up, down). For no-op action you can use space key. 

In food sharing environment there are two agents. In a human-playable demo these agents take turns. In an RL setting they agents take actions concurrently and the environment implements their actions in a random order (randomising the order for each turn).

The human-playable benchmark environments are in the following files:
```
food_unbounded.py
danger_tiles.py
predators.py
food_homeostasis.py
food_sustainability.py
food_drink_homeostasis.py
food_drink_homeostasis_gold.py
food_drink_homeostasis_gold_silver.py
food_sharing.py
```


## Logging

TODO


## Windows

Aintelope code base is compatible with Windows. No extra steps needed. GPU computation works fine as well. WSL is not needed.


# Differences to regular RL

For alignment and cognitive research, the internal reward of the agent and 
the actual score from the desired behaviour are measured separately. 
The reward comes from the agent.py itself, while the desired score comes from 
the environment (and thus the test). Both of these values are then recorded and 
compared during analysis.


# Papers

* A working paper related to this repo: Pihlakas, R & Pyykkö, J. **"From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks"**. Arxiv (2024-2025). https://arxiv.org/abs/2410.00081 
<br>Data files of the experiments performed for this paper will be available here: https://drive.google.com/drive/folders/1KBRcPLbM2a8Li3HQ92lGTEfizOGtZ7OM?usp=sharing These data files contain logs of the steps agents took across training and test episodes, and multi-objective environment scores calculated during each step.


# Blog posts

* **Why modelling multi-objective homeostasis is essential for AI alignment (and how it helps with AI safety as well)** (2025) https://www.lesswrong.com/posts/vGeuBKQ7nzPnn5f7A/why-modelling-multi-objective-homeostasis-is-essential-for


# Presentations

* At **VAISU unconference**, May 2024:
    - Demo and feedback session - AI safety benchmarking in multi-objective multi-agent gridworlds - Biologically essential yet neglected themes illustrating the weaknesses and dangers of current industry standard approaches to reinforcement learning. 
    - Video: https://www.youtube.com/watch?v=ydxMlGlQeco
    - Slides: https://bit.ly/bmmbs
* At **Foresight Institute's Intelligent Cooperation Group**, Nov 2024: 
    - The subject of the presentation was describing why we should consider fundamental yet neglected principles from biology and economics when thinking about AI alignment, and how these considerations will help with AI safety as well (alignment and safety were treated in this research explicitly as separate aspects, which both benefit from consideration of aforementioned principles). These principles include homeostasis and diminishing returns in utility functions, and sustainability. Next I introduce multi-objective and multi-agent gridworlds-based benchmark environments we have created for measuring the performance of machine learning algorithms and AI agents in relation to their capacity for biological and economical alignment. The benchmarks are now available as a public repo. At the end I mention some of the related themes and dilemmas not yet covered by these benchmarks, and describe new benchmark environments we have planned for future implementation.
    - Recording: https://www.youtube.com/watch?v=DCUqqyyhcko
    - Slides: https://bit.ly/beamm 


# Dependencies

* **Extended, multi-agent and multi-objective version of AI Safety Gridworlds** - Extended, multi-agent and multi-objective (MaMoRL / MoMaRL) environments based on DeepMind's AI Safety Gridworlds. This is a suite of reinforcement learning environments illustrating various safety properties of intelligent agents. It is made compatible with OpenAI's Gym/Gymnasium and Farama Foundation PettingZoo. https://github.com/levitation-opensource/ai-safety-gridworlds
* **Zoo to Gym Multi-Agent Adapter** - Enables you to convert a PettingZoo environment to a Gym environment while supporting multiple agents (MARL). Gym's default setup doesn't easily support multi-agent environments, but this wrapper resolves that by running each agent in its own process and sharing the environment across those processes. https://github.com/levitation-opensource/zoo_to_gym_multiagent_adapter


# Related work

* **Notable runaway-optimiser-like LLM failure modes on Biologically and Economically aligned AI safety benchmarks for LLMs with simplified observation format** https://www.lesswrong.com/posts/PejNckwQj3A2MGhMA/notable-runaway-optimiser-like-llm-failure-modes-on
* **BioBlue: Biologically and economically aligned AI safety benchmarks for LLM-s with simplified observation format** (Roland Pihlakas, Shruti Datta Gupta, Sruthi Kuriakose 2025) [repo](https://github.com/levitation-opensource/bioblue) and [PDF report](https://github.com/levitation-opensource/bioblue/blob/main/BioBlue%20-%20Biologically%20and%20economically%20aligned%20AI%20safety%20benchmarks%20for%20LLMs.pdf)


# License

This project is licensed under the Mozilla Public License 2.0. You are free to use, modify, and distribute this code under the terms of this license.

**Attribution Requirement**: If you use this benchmark suite, please cite the source as follows:

Roland Pihlakas and Joel Pyykkö. From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks. Arxiv, a working paper, September 2024 (https://arxiv.org/abs/2410.00081).

**Use of Entire Suite**: We encourage the inclusion of the entire benchmark suite in derivative works to maintain the integrity and comprehensiveness of AI safety assessments.

For more details, see the [LICENSE.txt](LICENSE.txt) file.
