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

### Installation

The project installation is managed via `make` and `pip`. Please see the
respective commanads in the `Makefile`. To setup the environment follow these
steps:

1. Create a virtual python environment: `make venv`
2. Activate the environment: `source venv_aintelope/bin/activate`
3. Install dependencies: `make install`

For development and testing follow (active environment):

1. Install development dependencies: `make install-dev`
2. Install project locally: `make build-local`
3. Run tests: `make tests-local`

### Code formatting and style

To autoformat python files you can use [`black`](https://github.com/psf/black).
To automatically sort the imports you can run
[`isort .`](https://github.com/PyCQA/isort) from the root level of the project.
Configurations of the formatters can be found in `pyproject.toml`.
For linting/code style use [`flake8`](https://flake8.pycqa.org/en/latest/).

These tools can be invoked via `make`:

```bash
make format
make isort
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

# Cygwin

If you are a Windows user, note that this project does work fine on linux ubuntu
using Windows wsl2. If you would prefer not to use wsl2, and instead run this
directly on Windows, you will need to figure out how to do that. We have not
managed to get aintelope to build under Windows/Cygwin!

You need at least the following prerequisites:

- Python: Cygwin modules Python 3.7 including python3-devel (but no 3.9+
  available)
- Pytorch: https://github.com/KoichiYasuoka/CygTorch

## Known Issues

- Even with CygTorch pytorch-lightning can't find a Torch version.

# Differences to regular RL


For alignment and cognitive research, the internal reward of the agent and 
the actual score from the desired behaviour are measured separately. 
The reward comes from the agent.py itself, while the desired score comes from 
the environment (and thus the test). Both of these values are then recorded and 
compared during analysis.
