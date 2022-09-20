# aintelope

We operationalize conjectures from Steven Byrnes’ 
[“Reverse-engineer human social instincts”](https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8/p/tj8AC3vhTnBywdZoA)
research program and extend existing research into brain-like AI. 
We simulate agents with reinforcement learning over selected cue-response patterns 
in environments that could give rise to humanl-iike complex behaviors. 
To do so we select cue-responses from a pre-existing list of more than 
60 candidates for human affects and instincts from affective neuroscience 
and other sources including original patterns. 
Cue-responses are conjectured to form hierarchies and the project will start 
the simulation of lower-level patterns first. 
We intend to verify the general principles and the ability of our software to 
model and simulate the agents and the environment to a sufficient degree.


Working document:
https://docs.google.com/document/d/1qc6a3MY2_guCZH8XJjutpaASNE7Zy6O5z1gblrfPemk/edit#

## Installation

### Poetry

Dependencies are managed via `poetry`. See installation instructions
[here](https://python-poetry.org/docs/#installation).

To install the python dependencies run `poetry install`.
To activate the environment via poetry run `poetry shell`. Alternatively, most
commands can also be executed without an active python environment via
`poetry run <command>`.

To execute the current trainer run `poetry run aintelope`.

Unfortunately, Poetry has a key outstanding incompatibility with PyTorch. 
PyTorch often needs you to specify a specific Cuda Version appropriate to your
local graphics card. The easiest way to fix this is to simply manually install
the correct version of PyTorch using pip. 
For example, Nathan needed to do the following to get PyTorch with Cuda version 113:
`pip3 install --use-deprecated=legacy-resolver torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

### Code formatting

To autoformat python files you can use [`black`](https://github.com/psf/black).
To automatically sort the imports you can run
[`isort .`](https://github.com/PyCQA/isort) from the root level of the project.

See the `Makefile` for further instructions.

## Running

Try `make run-training-short`.
Then look in `aintelope/checkpoints/memory_records`. There should be two new files
named `Record_{current timestamp}.csv` and `Record_{current timestamp}_plot..png`.
The plot will be an image of the path the agent took during the test episode, using
the best agent that the training produced. Green dots are food in the environment,
blue dots are water.


# Cygwin
If you are a Windows user, note that this project does work fine 
on linux ubuntu using Windows wsl2.
If you would prefer not to use wsl2, and instead run this directly on Windows,
you will need to figure out how to do that. We have not managed to get 
aintelope to build under Windows/Cygwin!

You need at least the following prerequisites:

* Python: Cygwin modules Python 3.7 including python3-devel (but no 3.9+ available)
* Pytorch: https://github.com/KoichiYasuoka/CygTorch

Problems:
* Windows Python versions don't work with the poetry commands.
* poetry shell doesn't work with paths with spaces.
* Even with CygTorch pytorch-lightning can't find a Torch version.
* other
