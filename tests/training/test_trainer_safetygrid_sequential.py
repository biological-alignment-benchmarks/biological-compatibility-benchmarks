# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import subprocess
import sys

import pytest

from aintelope.config.config_utils import register_resolvers
from aintelope.nonpipeline import aintelope_main
from tests.conftest import constants


def test_training_pipeline_main():
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-sequential-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.unit_test_mode=True",
        "hparams.num_episodes=5",
        "hparams.test_episodes=1",
        "hparams.env_params.num_iters=50",
        "hparams.warm_start_steps=10",
    ]
    aintelope_main()
    sys.argv = [""]


@pytest.mark.parametrize("execution_number", range(1))
def test_training_pipeline_main_with_dead_agents(execution_number):
    # run all code in single process always in order to pass seed argument
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-sequential-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.env_params.seed=" + str(execution_number),
        "hparams.env_params.test_death=True",
        "hparams.unit_test_mode=True",
        "hparams.num_episodes=5",
        "hparams.test_episodes=1",
        "hparams.env_params.num_iters=50",
        "hparams.warm_start_steps=10",
    ]
    aintelope_main()
    sys.argv = [""]


def test_training_pipeline_baseline():
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.BASELINE_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-sequential-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_class=q_agent",
        "hparams.agent_params.target_instincts=[]",
        "hparams.unit_test_mode=True",
        "hparams.num_episodes=5",
        "hparams.test_episodes=1",
        "hparams.env_params.num_iters=50",
        "hparams.warm_start_steps=10",
    ]
    aintelope_main()
    sys.argv = [""]


@pytest.mark.parametrize("execution_number", range(1))
def test_training_pipeline_baseline_with_dead_agents(execution_number):
    # run all code in single process always in order to pass seed argument
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.BASELINE_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-sequential-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_class=q_agent",
        "hparams.agent_params.target_instincts=[]",
        "hparams.env_params.seed=" + str(execution_number),
        "hparams.env_params.test_death=True",
        "hparams.unit_test_mode=True",
        "hparams.num_episodes=5",
        "hparams.test_episodes=1",
        "hparams.env_params.num_iters=50",
        "hparams.warm_start_steps=10",
    ]
    aintelope_main()
    sys.argv = [""]


def test_training_pipeline_instinct():
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.INSTINCT_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-sequential-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_class=instinct_agent",
        "hparams.agent_params.target_instincts=[food,drink,gold,silver,danger,predator,collision,antimovement]",
        "hparams.unit_test_mode=True",
        "hparams.num_episodes=5",
        "hparams.test_episodes=1",
        "hparams.env_params.num_iters=50",
        "hparams.warm_start_steps=10",
    ]
    aintelope_main()
    sys.argv = [""]


@pytest.mark.parametrize("execution_number", range(1))
def test_training_pipeline_instinct_with_dead_agents(execution_number):
    # run all code in single process always in order to pass seed argument
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.INSTINCT_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-sequential-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_class=instinct_agent",
        "hparams.agent_params.target_instincts=[food,drink,gold,silver,danger,predator,collision,antimovement]",
        "hparams.env_params.seed=" + str(execution_number),
        "hparams.env_params.test_death=True",
        "hparams.unit_test_mode=True",
        "hparams.num_episodes=5",
        "hparams.test_episodes=1",
        "hparams.env_params.num_iters=50",
        "hparams.warm_start_steps=10",
    ]
    aintelope_main()
    sys.argv = [""]


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
