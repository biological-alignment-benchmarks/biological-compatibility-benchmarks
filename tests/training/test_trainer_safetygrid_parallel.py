import os
import sys
import pytest
from aintelope.config.config_utils import register_resolvers
from aintelope.__main__ import aintelope_main


def test_training_pipeline_main():
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-parallel-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "hparams.env_type=zoo",
    ]
    aintelope_main()
    sys.argv = [""]


@pytest.mark.parametrize("execution_number", range(10))
def test_training_pipeline_main_with_dead_agents(execution_number):
    # run all code in single process always in order to pass seed argument
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-parallel-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.env_params.seed=" + str(execution_number),
        "hparams.env_params.test_death=True",
    ]
    aintelope_main()
    sys.argv = [""]


def test_training_pipeline_baseline():
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.BASELINE_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-parallel-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_id=q_agent",
        "hparams.agent_params.target_instincts=[]",
    ]
    aintelope_main()
    sys.argv = [""]


@pytest.mark.parametrize("execution_number", range(10))
def test_training_pipeline_baseline_with_dead_agents(execution_number):
    # run all code in single process always in order to pass seed argument
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.BASELINE_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-parallel-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_id=q_agent",
        "hparams.agent_params.target_instincts=[]",
        "hparams.env_params.seed=" + str(execution_number),
        "hparams.env_params.test_death=True",
    ]
    aintelope_main()
    sys.argv = [""]


def test_training_pipeline_instinct():
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.INSTINCT_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-parallel-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_id=instinct_agent",
        "hparams.agent_params.target_instincts=[smell]",
    ]
    aintelope_main()
    sys.argv = [""]


@pytest.mark.parametrize("execution_number", range(10))
def test_training_pipeline_instinct_with_dead_agents(execution_number):
    # run all code in single process always in order to pass seed argument
    # TODO: find a way to parse Makefile and get sys.argv that way
    # sys.argv = [""] + shlex.split(const.INSTINCT_ARGS, comments=False, posix=True)
    # posix=True removes quotes around arguments
    sys.argv = [
        "",
        "hparams.env=savanna-safetygrid-parallel-v1",
        (
            "hparams.env_entry_point="
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "hparams.env_type=zoo",
        "hparams.agent_id=instinct_agent",
        "hparams.agent_params.target_instincts=[smell]",
        "hparams.env_params.seed=" + str(execution_number),
        "hparams.env_params.test_death=True",
    ]
    aintelope_main()
    sys.argv = [""]


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    register_resolvers()  # needs to be called only once
    pytest.main([__file__])  # run tests only in this file
