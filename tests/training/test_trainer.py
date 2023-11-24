import subprocess
import sys
import pytest
from aintelope.config.config_utils import register_resolvers

from tests.test_config import constants


def test_training_pipeline_main():
    if (
        sys.gettrace() is not None
    ):  # run all code in single process in case of debugging
        from aintelope.__main__ import aintelope_main

        aintelope_main()
    else:
        const = constants()
        ret = subprocess.run(["python", "-m", f"{const.PROJECT}"])
        assert ret.returncode == 0, "Trainer from __main__ caused an error!"


def test_training_pipeline_baseline():
    const = constants()
    if (
        sys.gettrace() is not None
    ):  # run all code in single process in case of debugging
        sys.argv = [
            "",
            "hparams.agent_id=q_agent",
            "hparams.agent_params.target_instincts=[]",
        ]
        from aintelope.__main__ import aintelope_main

        aintelope_main()
        sys.argv = [""]
    else:
        ret = subprocess.run(["make", f"{const.BASELINE}"])
        assert ret.returncode == 0, "Trainer baseline caused an error!"


def test_training_pipeline_instinct():
    const = constants()
    if (
        sys.gettrace() is not None
    ):  # run all code in single process in case of debugging
        sys.argv = [
            "",
            "hparams.agent_id=instinct_agent",
            "hparams.agent_params.target_instincts=[smell]",
        ]
        from aintelope.__main__ import aintelope_main

        aintelope_main()
        sys.argv = [""]
    else:
        ret = subprocess.run(["make", f"{const.INSTINCT}"])
        assert ret.returncode == 0, "Trainer baseline caused an error!"


if __name__ == "__main__" and sys.gettrace() is not None:  # detect debugging
    register_resolvers()  # needs to be called only once
    # running via pytest does not work in case of subprocesses here for some reason
    pytest.main([__file__])  # run tests only in this file
